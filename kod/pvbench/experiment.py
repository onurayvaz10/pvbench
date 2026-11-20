"""Benchmark runner: reference models, neural models, hyper-parameter search, result files.

Stages
------
tune   Random search on the validation set only (no test metric is computed or stored).
       The same list of candidate configurations is tried for every architecture on a fixed
       subset of plants; the configuration with the lowest mean validation RMSE is selected
       per architecture x horizon x information set.
main   Reference models (once) and every architecture with `seeds` random seeds per
       plant x horizon x information set, using the selected (or default) configuration.

Work is split into tasks (plant, horizon, information set). Several worker processes can
run side by side (`--worker k`, one process per worker number); each appends to its own CSV file and skips
what any worker has already finished, so an interrupted run is resumed by starting it again.

Run from the `kod` folder:  python -m pvbench.experiment --stage main --out <folder> ...
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "2")        # workers are bound to one core; more BLAS threads only contend

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import dataset, features, metrics  # noqa: E402

REFERENCES = ["persistence", "seasonal_persistence", "smart_persistence", "climatology", "ridge", "lightgbm"]
DEFAULT = {"width": 1.0, "dropout": 0.2, "learning_rate": 1e-3, "batch_size": 128}
SPACE = {"width": [0.5, 1.0, 2.0], "dropout": [0.0, 0.1, 0.2, 0.3],
         "learning_rate": [3e-4, 1e-3, 3e-3], "batch_size": [64, 128, 256]}
MAX_EPOCHS, PATIENCE, LOOKBACK = 100, 10, 24
EVAL_BATCH = 8192                        # samples per forward pass outside training (pooled runs lower it: GPU memory)
CONFIG_INFO = {"S1a": "S1"}              # ablation sets reuse the configuration selected for their parent set
CPU_MODELS = {"dense", "simplernn"}      # faster on one CPU core than on a GPU shared by several workers
MIN_SAMPLES = {"train": 1000, "val": 200, "test": 1000}
FIELDS = ["stage", "plant", "name", "protocol", "horizon", "info", "model", "seed", "trial", "status",
          "width", "dropout", "learning_rate", "batch_size", "n_features", "n_train", "n_val", "n_test",
          "params", "epochs", "best_epoch", "fit_s", "device", "val_rmse", "val_mae",
          "n", "rmse", "mae", "nrmse", "nmae", "nmbe", "r2", "theil_u2", "mase",
          "n_day", "rmse_day", "mae_day", "nrmse_day", "nmae_day", "nmbe_day", "r2_day", "theil_u2_day",
          "mase_day", "mape_02", "mape_05", "mape_10", "finished"]


def candidate_configs(n_trials, seed=2026):
    """Trial 0 is the default configuration; the rest are drawn without replacement from the grid."""
    grid = [dict(zip(SPACE, values)) for values in itertools.product(*SPACE.values())]
    grid = [g for g in grid if g != DEFAULT]
    order = np.random.default_rng(seed).permutation(len(grid))
    return [dict(DEFAULT)] + [grid[i] for i in order[: max(0, n_trials - 1)]]


ADDED_AFTER_TUNING = {40}    # replacement plant (selection rule 6); it joined after the tuning subset was drawn


def tuning_plants(plants: pd.DataFrame, protocols: dict, every=4):
    """Outcome-blind subset: plants with the common protocol sorted by latitude, every 4th one
    (spreads the subset over the climates covered by the study). Plants that joined the study after the
    subset had been drawn are left out, so the rule keeps returning the subset that was actually used."""
    common = [no for no in plants.sort_values("lat").index
              if protocols[no]["name"] == "common" and no not in ADDED_AFTER_TUNING]
    return common[::every]


def boost_process(worker: int, p_cores: int = 8):
    """Windows only. Background processes are moved to efficiency cores / throttled ("EcoQoS") after a
    few seconds, which slows training 5-10x. Opt out, raise the priority, bind the worker to one
    performance core (two logical CPUs) and keep the system awake while the worker runs."""
    if os.name != "nt":
        return
    import ctypes

    class PowerThrottling(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    handle = k.GetCurrentProcess()
    state = PowerThrottling(1, 0x1, 0)                                    # execution-speed throttling off
    k.SetProcessInformation(handle, 4, ctypes.byref(state), ctypes.sizeof(state))
    k.SetPriorityClass(handle, 0x00008000)                                # ABOVE_NORMAL_PRIORITY_CLASS
    mask = os.environ.get("PVBENCH_AFFINITY")            # extra workers on efficiency cores: hex CPU mask
    if mask:
        k.SetProcessAffinityMask(handle, int(mask, 16))
    elif p_cores > 0:
        k.SetProcessAffinityMask(handle, 0b11 << (2 * (worker % p_cores)))
    k.SetThreadExecutionState(0x80000000 | 0x00000001)                    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def claim(out: Path, stage: str, task) -> bool:
    """Atomic claim of a task by one worker (the launcher clears old claims before a run)."""
    folder = Path(out) / "claims"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with open(folder / f"{stage}_{task[0]:02d}_h{task[1]}_{task[2]}.txt", "x") as f:
            f.write(f"pid {os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    except FileExistsError:
        return False


def release(out: Path, stage: str, task):
    """Give a claimed task back (used when this process cannot finish it)."""
    try:
        (Path(out) / "claims" / f"{stage}_{task[0]:02d}_h{task[1]}_{task[2]}.txt").unlink()
    except OSError:
        pass


class GpuFailure(RuntimeError):
    """The accelerator context of this process is unusable: every later fit in it would fail as well.
    No result row is written; `main` gives the task back and continues in a fresh process."""


# ----------------------------------------------------------------------------- result files

class Results:
    def __init__(self, out: Path, worker: int):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / f"metrics_w{worker}.csv"
        self.done = set()
        for path in self.out.glob("metrics_w*.csv"):
            self.done |= self._keys(path)

    @staticmethod
    def key(row):
        return (row["stage"], int(row["plant"]), int(row["horizon"]), row["info"], row["model"],
                int(row["seed"]), int(row["trial"]))

    def _keys(self, path):
        try:
            with open(path, encoding="utf-8", newline="") as f:
                return {self.key(r) for r in csv.DictReader(f) if r.get("finished")}
        except (OSError, ValueError, KeyError, TypeError):
            return set()

    def refresh(self):
        for path in self.out.glob("metrics_w*.csv"):
            if path != self.path:
                self.done |= self._keys(path)

    def write(self, row):
        row = {**row, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        new = not self.path.exists()
        for attempt in range(60):                       # tolerate short file locks (sync tools, viewers)
            try:
                with open(self.path, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                    if new:
                        writer.writeheader()
                    writer.writerow(row)
                break
            except OSError:
                time.sleep(5)
        self.done.add(self.key(row))


def read_metrics(out) -> pd.DataFrame:
    # Sorted, so that row order does not depend on the file system (Linux glob order is arbitrary).
    parts = [pd.read_csv(p) for p in sorted(Path(out).glob("metrics_w*.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=FIELDS)


# ----------------------------------------------------------------------------- models

def flatten(X):
    return X.reshape(len(X), -1)


def fit_ridge(s):
    from sklearn.linear_model import Ridge
    best = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha).fit(flatten(s.X[s.train]), s.y[s.train])
        pred = model.predict(flatten(s.X))
        score = metrics.rmse(s.y[s.val], features.postprocess(pred, s.aux["cs_lead"])[s.val])
        if best is None or score < best[0]:
            best = (score, pred, {"alpha": alpha})
    return best[1], best[2]


def fit_lightgbm(s, n_trials=12, seed=0):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    space = {"learning_rate": [0.03, 0.05, 0.1], "num_leaves": [15, 31, 63], "min_child_samples": [20, 50, 100],
             "feature_fraction": [0.6, 0.8, 1.0], "lambda_l2": [0.0, 1.0, 10.0]}
    configs = [{"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "feature_fraction": 0.8,
                "lambda_l2": 0.0}]
    configs += [{k: v[rng.integers(len(v))] for k, v in space.items()} for _ in range(n_trials - 1)]
    Xf = flatten(s.X)
    train = lgb.Dataset(Xf[s.train], s.y[s.train], free_raw_data=False)
    val = lgb.Dataset(Xf[s.val], s.y[s.val], reference=train, free_raw_data=False)
    best = None
    for cfg in configs:
        booster = lgb.train({"objective": "regression", "verbosity": -1, "seed": seed, "bagging_seed": seed,
                             "feature_fraction_seed": seed, "num_threads": 2, **cfg}, train, 2000,
                            valid_sets=[val], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = booster.predict(Xf, num_iteration=booster.best_iteration)
        score = metrics.rmse(s.y[s.val], features.postprocess(pred, s.aux["cs_lead"])[s.val])
        if best is None or score < best[0]:
            best = (score, pred, {**cfg, "trees": booster.best_iteration})
    return best[1], best[2]


def fit_neural(name, s, cfg, seed, probe=None):
    """Train one network; returns predictions for all samples and training information.

    `probe`, if given, is called with a function X -> raw forecasts of the trained network before the
    network is discarded; its return value is stored under info["probe"] (used for permutation importance)."""
    import keras
    import torch
    from .models import build_model

    # The architecture is a Keras model; the optimisation loop is plain PyTorch (data kept on the
    # device), which is several times faster than `model.fit` for networks of this size.
    # Adam, MSE loss, global gradient-norm clipping at 1.0, early stopping on the validation loss
    # (patience PATIENCE, best weights restored).
    device = "cuda" if torch.cuda.is_available() and name not in CPU_MODELS else "cpu"
    with keras.device(device):
        return _fit_neural(name, s, cfg, seed, device, probe)


def _fit_neural(name, s, cfg, seed, device, probe=None):
    import keras
    import torch
    from .models import build_model

    keras.utils.set_random_seed(seed)
    model = build_model(name, s.X.shape[1], s.X.shape[2], cfg["width"], cfg["dropout"], compile_model=False)
    trainable = [v.value for v in model.trainable_variables]
    optimizer = torch.optim.Adam(trainable, lr=float(cfg["learning_rate"]))
    X_train = torch.as_tensor(s.X[s.train], device=device)
    y_train = torch.as_tensor(s.y[s.train], device=device).unsqueeze(1)
    X_val = torch.as_tensor(s.X[s.val], device=device)
    y_val = torch.as_tensor(s.y[s.val], device=device).unsqueeze(1)
    generator = torch.Generator(device=device).manual_seed(seed)
    batch, n_train = int(cfg["batch_size"]), len(X_train)
    losses, best, best_weights = [], np.inf, None
    started = time.time()
    for epoch in range(MAX_EPOCHS):
        order = torch.randperm(n_train, device=device, generator=generator)
        for start in range(0, n_train, batch):
            rows = order[start:start + batch]
            loss = torch.nn.functional.mse_loss(model(X_train[rows], training=True), y_train[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        with torch.no_grad():                # one pass up to EVAL_BATCH samples (every single-plant run), else chunks
            out = torch.cat([model(X_val[i:i + EVAL_BATCH], training=False)
                             for i in range(0, max(len(X_val), 1), EVAL_BATCH)])
            val = float(torch.nn.functional.mse_loss(out, y_val))
        losses.append(val)
        if not np.isfinite(val):
            break
        if val < best:
            best, best_weights = val, model.get_weights()
        elif epoch + 1 - (int(np.argmin(losses)) + 1) >= PATIENCE:
            break
    if best_weights is None:
        raise FloatingPointError("validation loss is not finite")
    model.set_weights(best_weights)
    fit_s = time.time() - started
    def predict(X):
        with torch.no_grad():                # chunks are moved to the device one at a time (memory of pooled runs)
            out = torch.cat([model(torch.as_tensor(X[i:i + EVAL_BATCH], device=device), training=False)
                             for i in range(0, len(X), EVAL_BATCH)])
            return out.squeeze(1).cpu().numpy()

    pred = predict(s.X)
    info = {"params": model.count_params(), "epochs": len(losses), "best_epoch": int(np.argmin(losses)) + 1,
            "fit_s": round(fit_s, 1), "device": device}
    if probe is not None:
        info["probe"] = probe(predict)
    del optimizer, X_train, y_train, X_val, y_val
    del model, predict
    keras.backend.clear_session()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, info


_MATCHED = {}


def matched_width(name, target, n_features):
    """Width multiplier that brings architecture `name` closest to `target` parameters (cached per worker)."""
    key = (name, int(target), int(n_features))
    if key not in _MATCHED:
        import keras
        from .models import match_width

        with keras.device("cpu"):
            _MATCHED[key] = float(match_width(name, int(target), LOOKBACK, int(n_features))[0])
        keras.backend.clear_session()
    return _MATCHED[key]


# ----------------------------------------------------------------------------- one task

def score(s, pred, with_test):
    pred = features.postprocess(pred, s.aux["cs_lead"])
    out = {"val_rmse": metrics.rmse(s.y[s.val], pred[s.val]), "val_mae": metrics.mae(s.y[s.val], pred[s.val])}
    if with_test:
        naive = features.postprocess(s.aux["persistence"], s.aux["cs_lead"])
        out.update(metrics.evaluate(s.y[s.test], pred[s.test], 1.0, daytime=s.aux["daytime"][s.test],
                                    naive=naive[s.test], scale=s.mase_scale))
    return out, pred


def run_task(stage, plant_no, plant, horizon, info, jobs, results: Results, out: Path, log):
    """jobs: list of (model, seed, trial, config-or-None)."""
    base = {"stage": stage, "plant": plant_no, "name": plant["name"], "horizon": horizon, "info": info}
    todo = [j for j in jobs if (stage, plant_no, horizon, info, j[0], j[1], j[2]) not in results.done]
    if not todo:
        return
    table = dataset.load_table(plant_no)
    periods = features.protocol(table, plant_no)
    s = features.prepare(table, horizon, info, periods, LOOKBACK)
    counts = {"n_train": int(s.train.sum()), "n_val": int(s.val.sum()), "n_test": int(s.test.sum())}
    base.update({"protocol": periods["name"], "n_features": s.X.shape[2], **counts})
    if any(counts[f"n_{k}"] < v for k, v in MIN_SAMPLES.items()):
        for model, seed, trial, _ in todo:
            results.write({**base, "model": model, "seed": seed, "trial": trial, "status": "too_few_samples"})
        log(f"skip  plant {plant_no} h{horizon} {info}: too few samples {counts}")
        return
    with_test = stage == "main"
    pred_dir = out / "pred" / f"{plant_no:02d}_h{horizon}_{info}"
    if with_test:
        pred_dir.mkdir(parents=True, exist_ok=True)
        target = pd.DataFrame({"y": s.y[s.test], "cs_lead": s.aux["cs_lead"][s.test],
                               "daytime": s.aux["daytime"][s.test]}, index=s.target_time[s.test])
        target.to_parquet(pred_dir / "target.parquet")
    for model, seed, trial, cfg in todo:
        row = {**base, "model": model, "seed": seed, "trial": trial, "status": "ok"}
        started = time.time()
        try:
            if model in ("persistence", "seasonal_persistence", "smart_persistence", "climatology"):
                pred, extra = s.aux[model], {}
            elif model == "ridge":
                pred, extra = fit_ridge(s)
            elif model == "lightgbm":
                pred, extra = fit_lightgbm(s)
            else:
                if cfg.get("match_params"):     # parameter-matched run: only the width is replaced
                    cfg = {**cfg, "width": matched_width(model, cfg["match_params"], s.X.shape[2])}
                pred, extra = fit_neural(model, s, cfg, seed)
                row.update(cfg)
                row.update(extra)
            row.setdefault("fit_s", round(time.time() - started, 1))
            scores, pred = score(s, pred, with_test)
            row.update(scores)
            if with_test:
                np.save(pred_dir / f"{model}_s{seed}.npy", pred[s.test].astype(np.float32))
                if model in ("ridge", "lightgbm"):
                    (pred_dir / f"{model}_config.json").write_text(json.dumps(extra, default=float))
        except Exception as error:                                  # keep the run alive, record the failure
            if type(error).__name__ == "AcceleratorError" or "CUDA" in str(error):
                raise GpuFailure(f"{type(error).__name__}: {error}"[:200]) from error
            row["status"] = f"error: {type(error).__name__}: {error}"[:300].replace("\n", " ")
        results.write(row)
        shown = row.get("nrmse", "")
        log(f"{stage} plant {plant_no:>2} h{horizon:<2} {info} {model:<20} seed {seed} trial {trial}  "
            f"val {row.get('val_rmse', float('nan')):.4f}  "
            f"{'nRMSE ' + format(shown, '.2f') if shown != '' else ''}  {row.get('epochs', '')} ep  "
            f"{row.get('fit_s', '')} s  {row['status'] if row['status'] != 'ok' else ''}")


# ----------------------------------------------------------------------------- selection

def select_configs(tune_out, plants, horizons, infos, models, n_trials, wait=True, log=print):
    """Best trial per (horizon, info, model) by mean validation RMSE over the tuning plants."""
    path = Path(tune_out) / "selected.csv"
    expected = len(plants) * len(horizons) * len(infos) * len(models) * n_trials
    while True:
        if path.exists():
            return pd.read_csv(path)
        df = read_metrics(tune_out)
        df = df[(df["stage"] == "tune") & df["model"].isin(models)] if len(df) else df
        if len(df) >= expected:
            break
        if not wait:
            raise RuntimeError(f"tuning incomplete: {len(df)}/{expected} runs")
        log(f"waiting for tuning: {len(df)}/{expected} runs finished")
        time.sleep(120)
    usable = df[df["status"] == "ok"]
    mean = usable.groupby(["horizon", "info", "model", "trial"]).agg(
        val_rmse=("val_rmse", "mean"), n_plants=("val_rmse", "size"),
        **{k: (k, "first") for k in DEFAULT}).reset_index()
    mean = mean[mean["n_plants"] == mean.groupby(["horizon", "info", "model"])["n_plants"].transform("max")]
    best = mean.loc[mean.groupby(["horizon", "info", "model"])["val_rmse"].idxmin()].reset_index(drop=True)
    mean.to_csv(Path(tune_out) / "tuning_trials.csv", index=False)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    best.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
    except OSError:
        pass
    return pd.read_csv(path)


# ----------------------------------------------------------------------------- command line

def main(argv=None):
    from .models import MODEL_NAMES

    ap = argparse.ArgumentParser(description="PV forecasting benchmark runner")
    ap.add_argument("--stage", choices=["tune", "main"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plants", type=int, nargs="*")
    ap.add_argument("--models", nargs="*", default=MODEL_NAMES)
    ap.add_argument("--no-references", action="store_true")
    ap.add_argument("--horizons", type=int, nargs="*", default=[1, 24])
    ap.add_argument("--info", nargs="*", default=["S1", "S2"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--match-params", type=int, default=0,
                    help="main stage: scale every architecture to about this many parameters (other settings unchanged)")
    ap.add_argument("--tuning", help="folder of a finished/running tune stage; main waits for it and uses its selection")
    ap.add_argument("--worker", type=int, default=0, help="worker number (file names, core binding)")
    ap.add_argument("--p-cores", type=int, default=8, help="performance cores to bind workers to (0 = no binding)")
    args = ap.parse_args(argv)
    boost_process(args.worker, args.p_cores)
    import torch
    torch.set_num_threads(1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"log_w{args.worker}.txt"

    def log(message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
        print(line, flush=True)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    plants = dataset.load_plants()
    protocols = {no: features.protocol(dataset.load_table(no), no) for no in plants.index}
    tuners = tuning_plants(plants, protocols)
    configs = candidate_configs(args.trials)
    if args.stage == "tune":
        chosen = args.plants or tuners
        jobs_for = lambda h, info: [(m, 0, t, cfg) for m in args.models for t, cfg in enumerate(configs)]  # noqa: E731
    else:
        chosen = args.plants or list(plants.index)
        selected = None
        if args.tuning:
            selected = select_configs(args.tuning, tuners, args.horizons, args.info, args.models, args.trials, log=log)
            selected = selected.set_index(["horizon", "info", "model"])

        def jobs_for(h, info):
            jobs = [] if args.no_references else [(m, 0, 0, None) for m in REFERENCES]
            for m in args.models:
                cfg = dict(DEFAULT)
                # horizons that were not tuned (multi-horizon curve) use the configuration selected
                # for the nearest tuned horizon of the same information set and model
                source = CONFIG_INFO.get(info, info)
                tuned = [] if selected is None else sorted(k[0] for k in selected.index if k[1:] == (source, m))
                if tuned:
                    nearest = min(tuned, key=lambda x: (abs(x - h), x))
                    cfg = {k: selected.loc[(nearest, source, m), k] for k in DEFAULT}
                    cfg = {k: (int(v) if k == "batch_size" else float(v)) for k, v in cfg.items()}
                if args.match_params:
                    cfg["match_params"] = args.match_params
                jobs +=[(m, seed, 0, cfg) for seed in range(args.seeds)]
            return jobs

    tasks = [(no, h, info) for no in chosen for h in args.horizons for info in args.info]
    results = Results(out, args.worker)
    log(f"start stage {args.stage}: worker {args.worker}, {len(tasks)} tasks, plants {chosen}, "
        f"tuning plants {tuners}, device {'cuda' if torch.cuda.is_available() else 'cpu'}")
    for task in tasks:
        if not claim(out, args.stage, task):
            continue
        results.refresh()
        no, h, info = task
        try:
            run_task(args.stage, no, plants.loc[no], h, info, jobs_for(h, info), results, out, log)
        except GpuFailure as failure:
            # A broken accelerator context fails every later fit at once.  Nothing was written for the failed
            # fit; the task is given back and the same command continues in a child process (new context).
            import subprocess
            release(out, args.stage, task)
            restarts = int(os.environ.get("PVBENCH_RESTARTS", "0"))
            log(f"GPU failure in plant {no} h{h} {info}: {failure}  (restart {restarts + 1} of 3)")
            if restarts >= 3:
                raise SystemExit(1)
            command = [sys.executable, "-W", "ignore", "-m", "pvbench.experiment",
                       *(list(argv) if argv is not None else sys.argv[1:])]
            raise SystemExit(subprocess.call(command, env={**os.environ, "PVBENCH_RESTARTS": str(restarts + 1)}))
    log("finished")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
