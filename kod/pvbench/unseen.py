"""Unseen-plant experiment (E6): one pooled ("global") model for many plants, tested on plants it never saw.

Design
------
Training plants  The plants with the common protocol (training to Aug 2024, validation Sep - Nov 2024).
                 Plants with a short record (own protocol) never enter training: their training period
                 overlaps the test period of the other plants.
Folds            Training plants closer than 25 km to each other form one site group and are always left out
                 together.  The groups are left out one at a time (`--folds 0`, fold number = lowest plant
                 number of the group) or in K folds (`--folds K`: groups sorted by latitude, fold = rank mod K,
                 numbered 1..K, so every fold spans the latitude range).
                 For every fold one model is fitted on the pooled training samples of the remaining plants;
                 early stopping and model selection use their pooled validation samples.  Fold 0 is one
                 further model fitted on all training plants (nothing left out).
Test             Every plant on its own test period, with exactly the samples of the main benchmark:
                   unseen  plants of the left-out fold
                   short   short-record plants (unseen in every fold; fold 0 is the one to report)
                   seen    plants whose training period was part of the pooled training set
                 No training or validation sample is later than the start of any test period.
Inputs           As in the main benchmark.  Power is already in units of rated capacity; the weather channels
                 are standardised with the mean and standard deviation of the pooled training samples of the
                 fold (the same numbers for every plant, so the physical scale is shared between plants).
Settings         Networks use the candidate configurations and validation scores of the single-plant benchmark
                 (same horizon, information set and architecture); for every fold the configuration is
                 re-selected WITHOUT the tuning plants that the fold leaves out (`fold_configs`), so neither
                 the fit nor the model selection has seen a tested plant.  The batch size is `--batch-size`
                 (default 1024) because the pooled training set is about twenty times larger.  Ridge and
                 LightGBM run the same search as in the benchmark on the pooled data.

Run from the `kod` folder:
    python -m pvbench.unseen --out <cikti>/e6 --runs <cikti> --folds 5 --models gru cnn_lstm_gru --seeds 3 --worker 0
Several workers can run side by side; finished fits are skipped when the command is repeated.
Results: <out>/metrics_w*.csv (one row per fit x plant) and <out>/pred/<plant>_h<h>_<info>/<model>_f<fold>_s<seed>.npy
(test forecasts of the plants that were not in the training set).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataset, experiment, features, metrics

TUNING = {"S1": "ayar", "S2": "ayar_icon"}          # information set -> tuning run of the single-plant benchmark
REFERENCES = ("ridge", "lightgbm")
ALL = 0                                             # fold number of the model fitted on all training plants
N_BASE = 5                                          # p, clear sky (2), day of year (2): not standardised
CLUSTER_KM = 25.0                                   # plants closer than this are left out together
EVAL_BATCH = 2048                                   # forward-pass chunk outside training (GPU memory, 8 workers)
FIELDS = ["fold", "plant", "name", "role", "protocol", "horizon", "info", "model", "seed", "status",
          "n_plants_train", "n_train", "n_val", "n_test", "n_features", "width", "dropout", "learning_rate",
          "batch_size", "n_tune_plants", "tune_trial", "params", "epochs", "best_epoch", "fit_s", "device", "val_rmse",
          "n", "rmse", "mae", "nrmse", "nmae", "nmbe", "r2", "theil_u2", "mase",
          "n_day", "rmse_day", "mae_day", "nrmse_day", "nmae_day", "nmbe_day", "r2_day", "theil_u2_day",
          "mase_day", "mape_02", "mape_05", "mape_10", "finished"]


def distance_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 6371.0 * np.arcsin(np.sqrt(a)))


def site_groups(plants: pd.DataFrame, numbers) -> list:
    """Plants closer than CLUSTER_KM to each other (single linkage) form one group, so neighbouring plants
    are always left out together.  Groups are sorted by their mean latitude."""
    groups = []
    for no in numbers:
        near = [g for g in groups if any(distance_km(plants.loc[no, "lat"], plants.loc[no, "lon"],
                                                     plants.loc[o, "lat"], plants.loc[o, "lon"]) <= CLUSTER_KM for o in g)]
        groups = [g for g in groups if g not in near] + [sorted([int(no)] + [o for g in near for o in g])]
    return sorted(groups, key=lambda g: (float(plants.loc[g, "lat"].mean()), g[0]))


def make_folds(plants: pd.DataFrame, protocols: dict, k: int) -> dict:
    """fold number -> plants left out.  Outcome-blind: only the protocol and the coordinates are used."""
    common = [no for no in plants.sort_values("lat").index if protocols[no]["name"] == "common"]
    groups = site_groups(plants, common)
    if k <= 0:
        folds = {group[0]: group for group in groups}
    else:
        folds = {i + 1: sorted(no for group in groups[i::k] for no in group) for i in range(k)}
    return {ALL: [], **folds}


def load_samples(numbers, horizon, info):
    """Samples of every plant with the weather channels in physical units."""
    out = {}
    for no in numbers:
        table = dataset.load_table(no)
        out[no] = features.prepare(table, horizon, info, features.protocol(table, no), experiment.LOOKBACK,
                                   standardise=False)
    return out


def pooled_statistics(samples, train_plants):
    """Mean and standard deviation of every weather channel over the pooled training windows."""
    total = square = None
    count = 0
    for no in train_plants:
        s = samples[no]
        values = s.X[s.train][:, :, N_BASE:].astype(np.float64).reshape(-1, s.X.shape[2] - N_BASE)
        total = values.sum(axis=0) if total is None else total + values.sum(axis=0)
        square = (values ** 2).sum(axis=0) if square is None else square + (values ** 2).sum(axis=0)
        count += len(values)
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean ** 2, 0.0))
    return mean.astype(np.float32), np.where(std > 0, std, 1.0).astype(np.float32)


def build(samples, train_plants):
    """Pooled sample set: training rows, validation rows (training plants), then the test rows of all plants.

    Returns (pooled Samples, number of fit rows, {plant: (first, last) row of its test block})."""
    mean, std = pooled_statistics(samples, train_plants)

    def part(no, mask):
        s = samples[no]
        X = s.X[mask].copy()
        X[:, :, N_BASE:] = (X[:, :, N_BASE:] - mean) / std
        return X, s.y[mask], s.aux["cs_lead"][mask]

    blocks = [part(no, samples[no].train) for no in train_plants]
    n_train = sum(len(b[1]) for b in blocks)
    blocks += [part(no, samples[no].val) for no in train_plants]
    n_fit = sum(len(b[1]) for b in blocks)
    spans, position = {}, n_fit
    for no in samples:
        blocks.append(part(no, samples[no].test))
        spans[no] = (position, position + len(blocks[-1][1]))
        position = spans[no][1]
    X, y, cs_lead = (np.concatenate([b[i] for b in blocks]) for i in range(3))
    rows = np.arange(len(y))
    first = samples[train_plants[0]]
    pooled = features.Samples(X=X, y=y, target_time=None, train=rows < n_train, val=(rows >= n_train) & (rows < n_fit),
                              test=rows >= n_fit, names=first.names, periods={}, aux={"cs_lead": cs_lead})
    return pooled, n_fit, spans


def fit_view(pooled, n_fit):
    """The fit rows only (networks are trained on this view; the test rows are predicted block by block)."""
    return features.Samples(X=pooled.X[:n_fit], y=pooled.y[:n_fit], target_time=None, train=pooled.train[:n_fit],
                            val=pooled.val[:n_fit], test=pooled.test[:n_fit], names=pooled.names, periods={},
                            aux={"cs_lead": pooled.aux["cs_lead"][:n_fit]})


def plant_scores(s, pred):
    """Test metrics of one plant (same post-processing and metrics as the main benchmark)."""
    cs_lead = s.aux["cs_lead"][s.test]
    pred = features.postprocess(pred, cs_lead)
    naive = features.postprocess(s.aux["persistence"][s.test], cs_lead)
    return metrics.evaluate(s.y[s.test], pred, 1.0, daytime=s.aux["daytime"][s.test], naive=naive,
                            scale=s.mase_scale), pred


def done_keys(out: Path, n_plants: int):
    """Fits whose rows are complete (one scored row per plant); anything else is fitted again."""
    counts = {}
    for path in out.glob("metrics_w*.csv"):
        try:
            with open(path, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("finished") and r.get("status") == "ok":
                        key = (int(r["fold"]), int(r["horizon"]), r["info"], r["model"], int(r["seed"]))
                        counts.setdefault(key, set()).add(int(r["plant"]))
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return {key for key, plants in counts.items() if len(plants) >= n_plants}


def fold_configs(runs: Path, info: str, models, left_out) -> dict:
    """(horizon, model) -> configuration, selected as in the benchmark (lowest mean validation RMSE over the
    tuning plants) but WITHOUT the tuning plants that are left out in this fold, so that no step of model
    selection has seen a plant the fold is tested on.  With nothing left out this is the benchmark's selection."""
    df = experiment.read_metrics(runs / TUNING[info])
    df = df[(df["stage"] == "tune") & (df["status"] == "ok") & (df["info"] == info) & df["model"].isin(models)
            & ~df["plant"].isin(list(left_out))]
    mean = df.groupby(["horizon", "model", "trial"]).agg(
        val_rmse=("val_rmse", "mean"), n_plants=("val_rmse", "size"),
        **{k: (k, "first") for k in experiment.DEFAULT}).reset_index()
    mean = mean[mean["n_plants"] == mean.groupby(["horizon", "model"])["n_plants"].transform("max")]
    best = mean.loc[mean.groupby(["horizon", "model"])["val_rmse"].idxmin()]
    return {(int(r["horizon"]), r["model"]): {**{k: (int(r[k]) if k == "batch_size" else float(r[k])) for k in experiment.DEFAULT},
                                             "n_tune_plants": int(r["n_plants"]), "tune_trial": int(r["trial"])}
            for _, r in best.iterrows()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="pooled model tested on unseen plants")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", required=True, help="folder that holds the tuning runs of the benchmark (ayar, ayar_icon)")
    ap.add_argument("--folds", type=int, default=0, help="0 = leave one plant out; K = K groups by latitude rank")
    ap.add_argument("--only-folds", type=int, nargs="*", help="run these fold numbers only")
    ap.add_argument("--horizons", type=int, nargs="*", default=[1, 24])
    ap.add_argument("--info", nargs="*", default=["S1", "S2"])
    ap.add_argument("--models", nargs="*", default=["ridge", "lightgbm", "gru", "tcn", "cnn_lstm_gru"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--p-cores", type=int, default=8)
    args = ap.parse_args(argv)
    experiment.boost_process(args.worker, args.p_cores)
    experiment.EVAL_BATCH = EVAL_BATCH
    if any(m not in REFERENCES for m in args.models):
        import torch
        torch.set_num_threads(1)

    out, runs = Path(args.out), Path(args.runs)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"metrics_w{args.worker}.csv"

    def log(message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
        print(line, flush=True)
        try:
            with open(out / f"log_w{args.worker}.txt", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def write(rows):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        new = not path.exists()
        for attempt in range(60):                       # tolerate short file locks (sync tools, viewers)
            try:
                with open(path, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                    if new:
                        writer.writeheader()
                    writer.writerows({**r, "finished": stamp} for r in rows)
                return
            except OSError:
                time.sleep(5)

    plants = dataset.load_plants()
    protocols = {no: features.protocol(dataset.load_table(no), no) for no in plants.index}
    folds = make_folds(plants, protocols, args.folds)
    common = [int(no) for no in plants.index if protocols[no]["name"] == "common"]
    chosen = [f for f in folds if not args.only_folds or f in args.only_folds]
    neural = [m for m in args.models if m not in REFERENCES]
    selected = {(info, fold): fold_configs(runs, info, neural, folds[fold]) for info in args.info for fold in chosen} if neural else {}
    log(f"start: worker {args.worker}, folds {chosen}, horizons {args.horizons}, info {args.info}, models {args.models}")
    for h in args.horizons:
        for info in args.info:
            samples = None
            for fold in chosen:
                for model in args.models:
                    if not experiment.claim(out, f"e6-{model}", (fold, h, info)):
                        continue
                    seeds = [0] if model in REFERENCES else list(range(args.seeds))
                    done = done_keys(out, len(plants))
                    seeds = [seed for seed in seeds if (fold, h, info, model, seed) not in done]
                    if not seeds:
                        continue
                    if samples is None:
                        samples = load_samples(list(plants.index), h, info)
                    train_plants = [no for no in common if no not in folds[fold]]
                    latest = max(samples[no].periods["test_start"] for no in train_plants)
                    if not all(samples[no].periods["test_start"] >= latest for no in samples):
                        raise RuntimeError("a test period starts before the end of the pooled training/validation data")
                    pooled, n_fit, spans = build(samples, train_plants)
                    base = {"fold": fold, "horizon": h, "info": info, "model": model, "n_plants_train": len(train_plants),
                            "n_train": int(pooled.train.sum()), "n_val": int(pooled.val.sum()),
                            "n_features": pooled.X.shape[2]}
                    for seed in seeds:
                        row = {**base, "seed": seed, "status": "ok"}
                        started = time.time()
                        try:
                            if model in REFERENCES:
                                fit = experiment.fit_ridge if model == "ridge" else experiment.fit_lightgbm
                                pred, extra = fit(pooled)
                                row["fit_s"] = round(time.time() - started, 1)
                                val, test = pred[pooled.val], pred[n_fit:]
                                val_cs = pooled.aux["cs_lead"][pooled.val]
                                log(f"  {model} selected {extra}")
                            else:
                                cfg = dict(selected[(info, fold)][(h, model)])       # selection without the left-out plants
                                cfg["batch_size"] = args.batch_size
                                view = fit_view(pooled, n_fit)
                                pred, extra = experiment.fit_neural(model, view, cfg, seed,
                                                                    probe=lambda predict: predict(pooled.X[n_fit:]))
                                test = extra.pop("probe")
                                row.update(cfg)
                                row.update(extra)
                                val, val_cs = pred[view.val], view.aux["cs_lead"][view.val]
                            row["val_rmse"] = metrics.rmse(pooled.y[pooled.val], features.postprocess(val, val_cs))
                            rows, shown = [], {"unseen": [], "short": [], "seen": []}
                            for no, (a, b) in spans.items():
                                role = "seen" if no in train_plants else ("unseen" if no in folds[fold] else "short")
                                scores, forecast = plant_scores(samples[no], test[a - n_fit:b - n_fit])
                                rows.append({**row, "plant": no, "name": plants.loc[no, "name"], "role": role,
                                             "protocol": protocols[no]["name"], "n_test": b - a, **scores})
                                shown[role].append(scores["nrmse"])
                                if role != "seen":
                                    folder = out / "pred" / f"{no:02d}_h{h}_{info}"
                                    folder.mkdir(parents=True, exist_ok=True)
                                    np.save(folder / f"{model}_f{fold}_s{seed}.npy", forecast.astype(np.float32))
                            write(rows)
                            log(f"fold {fold:>2} h{h:<2} {info} {model:<13} seed {seed}  val {row['val_rmse']:.4f}  "
                                + "  ".join(f"{k} {np.mean(v):.2f} (n={len(v)})" for k, v in shown.items() if v)
                                + f"  {row.get('epochs', '')} ep  {row.get('fit_s', '')} s")
                        except Exception as error:                  # keep the run alive, record the failure
                            if type(error).__name__ == "AcceleratorError" or "CUDA" in str(error):
                                experiment.release(out, f"e6-{model}", (fold, h, info))   # broken GPU context: stop this
                                log(f"GPU failure in fold {fold} h{h} {info} {model}: {error}"[:300])   # worker; the
                                raise SystemExit(1)                                       # launcher runs it again
                            row["status"] = f"error: {type(error).__name__}: {error}"[:300].replace("\n", " ")
                            write([{**row, "plant": 0, "role": ""}])
                            log(f"fold {fold} h{h} {info} {model} seed {seed}  {row['status']}")
                    del pooled
    log("finished")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
