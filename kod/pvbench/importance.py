"""Permutation importance of input channel groups (experiment E5).

For every plant x horizon x information set a model is fitted exactly as in the main benchmark (same
samples, same selected configuration; Ridge and LightGBM are refitted with the hyper-parameters stored
by the main run).  Then, on the TEST samples only, the channels of one group at a time are exchanged
between samples (whole 24-hour windows of that group, all its channels together) and the forecast error
is recomputed.  Samples are exchanged only with samples that have the same target hour of day (UTC), so
the permutation breaks the link between the group and the target but keeps every window physically
plausible for its hour.  Importance = nRMSE after permutation - nRMSE before (percentage points of rated
capacity), mean over `--repeats` permutations.

Groups: pv (power history), geometry (clear-sky irradiance at the origin and target hour, day of year),
irradiance (global, direct normal, diffuse), temperature, humidity, wind, cloud.

Run from the `kod` folder:
    python -m pvbench.importance --out <cikti>/e5_onem --runs <cikti> --worker 0
Several workers can run side by side; finished model runs are skipped when the command is repeated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataset, experiment, features, metrics

SOURCES = {"S1": ("ana", "ayar"), "S2": ("ana_icon", "ayar_icon")}      # information set -> (main run, tuning run)
GROUP_ORDER = ["pv", "geometry", "irradiance", "temperature", "humidity", "wind", "cloud"]
FIELDS = ["plant", "name", "horizon", "info", "model", "seed", "group", "n_channels", "n_test", "repeats",
          "base_nrmse", "perm_nrmse", "delta", "delta_sd", "main_nrmse", "finished"]


def group_of(channel: str) -> str:
    if channel in ("p", "p_ma"):
        return "pv"
    if channel in ("cs", "cs_lead", "doy_sin", "doy_cos"):
        return "geometry"
    for key, group in (("radiation", "irradiance"), ("irradiance", "irradiance"), ("temperature", "temperature"),
                       ("humidity", "humidity"), ("wind", "wind"), ("cloud", "cloud")):
        if key in channel:
            return group
    raise ValueError(f"channel without a group: {channel}")


def permutations(hours: np.ndarray, repeats: int, seed: int):
    """`repeats` index arrays; each permutes the samples within groups of equal hour of day."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(repeats):
        index = np.arange(len(hours))
        for hour in np.unique(hours):
            rows = np.flatnonzero(hours == hour)
            index[rows] = rng.permutation(rows)
        out.append(index)
    return out


def importance(predict, s, repeats, seed):
    """Rows (one per group) for a fitted model given as a function X -> raw forecasts."""
    X, y, cs_lead = s.X[s.test], s.y[s.test], s.aux["cs_lead"][s.test]

    def nrmse(inputs):
        return 100.0 * metrics.rmse(y, features.postprocess(predict(inputs), cs_lead))

    base = nrmse(X)
    hours = np.asarray(s.target_time[s.test].hour)
    perms = permutations(hours, repeats, seed)
    groups = [group_of(name) for name in s.names]
    rows = []
    for group in GROUP_ORDER:
        columns = [i for i, g in enumerate(groups) if g == group]
        if not columns:
            continue
        scores = []
        for index in perms:
            changed = X.copy()
            changed[:, :, columns] = X[index][:, :, columns]
            scores.append(nrmse(changed))
        rows.append({"group": group, "n_channels": len(columns), "n_test": int(len(y)), "repeats": repeats,
                     "base_nrmse": base, "perm_nrmse": float(np.mean(scores)),
                     "delta": float(np.mean(scores) - base), "delta_sd": float(np.std(scores, ddof=1))})
    return rows


def fitted_reference(model, s, config):
    """Ridge / LightGBM refitted with the hyper-parameters the main run selected on the validation set."""
    flat = experiment.flatten
    if model == "ridge":
        from sklearn.linear_model import Ridge
        fitted = Ridge(alpha=float(config["alpha"])).fit(flat(s.X[s.train]), s.y[s.train])
        return lambda X: fitted.predict(flat(X))
    import lightgbm as lgb
    params = {k: v for k, v in config.items() if k != "trees"}
    params["num_leaves"], params["min_child_samples"] = int(params["num_leaves"]), int(params["min_child_samples"])
    booster = lgb.train({"objective": "regression", "verbosity": -1, "seed": 0, "bagging_seed": 0,
                         "feature_fraction_seed": 0, "num_threads": 2, **params},
                        lgb.Dataset(flat(s.X[s.train]), s.y[s.train]), int(config["trees"]))
    return lambda X: booster.predict(flat(X))


def done_keys(out: Path):
    groups = {}                                   # a model run is done only if every channel group has its row
    for path in out.glob("importance_w*.csv"):
        try:
            with open(path, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("finished"):
                        key = (int(r["plant"]), int(r["horizon"]), r["info"], r["model"], int(r["seed"]))
                        groups.setdefault(key, set()).add(r["group"])
        except (OSError, ValueError, KeyError):
            pass
    return {key for key, found in groups.items() if len(found) >= len(GROUP_ORDER)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="permutation importance of channel groups")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", required=True, help="folder that holds the main and tuning runs (ana, ana_icon, ayar, ayar_icon)")
    ap.add_argument("--plants", type=int, nargs="*")
    ap.add_argument("--horizons", type=int, nargs="*", default=[1, 24])
    ap.add_argument("--info", nargs="*", default=["S1", "S2"])
    ap.add_argument("--models", nargs="*", default=["gru", "cnn_lstm_gru"])
    ap.add_argument("--references", nargs="*", default=["ridge", "lightgbm"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--p-cores", type=int, default=8)
    args = ap.parse_args(argv)
    experiment.boost_process(args.worker, args.p_cores)
    import torch
    torch.set_num_threads(1)

    out, runs = Path(args.out), Path(args.runs)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"importance_w{args.worker}.csv"

    def log(message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
        print(line, flush=True)
        with open(out / f"log_w{args.worker}.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write(rows):
        new = not path.exists()
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerows(rows)

    plants = dataset.load_plants()
    chosen = args.plants or list(plants.index)
    selected = {info: pd.read_csv(runs / SOURCES[info][1] / "selected.csv").set_index(["horizon", "info", "model"])
                for info in args.info}
    main_scores = {info: experiment.read_metrics(runs / SOURCES[info][0]) for info in args.info}
    log(f"start: worker {args.worker}, plants {chosen}, horizons {args.horizons}, info {args.info}")
    for no in chosen:
        for h in args.horizons:
            for info in args.info:
                if not experiment.claim(out, "imp", (no, h, info)):
                    continue
                done = done_keys(out)
                jobs = [(m, 0) for m in args.references] + [(m, seed) for m in args.models for seed in range(args.seeds)]
                jobs = [j for j in jobs if (no, h, info, j[0], j[1]) not in done]
                if not jobs:
                    continue
                table = dataset.load_table(no)
                s = features.prepare(table, h, info, features.protocol(table, no), experiment.LOOKBACK)
                pred_dir = runs / SOURCES[info][0] / "pred" / f"{no:02d}_h{h}_{info}"
                ms = main_scores[info]
                ms = ms[(ms["stage"] == "main") & (ms["plant"] == no) & (ms["horizon"] == h) & (ms["info"] == info)]
                for model, seed in jobs:
                    base = {"plant": no, "name": plants.loc[no, "name"], "horizon": h, "info": info, "model": model,
                            "seed": seed}
                    reference = ms[(ms["model"] == model) & (ms["seed"] == seed)]["nrmse"]
                    base["main_nrmse"] = float(reference.iloc[0]) if len(reference) else np.nan
                    perm_seed = 100000 * no + 100 * h + seed
                    if model in ("ridge", "lightgbm"):
                        config = json.loads((pred_dir / f"{model}_config.json").read_text())
                        rows = importance(fitted_reference(model, s, config), s, args.repeats, perm_seed)
                    else:
                        cfg = {k: selected[info].loc[(h, info, model), k] for k in experiment.DEFAULT}
                        cfg = {k: (int(v) if k == "batch_size" else float(v)) for k, v in cfg.items()}
                        _, fit = experiment.fit_neural(model, s, cfg, seed,
                                                       probe=lambda predict: importance(predict, s, args.repeats, perm_seed))
                        rows = fit["probe"]
                    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    write([{**base, **r, "finished": stamp} for r in rows])
                    log(f"plant {no:>2} h{h:<2} {info} {model:<13} seed {seed}  nRMSE {rows[0]['base_nrmse']:.2f} "
                        f"(main {base['main_nrmse']:.2f})  " + "  ".join(f"{r['group']} {r['delta']:+.2f}" for r in rows))
    log("finished")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
