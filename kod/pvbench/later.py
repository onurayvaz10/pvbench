"""Second test year (experiment E10): how do the models of the main benchmark behave later on?

Every model is fitted exactly as in the main benchmark (same training and validation samples, same selected
configuration; ridge and LightGBM are refitted with the hyper-parameters stored by the main run) and is then scored,
without any refitting, on three sets of samples:

    y1    the test period of the main benchmark (Dec 2024 - Nov 2025); must reproduce the main run
    y1m   its first eleven months (Dec 2024 - Oct 2025): the calendar months of y2
    y2    Dec 2025 - Oct 2026: data that no other experiment uses, 15 to 26 months after the end of training
    y1p, y2p   the hours of y1m and y2 that have a valid sample exactly 365 days apart in the other year
               (identical dates and hours in both years; used for the comparison between the years)

Only common-protocol plants take part (the short-record plants are tested on Nov 2025 - Oct 2026 anyway).  A sample
belongs to a period if its whole input window and its target hour lie inside it, as in `features.prepare`.

Run from the `kod` folder:
    python -m pvbench.later --out <cikti>/e10_ref --runs <cikti> --worker 0                    (references, CPU)
    python -m pvbench.later --out <cikti>/e10 --runs <cikti> --references --models gru --worker 0
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

from . import dataset, experiment, features, importance, metrics

SOURCES = importance.SOURCES                                  # information set -> (main run, tuning run)
PERIODS = {"y1m": ("2024-12-01", "2025-11-01"), "y2": ("2025-12-01", "2026-11-01")}      # target hours in (start, end]
N_PERIODS = 5                                                 # y1, y1m, y2 and the paired hours y1p, y2p
NO_FIT = ("persistence", "smart_persistence", "climatology")
REFERENCES = [*NO_FIT, "ridge", "lightgbm"]
FIELDS = ["plant", "name", "horizon", "info", "model", "seed", "period", "status", "n", "rmse", "mae", "nrmse", "nmae",
          "nmbe", "r2", "theil_u2", "mase", "n_day", "nrmse_day", "nmae_day", "main_nrmse", "finished"]


def period_masks(s, horizon: int) -> dict:
    """Sample masks per period; `y1` is rebuilt from the dates and must equal the test mask of the benchmark."""
    window_start = s.target_time - pd.Timedelta(hours=experiment.LOOKBACK - 1 + horizon)

    def inside(lo, hi):
        return np.asarray((window_start > pd.Timestamp(lo, tz="UTC")) & (s.target_time <= pd.Timestamp(hi, tz="UTC")))

    masks = {name: inside(lo, hi) for name, (lo, hi) in PERIODS.items()}
    rebuilt = inside(s.periods["test_start"].tz_convert(None), s.periods["test_end"].tz_convert(None))
    if not np.array_equal(rebuilt, s.test):
        raise RuntimeError("period masks do not reproduce the test mask of the benchmark")
    # Paired hours: a sample of y1m and the sample exactly 365 days later in y2 (neither period contains a leap day),
    # so that both years are compared on the same dates and hours even where a record has gaps.
    year, t = pd.Timedelta(days=365), s.target_time
    masks["y1p"] = masks["y1m"] & np.asarray(t.isin(t[masks["y2"]] - year))
    masks["y2p"] = masks["y2"] & np.asarray(t.isin(t[masks["y1m"]] + year))
    assert masks["y1p"].sum() == masks["y2p"].sum()
    return {"y1": s.test, **masks}


def evaluate(s, raw, mask) -> dict:
    """Same post-processing and metrics as `experiment.score`, on the samples of `mask`."""
    pred = features.postprocess(raw, s.aux["cs_lead"])
    naive = features.postprocess(s.aux["persistence"], s.aux["cs_lead"])
    return metrics.evaluate(s.y[mask], pred[mask], 1.0, daytime=s.aux["daytime"][mask], naive=naive[mask],
                            scale=s.mase_scale)


def done_keys(out: Path) -> set:
    periods = {}                                              # a run is done only if all three periods have a row
    for path in out.glob("later_w*.csv"):
        try:
            with open(path, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("finished"):
                        key = (int(r["plant"]), int(r["horizon"]), r["info"], r["model"], int(r["seed"]))
                        periods.setdefault(key, set()).add(r["period"])
        except (OSError, ValueError, KeyError):
            pass
    return {key for key, found in periods.items() if len(found) >= N_PERIODS}


def main(argv=None):
    ap = argparse.ArgumentParser(description="second test year of the main benchmark")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", required=True, help="folder that holds the main and tuning runs (ana, ana_icon, ayar, ayar_icon)")
    ap.add_argument("--plants", type=int, nargs="*")
    ap.add_argument("--horizons", type=int, nargs="*", default=[1, 24])
    ap.add_argument("--info", nargs="*", default=["S1", "S2"])
    ap.add_argument("--models", nargs="*", default=[], help="networks (GPU)")
    ap.add_argument("--references", nargs="*", default=REFERENCES)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--p-cores", type=int, default=8)
    args = ap.parse_args(argv)
    experiment.boost_process(args.worker, args.p_cores)
    import torch
    torch.set_num_threads(1)

    out, runs = Path(args.out), Path(args.runs)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"later_w{args.worker}.csv"

    def log(message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
        print(line, flush=True)
        with open(out / f"log_w{args.worker}.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write(rows):
        new = not path.exists()
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerows(rows)

    plants = dataset.load_plants()
    chosen = args.plants or list(plants.index)
    selected = {info: pd.read_csv(runs / SOURCES[info][1] / "selected.csv").set_index(["horizon", "info", "model"])
                for info in args.info} if args.models else {}
    main_scores = {info: experiment.read_metrics(runs / SOURCES[info][0]) for info in args.info}
    log(f"start: worker {args.worker}, plants {chosen}, horizons {args.horizons}, info {args.info}, "
        f"references {args.references}, networks {args.models}")
    for no in chosen:
        table = None
        for h in args.horizons:
            for info in args.info:
                if not experiment.claim(out, "later", (no, h, info)):
                    continue
                done = done_keys(out)
                jobs = [(m, 0) for m in args.references] + [(m, seed) for m in args.models for seed in range(args.seeds)]
                jobs = [j for j in jobs if (no, h, info, j[0], j[1]) not in done]
                if not jobs:
                    continue
                table = dataset.load_table(no) if table is None else table
                periods = features.protocol(table, no)
                if periods["name"] != "common":
                    log(f"skip  plant {no}: own protocol")
                    break
                s = features.prepare(table, h, info, periods, experiment.LOOKBACK)
                masks = period_masks(s, h)
                counts = {name: int(mask.sum()) for name, mask in masks.items()}
                pred_dir = runs / SOURCES[info][0] / "pred" / f"{no:02d}_h{h}_{info}"
                ms = main_scores[info]
                ms = ms[(ms["stage"] == "main") & (ms["plant"] == no) & (ms["horizon"] == h) & (ms["info"] == info)]
                for model, seed in jobs:
                    base = {"plant": no, "name": plants.loc[no, "name"], "horizon": h, "info": info, "model": model,
                            "seed": seed}
                    stamp = lambda: time.strftime("%Y-%m-%d %H:%M:%S")                          # noqa: E731
                    if counts["y2"] < experiment.MIN_SAMPLES["test"]:
                        write([{**base, "period": name, "status": "too_few_samples", "n": counts[name], "finished": stamp()}
                               for name in masks])
                        continue
                    reference = ms[(ms["model"] == model) & (ms["seed"] == seed)]["nrmse"]
                    base["main_nrmse"] = float(reference.iloc[0]) if len(reference) else np.nan
                    try:
                        if model in NO_FIT:
                            raw = s.aux[model]
                        elif model in ("ridge", "lightgbm"):
                            config = json.loads((pred_dir / f"{model}_config.json").read_text())
                            raw = importance.fitted_reference(model, s, config)(s.X)
                        else:
                            cfg = {k: selected[info].loc[(h, info, model), k] for k in experiment.DEFAULT}
                            cfg = {k: (int(v) if k == "batch_size" else float(v)) for k, v in cfg.items()}
                            raw, _ = experiment.fit_neural(model, s, cfg, seed)
                        rows = [{**base, "period": name, "status": "ok", **evaluate(s, raw, mask), "finished": stamp()}
                                for name, mask in masks.items()]
                    except Exception as error:
                        if type(error).__name__ == "AcceleratorError" or "CUDA" in str(error):
                            experiment.release(out, "later", (no, h, info))            # give the task back and stop
                            log(f"GPU failure in plant {no} h{h} {info} {model}: {error}"[:300])
                            raise SystemExit(1)
                        rows = [{**base, "period": name, "status": f"error: {type(error).__name__}: {error}"[:300],
                                 "finished": stamp()} for name in masks]
                    write(rows)
                    by = {r["period"]: r for r in rows}
                    if by["y1"]["status"] == "ok":
                        log(f"plant {no:>2} h{h:<2} {info} {model:<17} seed {seed}  nRMSE y1 {by['y1']['nrmse']:.3f} "
                            f"(main {base['main_nrmse']:.3f})  y1m {by['y1m']['nrmse']:.3f}  y2 {by['y2']['nrmse']:.3f}  "
                            f"n {counts['y1']}/{counts['y1m']}/{counts['y2']}")
                    else:
                        log(f"plant {no:>2} h{h:<2} {info} {model:<17} seed {seed}  {by['y1']['status']}")
    log("finished")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
