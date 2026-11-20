"""Resolution sensitivity (experiment E8): one-step-ahead forecasts at 5, 10 and 20 minutes.

Data
----
Only PVOutput systems that record every 5 minutes are used (rule fixed before any result was seen):
    * median spacing of the records 300 s, at least 95 % of the record intervals with generation exactly 300 s
      and at least 99 % of those records on the 5-minute clock grid;
    * step of the energy counter at most 2.5 % of the energy generated in 5 minutes at rated power
      (a coarser counter makes the 5-minute mean power a staircase: quantisation error above 0.7 % of capacity).
The 5-minute target is built exactly like the hourly one (energy increments integrated over the bin, label =
END of the bin, UTC, nothing interpolated); a bin is valid only if it is covered by record intervals of at
most 5.5 minutes, and only if the hour it belongs to passed the hourly quality control.  The 10- and
20-minute series are means of two and four valid 5-minute bins, so all resolutions use the same records.

Samples
-------
Information set S0 of the benchmark (weather data are hourly, so they are not used): power, clear-sky
irradiance of the bin and of the target bin, day of year of the target.  Lookback 24 steps, horizon one step,
the chronological split of the plant in the benchmark.  The 60-minute point of the comparison is the S0,
h = 1 run of the benchmark itself.

Models
------
Persistence, clear-sky ("smart") persistence, Ridge, LightGBM (same search as the benchmark) and the networks
given by --models, with the configuration selected for S0, h = 1 in the benchmark; the batch size is
multiplied by 60 / resolution so that an epoch has the same number of updates as in the hourly run.

Run from the `kod` folder:
    python -m pvbench.resolution --build                               (tables -> veri/islenmis/altsaatlik)
    python -m pvbench.resolution --out <cikti>/e8 --tuning <cikti>/ayar_e2 --models gru tcn --worker 0
Result rows have the format of the benchmark (`metrics_w*.csv`), with info = S0r5 / S0r10 / S0r20 and horizon = 1 step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataset, experiment, features, pvoutput

RESOLUTIONS = (5, 10, 20)
BASE = 5                                    # minutes: resolution of the records
MAX_GAP_MIN = 5.5                           # a record interval longer than this is not a 5-minute measurement
MIN_SHARE_300, MIN_SHARE_GRID, MAX_COUNTER_SHARE = 0.95, 0.99, 0.025
SIMPLE = ("persistence", "smart_persistence")
REFERENCES = SIMPLE + ("ridge", "lightgbm")
FOLDER = dataset.OUT / "altsaatlik"
EVAL_BATCH = 2048


# ----------------------------------------------------------------------------- tables

def record_profile(raw: pd.DataFrame, capacity_kw: float) -> dict:
    """Spacing of the records and step of the energy counter of one system."""
    raw = raw.dropna(subset=["energy_wh"])
    same_day = raw["local"].dt.normalize() == raw["local"].dt.normalize().shift()
    spacing = raw["local"].diff().dt.total_seconds()[same_day]
    increment = raw["energy_wh"].diff()[same_day]
    positive = increment[increment > 0]
    # Counter step: the largest q (Wh) that divides at least 80 % of the positive energy increments.
    sample = np.rint(positive.to_numpy()).astype(np.int64)
    sample = sample[:: max(1, len(sample) // 50000)]
    counter = max((q for q in range(1, 2001) if len(sample) and (sample % q == 0).mean() >= 0.8), default=1)
    rated = capacity_kw * 1000.0 * BASE / 60.0                                  # Wh in one bin at rated power
    generating = raw["local"][same_day & (increment > 0)]
    profile = {"records": int(len(raw)), "spacing_median_s": float(spacing.median()),
               "share_300s": float((spacing[increment > 0] == 300).mean()) if len(positive) else np.nan,
               "share_on_grid": float((generating.dt.minute % BASE == 0).mean()) if len(generating) else np.nan,
               "counter_step_wh": counter, "counter_step_share": counter / rated}
    # records must also lie on the 5-minute grid: a record between two grid points would be split over two bins
    profile["eligible"] = bool(profile["spacing_median_s"] == 300 and profile["share_300s"] >= MIN_SHARE_300
                               and profile["share_on_grid"] >= MIN_SHARE_GRID
                               and profile["counter_step_share"] <= MAX_COUNTER_SHARE)
    return profile


def geometry(index: pd.DatetimeIndex, lat, lon, altitude) -> pd.DataFrame:
    """Clear-sky GHI and solar elevation at the middle of every bin."""
    import pvlib

    middle = index - pd.Timedelta(minutes=BASE / 2.0)
    site = pvlib.location.Location(lat, lon, altitude=0.0 if not np.isfinite(altitude) else altitude)
    position = site.get_solarposition(middle)
    clear = site.get_clearsky(middle, model="ineichen", solar_position=position)
    return pd.DataFrame({"cs_ghi": clear["ghi"].to_numpy(), "elev": position["apparent_elevation"].to_numpy()}, index=index)


def build_base(plant, hourly: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """5-minute table (p, cs_ghi, elev) of one plant, labels = end of the bin (UTC)."""
    folder = dataset.RAW / "pvoutput" / plant["sid"]
    totals = pvoutput.read_daily_totals(folder)
    intervals = pvoutput.build_intervals(raw, plant["tz"], plant["capacity_kw"], MAX_GAP_MIN, totals if len(totals) else None)
    fine = pvoutput.binned_from_intervals(intervals, plant["capacity_kw"], BASE * 60.0)
    index = pd.date_range(hourly.index[0] - pd.Timedelta(minutes=60 - BASE), hourly.index[-1], freq=f"{BASE}min",
                          name="time_utc_end")
    hour_ok = hourly["p"].notna().reindex(index.ceil("h"), fill_value=False).to_numpy(dtype=bool)
    _, altitude = dataset.read_openmeteo(dataset.RAW / "openmeteo" / f"{plant.name:02d}", "arsiv")
    table = geometry(index, plant["lat"], plant["lon"], altitude)
    table.insert(0, "p", (fine["power_kw"].reindex(index) / plant["capacity_kw"]).where(hour_ok))
    return table


def aggregate(base: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Means over `minutes` (a multiple of 5); a value is valid only if all its 5-minute bins are."""
    if minutes == BASE:
        return base
    k = minutes // BASE
    label = base.index.ceil(f"{minutes}min")
    grouped = base.groupby(label)
    out = grouped[["cs_ghi", "elev"]].mean()
    out.insert(0, "p", grouped["p"].sum(min_count=k) / k)
    out = out[grouped.size().to_numpy() == k]
    return out.rename_axis("time_utc_end")


def build_all(log=print):
    FOLDER.mkdir(parents=True, exist_ok=True)
    records = []
    for no, plant in dataset.load_plants().iterrows():
        record = {"no": no, "name": plant["name"], "source": plant["source"], "capacity_kw": plant["capacity_kw"]}
        if plant["source"] != "pvoutput":
            records.append({**record, "eligible": False})
            continue
        raw = pvoutput.read_raw(dataset.RAW / "pvoutput" / plant["sid"])
        record.update(record_profile(raw, plant["capacity_kw"]))
        if record["eligible"]:
            hourly = dataset.load_table(no)
            base = build_base(plant, hourly, raw)
            base.astype(np.float32).to_parquet(FOLDER / f"{no:02d}.parquet")
            # consistency with the hourly target: mean of the twelve bins of an hour = hourly value
            hour = aggregate(base, 60)["p"].reindex(hourly.index)
            both = hour.notna() & hourly["p"].notna()
            ok = base["p"].notna()
            span = (base.index >= ok[ok].index.min()) & (base.index <= ok[ok].index.max())
            record.update({"bins_valid": int(ok.sum()), "valid_share_in_span": float(ok[span].mean()),
                           "hours_valid_hourly": int(hourly["p"].notna().sum()), "hours_valid_from_bins": int(both.sum()),
                           "max_diff_to_hourly": float((hour - hourly["p"])[both].abs().max())})
        records.append(record)
        log(f"{no:>2} {plant['name'][:24]:<24} " + "  ".join(
            f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in record.items() if k not in ("no", "name", "source")))
    pd.DataFrame(records).to_csv(dataset.OUT / "altsaatlik_envanter.csv", sep=";", index=False, encoding="utf-8-sig")


def eligible_plants() -> list:
    inventory = pd.read_csv(dataset.OUT / "altsaatlik_envanter.csv", sep=";", encoding="utf-8-sig")
    return [int(no) for no in inventory.loc[inventory["eligible"].astype(bool), "no"]]


def load_table(no, minutes) -> pd.DataFrame:
    return aggregate(pd.read_parquet(FOLDER / f"{int(no):02d}.parquet").astype(np.float64), minutes)


# ----------------------------------------------------------------------------- samples

def prepare(table: pd.DataFrame, periods: dict, lookback: int = 24) -> features.Samples:
    """One-step-ahead samples on a table of any resolution (channels of information set S0)."""
    index = table.index
    step = index[1] - index[0]
    per_day = int(pd.Timedelta(days=1) / step)
    target_time = index + step
    angle = 2 * np.pi * (target_time.dayofyear + (target_time.hour + target_time.minute / 60.0) / 24.0) / 365.25
    cs = table["cs_ghi"] / 1000.0
    cs_lead = cs.shift(-1)
    ch = pd.DataFrame({"p": table["p"], "cs": cs, "cs_lead": cs_lead, "doy_sin": np.sin(angle), "doy_cos": np.cos(angle)},
                      index=index)
    train_rows = np.asarray((index > periods["train_start"]) & (index <= periods["val_start"]))
    a = ch.to_numpy(np.float32)
    p = table["p"].to_numpy(np.float32)
    n, L = len(a), lookback
    origin = np.arange(L - 1, n - 1)
    bad_rows = (~np.isfinite(a).all(axis=1)).astype(np.int32)
    bad_in_window = np.convolve(bad_rows, np.ones(L, np.int32), mode="valid")[: len(origin)]
    keep = (bad_in_window == 0) & np.isfinite(p[origin + 1])
    origin = origin[keep]
    windows = np.lib.stride_tricks.sliding_window_view(a, L, axis=0)      # (n-L+1, F, L)
    X = np.ascontiguousarray(windows[origin - (L - 1)].transpose(0, 2, 1))
    sample_target = index[origin + 1]
    window_start = index[origin - (L - 1)]

    def inside(lo, hi):
        return np.asarray((window_start > lo) & (sample_target <= hi))

    k = (table["p"] / cs).where(cs >= 0.05).ffill(limit=per_day)          # last clear-sky index of the past day
    smart = (k.clip(0, 3) * cs_lead).where(cs_lead > 0, 0.0)
    smart = smart.fillna(table["p"])      # no clear-sky index in the past day (after a long gap): plain persistence
    aux ={"persistence": p[origin], "smart_persistence": smart.to_numpy(np.float32)[origin],
           "cs_lead": cs_lead.to_numpy(np.float32)[origin], "daytime": table["elev"].to_numpy()[origin + 1] > 5.0}
    seasonal_diff = np.abs(p[per_day:] - p[:-per_day])[train_rows[per_day:]]
    return features.Samples(X=X, y=p[origin + 1], target_time=sample_target,
                            train=inside(periods["train_start"], periods["val_start"]),
                            val=inside(periods["val_start"], periods["test_start"]),
                            test=inside(periods["test_start"], periods["test_end"]),
                            names=list(ch.columns), periods=periods, aux=aux,
                            mase_scale=float(np.nanmean(seasonal_diff)) if np.isfinite(seasonal_diff).any() else np.nan)


# ----------------------------------------------------------------------------- runner

def settled(out) -> set:
    """(plant, info, model, seed) of the rows that need no further run: scored fits and too-short series."""
    rows = experiment.read_metrics(out)
    rows = rows[rows["status"].isin(["ok", "too_few_samples"])]
    return {(int(r.plant), r.info, r.model, int(r.seed)) for r in rows.itertuples()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="resolution sensitivity: one step ahead at 5, 10 and 20 minutes")
    ap.add_argument("--build", action="store_true", help="build the 5-minute tables and the inventory, then stop")
    ap.add_argument("--out")
    ap.add_argument("--tuning", help="tuning run of the benchmark with information set S0 (ayar_e2)")
    ap.add_argument("--plants", type=int, nargs="*")
    ap.add_argument("--resolutions", type=int, nargs="*", default=list(RESOLUTIONS))
    ap.add_argument("--models", nargs="*", default=["gru", "tcn"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--p-cores", type=int, default=8)
    args = ap.parse_args(argv)
    experiment.boost_process(args.worker, args.p_cores)
    if args.build:
        build_all()
        return
    experiment.EVAL_BATCH = EVAL_BATCH
    neural = [m for m in args.models if m not in REFERENCES]
    if neural:
        import torch
        torch.set_num_threads(1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
        print(line, flush=True)
        try:
            with open(out / f"log_w{args.worker}.txt", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    plants = dataset.load_plants()
    chosen = args.plants or eligible_plants()
    selected = pd.read_csv(Path(args.tuning) / "selected.csv").set_index(["horizon", "info", "model"]) if neural else None
    results = experiment.Results(out, args.worker)
    log(f"start: worker {args.worker}, plants {chosen}, resolutions {args.resolutions}, models {args.models}")
    for model in args.models:                                   # all plants with one model before the next model
        for minutes in sorted(args.resolutions, reverse=True):
            for no in chosen:
                info = f"S0r{minutes}"
                if not experiment.claim(out, f"e8-{model}", (no, 1, info)):
                    continue
                done = settled(out)                              # failed fits are not settled: they run again
                seeds = [0] if model in REFERENCES else list(range(args.seeds))
                seeds = [seed for seed in seeds if (no, info, model, seed) not in done]
                if not seeds:
                    continue
                periods = features.protocol(dataset.load_table(no), no)
                s = prepare(load_table(no, minutes), periods, experiment.LOOKBACK)
                counts = {"n_train": int(s.train.sum()), "n_val": int(s.val.sum()), "n_test": int(s.test.sum())}
                base = {"stage": "main", "plant": no, "name": plants.loc[no, "name"], "protocol": periods["name"],
                        "horizon": 1, "info": info, "model": model, "trial": 0, "n_features": s.X.shape[2], **counts}
                factor = 60 // minutes
                if any(counts[f"n_{k}"] < v * factor for k, v in experiment.MIN_SAMPLES.items()):
                    for seed in seeds:
                        results.write({**base, "seed": seed, "status": "too_few_samples"})
                    log(f"skip  plant {no} {info}: too few samples {counts}")
                    continue
                pred_dir = out / "pred" / f"{no:02d}_h1_{info}"
                pred_dir.mkdir(parents=True, exist_ok=True)
                if not (pred_dir / "target.parquet").exists():          # written once, published atomically
                    temporary = pred_dir / f"target.w{args.worker}.tmp"
                    pd.DataFrame({"y": s.y[s.test], "cs_lead": s.aux["cs_lead"][s.test], "daytime": s.aux["daytime"][s.test]},
                                 index=s.target_time[s.test]).to_parquet(temporary)
                    os.replace(temporary, pred_dir / "target.parquet")
                for seed in seeds:
                    row = {**base, "seed": seed, "status": "ok"}
                    started = time.time()
                    try:
                        if model in SIMPLE:
                            pred, extra = s.aux[model], {}
                        elif model == "ridge":
                            pred, extra = experiment.fit_ridge(s)
                        elif model == "lightgbm":
                            pred, extra = experiment.fit_lightgbm(s)
                        else:
                            cfg = {k: selected.loc[(1, "S0", model), k] for k in experiment.DEFAULT}
                            cfg = {k: (int(v) if k == "batch_size" else float(v)) for k, v in cfg.items()}
                            cfg["batch_size"] *= factor
                            pred, extra = experiment.fit_neural(model, s, cfg, seed)
                            row.update(cfg)
                            row.update(extra)
                        row.setdefault("fit_s", round(time.time() - started, 1))
                        scores, pred = experiment.score(s, pred, True)
                        row.update(scores)
                        np.save(pred_dir / f"{model}_s{seed}.npy", pred[s.test].astype(np.float32))
                        if model in ("ridge", "lightgbm"):
                            (pred_dir / f"{model}_config.json").write_text(json.dumps(extra, default=float))
                    except Exception as error:
                        if type(error).__name__ == "AcceleratorError" or "CUDA" in str(error):
                            experiment.release(out, f"e8-{model}", (no, 1, info))   # broken GPU context: stop this worker,
                            log(f"GPU failure in plant {no} {info} {model}: {error}"[:300])
                            raise SystemExit(1)                                     # the launcher starts the run again
                        row["status"] = f"error: {type(error).__name__}: {error}"[:300].replace("\n", " ")
                    results.write(row)
                    shown = row.get("nrmse", "")
                    log(f"plant {no:>2} {info:<6} {model:<17} seed {seed}  n {counts['n_train']}/{counts['n_val']}/{counts['n_test']}  "
                        f"{'nRMSE ' + format(shown, '.2f') if shown != '' else ''}  {row.get('epochs', '')} ep  "
                        f"{row.get('fit_s', '')} s  {row['status'] if row['status'] != 'ok' else ''}")
    log("finished")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
