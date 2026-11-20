"""Forecast samples: input windows, targets, chronological splits and reference forecasts.

Forecast origin t, horizon h, lookback L (all in hours, labels are hour ends):

    X[i]  rows t-L+1 .. t of the channel table (shape L x F)
    y[i]  p(t+h), power divided by rated capacity

Channels (row tau of the window):
    S0   p(tau), clear-sky GHI at tau, clear-sky GHI at tau+h, day-of-year sin/cos of tau+h
    S1   S0 + reanalysis weather at tau                       (history up to the origin)
    S2   S0 + archived NWP forecast valid at tau+h            (issued 24 h before its valid time; ICON global)
    S2B  as S2 with Open-Meteo's `best_match` archive         (source model changes in 2024; sensitivity only)
    S3   S0 + reanalysis weather at tau+h                     (not available in operation; upper bound)
    S12  S1 + S2: reanalysis weather at tau and the archived NWP forecast valid at tau+h
    S0m, S1m, S3m   the same channels as S0, S1, S3, restricted to the samples S2 has (equal training period)
    S1a  S1 + trailing moving average of p over MA_WINDOW hours (ablation of the input of the study's first version)

The last row of a window therefore carries the deterministic geometry (and, for S2/S3,
the weather) of the target hour. No measurement later than t enters X.

A sample is used only if every input value and the target are present. Sets are
defined by the target time, and a sample belongs to a set only if its whole input
window also lies inside that set, which leaves a gap of L+h-1 hours between sets.
Weather channels are standardised with statistics of the training period only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dataset import WEATHER

INFO_SETS = ("S0", "S1", "S2", "S2B", "S3", "S12", "S0m", "S1m", "S3m", "S1a", "S2d2")
# S2d2: as S2, but with the forecast value that was produced 48 h (not 24 h) before the target hour.  For h = 24 h
# the S2 value is the one "predicted 24 h before valid time", i.e. at the origin itself, and the archive does not
# say when that model run was disseminated; the 48-h-old value was certainly available (sensitivity analysis).
# S1a: S1 plus the trailing moving average of the power (the extra input of the first version of this study):
# mean of the values available among p(tau-MA_WINDOW+1) .. p(tau).  It uses no value later than tau and is
# defined wherever p(tau) is, so S1a has exactly the samples of S1.
MA_WINDOW = 3
# Matched variants: the channels of S0 / S1 / S3, but only the samples that S2 also has (the NWP
# archive starts on 2024-01-19, so S2 has a much shorter training period).  They separate the
# effect of the information from the effect of the amount of training data.
MATCHED = {"S0m": "S0", "S1m": "S1", "S3m": "S3"}
COMMON = {"train_start": "2022-12-01", "val_start": "2024-09-01", "test_start": "2024-12-01",
          "test_end": "2025-12-01"}
# Record limits imposed by quality control (plant number -> last usable hour label, UTC).
RECORD_END = {}
MIN_TRAIN_DAYS = 180


def _ts(value):
    return pd.Timestamp(value, tz="UTC")


def protocol(table: pd.DataFrame, plant_no=None) -> dict:
    """Chronological split for one plant.

    'common': test Dec 2024 - Nov 2025, validation Sep - Nov 2024, training before that.
    'own'   : for plants with fewer than MIN_TRAIN_DAYS valid days before the common
              validation period (or a record cut by quality control): test = last 12 full
              calendar months of the record, validation = the 3 months before, training = the rest.
    """
    ok = table["p"].notna()
    limit = RECORD_END.get(int(plant_no)) if plant_no is not None else None
    if limit:
        ok &= table.index <= _ts(limit)
    days_before = ok[table.index <= _ts(COMMON["val_start"])].sum() / 24.0
    if days_before >= MIN_TRAIN_DAYS and not limit:
        return {"name": "common", **{k: _ts(v) for k, v in COMMON.items()}, "record_end": None}
    last = ok[ok].index.max() + pd.Timedelta(days=2)
    test_end = _ts(f"{last.year}-{last.month:02d}-01")
    test_start = test_end - pd.DateOffset(months=12)
    return {"name": "own", "train_start": _ts(COMMON["train_start"]),
            "val_start": test_start - pd.DateOffset(months=3), "test_start": test_start,
            "test_end": test_end, "record_end": _ts(limit) if limit else None}


def channel_table(table: pd.DataFrame, horizon: int, info_set: str) -> pd.DataFrame:
    if info_set not in INFO_SETS:
        raise ValueError(f"unknown information set {info_set}")
    target_time = table.index + pd.Timedelta(hours=horizon)
    angle = 2 * np.pi * (target_time.dayofyear + target_time.hour / 24.0) / 365.25
    ch = pd.DataFrame({"p": table["p"], "cs": table["cs_ghi"] / 1000.0,
                       "cs_lead": table["cs_ghi"].shift(-horizon) / 1000.0,
                       "doy_sin": np.sin(angle), "doy_cos": np.cos(angle)}, index=table.index)
    for name in WEATHER:
        if info_set in ("S1", "S12", "S1a"):
            ch[f"era_{name}"] = table[f"era_{name}"]
        elif info_set == "S2":
            ch[f"nwp_{name}_lead"] = table[f"nwp_{name}"].shift(-horizon)
        elif info_set == "S2d2":
            ch[f"nwp2_{name}_lead"] = table[f"nwp2_{name}"].shift(-horizon)
        elif info_set == "S2B":
            ch[f"nwpbm_{name}_lead"] = table[f"nwpbm_{name}"].shift(-horizon)
        elif info_set == "S3":
            ch[f"era_{name}_lead"] = table[f"era_{name}"].shift(-horizon)
    if info_set == "S12":                                      # history (S1) followed by the forecast (S2)
        for name in WEATHER:
            ch[f"nwp_{name}_lead"] = table[f"nwp_{name}"].shift(-horizon)
    if info_set == "S1a":
        ch["p_ma"] = table["p"].rolling(MA_WINDOW, min_periods=1).mean().where(table["p"].notna())
    return ch


def reference_forecasts(table: pd.DataFrame, horizon: int, train_rows: np.ndarray) -> pd.DataFrame:
    """Reference forecasts of p(t+h), indexed by the ORIGIN t (same rows as `table`)."""
    p, cs = table["p"], table["cs_ghi"] / 1000.0
    cs_lead = cs.shift(-horizon)
    out = pd.DataFrame(index=table.index)
    out["persistence"] = p
    out["seasonal_persistence"] = p.shift(24 * int(np.ceil(horizon / 24)) - horizon)
    k = (p / cs).where(cs >= 0.05).ffill(limit=24)          # last clear-sky index seen in the past 24 h
    out["smart_persistence"] = (k.clip(0, 3) * cs_lead).where(cs_lead > 0, 0.0)
    target_time = table.index + pd.Timedelta(hours=horizon)
    train = pd.DataFrame({"p": p.to_numpy(), "m": table.index.month, "h": table.index.hour})[train_rows]
    by_month_hour = train.groupby(["m", "h"])["p"].mean()
    by_hour = train.groupby("h")["p"].mean()
    keys = pd.MultiIndex.from_arrays([target_time.month, target_time.hour])
    clim = by_month_hour.reindex(keys).to_numpy()
    out["climatology"] = np.where(np.isfinite(clim), clim, by_hour.reindex(target_time.hour).to_numpy())
    return out


@dataclass
class Samples:
    X: np.ndarray                 # (n, L, F) float32
    y: np.ndarray                 # (n,) float32
    target_time: pd.DatetimeIndex
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    names: list
    periods: dict
    aux: dict = field(default_factory=dict)   # arrays aligned with the samples
    mase_scale: float = np.nan


def prepare(table: pd.DataFrame, horizon: int, info_set: str, periods: dict, lookback: int = 24,
            standardise: bool = True) -> Samples:
    """`standardise=False` leaves the weather channels in physical units (the pooled model of
    `pvbench.unseen` standardises them with statistics of its own training plants)."""
    if info_set in MATCHED:
        s = prepare(table, horizon, MATCHED[info_set], periods, lookback, standardise)
        shared = np.asarray(s.target_time.isin(prepare(table, horizon, "S2", periods, lookback).target_time))
        s.train, s.val, s.test = s.train & shared, s.val & shared, s.test & shared
        return s
    if periods.get("record_end") is not None:
        table = table[table.index <= periods["record_end"]]
    index = table.index
    ch = channel_table(table, horizon, info_set)
    train_rows = np.asarray((index > periods["train_start"]) & (index <= periods["val_start"]))
    for col in ch.columns[5:] if standardise else []:         # weather channels only
        if col == "p_ma":                                     # power channel: already in units of capacity
            continue
        ref =ch[col].shift(horizon)[train_rows] if col.endswith("_lead") else ch[col][train_rows]
        mean, std = ref.mean(), ref.std()
        ch[col] = (ch[col] - mean) / (std if std and np.isfinite(std) and std > 0 else 1.0)
    a = ch.to_numpy(np.float32)
    p = table["p"].to_numpy(np.float32)
    n, L = len(a), lookback
    origin = np.arange(L - 1, n - horizon)
    bad_rows = (~np.isfinite(a).all(axis=1)).astype(np.int32)
    bad_in_window = np.convolve(bad_rows, np.ones(L, np.int32), mode="valid")[: len(origin)]
    keep = (bad_in_window == 0) & np.isfinite(p[origin + horizon])
    origin = origin[keep]
    windows = np.lib.stride_tricks.sliding_window_view(a, L, axis=0)      # (n-L+1, F, L)
    X = np.ascontiguousarray(windows[origin - (L - 1)].transpose(0, 2, 1))
    target_time = index[origin + horizon]
    window_start = index[origin - (L - 1)]

    def inside(lo, hi):
        return np.asarray((window_start > lo) & (target_time <= hi))

    refs = reference_forecasts(table, horizon, train_rows & table["p"].notna().to_numpy())
    aux = {name: refs[name].to_numpy(np.float32)[origin] for name in refs.columns}
    aux["cs_lead"] = ch["cs_lead"].to_numpy(np.float32)[origin]
    aux["daytime"] = table["elev"].to_numpy()[origin + horizon] > 5.0
    seasonal_diff = np.abs(p[24:] - p[:-24])[train_rows[24:]]
    return Samples(X=X, y=p[origin + horizon], target_time=target_time,
                   train=inside(periods["train_start"], periods["val_start"]),
                   val=inside(periods["val_start"], periods["test_start"]),
                   test=inside(periods["test_start"], periods["test_end"]),
                   names=list(ch.columns), periods=periods, aux=aux,
                   mase_scale=float(np.nanmean(seasonal_diff)) if np.isfinite(seasonal_diff).any() else np.nan)


def postprocess(yhat, cs_lead, upper=1.15):
    """Applied to every model and reference: clip to the physical range, zero when the sun is down."""
    yhat = np.clip(np.asarray(yhat, np.float32), 0.0, upper)
    return np.where(cs_lead > 0, yhat, 0.0).astype(np.float32)
