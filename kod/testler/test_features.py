"""Alignment and leakage checks for pvbench.features (run: python kod/testler/test_features.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pvbench import features  # noqa: E402
from pvbench.dataset import WEATHER  # noqa: E402


def synthetic(n=24 * 1200, seed=0):
    """Every column encodes its own row number, so alignment can be read back from the values."""
    index = pd.date_range("2022-12-01 01:00", periods=n, freq="h", tz="UTC", name="time_utc_end")
    row = np.arange(n, dtype=float)
    table = pd.DataFrame({"p": row / n, "cs_ghi": 1000.0 * row, "elev": 10.0}, index=index)
    for name in WEATHER:
        table[f"era_{name}"] = row
        table[f"nwp_{name}"] = row
        table[f"nwp2_{name}"] = row
        table[f"nwpbm_{name}"] = row
    table.loc[index[5000], "p"] = np.nan                      # one missing target hour
    return table


def check(horizon, info):
    table = synthetic()
    n, L = len(table), 24
    periods = features.protocol(table)
    assert periods["name"] == "common"
    s = features.prepare(table, horizon, info, periods, L)
    target_row = np.rint(s.y * n).astype(int)
    origin_row = target_row - horizon
    assert np.allclose(s.X[:, -1, 0] * n, origin_row, atol=1e-2), "last p in the window must be p(t)"
    assert np.allclose(s.X[:, 0, 0] * n, origin_row - (L - 1), atol=1e-2), "window must start at t-L+1"
    assert np.allclose(s.X[:, -1, 1], origin_row), "clear-sky channel at the origin"
    assert np.allclose(s.X[:, -1, 2], target_row), "lead clear-sky channel must be the target hour"
    assert (s.X[:, :, 0].max(axis=1) * n < target_row - horizon + 0.5).all(), "no power value after the origin"
    assert 5000 not in target_row and not ((origin_row - (L - 1) <= 5000) & (origin_row >= 5000)).any(), \
        "windows touching a missing hour must be dropped"
    assert np.isfinite(s.X).all() and np.isfinite(s.y).all()
    # sets: disjoint, ordered, with a gap of L + horizon - 1 hours
    assert not (s.train & s.val).any() and not (s.val & s.test).any() and not (s.train & s.test).any()
    t = s.target_time
    assert t[s.train].max() <= periods["val_start"] < t[s.val].min()
    assert t[s.val].max() <= periods["test_start"] < t[s.test].min()
    assert t[s.val].min() - periods["val_start"] == pd.Timedelta(hours=L + horizon)   # L+h-1 hours left out
    assert t[s.test].max() <= periods["test_end"]
    # reference forecasts
    assert np.allclose(s.aux["persistence"] * n, origin_row, atol=1e-2)
    lag = 24 * int(np.ceil(horizon / 24))
    assert np.allclose(s.aux["seasonal_persistence"] * n, target_row - lag, atol=1e-2)
    # weather channels: S1 at tau, S2/S3 at tau + h (standardised, so compare differences)
    if info != "S0":
        w = s.X[:, :, 5]
        steps = np.diff(w.astype(float), axis=1)
        assert np.allclose(steps, steps[0, 0], rtol=0, atol=2e-6), "weather channel must be contiguous"
        ch = features.channel_table(table, horizon, info)
        col = ch.columns[5]
        expected = np.arange(n, dtype=float) + (0 if info in ("S1", "S12", "S1a") else horizon)
        assert np.allclose(ch[col].to_numpy()[: n - horizon], expected[: n - horizon])
        raw = features.prepare(table, horizon, info, periods, L, standardise=False)     # physical units
        assert np.array_equal(raw.X[:, :, :5], s.X[:, :, :5]) and np.array_equal(raw.y, s.y)
        assert np.allclose(raw.X[:, -1, 5], expected[origin_row]), "unstandardised weather must be the raw value"
        if info == "S1a":                                     # S1 plus the trailing moving average of the power
            base = features.prepare(table, horizon, "S1", periods, L)
            assert s.names == base.names + ["p_ma"] and s.target_time.equals(base.target_time)
            assert np.array_equal(s.X[:, :, :-1], base.X), "S1a must contain the S1 channels unchanged"
            k = features.MA_WINDOW
            mean = np.mean([s.X[:, k - 1 - i: L - i, 0] for i in range(k)], axis=0)
            assert np.allclose(s.X[:, k - 1:, -1], mean, atol=1e-5), "p_ma must be the mean of p(tau-k+1..tau)"
            assert (s.X[:, :, -1].max(axis=1) * n < target_row - horizon + 0.5).all(), "no power value after the origin"
        if info == "S12":                                     # history channels first, then the forecast channels
            assert len(ch.columns) == 5 + 2 * len(WEATHER)
            lead = ch[ch.columns[5 + len(WEATHER)]].to_numpy()
            assert ch.columns[5 + len(WEATHER)].endswith("_lead")
            assert np.allclose(lead[: n - horizon], np.arange(n, dtype=float)[: n - horizon] + horizon)
    print(f"ok  h={horizon:<2} {info}  samples {len(s.y)}  train/val/test {s.train.sum()}/{s.val.sum()}/{s.test.sum()}")


def check_own_protocol():
    table = synthetic()
    table.loc[table.index <= pd.Timestamp("2024-12-20", tz="UTC"), "p"] = np.nan
    periods = features.protocol(table)
    assert periods["name"] == "own"
    last = table["p"].dropna().index.max()
    assert periods["test_end"] <= last + pd.Timedelta(days=2)
    assert periods["test_end"] - pd.DateOffset(months=12) == periods["test_start"]
    assert periods["test_start"] - pd.DateOffset(months=3) == periods["val_start"]
    print("ok  own protocol", {k: str(v)[:10] for k, v in periods.items() if k != "name"})


def check_matched(horizon):
    """S0m / S1m / S3m: channels of S0 / S1 / S3, sets restricted to the samples S2 has."""
    table = synthetic()
    late = table.index < pd.Timestamp("2024-01-19 12:00", tz="UTC")          # NWP archive starts later
    for name in WEATHER:
        table.loc[late, f"nwp_{name}"] = np.nan
    periods = features.protocol(table)
    s2 = features.prepare(table, horizon, "S2", periods)
    assert s2.target_time[s2.train].min() > pd.Timestamp("2024-01-19", tz="UTC")
    for matched, base in features.MATCHED.items():
        full = features.prepare(table, horizon, base, periods)
        m = features.prepare(table, horizon, matched, periods)
        assert np.array_equal(m.X, full.X) and np.array_equal(m.y, full.y), "inputs must be those of the base set"
        for part in ("train", "val", "test"):
            a, b = m.target_time[getattr(m, part)], s2.target_time[getattr(s2, part)]
            assert a.equals(b), f"{matched} {part}: target hours must equal those of S2"
        assert m.train.sum() < full.train.sum() and m.test.sum() == full.test.sum()
    print(f"ok  h={horizon:<2} matched sets  train {s2.train.sum()} (S1 full {full.train.sum()})  "
          f"val {s2.val.sum()}  test {s2.test.sum()}")


if __name__ == "__main__":
    for h in (1, 24):
        for info in features.INFO_SETS:
            if info not in features.MATCHED:
                check(h, info)
        check_matched(h)
    check_own_protocol()
    print("all feature tests passed")
