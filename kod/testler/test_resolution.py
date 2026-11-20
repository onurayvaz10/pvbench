"""Alignment checks for pvbench.resolution (run: python kod/testler/test_resolution.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pvbench import features, pvoutput, resolution  # noqa: E402


def synthetic(minutes, days=1200):
    """Every column encodes its own row number, so alignment can be read back from the values."""
    n = days * 24 * 60 // minutes
    index = pd.date_range("2022-12-01", periods=n + 1, freq=f"{minutes}min", tz="UTC", name="time_utc_end")[1:]
    row = np.arange(n, dtype=float)
    table = pd.DataFrame({"p": row / n, "cs_ghi": 1000.0 * row, "elev": 10.0}, index=index)
    table.loc[index[5000], "p"] = np.nan
    return table


def check_prepare(minutes):
    table = synthetic(minutes)
    n, L = len(table), 24
    step = pd.Timedelta(minutes=minutes)
    periods = {k: pd.Timestamp(v, tz="UTC") for k, v in features.COMMON.items()}
    s = resolution.prepare(table, periods, L)
    target_row = np.rint(s.y.astype(float) * n).astype(int)
    origin_row = target_row - 1
    assert (table.index[target_row] == s.target_time).all(), "target time must be the label of the target row"
    assert np.allclose(s.X[:, -1, 0] * n, origin_row, atol=0.1), "last p in the window must be p(t)"
    assert np.allclose(s.X[:, 0, 0] * n, origin_row - (L - 1), atol=0.1), "window must start at t-L+1"
    assert np.allclose(s.X[:, -1, 1], origin_row, rtol=0, atol=0.25), "clear-sky channel at the origin"
    assert np.allclose(s.X[:, -1, 2], target_row, rtol=0, atol=0.25), "lead clear-sky channel must be the target bin"
    assert (s.X[:, :, 0].max(axis=1) * n < origin_row + 0.5).all(), "no power value after the origin"
    assert 5000 not in target_row and not ((origin_row - (L - 1) <= 5000) & (origin_row >= 5000)).any(), \
        "windows touching a missing bin must be dropped"
    assert np.isfinite(s.X).all() and np.isfinite(s.y).all()
    assert not (s.train & s.val).any() and not (s.val & s.test).any() and not (s.train & s.test).any()
    t = s.target_time
    assert t[s.train].max() <= periods["val_start"] < t[s.val].min()
    assert t[s.val].max() <= periods["test_start"] < t[s.test].min()
    assert t[s.val].min() - periods["val_start"] == (L + 1) * step, "L steps are left out between the sets"
    assert t[s.test].max() <= periods["test_end"]
    assert np.allclose(s.aux["persistence"] * n, origin_row, atol=0.1)
    assert np.isclose(s.mase_scale, (24 * 60 // minutes) / n, rtol=1e-3), "MASE scale must use a lag of one day"
    doy = s.target_time.dayofyear + (s.target_time.hour + s.target_time.minute / 60.0) / 24.0
    assert np.allclose(s.X[:, -1, 3], np.sin(2 * np.pi * doy / 365.25), atol=1e-5), "day-of-year channel of the target"
    print(f"ok  {minutes:>2} min  samples {len(s.y)}  train/val/test {s.train.sum()}/{s.val.sum()}/{s.test.sum()}")


def check_smart_persistence():
    """Without a clear-sky index in the past day the clear-sky persistence falls back to persistence."""
    minutes, L = 5, 24
    table = synthetic(minutes)
    n, per_day = len(table), 24 * 60 // minutes
    start = n - 100 * per_day                                             # inside the test period
    table.iloc[start:start + 2 * per_day, table.columns.get_loc("cs_ghi")] = 0.0
    periods = {k: pd.Timestamp(v, tz="UTC") for k, v in features.COMMON.items()}
    s = resolution.prepare(table, periods, L)
    origin_row = np.rint(s.y.astype(float) * n).astype(int) - 1
    smart, plain = s.aux["smart_persistence"], s.aux["persistence"]
    assert np.isfinite(smart).all(), "clear-sky persistence must be defined for every sample"
    last = origin_row == start + 2 * per_day - 1                          # dark origin, sunlit target, no index for 2 days
    assert last.sum() == 1 and np.allclose(smart[last], plain[last]), "fallback must be the persistence forecast"
    dark = (origin_row >= start - 1) & (origin_row < start + 2 * per_day - 1)          # the target row is dark
    assert dark.any() and (smart[dark] == 0).all(), "zero clear-sky irradiance at the target gives a zero forecast"
    usual = (origin_row < start - 1) | (origin_row >= start + 2 * per_day)  # p / cs = 1 / n, so k(t) * cs(t+1) = (t+1) / n
    deviation = np.abs(smart[usual] * n / (origin_row[usual] + 1) - 1)
    assert deviation.max() < 1e-5, f"k(t) * cs(t+1) elsewhere (largest relative deviation {deviation.max():.2g})"
    print("ok  clear-sky persistence fallback")


def check_aggregate():
    index = pd.date_range("2024-01-01 00:05", periods=48, freq="5min", tz="UTC", name="time_utc_end")
    base = pd.DataFrame({"p": np.arange(48.0), "cs_ghi": np.arange(48.0), "elev": 1.0}, index=index)
    base.loc[index[9], "p"] = np.nan
    for minutes, k in ((10, 2), (20, 4), (60, 12)):
        out = resolution.aggregate(base, minutes)
        assert len(out) == 48 // k and out.index[0] == index[k - 1], "labels are bin ends"
        expected = np.arange(48.0).reshape(-1, k).mean(axis=1)
        bad = 9 // k
        assert np.isnan(out["p"].iloc[bad]) and np.allclose(np.delete(out["p"].to_numpy(), bad), np.delete(expected, bad))
        assert np.allclose(out["cs_ghi"], expected)
    assert resolution.aggregate(base, 5) is base
    print("ok  aggregate")


def check_bins():
    """Records every 5 minutes: bin power = energy increment / bin length; a gap invalidates its bins only."""
    stamps = pd.date_range("2024-06-01 05:00", "2024-06-01 19:00", freq="5min")
    energy = np.cumsum(np.r_[0.0, np.full(len(stamps) - 1, 250.0)])                 # 250 Wh per 5 min = 3 kW
    raw = pd.DataFrame({"local": stamps, "energy_wh": energy, "power_w": 3000.0})
    raw = raw.drop(index=[50, 51]).reset_index(drop=True)                            # a 15-minute gap
    intervals = pvoutput.build_intervals(raw, "UTC", 10.0, resolution.MAX_GAP_MIN)
    fine = pvoutput.binned_from_intervals(intervals, 10.0, 300.0)
    hourly = pvoutput.hourly_from_intervals(pvoutput.build_intervals(raw, "UTC", 10.0, 30.0), 10.0)
    day = fine[(fine.index > pd.Timestamp("2024-06-01 05:00", tz="UTC")) & (fine.index <= pd.Timestamp("2024-06-01 19:00", tz="UTC"))]
    gap = (day.index > pd.Timestamp("2024-06-01 09:05", tz="UTC")) & (day.index <= pd.Timestamp("2024-06-01 09:20", tz="UTC"))
    assert day["power_kw"][gap].isna().all() and np.allclose(day["power_kw"][~gap], 3.0), "5-minute bins"
    assert np.allclose(fine.loc[pd.Timestamp("2024-06-01 03:00", tz="UTC"), "power_kw"], 0.0), "night is zero"
    full = hourly.loc[pd.Timestamp("2024-06-01 12:00", tz="UTC"), "power_kw"]
    assert np.isclose(full, 3.0) and np.isclose(day["power_kw"][day.index.ceil("h") == pd.Timestamp("2024-06-01 12:00", tz="UTC")].mean(), full)
    print("ok  5-minute bins")


if __name__ == "__main__":
    for minutes in (5, 10, 20):
        check_prepare(minutes)
    check_smart_persistence()
    check_aggregate()
    check_bins()
    print("all resolution tests passed")
