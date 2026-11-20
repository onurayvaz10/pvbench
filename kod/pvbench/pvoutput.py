"""Raw PVOutput status records -> hourly UTC series of interval-averaged AC power.

Target definition
-----------------
PVOutput reports, for each status record, the cumulative energy generated since
local midnight (Wh). The hourly target is the energy generated in the hour
divided by one hour, i.e. the interval-averaged AC power in kW. It is obtained
by distributing each energy increment uniformly over the interval between two
consecutive records and integrating over UTC hours.

Time convention
---------------
Records are stamped in the system's configured local time zone. They are
converted to UTC, and every hourly value is labelled with the END of its
averaging hour (the convention Open-Meteo uses for radiation).

Missing data
------------
Nothing is interpolated. An hour is valid only if it is fully explained by
valid record intervals. Time before the first and after the last record of a
day that has records is treated as zero generation (inverters do not report at
night), unless the counter shows that records are missing at the start or the
end of the day. Days without records are missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_FIELDS = ["date", "time", "energy_wh", "efficiency", "power_w", "avg_power_w", "normalised",
              "cons_energy_wh", "cons_power_w", "temperature_c", "voltage_v"]


def read_raw(folder) -> pd.DataFrame:
    """Read all monthly raw files of one system. Returns columns
    local (naive timestamp), energy_wh, power_w; duplicates removed, sorted."""
    rows = []
    for path in sorted(Path(folder).glob("20??-??.csv")):
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split(",")
            if len(parts) >= 5:
                rows.append((parts[0] + " " + parts[1], parts[2], parts[4]))
    raw = pd.DataFrame(rows, columns=["local", "energy_wh", "power_w"])
    raw["local"] = pd.to_datetime(raw["local"], format="%Y%m%d %H:%M", errors="coerce")
    for col in ("energy_wh", "power_w"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["local"]).drop_duplicates("local", keep="last").sort_values("local")
    return raw.reset_index(drop=True)


def read_daily_totals(folder) -> pd.Series:
    """Daily generation (Wh) from the Get Output file, indexed by local date."""
    path = Path(folder) / "gunluk.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) > 1:
            try:
                data[pd.Timestamp(parts[0]).date()] = float(parts[1])
            except ValueError:
                continue
    return pd.Series(data, dtype=float)


def _to_utc_seconds(local: pd.Series, tz: str) -> np.ndarray:
    """Epoch seconds (float, NaN where the local time is ambiguous or non-existent)."""
    aware = pd.DatetimeIndex(local).tz_localize(tz, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    seconds = (aware.tz_localize(None) - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    return np.asarray(seconds, dtype=float)


def build_intervals(raw: pd.DataFrame, tz: str, capacity_kw: float, max_gap_min=30.0,
                    daily_totals: pd.Series | None = None, counter_slack_wh=150.0) -> pd.DataFrame:
    """Non-overlapping intervals [start, end] (UTC epoch seconds) with the energy
    generated in each and a validity flag.

    For every local day that has records the intervals cover the full day:
    midnight -> first record, record -> record, last record -> next midnight.
    """
    raw = raw.dropna(subset=["energy_wh"]).copy()
    raw["utc"] = _to_utc_seconds(raw["local"], tz)
    raw = raw.dropna(subset=["utc"])
    raw["day"] = raw["local"].dt.normalize()
    step = raw.groupby("day")["utc"].diff().median()
    step = float(step) if np.isfinite(step) and step > 0 else 300.0
    cap_wh_per_s = capacity_kw * 1000.0 / 3600.0
    out = []
    days = raw["day"].drop_duplicates()
    midnights = dict(zip(days, _to_utc_seconds(days, tz)))
    next_midnights = dict(zip(days, _to_utc_seconds(days + pd.Timedelta(days=1), tz)))
    for day, group in raw.groupby("day", sort=True):
        t, energy = group["utc"].to_numpy(), group["energy_wh"].to_numpy()
        day_start, day_end = midnights[day], next_midnights[day]
        if not (np.isfinite(day_start) and np.isfinite(day_end)):
            continue
        # midnight -> first record
        first_start = max(day_start, t[0] - step)
        first_ok = energy[0] <= 1.2 * cap_wh_per_s * (t[0] - first_start) + counter_slack_wh
        if first_start > day_start:
            out.append((day_start, first_start, 0.0, first_ok))      # invalid if records are missing at day start
        out.append((first_start, t[0], energy[0] if first_ok else 0.0, first_ok))
        # record -> record
        duration, delta = np.diff(t), np.diff(energy)
        plausible = (delta >= 0) & (delta <= 1.2 * cap_wh_per_s * duration + counter_slack_wh)
        ok = plausible & ((duration <= max_gap_min * 60.0) | (delta == 0))
        out.extend(zip(t[:-1], t[1:], np.where(ok, delta, 0.0), ok))
        # last record -> next midnight
        end_ok = True
        if daily_totals is not None and day.date() in daily_totals.index:
            total = daily_totals[day.date()]
            end_ok = bool(energy[-1] >= 0.95 * total - counter_slack_wh)   # counter reached the daily total
        if day_end > t[-1]:
            out.append((t[-1], day_end, 0.0, end_ok))
    intervals = pd.DataFrame(out, columns=["start", "end", "energy_wh", "valid"])
    return intervals[intervals["end"] > intervals["start"]].reset_index(drop=True)


def hourly_from_intervals(intervals: pd.DataFrame, capacity_kw: float, max_ratio=1.15, step_s=3600.0) -> pd.DataFrame:
    """Integrate intervals over UTC hours. Index: end of hour (UTC).
    Columns: power_kw (NaN if the hour is not fully valid), valid_fraction.
    `step_s` other than 3600 gives the same table on a finer grid (resolution experiment)."""
    if intervals.empty:
        return pd.DataFrame(columns=["power_kw", "valid_fraction"])
    if step_s != 3600.0:
        return binned_from_intervals(intervals, capacity_kw, float(step_s), max_ratio)
    start, end = intervals["start"].to_numpy(), intervals["end"].to_numpy()
    duration = end - start
    valid = intervals["valid"].to_numpy(bool)
    knots = np.empty(2 * len(intervals))
    knots[0::2], knots[1::2] = start, end

    def cumulative(per_interval):
        total = np.cumsum(per_interval)
        values = np.empty(2 * len(intervals))
        values[0::2], values[1::2] = total - per_interval, total
        return values

    grid = np.arange(np.floor(start.min() / 3600.0) * 3600.0, np.ceil(end.max() / 3600.0) * 3600.0 + 1.0, 3600.0)
    energy = np.diff(np.interp(grid, knots, cumulative(intervals["energy_wh"].to_numpy())))
    valid_s = np.diff(np.interp(grid, knots, cumulative(np.where(valid, duration, 0.0))))
    power = np.where(valid_s >= 3599.0, energy / 1000.0, np.nan)
    power[power > max_ratio * capacity_kw] = np.nan
    index = pd.to_datetime(grid[1:], unit="s", utc=True)
    return pd.DataFrame({"power_kw": power, "valid_fraction": valid_s / 3600.0}, index=index.rename("time_utc_end"))


def binned_from_intervals(intervals: pd.DataFrame, capacity_kw: float, step_s: float, max_ratio=1.15) -> pd.DataFrame:
    """As `hourly_from_intervals` for bins of `step_s` seconds: interval-averaged power in kW, labelled
    with the END of the bin (UTC); NaN unless the whole bin is covered by valid record intervals."""
    start, end = intervals["start"].to_numpy(), intervals["end"].to_numpy()
    duration = end - start
    valid = intervals["valid"].to_numpy(bool)
    knots = np.empty(2 * len(intervals))
    knots[0::2], knots[1::2] = start, end

    def cumulative(per_interval):
        total = np.cumsum(per_interval)
        values = np.empty(2 * len(intervals))
        values[0::2], values[1::2] = total - per_interval, total
        return values

    grid = np.arange(np.floor(start.min() / step_s) * step_s, np.ceil(end.max() / step_s) * step_s + 1.0, step_s)
    energy = np.diff(np.interp(grid, knots, cumulative(intervals["energy_wh"].to_numpy())))
    valid_s = np.diff(np.interp(grid, knots, cumulative(np.where(valid, duration, 0.0))))
    power = np.where(valid_s >= step_s - 1.0, energy / 1000.0 * (3600.0 / step_s), np.nan)
    power[power > max_ratio * capacity_kw] = np.nan
    index = pd.to_datetime(grid[1:], unit="s", utc=True)
    return pd.DataFrame({"power_kw": power, "valid_fraction": valid_s / step_s}, index=index.rename("time_utc_end"))


def hourly_instantaneous(raw: pd.DataFrame, tz: str) -> pd.Series:
    """Hourly mean of the instantaneous power field (kW), labelled by hour end.
    Used only as a cross-check of the energy-based target."""
    utc = _to_utc_seconds(raw["local"], tz)
    keep = np.isfinite(utc) & raw["power_w"].notna().to_numpy()
    stamps = pd.to_datetime(utc[keep], unit="s", utc=True)
    series = pd.Series(raw["power_w"].to_numpy()[keep] / 1000.0, index=stamps)
    return series.groupby(series.index.ceil("h")).mean().rename("power_inst_kw")


def hourly_power(folder, tz: str, capacity_kw: float, max_gap_min=30.0) -> pd.DataFrame:
    """Full pipeline for one system folder: raw files -> hourly UTC table."""
    raw = read_raw(folder)
    totals = read_daily_totals(folder)
    intervals = build_intervals(raw, tz, capacity_kw, max_gap_min, totals if len(totals) else None)
    hourly = hourly_from_intervals(intervals, capacity_kw)
    return hourly.join(hourly_instantaneous(raw, tz), how="left")
