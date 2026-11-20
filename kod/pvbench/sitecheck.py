"""Check a plant's reported location against its own generation records.

The daily production window carries the location: its midpoint (in UTC) gives
the longitude through the time of solar noon, and its length gives the
latitude through the sunrise equation. Estimates are medians over many days;
they are coarse (a few degrees) but enough to detect a wrong country, a wrong
hemisphere or a wrong time-zone setting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pvoutput import _to_utc_seconds


def equation_of_time_minutes(day_of_year):
    b = 2.0 * np.pi * (np.asarray(day_of_year, float) - 81.0) / 364.0
    return 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)


def declination_deg(day_of_year):
    return 23.45 * np.sin(2.0 * np.pi * (284.0 + np.asarray(day_of_year, float)) / 365.0)


def daily_windows(raw: pd.DataFrame, tz: str, threshold=0.05, min_records=20) -> pd.DataFrame:
    """Per local day: UTC time (hours) of the midpoint and the length of the window
    in which power exceeds `threshold` x that day's maximum."""
    data = raw.dropna(subset=["power_w"]).copy()
    data["utc"] = _to_utc_seconds(data["local"], tz)
    data = data.dropna(subset=["utc"])
    data["day"] = data["local"].dt.normalize()
    rows = []
    for day, g in data.groupby("day"):
        peak = g["power_w"].max()
        on = g[g["power_w"] > threshold * peak] if peak > 0 else g.iloc[0:0]
        if len(on) < min_records:
            continue
        first, last = on["utc"].iloc[0], on["utc"].iloc[-1]
        if g["utc"].iloc[0] == first and g["power_w"].iloc[0] > 0.3 * peak:
            continue                                    # records start mid-production: window not observed
        mid = (first + last) / 2.0
        rows.append({"day": day, "doy": day.dayofyear, "noon_utc_h": (mid % 86400.0) / 3600.0,
                     "length_h": (last - first) / 3600.0})
    return pd.DataFrame(rows)


def estimate_location(raw: pd.DataFrame, tz: str) -> dict:
    """Median longitude from solar noon and latitude from day length.

    Latitude uses only days with |declination| > 15 degrees and is reported
    separately for the two solstice seasons; the production window is shorter
    than the true day, which biases the two seasons in opposite directions, so
    their mean is the estimate.
    """
    w = daily_windows(raw, tz)
    if w.empty:
        return {"n_days": 0, "longitude": np.nan, "latitude": np.nan, "lat_jun": np.nan, "lat_dec": np.nan}
    noon = w["noon_utc_h"].to_numpy()
    lon = 15.0 * (12.0 - equation_of_time_minutes(w["doy"]) / 60.0 - noon)
    lon = (lon + 180.0) % 360.0 - 180.0
    # circular median: centre on the circular mean first so that values near +-180 do not split
    centre = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(lon)))))
    longitude = centre + np.median((lon - centre + 180.0) % 360.0 - 180.0)
    decl = declination_deg(w["doy"])
    half = np.radians(np.clip(w["length_h"].to_numpy(), 0.5, 23.5) * 15.0 / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lat = np.degrees(np.arctan(-np.cos(half) / np.tan(np.radians(decl))))
    jun, dec = lat[decl > 15.0], lat[decl < -15.0]
    lat_jun = float(np.median(jun)) if jun.size else np.nan
    lat_dec = float(np.median(dec)) if dec.size else np.nan
    latitude = np.nanmean([lat_jun, lat_dec]) if (jun.size or dec.size) else np.nan
    return {"n_days": int(len(w)), "longitude": float(longitude), "latitude": float(latitude),
            "lat_jun": lat_jun, "lat_dec": lat_dec, "n_jun": int(jun.size), "n_dec": int(dec.size)}
