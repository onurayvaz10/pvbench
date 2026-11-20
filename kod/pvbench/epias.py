"""EPİAŞ UEVM records -> hourly UTC table in the same format as `pvbench.pvoutput.hourly_power`.

Source
------
`kod/veri_cekme/epias_cek.py` writes one CSV per plant (veri/ham/epias/uevm/<id>.csv) with columns
`zaman_tr` (start of the settlement hour, Türkiye time, UTC+3, no DST since 2016), `gunes` (MWh injected
by the solar unit in that hour), `toplam`, `diger`.

Target definition
-----------------
Energy injected in the hour divided by one hour = interval-averaged AC power. MWh/h -> kW (x 1000).
This is the same quantity as the PVOutput target (energy-based hourly mean power).

Time convention
---------------
Labelled with the END of the averaging hour in UTC (start + 1 h - 3 h), like the PVOutput table.

Missing data
------------
Nothing is interpolated. Hours without a record are NaN with valid_fraction 0. Values above
max_ratio x capacity are set to NaN, as for PVOutput.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_uevm(path) -> pd.Series:
    """Hourly solar injection (MWh) indexed by hour START in UTC."""
    raw = pd.read_csv(Path(path), dtype={"zaman_tr": str})
    if raw.empty:
        return pd.Series(dtype=float, name="mwh")
    start = pd.to_datetime(raw["zaman_tr"], utc=True)
    series = pd.Series(raw["gunes"].astype(float).to_numpy(), index=start)
    return series.groupby(series.index.floor("h")).sum().rename("mwh")      # tolerates sub-hourly records


def hourly_power(path, capacity_kw: float, realtime_path=None, max_ratio=1.15) -> pd.DataFrame:
    """Columns: power_kw, valid_fraction, power_inst_kw (TEİAŞ real-time generation if given, else NaN).
    Index: end of hour (UTC), named time_utc_end, on a complete hourly grid."""
    mwh = read_uevm(path)
    if mwh.empty:
        return pd.DataFrame(columns=["power_kw", "valid_fraction", "power_inst_kw"])
    grid = pd.date_range(mwh.index.min(), mwh.index.max(), freq="h", tz="UTC")
    power = mwh.reindex(grid) * 1000.0
    valid = power.notna().astype(float)
    power[power > max_ratio * capacity_kw] = np.nan
    out = pd.DataFrame({"power_kw": power.to_numpy(), "valid_fraction": valid.to_numpy()},
                       index=(grid + pd.Timedelta(hours=1)).rename("time_utc_end"))
    if realtime_path is not None and Path(realtime_path).exists():
        rt = read_uevm(realtime_path) * 1000.0
        rt.index = rt.index + pd.Timedelta(hours=1)
        out["power_inst_kw"] = rt.reindex(out.index).to_numpy()
    else:
        out["power_inst_kw"] = np.nan
    return out
