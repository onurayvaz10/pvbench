"""Hourly modelling table for every plant: target, quality control, solar geometry, weather.

One table per plant, indexed by the END of the averaging hour in UTC:

    power_kw, p            interval-averaged AC power; p = power_kw / rated capacity (NaN = missing)
    cs_ghi, elev, elev_max clear-sky GHI (W/m2, hourly mean), solar elevation (mean, max over the hour)
    era_*                  ERA5-based reanalysis (Open-Meteo archive) valid at the label time
    nwp_*, nwp2_*          archived NWP forecast for the label time, issued 24 h / 48 h earlier

Radiation variables of Open-Meteo are means over the preceding hour, i.e. the same
hour as the power value. Nothing is interpolated or smoothed.

Quality control (target only; flagged hours become NaN and their windows are dropped):
    parser      hour not fully covered by valid records, counter jumps, > 1.15 x capacity
    night       sun below -3 deg during the whole hour but power > 1 % of capacity
    stuck       eight or more consecutive hours with the same positive value

Run:  python -m pvbench.dataset        (from the `kod` folder)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import epias, pvoutput

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "veri" / "ham"
OUT = ROOT / "veri" / "islenmis"
START = pd.Timestamp("2022-12-01", tz="UTC")           # first hour label is START + 1 h
WEATHER = ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "temperature_2m",
           "relative_humidity_2m", "wind_speed_10m", "cloud_cover"]
NIGHT_ELEVATION, NIGHT_SHARE, STUCK_HOURS = -3.0, 0.01, 8


def load_plants(root=ROOT) -> pd.DataFrame:
    """Active plants with capacity, source id, time zone and verified coordinates."""
    plants = pd.read_csv(root / "veri" / "santraller.csv", sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    plants = plants[(plants["atla"] == "") & plants["kaynak"].isin(["pvoutput", "epias"])]
    sites = pd.read_csv(root / "veri" / "konum_dogrulama.csv", sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    sites = sites.set_index("no")
    rows = []
    for _, r in plants.iterrows():
        s = sites.loc[r["no"]]
        rows.append({"no": int(r["no"]), "name": r["ad"], "country": r["ulke"], "source": r["kaynak"],
                     "sid": r["sid"], "capacity_kw": float(r["kapasite_kw"]),
                     "tz": s["saat_dilimi"] or "Europe/Istanbul",
                     "lat": float(s["enlem"]), "lon": float(s["boylam"])})
    return pd.DataFrame(rows).set_index("no")


def read_openmeteo(folder, kind) -> tuple[pd.DataFrame, float]:
    """All yearly files of one kind ('arsiv', 'icon' or 'tahmin') as one UTC-indexed table, and the grid elevation."""
    parts, elevation = [], np.nan
    for path in sorted(Path(folder).glob(f"{kind}_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        hourly = pd.DataFrame(data["hourly"])
        hourly.index = pd.to_datetime(hourly.pop("time"), utc=True)
        parts.append(hourly.astype(float))
        elevation = float(data.get("elevation", elevation))
    table = pd.concat(parts)
    return table[~table.index.duplicated(keep="last")].sort_index(), elevation


def solar_geometry(index: pd.DatetimeIndex, lat, lon, altitude=0.0) -> pd.DataFrame:
    """Hourly means over the hour ending at each label (six 10-minute points)."""
    import pvlib

    offsets = pd.to_timedelta(np.arange(55, 0, -10), unit="min")
    naive = index.tz_convert("UTC").tz_localize(None)
    times = pd.DatetimeIndex(np.concatenate([(naive - o).to_numpy() for o in offsets])).tz_localize("UTC")
    site = pvlib.location.Location(lat, lon, altitude=0.0 if not np.isfinite(altitude) else altitude)
    position = site.get_solarposition(times)
    clear = site.get_clearsky(times, model="ineichen", solar_position=position)
    n = len(index)
    ghi = clear["ghi"].to_numpy().reshape(len(offsets), n)
    elevation = position["apparent_elevation"].to_numpy().reshape(len(offsets), n)
    return pd.DataFrame({"cs_ghi": ghi.mean(0), "elev": elevation.mean(0), "elev_max": elevation.max(0)}, index=index)


def _runs(values: np.ndarray) -> np.ndarray:
    """Length of the run of identical consecutive values that each element belongs to."""
    change = np.r_[True, values[1:] != values[:-1]]
    run_id = np.cumsum(change) - 1
    return np.bincount(run_id)[run_id]


def build_plant(plant, root=ROOT) -> tuple[pd.DataFrame, dict]:
    """Hourly table and an inventory record for one plant (a row of `load_plants`)."""
    raw = root / "veri" / "ham"
    cap = plant["capacity_kw"]
    if plant["source"] == "epias":
        power = epias.hourly_power(raw / "epias" / "uevm" / f"{plant['sid']}.csv", cap,
                                   raw / "epias" / "gercek_zamanli" / f"{plant['sid']}.csv")
    else:
        power = pvoutput.hourly_power(raw / "pvoutput" / plant["sid"], plant["tz"], cap)
    weather_dir = raw / "openmeteo" / f"{plant.name:02d}"
    era, altitude = read_openmeteo(weather_dir, "arsiv")
    # Archived forecasts. `icon`: one global model (DWD ICON) for every plant -> columns nwp_*, nwp2_*.
    # `tahmin`: Open-Meteo's default `best_match`, whose source model changes in summer 2024 for the
    # plants in NL, BE, LU, HU and UK -> columns nwpbm_*, kept only for the sensitivity comparison.
    nwp, _ = read_openmeteo(weather_dir, "icon")
    best, _ = read_openmeteo(weather_dir, "tahmin")

    end = min(power.index.max(), era.index.max())
    index = pd.date_range(START + pd.Timedelta(hours=1), end, freq="h", name="time_utc_end")
    table = power.reindex(index)
    table["valid_fraction"] = table["valid_fraction"].fillna(0.0)
    table = table.join(solar_geometry(index, plant["lat"], plant["lon"], altitude))

    parsed = table["power_kw"].notna()
    night = parsed & (table["elev_max"] < NIGHT_ELEVATION) & (table["power_kw"] > NIGHT_SHARE * cap)
    values = table["power_kw"].to_numpy()
    stuck = parsed.to_numpy() & (values > 0) & (_runs(np.where(np.isfinite(values), values, -1.0)) >= STUCK_HOURS)
    table.loc[night | stuck, "power_kw"] = np.nan
    table["p"] = table["power_kw"] / cap

    for name in WEATHER:
        table[f"era_{name}"] = era[name].reindex(index)
        table[f"nwp_{name}"] = nwp[f"{name}_previous_day1"].reindex(index)
        table[f"nwp2_{name}"] = nwp[f"{name}_previous_day2"].reindex(index)
        table[f"nwpbm_{name}"] = best[f"{name}_previous_day1"].reindex(index)

    ok = table["p"].notna()
    have = power["valid_fraction"].reindex(index).fillna(0) > 0
    first, last = (ok[ok].index.min(), ok[ok].index.max()) if ok.any() else (pd.NaT, pd.NaT)
    span = (index >= first) & (index <= last) if ok.any() else np.zeros(len(index), bool)
    day = table["elev"] > 5.0

    def share(lo, hi):
        sel = (index > pd.Timestamp(lo, tz="UTC")) & (index <= pd.Timestamp(hi, tz="UTC"))
        return float(ok[sel].mean()) if sel.any() else np.nan

    record = {"no": plant.name, "name": plant["name"], "country": plant["country"], "source": plant["source"],
              "id": plant["sid"], "capacity_kw": cap, "tz": plant["tz"], "lat": plant["lat"], "lon": plant["lon"],
              "first_valid": first, "last_valid": last, "hours_in_span": int(span.sum()),
              "hours_valid": int(ok.sum()), "valid_share_in_span": float(ok[span].mean()) if span.any() else np.nan,
              "hours_with_records": int(have.sum()),
              "excluded_parser": int((have & ~parsed).sum()), "excluded_night": int(night.sum()),
              "excluded_stuck": int((stuck & ~night.to_numpy()).sum()),
              "daytime_valid_share_in_span": float(ok[span & day.to_numpy()].mean()) if span.any() else np.nan,
              "valid_2022-12_2024-11": share("2022-12-01", "2024-12-01"),
              "valid_2024-12_2025-11": share("2024-12-01", "2025-12-01"),
              "valid_2025-12_end": share("2025-12-01", "2100-01-01"),
              "nwp_first": table["nwp_shortwave_radiation"].first_valid_index(),
              "p_max": float(table["p"].max()), "p99": float(table["p"].quantile(0.99)),
              "mean_daytime_p": float(table.loc[day, "p"].mean())}
    return table, record


def load_table(no, root=ROOT) -> pd.DataFrame:
    return pd.read_parquet(root / "veri" / "islenmis" / "saatlik" / f"{int(no):02d}.parquet")


def main(root=ROOT):
    out = root / "veri" / "islenmis"
    (out / "saatlik").mkdir(parents=True, exist_ok=True)
    records = []
    for no, plant in load_plants(root).iterrows():
        table, record = build_plant(plant, root)
        table.to_parquet(out / "saatlik" / f"{no:02d}.parquet")
        records.append(record)
        print(f"{no:>2} {plant['name']:<28} {record['first_valid']:%Y-%m-%d}..{record['last_valid']:%Y-%m-%d}  "
              f"valid {record['valid_share_in_span']:.3f}  pre {record['valid_2022-12_2024-11']:.2f}  "
              f"test {record['valid_2024-12_2025-11']:.2f}  post {record['valid_2025-12_end']:.2f}  "
              f"excl p/n/s {record['excluded_parser']}/{record['excluded_night']}/{record['excluded_stuck']}  "
              f"p99 {record['p99']:.2f}", flush=True)
    pd.DataFrame(records).to_csv(out / "envanter.csv", sep=";", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
