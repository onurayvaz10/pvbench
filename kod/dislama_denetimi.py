"""Dislanan PVOutput sistemleri icin Ek Not 1'de verilen sayilarin ham kayitlardan yeniden uretimi.

Denetlenen sistemler, gece gucu araligi ve tarihler veri/dislama_denetimi.json dosyasindadir; dosya veriyle birlikte
saklanir ve dagitilmaz. Uc denetim:
1. Gece kayitlari: indirilen butun gun ici kayitlarda yerel 00:00-03:59 arasindaki kayit sayisi; gucu sifirdan buyuk
   olanlarin, verilen guc araliginda olanlarin ve ayni gunun onceki kaydina gore enerji sayaci artmis olanlarin payi.
2. Cift kayit: iki sistemin ortak zaman damgalarinda guc ve enerji degeri birebir ayni olan kayitlarin payi.
3. Gece kurali: saatlik tabloda gece kuralinin (gunes butun saat boyunca ufkun 3 dereceden fazla altinda ve guc kurulu
   gucun %1'inden buyuk) cikardigi saat sayisi, bu saatlerdeki ortalama guc (kurulu gucun payi), ilk gorulme zamani,
   verilen tarihten sonraki gece saatleri ve verilen aralikta kaydi olan saat sayisi.

Cikti: <sonuclar>/ek_dislama_denetimi.csv (sutun adlarindaki guc araligi ve tarihler girdi dosyasindan gelir)
Kullanim:  python dislama_denetimi.py <sonuclar klasoru>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import dataset, pvoutput  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAW = dataset.RAW / "pvoutput"
INPUTS = dataset.ROOT / "veri" / "dislama_denetimi.json"


def _utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def night_records(sid: str, name: str, band) -> dict:
    low, high = band
    raw = pvoutput.read_raw(RAW / sid).dropna(subset=["power_w"])
    raw["day"] = raw["local"].dt.normalize()
    raw["rise"] = raw.groupby("day")["energy_wh"].diff() > 0
    night = raw[raw["local"].dt.hour < 4]
    return {"check": "night records 00:00-03:59 local", "system": name, "sid": sid,
            "days_downloaded": int(raw["day"].nunique()), "n": len(night),
            "share_power_positive": float((night["power_w"] > 0).mean()),
            f"share_{low}_{high}_w": float(night["power_w"].between(low, high).mean()),
            "share_positive_and_counter_rising": float(((night["power_w"] > 0) & night["rise"]).mean()),
            "median_power_w_when_positive": float(night.loc[night["power_w"] > 0, "power_w"].median())}


def duplicates(a: str, b: str, name: str) -> dict:
    ra = pvoutput.read_raw(RAW / a).set_index("local")[["energy_wh", "power_w"]]
    rb = pvoutput.read_raw(RAW / b).set_index("local")[["energy_wh", "power_w"]]
    both = ra.join(rb, how="inner", lsuffix="_a", rsuffix="_b").dropna()
    same = (both["energy_wh_a"] == both["energy_wh_b"]) & (both["power_w_a"] == both["power_w_b"])
    monthly = same.groupby(both.index.to_period("M")).mean()
    return {"check": "identical records", "system": name, "sid": f"{a} / {b}", "n": len(both),
            "share_identical": float(same.mean()), "share_identical_min_month": float(monthly.min()),
            "share_identical_max_month": float(monthly.max())}


def night_hours(no: int, name: str, after: str, period) -> dict:
    """Saatlik tablodaki gece kurali; `after` ve `period` (baslangic, bitis] girdi dosyasindaki UTC tarihleridir."""
    plants = pd.read_csv(dataset.ROOT / "veri" / "santraller.csv", sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    sites = pd.read_csv(dataset.ROOT / "veri" / "konum_dogrulama.csv", sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    p = plants[plants["no"] == str(no)].iloc[0]
    s = sites[sites["no"] == str(no)].iloc[0]
    cap = float(p["kapasite_kw"])
    power = pvoutput.hourly_power(RAW / p["sid"], s["saat_dilimi"], cap)
    index = pd.date_range(dataset.START + pd.Timedelta(hours=1), power.index.max(), freq="h", name="time_utc_end")
    table = power.reindex(index).join(dataset.solar_geometry(index, float(s["enlem"]), float(s["boylam"])))
    parsed = table["power_kw"].notna()
    night = parsed & (table["elev_max"] < dataset.NIGHT_ELEVATION) & (table["power_kw"] > dataset.NIGHT_SHARE * cap)
    start, end, change = _utc(period[0]), _utc(period[1]), _utc(after)
    gap = (index > start) & (index <= end)
    label = f"{start:%Y_%m}_to_{end - pd.Timedelta(days=1):%Y_%m}"
    return {"check": "night rule in the hourly table", "system": name, "sid": p["sid"], "n": int(parsed.sum()),
            "night_hours_removed": int(night.sum()),
            "first_night_hour": str(table.index[night.to_numpy()].min()) if night.any() else "",
            "mean_power_share_in_night_hours": float((table.loc[night, "power_kw"] / cap).mean()) if night.any() else np.nan,
            f"night_hours_from_{change:%Y_%m}": int((night & (index > change)).sum()),
            f"hours_with_records_{label}": int(parsed[gap].sum()), f"hours_in_{label}": int(gap.sum())}


def main():
    target = Path(sys.argv[1])
    spec = json.loads(INPUTS.read_text(encoding="utf-8"))
    band = spec["gece_kayitlari"]["guc_araligi_w"]
    rows = [night_records(s["sid"], s["ad"], band) for s in spec["gece_kayitlari"]["sistemler"]]
    rows += [duplicates(d["a"], d["b"], d["ad"]) for d in spec["cift_kayit"]]
    rows += [night_hours(int(g["no"]), g["ad"], g["sonrasi"], g["aralik"]) for g in spec["gece_kurali"]]
    out = pd.DataFrame(rows)
    out.to_csv(target / "ek_dislama_denetimi.csv", index=False)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
