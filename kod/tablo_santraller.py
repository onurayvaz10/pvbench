"""Santral envanteri tablosu (makale Tablo 1 ve Ek tablo): veri/islenmis ve kosu ciktilarindan dogrudan uretilir.

Kaynak:
    veri/islenmis/envanter.csv              kayit donemi, gecerli saatler, kalite denetiminde elenen saatler
    veri/islenmis/altsaatlik_envanter.csv   PVOutput ham kayit araligi (ortanca, saniye) ve sayac adimi
    veri/islenmis/iklim.csv                 Koppen-Geiger sinifi
    veri/islenmis/saatlik/<no>.parquet      santralin zaman bolmesi (features.protocol) ve donem kapsamalari
    <cikti>/ana, <cikti>/ana_icon           egitim / dogrulama / test ornek sayilari (S1 ve S2; h = 1 ve 24)

Kapsama = donemdeki saatlerden hedefi gecerli olanlarin payi (donem: (baslangic, bitis] saat sonu etiketleri).
Cikti: tablo_santraller.csv
Kullanim:  python tablo_santraller.py <cikti> <sonuclar klasoru>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import dataset, features  # noqa: E402
from pvbench.experiment import read_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COUNTRY = {"Yunanistan": "Greece", "Almanya": "Germany", "Birleşik Krallık": "United Kingdom", "Macaristan": "Hungary",
           "Tayland": "Thailand", "Japonya": "Japan", "Hollanda": "Netherlands", "Hindistan": "India",
           "Belçika": "Belgium", "Türkiye": "Türkiye", "Lüksemburg": "Luxembourg", "İtalya": "Italy"}
SOURCES = {"S1": "ana", "S2": "ana_icon"}


def sample_counts(out: Path) -> pd.DataFrame:
    """n_train / n_val / n_test per plant, information set and horizon (identical for every method of a cell)."""
    parts = []
    for info, folder in SOURCES.items():
        m = read_metrics(out / folder)
        m = m[(m["stage"] == "main") & (m["info"] == info) & (m["status"] == "ok")]
        g = m.groupby(["plant", "horizon"])[["n_train", "n_val", "n_test"]]
        assert (g.nunique() == 1).all().all(), f"{info}: yontemler farkli ornek sayilari kullanmis"
        wide = g.first().unstack("horizon")
        wide.columns = [f"{name}_{info}_h{h}" for name, h in wide.columns]
        parts.append(wide)
    return pd.concat(parts, axis=1).astype(int)


def main():
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    processed = ROOT / "veri" / "islenmis"
    inv = pd.read_csv(processed / "envanter.csv", sep=";", encoding="utf-8-sig").set_index("no")
    fine = pd.read_csv(processed / "altsaatlik_envanter.csv", sep=";").set_index("no")
    climate = pd.read_csv(processed / "iklim.csv").set_index("no")
    counts = sample_counts(out)
    assert set(counts.index) == set(inv.index), "kosulardaki santraller envanterle ayni degil"

    rows = []
    for no, r in inv.iterrows():
        table = dataset.load_table(no)
        periods = features.protocol(table, no)
        ok, day = table["p"].notna(), table["elev"] > 5.0

        def coverage(lo, hi, mask=None):
            sel = (table.index > lo) & (table.index <= hi)
            sel = sel & mask.to_numpy() if mask is not None else sel
            return float(ok[sel].mean()), int(sel.sum())

        first = pd.Timestamp(r["first_valid"])
        train_cov, _ = coverage(max(periods["train_start"], first - pd.Timedelta(hours=1)), periods["val_start"])
        test_cov, test_hours = coverage(periods["test_start"], periods["test_end"])
        test_day_cov, _ = coverage(periods["test_start"], periods["test_end"], day)
        is_pvoutput = r["source"] == "pvoutput"
        rows.append({
            "no": no, "name": r["name"], "country": COUNTRY[r["country"]],
            "source": "PVOutput" if is_pvoutput else "EPİAŞ", "source_id": r["id"],
            "capacity_kw": r["capacity_kw"], "lat": round(r["lat"], 2), "lon": round(r["lon"], 2),
            "koppen": climate.loc[no, "koppen"],
            "raw_resolution_min": fine.loc[no, "spacing_median_s"] / 60.0 if is_pvoutput else 60.0,
            "counter_step_wh": fine.loc[no, "counter_step_wh"] if is_pvoutput else float("nan"),
            "first_valid": first.strftime("%Y-%m-%d"), "last_valid": pd.Timestamp(r["last_valid"]).strftime("%Y-%m-%d"),
            "hours_in_span": int(r["hours_in_span"]), "hours_valid": int(r["hours_valid"]),
            "missing_share_pct": 100.0 * (1.0 - r["valid_share_in_span"]),
            "hours_no_or_incomplete_record": int(r["hours_in_span"] - r["hours_valid"] - r["excluded_night"] - r["excluded_stuck"]),
            "excluded_parser": int(r["excluded_parser"]), "excluded_night": int(r["excluded_night"]),
            "excluded_stuck": int(r["excluded_stuck"]),
            "protocol": periods["name"],
            "val_start": periods["val_start"].strftime("%Y-%m-%d"),
            "test_start": periods["test_start"].strftime("%Y-%m-%d"),
            "test_end": (periods["test_end"] - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "train_coverage_pct": 100.0 * train_cov, "test_hours": test_hours, "test_coverage_pct": 100.0 * test_cov,
            "test_daytime_coverage_pct": 100.0 * test_day_cov,
            "p99": r["p99"], "mean_daytime_p": r["mean_daytime_p"], **counts.loc[no].to_dict()})
    table = pd.DataFrame(rows)
    lat, lon = np.radians(inv["lat"].to_numpy()), np.radians(inv["lon"].to_numpy())      # great-circle distances, km
    a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
    distance = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    np.fill_diagonal(distance, np.inf)
    table["nearest_plant_km"] = distance.min(axis=1).round(1)
    table["nearest_plant"] = inv["name"].to_numpy()[distance.argmin(axis=1)]
    table.to_csv(target / "tablo_santraller.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 320, "display.max_columns", 60)
    show = ["no", "name", "country", "source_id", "capacity_kw", "koppen", "raw_resolution_min", "first_valid",
            "last_valid", "hours_valid", "missing_share_pct", "protocol", "test_start", "test_end", "test_coverage_pct"]
    print(table[show].round(1).to_string(index=False))
    print()
    print(table[["no", "protocol", "train_coverage_pct", "test_daytime_coverage_pct", "excluded_parser",
                 "excluded_night", "excluded_stuck"] + list(counts.columns)].round(1).to_string(index=False))
    common = table[table["protocol"] == "common"]
    print(f"\nsantral {len(table)}, ulke {table['country'].nunique()}, ortak protokol {len(common)}; "
          f"kurulu guc {table['capacity_kw'].min()} - {table['capacity_kw'].max()} kW (ortanca {table['capacity_kw'].median():.1f}); "
          f"gecerli saat toplam {table['hours_valid'].sum()}; elenen (ayristirici/gece/takili) "
          f"{table['excluded_parser'].sum()}/{table['excluded_night'].sum()}/{table['excluded_stuck'].sum()}")
    near = table["nearest_plant_km"]
    print(f"en yakin santrale uzaklik (km): ortanca {near.median():.0f}, en az {near.min():.0f}, en cok {near.max():.0f}; "
          f"< 50 km: {int((near < 50).sum())} santral, < 100 km: {int((near < 100).sum())}, > 500 km: {int((near > 500).sum())}")
    per_country = table["country"].value_counts()
    print(f"bir ya da iki santralli ulke sayisi: {int((per_country <= 2).sum())} / {len(per_country)}")
    print("ulke basina santral:", table["country"].value_counts().to_dict())
    print("ham cozunurluk (dk):", table["raw_resolution_min"].value_counts().to_dict())
    print("Koppen:", table["koppen"].value_counts().to_dict())
    for col in counts.columns:
        print(f"{col:<16} toplam {table[col].sum():>8}  ortak protokol ortanca {int(common[col].median()):>6}  "
              f"en az {table[col].min():>6}  en cok {table[col].max():>6}")
    print(f"test kapsama (%): ortak protokol en az {common['test_coverage_pct'].min():.1f}, ortanca "
          f"{common['test_coverage_pct'].median():.1f}; < 90: {common.loc[common['test_coverage_pct'] < 90, 'name'].tolist()}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
