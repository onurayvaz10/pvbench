"""Ek dosya tablolari (Supplementary Tables): kosu ciktilarindan ve sonuclar/*.csv dosyalarindan uretilir.

    ek_s5_olcutler.csv        ana kiyas (E1): yontem x hucre, butun olcutlerin santral ortalamasi ve ortancasi
                              (nRMSE, nMAE, nMBE, R2 = NSE, Theil U2, MASE; yalniz gunduz saatleri; MAPE %2 / %5 / %10)
    ek_mape_duyarlilik.csv    MAPE esigi duyarliligi: esikler arasinda yontem siralamasinin Spearman iliskisi,
                              her esikte en iyi yontem ve en iyi ag
    ek_s2_hizalama_konum.csv  zaman hizalamasi (hizalama_denetimi.py) ve konum denetimi (veri/konum_dogrulama.csv)

Her (bilgi, ufuk, santral, yontem) hucresinde once tohum ortalamasi alinir; ozetler santraller uzerindendir.
MAPE yalniz olculen gucun kurulu gucun esik payindan buyuk oldugu saatlerde hesaplanir (pvbench.metrics.mape).

Kullanim:  python ek_tablolar.py <cikti> <sonuclar klasoru>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench.experiment import REFERENCES, read_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CELLS = [("S1", 1), ("S1", 24), ("S2", 1), ("S2", 24)]
SOURCES = {"S1": "ana", "S2": "ana_icon"}
METRICS = ["nrmse", "nmae", "nmbe", "r2", "theil_u2", "mase", "nrmse_day", "nmae_day", "nmbe_day", "r2_day",
           "theil_u2_day", "mase_day", "mape_02", "mape_05", "mape_10"]
MAPE = ["mape_02", "mape_05", "mape_10"]


def main():
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    parts = []
    for info, folder in SOURCES.items():
        m = read_metrics(out / folder)
        parts.append(m[(m["stage"] == "main") & (m["info"] == info) & (m["status"] == "ok")])
    df = pd.concat(parts, ignore_index=True)
    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    keys = ["info", "horizon", "plant", "model"]
    cell = df.groupby(keys)[METRICS].mean().reset_index()
    assert (cell.groupby(["info", "horizon", "model"])["plant"].nunique() == 25).all()

    wide = {}
    for stat in ("mean", "median"):
        g = cell.groupby(["info", "horizon", "model"])[METRICS].agg(stat)
        wide[stat] = g.add_suffix(f"_{stat}")
    table = pd.concat(wide.values(), axis=1).reset_index()
    table.insert(3, "kind", ["reference" if m in REFERENCES else "network" for m in table["model"]])
    table = table.sort_values(["info", "horizon", "nrmse_mean"])
    table.to_csv(target / "ek_s5_olcutler.csv", index=False)

    rows = []
    mean = cell.groupby(["info", "horizon", "model"])[MAPE + ["nrmse"]].mean()
    for info, h in CELLS:
        part = mean.loc[(info, h)]
        learned = part.drop(index=[m for m in ("persistence", "seasonal_persistence", "climatology") if m in part.index])
        networks = part.loc[[m for m in part.index if m not in REFERENCES]]
        row = {"info": info, "horizon": h, "n_methods": len(part)}
        for a, b in (("mape_02", "mape_05"), ("mape_10", "mape_05"), ("mape_02", "mape_10"), ("mape_05", "nrmse")):
            row[f"spearman_{a}_{b}"] = part[a].corr(part[b], method="spearman")
        for col in MAPE:
            row[f"best_{col}"] = part[col].idxmin()
            row[f"best_network_{col}"] = networks[col].idxmin()
            row[f"top5_{col}"] = " ".join(learned[col].nsmallest(5).index)
            row[f"hybrid_rank_networks_{col}"] = int(networks[col].rank()["cnn_lstm_gru"])
        rows.append(row)
    sens = pd.DataFrame(rows)
    sens.to_csv(target / "ek_mape_duyarlilik.csv", index=False)

    pd.set_option("display.width", 320, "display.max_columns", 60, "display.max_rows", 200)
    show = ["model", "nrmse_mean", "nmae_mean", "nmbe_mean", "r2_median", "theil_u2_median", "mase_median",
            "nrmse_day_mean", "mape_02_mean", "mape_05_mean", "mape_10_mean"]
    for (info, h), part in table.groupby(["info", "horizon"], sort=False):
        print(f"\n=== {info} h={h} ===")
        print(part[show].round(3).to_string(index=False))
    print("\n=== MAPE esik duyarliligi ===")
    print(sens.T.to_string())

    # The alignment and location table describes the records of individual plants. The public deposit omits its
    # inputs (PVOutput records may not be redistributed); it is then skipped.
    align_path, site_path = target / "hizalama_denetimi.csv", ROOT / "veri" / "konum_dogrulama.csv"
    if not (align_path.is_file() and site_path.is_file()):
        print("\nek_s2_hizalama_konum.csv atlandi: hizalama_denetimi.csv ya da veri/konum_dogrulama.csv yok")
        return
    align = pd.read_csv(align_path, encoding="utf-8-sig")
    site = pd.read_csv(site_path, sep=";", encoding="utf-8-sig")
    site = site[["no", "saat_dilimi", "enlem", "boylam", "n_gun", "boylam_tahmin", "boylam_farki"]].rename(columns={
        "saat_dilimi": "time_zone", "enlem": "lat", "boylam": "lon", "n_gun": "days_used",
        "boylam_tahmin": "lon_from_generation_midpoint", "boylam_farki": "lon_difference_deg"})
    merged = align.merge(site, on="no", how="left")
    merged["solar_time_difference_min"] = merged["lon_difference_deg"] * 4.0
    merged.to_csv(target / "ek_s2_hizalama_konum.csv", index=False, encoding="utf-8-sig")
    d = merged["lon_difference_deg"].abs()
    print(f"\nboylam farki: < 1 derece {int((d < 1).sum())} santral, 1-5 derece {int(((d >= 1) & (d <= 5)).sum())}, en cok {d.max():.2f}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
