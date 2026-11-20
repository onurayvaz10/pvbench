"""E7: katmanli analiz (kesif niteliginde).  Ana kiyasin santral bazli sonuclari santral ozelliklerine gore
katmanlara ayrilir; yeni model egitilmez.

Girdi : <sonuclar>/e1_hucreler.csv (sonuc_tablolari.py), veri/islenmis/envanter.csv, veri/islenmis/iklim.csv
        (veri_cekme/iklim_cek.py; Koppen-Geiger sinifi, ERA5 1991-2020 normalleri).
Katman: iklim ana grubu ve sinifi, kurulu guc sinifi (< 10 kW, 10-100 kW, >= 100 kW), gecerli veri orani
        (>= %95, < %95), veri kaynagi (PVOutput, EPIAS), bolme protokolu (ortak, kendi).
Cikti : e7_santraller.csv        santral ozellikleri ve katmanlari
        e7_katmanlar.csv         katman x bilgi kumesi x ufuk x yontem: ortalama/medyan nRMSE, beceri, ortalama sira
        e7_korelasyon.csv        santral ozelligi ile hata/beceri arasinda Spearman rho, bootstrap %95 GA, p, Holm
        e7_grup_testleri.csv     en az 3 santralli duzeyler arasinda Kruskal-Wallis (iki duzeyde Mann-Whitney ile ayni)
        e7_sira_tutarliligi.csv  katman ici yontem siralamasinin butun santrallerdeki siralamayla Spearman iliskisi
Butun testler cift yonludur; alfa = 0.05.  Holm ailesi: ayni olcut, hucre ve yontem icindeki santral ozellikleri.

Kullanim:  python katmanli_analiz.py <sonuclar klasoru>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import dataset, features, stats  # noqa: E402

KOK = Path(__file__).resolve().parents[1]
CELLS = [("S1", 1), ("S1", 24), ("S2", 1), ("S2", 24)]
FOCUS = ["ridge", "lightgbm", "gru", "cnn_lstm_gru", "deep_mean"]
COVARIATES = ["log10_capacity_kw", "abs_lat", "valid_share", "mean_daytime_p", "mat_c", "map_mm"]
STRATA = ["koppen_main", "koppen", "capacity_class", "coverage_class", "source", "protocol"]
MEASURES = ["nrmse", "skill_smart_persistence"]


def plant_table():
    env = pd.read_csv(KOK / "veri" / "islenmis" / "envanter.csv", sep=";").set_index("no")
    climate = pd.read_csv(KOK / "veri" / "islenmis" / "iklim.csv").set_index("no")
    t = env[["name", "country", "source", "capacity_kw", "lat", "lon", "mean_daytime_p"]].copy()
    t["valid_share"] = env["valid_share_in_span"]
    t = t.join(climate[["koppen", "koppen_main", "mat_c", "map_mm"]])
    t["protocol"] = [features.protocol(dataset.load_table(no), no)["name"] for no in t.index]
    t["log10_capacity_kw"] = np.log10(t["capacity_kw"])
    t["abs_lat"] = t["lat"].abs()
    t["capacity_class"] = pd.cut(t["capacity_kw"], [0, 10, 100, np.inf], right=False,
                                 labels=["<10 kW", "10-100 kW", ">=100 kW"]).astype(str)
    t["coverage_class"] = np.where(t["valid_share"] >= 0.95, ">=95 %", "<95 %")
    return t


def main():
    target = Path(sys.argv[1])
    cells = pd.read_csv(target / "e1_hucreler.csv")
    cells = cells[[(i, h) in CELLS for i, h in zip(cells["info"], cells["horizon"])]]
    deep = cells[cells["kind"] == "deep"].groupby(["info", "horizon", "plant"], as_index=False)[MEASURES].mean()
    deep["model"] = "deep_mean"
    plants = plant_table()
    if set(cells["plant"]) != set(plants.index):
        raise SystemExit("santral kumeleri uyusmuyor (e1_hucreler.csv ile envanter.csv)")
    plants.to_csv(target / "e7_santraller.csv", encoding="utf-8")
    full = cells.join(plants, on="plant")

    rows = []
    for var in STRATA:
        g = full.groupby([var, "info", "horizon", "model"]).agg(
            n_plants=("plant", "nunique"), nrmse_mean=("nrmse", "mean"), nrmse_median=("nrmse", "median"),
            skill_smart_persistence=("skill_smart_persistence", "mean"), mean_rank_all=("rank_all", "mean")).reset_index()
        g = g.rename(columns={var: "level"})
        g.insert(0, "stratum", var)
        rows.append(g)
    pd.concat(rows, ignore_index=True).to_csv(target / "e7_katmanlar.csv", index=False, encoding="utf-8")

    focus = pd.concat([cells[cells["model"].isin(FOCUS)][["info", "horizon", "plant", "model"] + MEASURES], deep])
    focus = focus.join(plants, on="plant")
    corr, tests = [], []
    for (info, h, model), part in focus.groupby(["info", "horizon", "model"]):
        for measure in MEASURES:
            family = []
            for cov in COVARIATES:
                r = stats.spearman_ci(part[cov], part[measure])
                family.append({"info": info, "horizon": h, "model": model, "measure": measure, "covariate": cov,
                               "n": r["n"], "rho": r["rho"], "ci_lo": r["ci"][0], "ci_hi": r["ci"][1], "p": r["p"]})
            for row, adjusted in zip(family, stats.holm([f["p"] for f in family])):
                row["p_holm"] = adjusted
            corr += family
            for var in STRATA:
                groups = {k: v[measure].dropna().to_numpy() for k, v in part.groupby(var)}
                groups = {k: v for k, v in groups.items() if len(v) >= 3}
                if len(groups) < 2:
                    continue
                H, p = sps.kruskal(*groups.values())
                tests.append({"stratum": var, "info": info, "horizon": h, "model": model, "measure": measure,
                              "levels": "; ".join(f"{k} (n={len(v)})" for k, v in groups.items()),
                              "plants_used": int(sum(len(v) for v in groups.values())), "H": float(H), "p": float(p)})
    pd.DataFrame(corr).to_csv(target / "e7_korelasyon.csv", index=False)
    pd.DataFrame(tests).to_csv(target / "e7_grup_testleri.csv", index=False, encoding="utf-8")

    consistency = []
    references = set(cells.loc[cells["kind"] == "reference", "model"])
    overall =full.groupby(["info", "horizon", "model"])["rank_all"].mean()
    for var in STRATA:
        for (level, info, h), part in full.groupby([var, "info", "horizon"]):
            n = part["plant"].nunique()
            if n < 3:
                continue
            inside = part.groupby("model")["rank_all"].mean()
            ref = overall.loc[(info, h)].reindex(inside.index)
            consistency.append({"stratum": var, "level": level, "info": info, "horizon": h, "n_plants": n,
                                "rho_with_overall_ranking": float(sps.spearmanr(inside, ref)[0]),
                                "best_method": inside.idxmin(),
                                "best_deep": inside[[m for m in inside.index if m not in references]].idxmin(),
                                "hybrid_rank_among_17": float(inside.rank()["cnn_lstm_gru"])})
    pd.DataFrame(consistency).to_csv(target / "e7_sira_tutarliligi.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 250)
    print(plants[["name", "country", "capacity_kw", "koppen", "capacity_class", "coverage_class", "source", "protocol"]].to_string())
    for var in STRATA:
        print(f"\n{var}: {plants[var].value_counts().to_dict()}")
    c = pd.DataFrame(corr)
    print("\nkurulu guc ile iliski (Spearman rho [GA], p, Holm):")
    print(c[c["covariate"] == "log10_capacity_kw"].round(3).to_string(index=False))
    print("\nHolm sonrasi p < 0.05 olan iliskiler:")
    print(c[c["p_holm"] < 0.05].round(3).to_string(index=False))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
