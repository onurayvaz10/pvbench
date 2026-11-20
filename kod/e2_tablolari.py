"""E2: bilgi kumesi karsilastirmasi, eslenmis ornekler (her bilgi kumesi S2 ile ayni egitim/dogrulama/test saatleri).

Kaynak (<cikti> = pvbench_cikti):
    <cikti>/ana_e2m   S0m (yalniz uretim gecmisi), S1m (+ ERA5 gecmisi), S3m (+ hedef saatin ERA5 degeri; ust sinir),
                      S12 (ERA5 gecmisi + ICON tahmini)
    <cikti>/ana_icon  S2 (+ ICON tahmini)
Yontemler: E2 icin secilen dort derin model (<cikti>/e2_modeller.json) ile Ridge ve LightGBM.
Her (bilgi, ufuk, santral, yontem) hucresinde tohum ortalamasi alinir; hucrelerin ayni test saatlerini kullandigi denetlenir.

Onceden belirlenen eslenmis karsilastirmalar (santral = blok, cift yonlu Wilcoxon, alfa 0.05; Holm ailesi = ayni ufuk
ve karsilastirmadaki alti yontem):  S1m-S0m, S2-S0m, S2-S1m, S12-S2, S3m-S2.

Cikti: e2_hucreler.csv, e2_ozet.csv, e2_wilcoxon.csv

Tam donem (ICON arsivinin kapsamadigi saatler dahil, her santralin butun test donemi), ayni yontemler:
    <cikti>/ana_e2    S0 (yalniz uretim gecmisi), S3 (+ hedef saatin ERA5 degeri; ust sinir)
    <cikti>/ana       S1 (+ ERA5 gecmisi; ana kiyas)
Karsilastirmalar: S1-S0, S3-S1, S3-S0.  Cikti: e2_tam_hucreler.csv, e2_tam_ozet.csv, e2_tam_wilcoxon.csv

Ayni bilgi kumesinde yontemler (ana kiyastaki ailelerin aynisi; S1 ve S2 disindaki kumelerde sonuclar goruldukten
sonra eklendi, makalede boyle belirtilir): her ag - LightGBM, her ag - Ridge (Holm aile icinde), Ridge - LightGBM.
Cikti: e2_yontem_wilcoxon.csv, e2_tam_yontem_wilcoxon.csv
Kullanim:  python e2_tablolari.py <cikti> <sonuclar klasoru>
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import stats  # noqa: E402
from pvbench.experiment import read_metrics  # noqa: E402

ORDER = ["S0m", "S1m", "S2", "S12", "S3m"]
PAIRS = [("S1m", "S0m"), ("S2", "S0m"), ("S2", "S1m"), ("S12", "S2"), ("S3m", "S2")]
FULL_ORDER = ["S0", "S1", "S3"]
FULL_PAIRS = [("S1", "S0"), ("S3", "S1"), ("S3", "S0")]
METRICS = ["nrmse", "nmae", "nrmse_day", "rmse"]


def tables(df, models, order, pairs, target, prefix, label):
    df = df[df["model"].isin(models + ["smart_persistence"]) & df["info"].isin(order)]
    bad = df[df["status"] != "ok"]
    if len(bad):
        print("UYARI: basarisiz kosular:", bad.groupby(["info", "status"]).size().to_dict())
    df = df[df["status"] == "ok"]
    keys = ["info", "horizon", "plant", "model"]
    cell = df.groupby(keys).agg(**{m: (m, "mean") for m in METRICS}, seeds=("seed", "nunique"), n=("n", "first"),
                                n_min=("n", "min"), n_max=("n", "max"), n_train=("n_train", "first")).reset_index()
    assert (cell["n_min"] == cell["n_max"]).all(), "bir hucrede tohumlar farkli test saati sayisi kullanmis"
    same = cell.groupby(["horizon", "plant"])[["n", "n_train"]].nunique()
    assert (same == 1).all().all(), f"bilgi kumeleri ayni orneklerde degil: {same[(same > 1).any(axis=1)].index.tolist()[:5]}"
    complete = cell.groupby(["info", "horizon", "model"])["plant"].nunique()
    assert (complete == 25).all(), f"eksik santral: {complete[complete < 25].to_dict()}"
    wide = cell.pivot_table(index=["info", "horizon", "plant"], columns="model", values="rmse")
    skill = (1 - wide.div(wide["smart_persistence"], axis=0)).stack().rename("skill_smart_persistence").reset_index()
    cell = cell.merge(skill, on=keys, how="left")
    cell.to_csv(target / f"{prefix}_hucreler.csv", index=False)

    cell = cell[cell["model"].isin(models)]
    summary = cell.groupby(["horizon", "model", "info"]).agg(
        plants=("plant", "nunique"), nrmse_mean=("nrmse", "mean"), nrmse_median=("nrmse", "median"),
        nmae_mean=("nmae", "mean"), skill_smart_persistence=("skill_smart_persistence", "mean")).reset_index()
    base = summary[summary["info"] == order[0]].set_index(["horizon", "model"])["nrmse_mean"]
    summary[f"change_vs_{order[0]}_pct"] = [100 * (r.nrmse_mean / base.loc[(r.horizon, r.model)] - 1) for r in summary.itertuples()]
    summary.to_csv(target / f"{prefix}_ozet.csv", index=False)

    tests = []
    scores = cell.pivot_table(index=["horizon", "model", "plant"], columns="info", values="nrmse")
    for h in sorted(cell["horizon"].unique()):
        for x, y in pairs:
            family = []
            for m in models:
                part = scores.loc[(h, m)]
                family.append({"horizon": h, "a": x, "b": y, "model": m,
                               **stats.paired_wilcoxon(part[x].to_numpy(), part[y].to_numpy())})
            for row, p in zip(family, stats.holm([f["p"] for f in family])):
                row["p_holm"] = p
                row["mean_ci_lo"], row["mean_ci_hi"] = row.pop("mean_ci")
                row["median_ci_lo"], row["median_ci_hi"] = row.pop("median_ci")
            tests += family
    tests = pd.DataFrame(tests)
    tests.to_csv(target / f"{prefix}_wilcoxon.csv", index=False)

    # Methods within one information set (same families as in the main benchmark: every network against LightGBM and
    # against ridge, Holm within the family; ridge against LightGBM as a single comparison).
    networks = [m for m in models if m not in ("ridge", "lightgbm")]
    by_model = cell.pivot_table(index=["horizon", "info", "plant"], columns="model", values="nrmse")
    between = []
    for h in sorted(cell["horizon"].unique()):
        for info in order:
            part = by_model.loc[(h, info)]
            for a_list, b in ((networks, "lightgbm"), (networks, "ridge"), (["ridge"], "lightgbm")):
                family = [{"horizon": h, "info": info, "a": a, "b": b,
                           **stats.paired_wilcoxon(part[a].to_numpy(), part[b].to_numpy())} for a in a_list]
                for row, p in zip(family, stats.holm([f["p"] for f in family])):
                    row["p_holm"] = p
                    row["mean_ci_lo"], row["mean_ci_hi"] = row.pop("mean_ci")
                    row["median_ci_lo"], row["median_ci_hi"] = row.pop("median_ci")
                between += family
    between = pd.DataFrame(between)
    between.to_csv(target / f"{prefix}_yontem_wilcoxon.csv", index=False)

    pd.set_option("display.width", 250)
    for h in sorted(cell["horizon"].unique()):
        part = summary[summary["horizon"] == h]
        print(f"\n=== h={h}: ortalama test nRMSE (%), 25 santral, {label} ===")
        print(part.pivot(index="model", columns="info", values="nrmse_mean")[order].loc[models].round(2).to_string())
        t = tests[tests["horizon"] == h]
        print("\nfark a-b (nRMSE puani; eksi = a daha iyi), a'nin kazandigi santral sayisi, Holm p:")
        show = t.assign(pair=t["a"] + "-" + t["b"], cell=[f"{r.mean_diff:+.2f} ({r.wins_a}/25, p={r.p_holm:.3g})"
                                                         for r in t.itertuples()])
        print(show.pivot(index="model", columns="pair", values="cell")[[f"{x}-{y}" for x, y in pairs]].loc[models].to_string())
        b = between[between["horizon"] == h]
        print("\nayni bilgi kumesinde yontemler: fark a-b (eksi = a daha iyi), a'nin kazandigi santral sayisi, Holm p:")
        show = b.assign(pair=b["a"] + " - " + b["b"], cell=[f"{r.mean_diff:+.2f} ({r.wins_a}/25, p={r.p_holm:.3g})"
                                                            for r in b.itertuples()])
        print(show.pivot(index="pair", columns="info", values="cell")[order].to_string())


def main():
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    models = json.loads((out / "e2_modeller.json").read_text(encoding="utf-8"))["chosen"] + ["ridge", "lightgbm"]
    a = read_metrics(out / "ana_e2m")
    b = read_metrics(out / "ana_icon")
    df = pd.concat([a[a["stage"] == "main"], b[(b["stage"] == "main") & (b["info"] == "S2")]], ignore_index=True)
    tables(df, models, ORDER, PAIRS, target, "e2", "eslenmis ornekler")
    a = read_metrics(out / "ana_e2")
    b = read_metrics(out / "ana")
    df = pd.concat([a[a["stage"] == "main"], b[(b["stage"] == "main") & (b["info"] == "S1")]], ignore_index=True)
    tables(df, models, FULL_ORDER, FULL_PAIRS, target, "e2_tam", "tam test donemi")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
