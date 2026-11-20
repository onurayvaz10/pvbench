"""E9: cok ufuklu egri (1, 2, 3, 6, 12, 18, 24 saat), bilgi kumeleri S1 ve S2.

Kaynak (<cikti> = pvbench_cikti):
    <cikti>/ana_e9    ufuk 2, 3, 6, 12, 18;  S1 ve S2 (ICON);  dort ag (gru, tcn, lstm, cnn_gru) x 3 tohum + 6 referans
    <cikti>/ana       ufuk 1 ve 24, S1  (ana kiyas)
    <cikti>/ana_icon  ufuk 1 ve 24, S2  (ana kiyas)
Ana kiyastan yalniz ayni on yontem ve E9'daki tohumlar (0, 1, 2) alinir; boylece egrinin her noktasi ayni sayida
tohumun ortalamasidir (ana kiyas tablosundaki bes tohumlu degerlerden ucuncu basamakta ayrilabilir).
Ara ufuklarda ayar aramasi yapilmamistir: her ufuk en yakin ayarli ufkun (1 ya da 24 saat) secilmis ayarini kullanir.

Karsilastirmalar (santral = blok, cift yonlu Wilcoxon; bu liste E9 sonuclari gorulmeden once yazilmistir):
    A  her ag - Ridge       Holm ailesi = ayni bilgi kumesi ve ufuktaki dort ag
    B  her ag - LightGBM    Holm ailesi = ayni bilgi kumesi ve ufuktaki dort ag
    C  S2 - S1 (ayni yontem) Holm ailesi = ayni ufuktaki alti yontem; yalniz iki kumenin test ornek sayisi her
       santralde esitse "same_samples" = True

Cikti: e9_hucreler.csv, e9_ozet.csv, e9_wilcoxon.csv
Kullanim:  python e9_tablolari.py <cikti> <sonuclar klasoru>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import stats  # noqa: E402
from pvbench.experiment import REFERENCES, read_metrics  # noqa: E402

NETWORKS = ["gru", "tcn", "lstm", "cnn_gru"]
HORIZONS = [1, 2, 3, 6, 12, 18, 24]


def load(out: Path) -> pd.DataFrame:
    e9 = read_metrics(out / "ana_e9")
    seeds = sorted(e9["seed"].dropna().unique())
    s1 = read_metrics(out / "ana")
    s1 = s1[s1["info"] == "S1"]
    s2 = read_metrics(out / "ana_icon")
    s2 = s2[s2["info"] == "S2"]
    df = pd.concat([e9, s1, s2], ignore_index=True)
    df = df[(df["stage"] == "main") & df["model"].isin(NETWORKS + list(REFERENCES))]
    df = df[df["model"].isin(REFERENCES) | df["seed"].isin(seeds)]
    bad = df[df["status"] != "ok"]
    assert not len(bad), f"basarisiz kosular: {bad.groupby(['info', 'horizon', 'status']).size().to_dict()}"
    return df


def wilcoxon_family(scores, rows, label):
    family = []
    for key, a, b in rows:
        test = stats.paired_wilcoxon(scores[a].to_numpy(), scores[b].to_numpy())
        family.append({**key, "comparison": label, **test})
    for row, p in zip(family, stats.holm([f["p"] for f in family])):
        row["p_holm"] = p
        row["mean_ci_lo"], row["mean_ci_hi"] = row.pop("mean_ci")
        row["median_ci_lo"], row["median_ci_hi"] = row.pop("median_ci")
    return family


def main():
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    df = load(out)
    keys = ["info", "horizon", "plant", "model"]
    cell = df.groupby(keys).agg(nrmse=("nrmse", "mean"), nmae=("nmae", "mean"), rmse=("rmse", "mean"),
                                nrmse_day=("nrmse_day", "mean"), seeds=("seed", "nunique"),
                                seed_sd_nrmse=("nrmse", "std"), n=("n", "first"), n_min=("n", "min"), n_max=("n", "max"),
                                n_train=("n_train", "first")).reset_index()
    assert (cell["n_min"] == cell["n_max"]).all(), "bir hucrede tohumlar farkli test saati sayisi kullanmis"
    assert sorted(cell["horizon"].unique()) == HORIZONS, sorted(cell["horizon"].unique())
    complete = cell.groupby(["info", "horizon", "model"])["plant"].nunique()
    assert (complete == 25).all(), f"eksik santral: {complete[complete < 25].to_dict()}"
    net = cell["model"].isin(NETWORKS)
    assert (cell.loc[net, "seeds"] == 3).all(), "ag hucrelerinde uc tohum yok"
    same = cell.groupby(["info", "horizon", "plant"])["n"].nunique()
    assert (same == 1).all(), f"yontemler farkli test saatlerinde puanlanmis: {same[same > 1].index.tolist()[:5]}"
    wide = cell.pivot_table(index=["info", "horizon", "plant"], columns="model", values="rmse")
    skill = (1 - wide.div(wide["smart_persistence"], axis=0)).stack().rename("skill_smart_persistence").reset_index()
    cell = cell.merge(skill, on=keys, how="left")
    cell["rank"] = cell.groupby(["info", "horizon", "plant"])["nrmse"].rank(method="average")
    cell = cell.drop(columns=["n_min", "n_max"])
    cell.to_csv(target / "e9_hucreler.csv", index=False)

    summary = cell.groupby(["info", "horizon", "model"]).agg(
        plants=("plant", "nunique"), nrmse_mean=("nrmse", "mean"), nrmse_median=("nrmse", "median"),
        nmae_mean=("nmae", "mean"), nrmse_day_mean=("nrmse_day", "mean"),
        skill_smart_persistence=("skill_smart_persistence", "mean"), rank_mean=("rank", "mean"),
        wins=("rank", lambda s: int((s == 1).sum())), seed_sd_nrmse=("seed_sd_nrmse", "mean"),
        n_test_median=("n", "median")).reset_index()
    summary = summary.sort_values(["info", "horizon", "nrmse_mean"])
    summary.to_csv(target / "e9_ozet.csv", index=False)

    tests = []
    by_model = cell.pivot_table(index=["info", "horizon", "plant"], columns="model", values="nrmse")
    for info in ("S1", "S2"):
        for h in HORIZONS:
            part = by_model.loc[(info, h)]
            for reference, label in (("ridge", "network - ridge"), ("lightgbm", "network - lightgbm")):
                tests += wilcoxon_family(part, [({"info": info, "horizon": h, "a": m, "b": reference}, m, reference)
                                                for m in NETWORKS], label)
    by_info = cell.pivot_table(index=["horizon", "model", "plant"], columns="info", values="nrmse")
    n_info = cell.pivot_table(index=["horizon", "plant"], columns="info", values="n", aggfunc="first")
    for h in HORIZONS:
        family = wilcoxon_family_models(by_info, h)
        matched = bool((n_info.loc[h, "S1"] == n_info.loc[h, "S2"]).all())
        for row in family:
            row["same_samples"] = matched
        tests += family
    tests = pd.DataFrame(tests)
    tests.to_csv(target / "e9_wilcoxon.csv", index=False)

    pd.set_option("display.width", 250)
    for info in ("S1", "S2"):
        print(f"\nortalama test nRMSE (%), 25 santral, {info}:")
        print(summary[summary["info"] == info].pivot(index="model", columns="horizon", values="nrmse_mean")
              .sort_values(24).round(3).to_string())
        print(f"beceri (akilli kalicilik), {info}:")
        print(summary[summary["info"] == info].pivot(index="model", columns="horizon", values="skill_smart_persistence")
              .sort_values(24, ascending=False).round(3).to_string())
    print("\nS2 - S1 (nRMSE puani; eksi = tahmin girdisi yararli):")
    c = tests[tests["comparison"] == "S2 - S1"]
    print(c.pivot(index="a_model", columns="horizon", values="mean_diff").round(3).to_string())
    print("Holm p:")
    print(c.pivot(index="a_model", columns="horizon", values="p_holm").map(lambda p: f"{p:.2g}").to_string())
    print("ayni test ornekleri:", c.groupby("horizon")["same_samples"].first().to_dict())
    for label in ("network - ridge", "network - lightgbm"):
        c = tests[tests["comparison"] == label]
        for info in ("S1", "S2"):
            d = c[c["info"] == info]
            print(f"\n{label}, {info}: ortalama fark (eksi = ag daha iyi) / agin kazandigi santral / Holm p")
            show = d.assign(text=[f"{r.mean_diff:+.2f} {r.wins_a:>2}/25 {r.p_holm:.2g}" for r in d.itertuples()])
            print(show.pivot(index="a", columns="horizon", values="text").to_string())


def wilcoxon_family_models(by_info, h):
    part = by_info.loc[h]
    family = []
    for m in NETWORKS + ["ridge", "lightgbm"]:
        test = stats.paired_wilcoxon(part.loc[m, "S2"].to_numpy(), part.loc[m, "S1"].to_numpy())
        family.append({"info": "S2-S1", "horizon": h, "a": "S2", "b": "S1", "a_model": m, "comparison": "S2 - S1", **test})
    for row, p in zip(family, stats.holm([f["p"] for f in family])):
        row["p_holm"] = p
        row["mean_ci_lo"], row["mean_ci_hi"] = row.pop("mean_ci")
        row["median_ci_lo"], row["median_ci_hi"] = row.pop("median_ci")
    return family


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
