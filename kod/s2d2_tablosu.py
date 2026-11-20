"""Duyarlilik: arsivlenmis tahminin yasi.  S2 = hedef saat icin 24 saat once uretilmis ICON degeri;
S2d2 = 48 saat once uretilmis deger (her iki ufukta da tahmin aninda kesin olarak yayimlanmis).

Kaynak (<cikti> = pvbench_cikti):
    <cikti>/ana_s2d2  S2d2, referans yontemler (egitim_s2d2.ps1)
    <cikti>/ana_icon  S2
    <cikti>/ana_e2m   S0m (yalniz uretim gecmisi; S2 ile ayni saatler)
Yontemler: Ridge ve LightGBM.  Test ornekleri uc kumede ayni olmalidir (denetlenir); S2d2'nin egitim donemi bir gun
kisadir (48 saatlik deger arsivde bir gun sonra baslar).
Eslenmis karsilastirmalar (santral = blok, cift yonlu Wilcoxon, Holm ailesi = ayni ufuk ve karsilastirmadaki iki
yontem):  S2d2 - S2,  S2d2 - S0m.

Cikti: s2d2_hucreler.csv, s2d2_ozet.csv, s2d2_wilcoxon.csv
Kullanim:  python s2d2_tablosu.py <cikti> <sonuclar klasoru>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import stats  # noqa: E402
from pvbench.experiment import read_metrics  # noqa: E402

MODELS = ["ridge", "lightgbm"]
SOURCES = {"S0m": "ana_e2m", "S2": "ana_icon", "S2d2": "ana_s2d2"}
PAIRS = [("S2d2", "S2"), ("S2d2", "S0m")]


def main():
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    parts = []
    for info, folder in SOURCES.items():
        m = read_metrics(out / folder)
        parts.append(m[(m["stage"] == "main") & (m["info"] == info) & m["model"].isin(MODELS + ["smart_persistence"])])
    df = pd.concat(parts, ignore_index=True)
    bad = df[df["status"] != "ok"]
    assert not len(bad), f"basarisiz kosular: {bad.groupby(['info', 'status']).size().to_dict()}"
    keys = ["info", "horizon", "plant", "model"]
    assert not df.duplicated(keys).any(), "ayni kosu iki kez yazilmis"
    cell = df[keys + ["nrmse", "nmae", "rmse", "n_train", "n_test"]].copy()
    complete = cell.groupby(["info", "horizon", "model"])["plant"].nunique()
    assert (complete == 25).all(), f"eksik santral: {complete[complete < 25].to_dict()}"
    same = cell.groupby(["horizon", "plant"])["n_test"].nunique()
    assert (same == 1).all(), f"bilgi kumeleri ayni test orneklerinde degil: {same[same > 1].index.tolist()[:5]}"
    wide = cell.pivot_table(index=["info", "horizon", "plant"], columns="model", values="rmse")
    skill = (1 - wide.div(wide["smart_persistence"], axis=0)).stack().rename("skill_smart_persistence").reset_index()
    cell = cell.merge(skill, on=keys, how="left")
    cell.to_csv(target / "s2d2_hucreler.csv", index=False)

    cell = cell[cell["model"].isin(MODELS)]
    summary = cell.groupby(["horizon", "model", "info"]).agg(
        plants=("plant", "nunique"), nrmse_mean=("nrmse", "mean"), nrmse_median=("nrmse", "median"),
        skill_smart_persistence=("skill_smart_persistence", "mean"), n_train_median=("n_train", "median")).reset_index()
    summary.to_csv(target / "s2d2_ozet.csv", index=False)

    scores = cell.pivot_table(index=["horizon", "model", "plant"], columns="info", values="nrmse")
    tests = []
    for h in sorted(cell["horizon"].unique()):
        for a, b in PAIRS:
            family = [{"horizon": h, "a": a, "b": b, "model": m,
                       **stats.paired_wilcoxon(scores.loc[(h, m), a].to_numpy(), scores.loc[(h, m), b].to_numpy())}
                      for m in MODELS]
            for row, p in zip(family, stats.holm([f["p"] for f in family])):
                row["p_holm"] = p
                row["mean_ci_lo"], row["mean_ci_hi"] = row.pop("mean_ci")
                row["median_ci_lo"], row["median_ci_hi"] = row.pop("median_ci")
            tests += family
    tests = pd.DataFrame(tests)
    tests.to_csv(target / "s2d2_wilcoxon.csv", index=False)

    pd.set_option("display.width", 250)
    print("ortalama test nRMSE (%), 25 santral:")
    print(summary.pivot(index=["horizon", "model"], columns="info", values="nrmse_mean")[list(SOURCES)].round(3).to_string())
    print("\nfark a-b (nRMSE puani; eksi = a daha iyi), a'nin kazandigi santral sayisi, W, p, Holm p, %95 GA (ortalama fark):")
    for r in tests.itertuples():
        print(f"h={r.horizon:<2} {r.model:<9} {r.a}-{r.b}: {r.mean_diff:+.3f} [{r.mean_ci_lo:+.3f}, {r.mean_ci_hi:+.3f}]  "
              f"{r.wins_a}/{r.n}  W={r.W:.0f}  p={r.p:.3g}  p_holm={r.p_holm:.3g}  r={r.rank_biserial:+.2f}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
