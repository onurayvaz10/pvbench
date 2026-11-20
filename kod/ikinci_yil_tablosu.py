"""E10: ikinci test yili (Aralik 2025 - Ekim 2026).  Ana kiyasin modelleri yeniden egitilmeden puanlanir.

Kaynak (<cikti> = pvbench_cikti):  <cikti>/e10_ref (sureklilikler, klimatoloji, Ridge, LightGBM),  <cikti>/e10 (GRU)
                                   [pvbench.later; donemler y1, y1m, y2, y1p, y2p]
Aglarda once tohum ortalamasi alinir.  Denetimler: Ridge ve LightGBM'in y1 puani ana kosuyla ayni; bir hucrede butun
yontemler ayni ornek sayisini kullanmis.

Tablolar
    e10_hucreler.csv   (bilgi, ufuk, santral, yontem, donem): nRMSE, nMAE, yanlilik, aciksemaya gore beceri, n
    e10_ozet.csv       donem x yontem: santral ortalamalari (y1m ve y2: donemin butun ornekleri; y1p ve y2p: iki
                       yilda da ayni gun ve saatte gecerli ornegi olan saatler)
    e10_wilcoxon.csv   onceden belirlenen karsilastirmalar (santral = blok, cift yonlu Wilcoxon, alfa 0.05):
                       (1) her ogrenen yontemde becerinin degisimi, y2p - y1p  (Holm: ayni hucredeki yontemler)
                       (2) y2 icinde LightGBM - Ridge, GRU - Ridge, GRU - LightGBM (nRMSE; Holm: ayni hucredeki ciftler)
                       (3) ayni ciftler y1m icinde (karsilastirma icin)
Karsilastirma (1)'e, eslenmis saat sayisi en az MIN_PAIRED olan santraller girer.
Duyarlilik: duzey kaymasi bayrakli santraller olmadan (duzey_kaymasi.csv) ozet ayrica yazdirilir.

Kullanim:  python ikinci_yil_tablosu.py <cikti> <sonuclar klasoru>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import stats  # noqa: E402

LEARNED = ["ridge", "lightgbm", "gru"]
PAIRS = [("lightgbm", "ridge"), ("gru", "ridge"), ("gru", "lightgbm")]
KEYS = ["info", "horizon", "plant", "model", "period"]
MIN_PAIRED = 1000


def unpack(row: dict) -> dict:
    row["mean_ci_lo"], row["mean_ci_hi"] = row.pop("mean_ci")
    row["median_ci_lo"], row["median_ci_hi"] = row.pop("median_ci")
    return row


def main():
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    files = [f for name in ("e10_ref", "e10") for f in sorted((out / name).glob("later_w*.csv"))]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    skipped = df[df["status"] == "too_few_samples"]
    if len(skipped):
        cells = skipped[skipped["period"] == "y2"].drop_duplicates(["plant", "horizon", "info"])
        print("ikinci donemde ornegi yetersiz hucreler (disarida):",
              cells.groupby("name")["n"].agg(["size", "max"]).to_dict("index"))
    bad = df[~df["status"].isin(["ok", "too_few_samples"])]
    assert not len(bad), f"basarisiz kosular: {bad.groupby(['model', 'status']).size().to_dict()}"
    df = df[df["status"] == "ok"]
    assert not df.duplicated(KEYS + ["seed"]).any(), "ayni kosu iki kez yazilmis"
    check = df[(df["period"] == "y1") & df["model"].isin(["ridge", "lightgbm"])]
    worst = (check["nrmse"] - check["main_nrmse"]).abs().max()
    assert worst < 1e-6, f"y1 puani ana kosuyla ayni degil (en buyuk fark {worst})"

    cell = df.groupby(KEYS).agg(nrmse=("nrmse", "mean"), nmae=("nmae", "mean"), nmbe=("nmbe", "mean"), rmse=("rmse", "mean"),
                                seeds=("seed", "nunique"), n=("n", "first"), n_min=("n", "min"), n_max=("n", "max"),
                                name=("name", "first")).reset_index()
    assert (cell["n_min"] == cell["n_max"]).all()
    same = cell.groupby(["info", "horizon", "plant", "period"])["n"].nunique()
    assert (same == 1).all(), "yontemler ayni orneklerde degil"
    wide = cell.pivot_table(index=["info", "horizon", "plant", "period"], columns="model", values="rmse")
    skill = (1 - wide.div(wide["smart_persistence"], axis=0)).stack().rename("skill").reset_index()
    cell = cell.merge(skill, on=KEYS, how="left").drop(columns=["n_min", "n_max"])
    cell.to_csv(target / "e10_hucreler.csv", index=False)

    models = [m for m in ["persistence", "smart_persistence", "climatology"] + LEARNED if m in set(cell["model"])]
    learned = [m for m in LEARNED if m in models]
    summary = cell.groupby(["info", "horizon", "period", "model"]).agg(
        plants=("plant", "nunique"), n_median=("n", "median"), nrmse_mean=("nrmse", "mean"),
        nrmse_median=("nrmse", "median"), nmbe_mean=("nmbe", "mean"), skill_mean=("skill", "mean")).reset_index()
    summary.to_csv(target / "e10_ozet.csv", index=False)

    tests = []
    for (info, h), part in cell.groupby(["info", "horizon"]):
        paired = part[part["period"].isin(["y1p", "y2p"])]
        enough = paired[paired["period"] == "y2p"].groupby("plant")["n"].first()
        keep = enough[enough >= MIN_PAIRED].index
        s = paired[paired["plant"].isin(keep)].pivot_table(index=["model", "plant"], columns="period", values="skill")
        family = [unpack({"info": info, "horizon": h, "comparison": "skill y2p - y1p", "a": m, "b": m,
                          **stats.paired_wilcoxon(s.loc[m, "y2p"].to_numpy(), s.loc[m, "y1p"].to_numpy())}) for m in learned]
        for row, p in zip(family, stats.holm([f["p"] for f in family])):
            row["p_holm"] = p
        tests += family
        for period in ("y2", "y1m"):
            scores = part[part["period"] == period].pivot(index="plant", columns="model", values="nrmse")
            family = [unpack({"info": info, "horizon": h, "comparison": f"nrmse in {period}", "a": a, "b": b,
                              **stats.paired_wilcoxon(scores[a].to_numpy(), scores[b].to_numpy())})
                      for a, b in PAIRS if a in scores and b in scores]
            for row, p in zip(family, stats.holm([f["p"] for f in family])):
                row["p_holm"] = p
            tests += family
    tests = pd.DataFrame(tests)
    tests.to_csv(target / "e10_wilcoxon.csv", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 300)
    for value, label in (("nrmse_mean", "ortalama nRMSE (%)"), ("skill_mean", "acik gokyuzu surekliligine gore ortalama beceri")):
        t = summary.pivot_table(index=["info", "horizon", "model"], columns="period", values=value)
        print(f"\n=== {label} ===")
        print(t.loc[(slice(None), slice(None), models), ["y1", "y1m", "y2", "y1p", "y2p"]].round(3).to_string())
    counts = summary[summary["model"] == "ridge"].pivot_table(index=["info", "horizon"], columns="period", values="plants")
    print("\nsantral sayisi:", counts.to_dict("index"))
    print("\nkarsilastirmalar (fark a-b; eksi = a daha iyi / beceride arti = ikinci yil daha iyi):")
    for r in tests.itertuples():
        print(f"{r.info} h={r.horizon:<2} {r.comparison:<16} {r.a:>9} - {r.b:<9} {r.mean_diff:+.3f} "
              f"[{r.mean_ci_lo:+.3f}, {r.mean_ci_hi:+.3f}]  {r.wins_a}/{r.n}  W={r.W:.0f}  p={r.p:.3g}  p_holm={r.p_holm:.3g}")
    if not (target / "duzey_kaymasi.csv").is_file():  # not in the public deposit (describes the records)
        print("\nduzey_kaymasi.csv yok: bayrakli santraller olmadan duyarlilik ozeti atlandi")
        return
    flags = pd.read_csv(target / "duzey_kaymasi.csv", encoding="utf-8-sig")
    flagged = flags.loc[flags["flag"], "no"].tolist()
    rest = cell[~cell["plant"].isin(flagged) & cell["period"].isin(["y1p", "y2p"]) & cell["model"].isin(learned)]
    print(f"\n=== duzey kaymasi bayrakli santraller olmadan {flagged}: ortalama beceri ===")
    print(rest.pivot_table(index=["info", "horizon", "model"], columns="period", values="skill").round(3).to_string())


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
