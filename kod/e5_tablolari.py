"""E5: (1) hareketli ortalama ablasyonu, (2) girdi gruplarinin permutasyon onemi.

Kaynak (<cikti> = pvbench_cikti):
    <cikti>/ana_e5ma   S1a = S1 + guc serisinin geriye donuk 3 saatlik ortalamasi; GRU ve hibrit (3 tohum) + referanslar
    <cikti>/ana        S1 (ana kiyas; aglarda yalniz ayni tohumlar 0-2)
    <cikti>/e5_onem    importance_w*.csv (pvbench.importance): S1 ve S2, h = 1 ve 24; Ridge, LightGBM, GRU, hibrit

(1) Eslenmis karsilastirma S1a - S1 (santral = blok, cift yonlu Wilcoxon; Holm ailesi = ayni ufuktaki dort yontem:
    ridge, lightgbm, gru, cnn_lstm_gru).  Liste sonuclar gorulmeden once yazilmistir.
(2) Onem = bir grubun kanallari test ornekleri arasinda (gunun ayni saatinde) karistirildiginda test nRMSE'sindeki
    artis (puan; 5 tekrarin ortalamasi).  Aglar onem icin ayni ayar ve tohumla yeniden egitilir; "base_nrmse" bu
    yeniden egitilen modelin, "main_nrmse" ana kosunun hatasidir - aralarindaki fark tabloya yazilir.
    Ozet: once tohum ortalamasi (santral basina), sonra 25 santral uzerinden ortalama / medyan; grubun o santralde
    en onemli grup oldugu santral sayisi.  Betimleyicidir; test yapilmaz.

Cikti: e5_ma_hucreler.csv, e5_ma_ozet.csv, e5_onem_hucreler.csv, e5_onem_ozet.csv, e5_onem_yeniden_egitim.csv
Kullanim:  python e5_tablolari.py <cikti> <sonuclar klasoru>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import stats  # noqa: E402
from pvbench.experiment import read_metrics  # noqa: E402
from pvbench.importance import GROUP_ORDER  # noqa: E402

MODELS = ["ridge", "lightgbm", "gru", "cnn_lstm_gru"]


def moving_average(out: Path, target: Path):
    a = read_metrics(out / "ana_e5ma")
    seeds = sorted(a["seed"].dropna().unique())
    b = read_metrics(out / "ana")
    b = b[(b["info"] == "S1") & (b["seed"].isin(seeds) | b["seed"].isna() | ~b["model"].isin(["gru", "cnn_lstm_gru"]))]
    df = pd.concat([a, b], ignore_index=True)
    df = df[(df["stage"] == "main") & df["model"].isin(MODELS)]
    df = df[~df["model"].isin(["gru", "cnn_lstm_gru"]) | df["seed"].isin(seeds)]
    assert (df["status"] == "ok").all()
    keys = ["info", "horizon", "plant", "model"]
    cell = df.groupby(keys).agg(nrmse=("nrmse", "mean"), nmae=("nmae", "mean"), seeds=("seed", "nunique"),
                                n=("n", "first"), n_features=("n_features", "first")).reset_index()
    complete = cell.groupby(["info", "horizon", "model"])["plant"].nunique()
    assert len(complete) == 16 and (complete == 25).all(), complete.to_dict()
    net = cell["model"].isin(["gru", "cnn_lstm_gru"])
    assert (cell.loc[net, "seeds"] == 3).all() and (cell.loc[~net, "seeds"] <= 1).all()
    same = cell.groupby(["horizon", "plant"])["n"].nunique()
    assert (same == 1).all(), f"S1 ve S1a farkli test saatlerinde: {same[same > 1].index.tolist()[:5]}"
    cell.to_csv(target / "e5_ma_hucreler.csv", index=False)
    wide = cell.pivot_table(index=["horizon", "model", "plant"], columns="info", values="nrmse")
    rows = []
    for h in sorted(cell["horizon"].unique()):
        family = [{"horizon": h, "model": m, "nrmse_S1": wide.loc[(h, m), "S1"].mean(),
                   "nrmse_S1a": wide.loc[(h, m), "S1a"].mean(),
                   **stats.paired_wilcoxon(wide.loc[(h, m), "S1a"].to_numpy(), wide.loc[(h, m), "S1"].to_numpy())}
                  for m in MODELS]
        for row, p in zip(family, stats.holm([f["p"] for f in family])):
            row["p_holm"] = p
            row["mean_ci_lo"], row["mean_ci_hi"] = row.pop("mean_ci")
            row["median_ci_lo"], row["median_ci_hi"] = row.pop("median_ci")
        rows += family
    table = pd.DataFrame(rows).rename(columns={"mean_diff": "diff_S1a_minus_S1", "wins_a": "plants_S1a_better"})
    table.to_csv(target / "e5_ma_ozet.csv", index=False)
    print("hareketli ortalama ablasyonu (S1a - S1, nRMSE puani; eksi = ek girdi yararli):")
    print(table[["horizon", "model", "nrmse_S1", "nrmse_S1a", "diff_S1a_minus_S1", "mean_ci_lo", "mean_ci_hi",
                 "plants_S1a_better", "n", "W", "p", "p_holm"]].round(4).to_string(index=False))


def importance(out: Path, target: Path):
    d = pd.concat([pd.read_csv(f) for f in sorted((out / "e5_onem").glob("importance_w*.csv"))], ignore_index=True)
    run = ["plant", "horizon", "info", "model", "seed"]
    assert not d.duplicated(run + ["group"]).any()
    assert (d.groupby(run)["group"].nunique() == len(GROUP_ORDER)).all()
    runs = d.drop_duplicates(run).assign(refit_minus_main=lambda x: x["base_nrmse"] - x["main_nrmse"])
    refit = runs.groupby(["info", "horizon", "model"]).agg(
        runs=("seed", "size"), base_nrmse=("base_nrmse", "mean"), main_nrmse=("main_nrmse", "mean"),
        mean_diff=("refit_minus_main", "mean"), mean_abs_diff=("refit_minus_main", lambda s: s.abs().mean()),
        max_abs_diff=("refit_minus_main", lambda s: s.abs().max())).reset_index()
    refit.to_csv(target / "e5_onem_yeniden_egitim.csv", index=False)
    cell = d.groupby(["info", "horizon", "model", "plant", "group"]).agg(
        delta=("delta", "mean"), base_nrmse=("base_nrmse", "mean"), seeds=("seed", "nunique"),
        n_channels=("n_channels", "first")).reset_index()
    complete = cell.groupby(["info", "horizon", "model", "group"])["plant"].nunique()
    assert len(complete) == 2 * 2 * 4 * 7 and (complete == 25).all()
    cell["relative"] = cell["delta"] / cell["base_nrmse"]
    cell["rank"] = cell.groupby(["info", "horizon", "model", "plant"])["delta"].rank(ascending=False, method="min")
    cell.to_csv(target / "e5_onem_hucreler.csv", index=False)
    summary = cell.groupby(["info", "horizon", "model", "group"]).agg(
        n_channels=("n_channels", "first"), delta_mean=("delta", "mean"), delta_median=("delta", "median"),
        delta_min=("delta", "min"), delta_max=("delta", "max"), relative_mean=("relative", "mean"),
        plants_top=("rank", lambda s: int((s == 1).sum())), plants_top2=("rank", lambda s: int((s <= 2).sum())),
        plants_negative=("delta", lambda s: int((s < 0).sum()))).reset_index()
    summary["group"] = pd.Categorical(summary["group"], GROUP_ORDER, ordered=True)
    summary = summary.sort_values(["info", "horizon", "model", "group"])
    summary.to_csv(target / "e5_onem_ozet.csv", index=False)
    print("\nyeniden egitilen model - ana kosu (nRMSE puani):")
    print(refit.round(3).to_string(index=False))
    for value, text in (("delta_mean", "ortalama artis (nRMSE puani)"), ("plants_top", "grubun en onemli oldugu santral sayisi")):
        print(f"\npermutasyon onemi, {text}:")
        print(summary.pivot(index=["info", "horizon", "model"], columns="group", values=value).round(2).to_string())


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 250)
    moving_average(Path(sys.argv[1]), Path(sys.argv[2]))
    importance(Path(sys.argv[1]), Path(sys.argv[2]))
