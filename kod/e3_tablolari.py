"""E3: parametre sayisi eslenmis kiyas.  On bir agin hepsi genislik carpaniyla yaklasik 50 000 parametreye getirilir;
dropout, ogrenme hizi ve yigin boyutu ana kiyasta secilen degerlerde kalir (yeni ayar aramasi yok).

Kaynak (<cikti> = pvbench_cikti):
    <cikti>/ana_e3    eslenmis boyut ("matched"), S1 ve S2 (ICON), h = 1 ve 24, 3 tohum
    <cikti>/ana       ayarli boyut ("tuned"), S1;   <cikti>/ana_icon  ayarli boyut, S2
Ana kiyastan yalniz ayni tohumlar (0, 1, 2) alinir; boylece iki boyut ayni sayida tohumun ortalamasidir.

Karsilastirmalar (santral = blok, cift yonlu Wilcoxon; bu liste E3 sonuclari gorulmeden once yazilmistir):
    A  ayni ag, eslenmis - ayarli boyut          Holm ailesi = ayni hucredeki (bilgi kumesi, ufuk) 11 ag
    B  eslenmis boyutta hibrit - diger her ag    Holm ailesi = ayni hucredeki 10 ag
    C  eslenmis boyutta 11 agin Friedman testi (Iman-Davenport) ve ortalama siralar; ayarli boyuttaki ortalama
       siralarla Spearman korelasyonu

Cikti: e3_hucreler.csv, e3_ozet.csv, e3_wilcoxon.csv, e3_friedman.csv
Kullanim:  python e3_tablolari.py <cikti> <sonuclar klasoru>
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import stats  # noqa: E402
from pvbench.experiment import REFERENCES, read_metrics  # noqa: E402

CELLS = [("S1", 1), ("S1", 24), ("S2", 1), ("S2", 24)]
HYBRID = "cnn_lstm_gru"


def load(out: Path) -> pd.DataFrame:
    matched = read_metrics(out / "ana_e3").assign(size="matched")
    seeds = sorted(matched["seed"].dropna().unique())
    s1 = read_metrics(out / "ana")
    s2 = read_metrics(out / "ana_icon")
    tuned = pd.concat([s1[s1["info"] == "S1"], s2[s2["info"] == "S2"]]).assign(size="tuned")
    tuned = tuned[tuned["seed"].isin(seeds)]
    df = pd.concat([matched, tuned], ignore_index=True)
    df = df[(df["stage"] == "main") & ~df["model"].isin(REFERENCES)]
    bad = df[df["status"] != "ok"]
    assert not len(bad), f"basarisiz kosular: {bad.groupby(['size', 'info', 'status']).size().to_dict()}"
    return df


def unpack(rows):
    for row in rows:
        row["mean_ci_lo"], row["mean_ci_hi"] = row.pop("mean_ci")
        row["median_ci_lo"], row["median_ci_hi"] = row.pop("median_ci")
    return rows


def main():
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    df = load(out)
    keys = ["size", "info", "horizon", "plant", "model"]
    cell = df.groupby(keys).agg(nrmse=("nrmse", "mean"), nmae=("nmae", "mean"), seeds=("seed", "nunique"),
                                seed_sd_nrmse=("nrmse", "std"), n=("n", "first"), n_min=("n", "min"), n_max=("n", "max"),
                                params=("params", "first"), params_max=("params", "max"), width=("width", "first"),
                                fit_s=("fit_s", "mean"), epochs=("epochs", "mean")).reset_index()
    assert (cell["n_min"] == cell["n_max"]).all() and (cell["params"] == cell["params_max"]).all()
    assert (cell["seeds"] == 3).all(), cell["seeds"].value_counts().to_dict()
    complete = cell.groupby(["size", "info", "horizon", "model"])["plant"].nunique()
    assert len(complete) == 2 * 4 * 11 and (complete == 25).all(), f"eksik: {complete[complete < 25].to_dict()}"
    same = cell.groupby(["info", "horizon", "plant"])["n"].nunique()
    assert (same == 1).all(), f"iki boyut farkli test saatlerinde puanlanmis: {same[same > 1].index.tolist()[:5]}"
    cell["rank"] = cell.groupby(["size", "info", "horizon", "plant"])["nrmse"].rank(method="average")
    cell = cell.drop(columns=["n_min", "n_max", "params_max"])
    cell.to_csv(target / "e3_hucreler.csv", index=False)

    wide = cell.pivot_table(index=["info", "horizon", "model", "plant"], columns="size", values="nrmse")
    agg = cell.groupby(["info", "horizon", "model", "size"]).agg(
        nrmse=("nrmse", "mean"), rank=("rank", "mean"), params=("params", "median"), params_min=("params", "min"),
        params_max=("params", "max"), fit_s=("fit_s", "mean"), seed_sd=("seed_sd_nrmse", "mean")).unstack("size")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    summary, tests, fried = [], [], []
    for info, h in CELLS:
        models = sorted(cell["model"].unique())
        family = unpack([{"info": info, "horizon": h, "model": m,
                          **stats.paired_wilcoxon(wide.loc[(info, h, m), "matched"].to_numpy(),
                                                  wide.loc[(info, h, m), "tuned"].to_numpy())} for m in models])
        for row, p in zip(family, stats.holm([f["p"] for f in family])):
            row["p_holm"] = p
        summary += family
        scores = cell[(cell["size"] == "matched") & (cell["info"] == info) & (cell["horizon"] == h)].pivot(
            index="plant", columns="model", values="nrmse")
        t = stats.compare_to_reference(scores[[HYBRID] + [m for m in models if m != HYBRID]], HYBRID)
        for col in ("mean_ci", "median_ci"):
            t[f"{col}_lo"], t[f"{col}_hi"] = t[col].str[0], t[col].str[1]
        tests.append(t.drop(columns=["mean_ci", "median_ci"]).assign(info=info, horizon=h))
        r = stats.friedman_nemenyi(scores, 0.05)
        ranks = agg.loc[(info, h)]
        rho, p = spearmanr(ranks["rank_matched"], ranks["rank_tuned"])
        fried.append({"info": info, "horizon": h, **{k: v for k, v in r.items() if k != "mean_ranks"},
                      "spearman_rank_matched_tuned": rho, "spearman_p": p})
    summary = pd.DataFrame(summary).rename(columns={"wins_a": "plants_matched_better", "mean_diff": "diff_matched_minus_tuned"})
    summary = summary.merge(agg.reset_index(), on=["info", "horizon", "model"])
    first = ["info", "horizon", "model", "params_tuned", "params_matched", "nrmse_tuned", "nrmse_matched",
             "diff_matched_minus_tuned", "mean_ci_lo", "mean_ci_hi", "plants_matched_better", "n", "W", "p", "p_holm",
             "rank_biserial", "rank_tuned", "rank_matched", "fit_s_tuned", "fit_s_matched", "seed_sd_tuned",
             "seed_sd_matched", "params_min_matched", "params_max_matched"]
    summary = summary[first].sort_values(["info", "horizon", "nrmse_matched"])
    summary.to_csv(target / "e3_ozet.csv", index=False)
    tests = pd.concat(tests, ignore_index=True)
    tests.to_csv(target / "e3_wilcoxon.csv", index=False)
    fried = pd.DataFrame(fried)
    fried.to_csv(target / "e3_friedman.csv", index=False)

    pd.set_option("display.width", 250)
    for (info, h), part in summary.groupby(["info", "horizon"], sort=False):
        print(f"\n=== {info}  h={h} ===")
        print(part[["model", "params_tuned", "params_matched", "nrmse_tuned", "nrmse_matched", "diff_matched_minus_tuned",
                    "plants_matched_better", "p_holm", "rank_tuned", "rank_matched", "fit_s_tuned", "fit_s_matched"]]
              .round(3).to_string(index=False))
    print("\nFriedman (eslenmis boyut, 11 ag) ve siralarin ayarli boyutla Spearman korelasyonu:")
    print(fried.round(4).to_string(index=False))
    print("\neslenmis boyutta hibrit - diger ag (negatif fark: hibrit daha iyi):")
    show = tests.assign(text=[f"{r.mean_diff:+.2f} {r.wins_a:>2}/{r.n} {r.p_holm:.2g}" for r in tests.itertuples()])
    print(show.pivot(index="model", columns=["info", "horizon"], values="text").to_string())
    print("parametre araligi (eslenmis):", int(cell.loc[cell['size'] == 'matched', 'params'].min()),
          int(cell.loc[cell['size'] == 'matched', 'params'].max()))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
