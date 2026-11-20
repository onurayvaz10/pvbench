"""Duyarlilik: ana kiyasin (E1) ozetleri ve E4 testleri yalniz ortak protokollu santrallerle yinelenir.

Ana analiz 25 santralin hepsini blok olarak kullanir. Uc santral (kisa kayit; kendi protokolu) baska bir test
doneminde ve cok daha kisa bir egitim donemiyle degerlendirildigi icin, ayni ozet ve testler burada bu uc santral
olmadan (ortak protokol: test Aralik 2024 - Kasim 2025) yeniden hesaplanir.  Girdi: sonuc_tablolari.py'nin
urettigi e1_hucreler.csv; protokol, kosularin kaydettigi `protocol` alanindan okunur.

Cikti: e1_ozet_ortak.csv, e4_friedman_ortak.csv, e4_wilcoxon_ortak.csv, e1_kisa_kayit.csv (uc santralin hucreleri)
Kullanim:  python duyarlilik_ortak_protokol.py <cikti> <sonuclar klasoru>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import stats  # noqa: E402
from pvbench.experiment import REFERENCES, read_metrics  # noqa: E402

CELLS = [("S1", 1), ("S1", 24), ("S2", 1), ("S2", 24)]
ALPHA = 0.05


def main():
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    runs = read_metrics(out / "ana")
    protocol = runs[runs["stage"] == "main"].groupby("plant")["protocol"].agg(lambda s: s.unique().tolist())
    assert (protocol.str.len() == 1).all(), "bir santralde birden cok protokol"
    protocol = protocol.str[0]
    common = sorted(protocol[protocol == "common"].index.astype(int))
    own = sorted(protocol[protocol == "own"].index.astype(int))

    cell = pd.read_csv(target / "e1_hucreler.csv")
    cell = cell[cell[["info", "horizon"]].apply(tuple, axis=1).isin(CELLS)]
    cell[cell["plant"].isin(own)].to_csv(target / "e1_kisa_kayit.csv", index=False)
    full = cell.copy()
    cell = cell[cell["plant"].isin(common)].copy()
    keys = ["info", "horizon", "plant"]
    cell["rank_all"] = cell.groupby(keys)["nrmse"].rank()
    deep = ~cell["model"].isin(REFERENCES)
    cell.loc[deep, "rank_dl"] = cell[deep].groupby(keys)["nrmse"].rank()

    summary = cell.groupby(["info", "horizon", "model"]).agg(
        plants=("plant", "nunique"), nrmse_mean=("nrmse", "mean"), nrmse_median=("nrmse", "median"),
        skill_smart_persistence=("skill_smart_persistence", "mean"), rank_all=("rank_all", "mean"),
        rank_dl=("rank_dl", "mean"), wins_all=("rank_all", lambda s: int((s == 1).sum()))).reset_index()
    all25 = full.groupby(["info", "horizon", "model"]).agg(nrmse_mean_25=("nrmse", "mean"),
                                                             rank_all_25=("rank_all", "mean")).reset_index()
    summary = summary.merge(all25, on=["info", "horizon", "model"]).sort_values(["info", "horizon", "rank_all"])
    summary.to_csv(target / "e1_ozet_ortak.csv", index=False)

    fried, wil = [], []
    for info, h in CELLS:
        scores = cell[(cell["info"] == info) & (cell["horizon"] == h)].pivot(index="plant", columns="model", values="nrmse")
        nets = [m for m in scores.columns if m not in REFERENCES]
        for label, cols in (("all17", list(scores.columns)), ("deep11", nets)):
            r = stats.friedman_nemenyi(scores[cols], ALPHA)
            fried.append({"info": info, "horizon": h, "set": label, **{k: v for k, v in r.items() if k != "mean_ranks"}})
        for family, ref, others in ((1, "cnn_lstm_gru", [m for m in scores.columns if m != "cnn_lstm_gru"]),
                                    (2, "lightgbm", nets), (3, "ridge", nets)):
            t = stats.compare_to_reference(scores[[ref] + others], ref)
            for col in ("mean_ci", "median_ci"):
                t[f"{col}_lo"], t[f"{col}_hi"] = t[col].str[0], t[col].str[1]
            wil.append(t.drop(columns=["mean_ci", "median_ci"]).assign(info=info, horizon=h, family=family))
    fried, wil = pd.DataFrame(fried), pd.concat(wil, ignore_index=True)
    fried.to_csv(target / "e4_friedman_ortak.csv", index=False)
    wil.to_csv(target / "e4_wilcoxon_ortak.csv", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(f"ortak protokol: {len(common)} santral; kendi protokolu: {own}")
    for (info, h), part in summary.groupby(["info", "horizon"], sort=False):
        print(f"\n=== {info}  h={h} ===")
        print(part.drop(columns=["info", "horizon"]).round(3).to_string(index=False))
    print("\n=== Friedman (ortak protokol) ===")
    print(fried.round(4).to_string(index=False))

    # Does any decision at alpha = 0.05 (Holm) differ from the analysis with all plants?
    base = pd.read_csv(target / "e4_wilcoxon.csv")
    both = wil.merge(base, on=["family", "info", "horizon", "reference", "model"], suffixes=("_22", "_25"))
    changed = both[(both["p_holm_22"] < ALPHA) != (both["p_holm_25"] < ALPHA)]
    print(f"\nWilcoxon + Holm: {len(both)} karsilastirmanin {len(changed)} tanesinde karar (alfa {ALPHA}) degisti")
    if len(changed):
        print(changed[["family", "info", "horizon", "reference", "model", "mean_diff_22", "p_holm_22", "mean_diff_25",
                       "p_holm_25"]].round(4).to_string(index=False))
    short = full[full["plant"].isin(own)]
    print("\n=== kisa kayitli uc santral: nRMSE (%) ===")
    print(short.pivot_table(index=["info", "horizon", "model"], columns="plant", values="nrmse").round(2)
          .loc[lambda t: t.index.get_level_values("model").isin(
              ["ridge", "lightgbm", "gru", "tcn", "cnn_lstm_gru", "smart_persistence", "persistence"])].to_string())


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
