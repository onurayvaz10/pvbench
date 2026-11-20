"""E4: ana kiyasin istatistik testleri (santral = blok).  Girdi: sonuc_tablolari.py'nin urettigi
e1_hucreler.csv (tohum ortalamali test nRMSE) ve kosularin kaydettigi test tahminleri (pred/).

Onceden belirlenen karsilastirmalar (bkz. makale, Yontem: istatistiksel analiz):
  A. Friedman + Iman-Davenport + Nemenyi kritik farki: 17 yontem ve yalniz 11 derin model.
  B. Eslenmis Wilcoxon (cift yonlu), aile ici Holm duzeltmesi, rank-biserial etki buyuklugu,
     ortalama/medyan farkin bootstrap %95 guven araligi:
       aile 1: hibrit CNN-LSTM-GRU'ya karsi diger her yontem (calismanin ilk surumunun iddiasi),
       aile 2: LightGBM'e karsi her derin model,   aile 3: Ridge'e karsi her derin model.
  C. Santral icinde Diebold-Mariano (gunluk ortalama kare hata, Bartlett cekirdegi), 25 santral
     uzerinde Holm: GRU-LightGBM, GRU-Ridge, hibrit-GRU, hibrit-LightGBM, Ridge-LightGBM.
     Derin modellerde kayip, bes tohumun gunluk kayiplarinin ortalamasidir.
Alfa = 0.05; butun testler cift yonludur.

Kullanim:  python istatistik.py <cikti> <sonuclar klasoru>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import dataset, stats  # noqa: E402
from pvbench.experiment import REFERENCES  # noqa: E402

CELLS = [("S1", 1), ("S1", 24), ("S2", 1), ("S2", 24)]
FOLDER = {"S1": ("ana", "S1"), "S2": ("ana_icon", "S2")}
DM_PAIRS = [("gru", "lightgbm"), ("gru", "ridge"), ("cnn_lstm_gru", "gru"), ("cnn_lstm_gru", "lightgbm"),
            ("ridge", "lightgbm")]
ALPHA = 0.05


def day_losses(out, info, horizon, plant, model, offset):
    folder, name = FOLDER[info]
    d = out / folder / "pred" / f"{plant:02d}_h{horizon}_{name}"
    target = pd.read_parquet(d / "target.parquet")
    runs = sorted(d.glob(f"{model}_s*.npy"))
    if not runs:
        return None
    losses = [stats.daily_loss(np.load(r) - target["y"].to_numpy(), target.index, offset) for r in runs]
    return pd.concat(losses, axis=1).mean(axis=1)


def main():
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    cell = pd.read_csv(target / "e1_hucreler.csv")
    plants = dataset.load_plants()
    offset = {no: int(round(float(lon) / 15.0)) for no, lon in plants["lon"].items()}
    fried, ranks, wil, dm = [], [], [], []
    for info, h in CELLS:
        part = cell[(cell["info"] == info) & (cell["horizon"] == h)]
        scores = part.pivot(index="plant", columns="model", values="nrmse")
        deep = [m for m in scores.columns if m not in REFERENCES]
        for label, cols in (("all17", list(scores.columns)), ("deep11", deep)):
            r = stats.friedman_nemenyi(scores[cols], ALPHA)
            fried.append({"info": info, "horizon": h, "set": label, **{k: v for k, v in r.items() if k != "mean_ranks"}})
            ranks += [{"info": info, "horizon": h, "set": label, "model": m, "mean_rank": v}
                      for m, v in r["mean_ranks"].items()]
        for family, ref, others in ((1, "cnn_lstm_gru", [m for m in scores.columns if m != "cnn_lstm_gru"]),
                                    (2, "lightgbm", deep), (3, "ridge", deep)):
            t = stats.compare_to_reference(scores[[ref] + others], ref)
            for col in ("mean_ci", "median_ci"):
                t[f"{col}_lo"], t[f"{col}_hi"] = t[col].str[0], t[col].str[1]
            wil.append(t.drop(columns=["mean_ci", "median_ci"]).assign(info=info, horizon=h, family=family))
        for a, b in DM_PAIRS:
            rows = []
            for plant in scores.index:
                la = day_losses(out, info, h, int(plant), a, offset[plant])
                lb = day_losses(out, info, h, int(plant), b, offset[plant])
                if la is None or lb is None:
                    continue
                rows.append({"info": info, "horizon": h, "a": a, "b": b, "plant": int(plant),
                             **stats.diebold_mariano(la, lb)})
            t = pd.DataFrame(rows)
            t["p_holm"] = stats.holm(t["p"].to_numpy())
            dm.append(t)
    fried, ranks = pd.DataFrame(fried), pd.DataFrame(ranks)
    wil, dm = pd.concat(wil, ignore_index=True), pd.concat(dm, ignore_index=True)
    fried.to_csv(target / "e4_friedman.csv", index=False)
    ranks.to_csv(target / "e4_ortalama_siralar.csv", index=False)
    wil.to_csv(target / "e4_wilcoxon.csv", index=False)
    dm.to_csv(target / "e4_diebold_mariano.csv", index=False)

    pd.set_option("display.width", 250)
    print("=== Friedman / Nemenyi ===")
    print(fried.round(4).to_string(index=False))
    print("\n=== Wilcoxon + Holm (negatif fark: referans daha iyi) ===")
    show = wil[["family", "info", "horizon", "reference", "model", "n", "wins_a", "wins_b", "median_diff",
                "median_ci_lo", "median_ci_hi", "W", "p", "p_holm", "rank_biserial"]]
    for (fam, info, h), t in show.groupby(["family", "info", "horizon"]):
        if fam == 1 and (info, h) not in (("S1", 1), ("S2", 24)):
            continue                                              # the full table is in the CSV
        print(f"\n-- aile {fam}  {info} h={h}")
        print(t.drop(columns=["family", "info", "horizon"]).round(4).to_string(index=False))
    print("\n=== Diebold-Mariano: 25 santralde Holm sonrasi anlamli (a daha iyi / b daha iyi / fark yok) ===")
    for (info, h, a, b), t in dm.groupby(["info", "horizon", "a", "b"], sort=False):
        sig = t["p_holm"] < ALPHA
        print(f"{info} h={h:<2} {a:>13} - {b:<9}: a {int((sig & (t['stat'] < 0)).sum()):>2} / "
              f"b {int((sig & (t['stat'] > 0)).sum()):>2} / yok {int((~sig).sum()):>2}   "
              f"(test edilemeyen {int(t['p'].isna().sum())}, gun medyan {int(t['n'].median())})")
    print(f"\nyazildi: {target}")


if __name__ == "__main__":
    main()
