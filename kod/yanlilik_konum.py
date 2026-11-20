"""Yanlilik ve konum (E7'ye ek, kesifsel): tahminlerin ortalama yanliligi santralin konumuyla iliskili mi?

Olcut: test donemi ortalama yanliligi (nMBE,
kurulu gucun yuzdesi; arti = fazla tahmin), ana kiyasin dort hucresinde, Ridge, LightGBM, GRU ve hibrit icin
(aglarda bes tohumun ortalamasi).  Santral ozellikleri e7_santraller.csv dosyasindan: mutlak enlem, gunduz ortalama
uretimi (kurulu guce oran), kurulu gucun logaritmasi.  Spearman korelasyonu, % 95 bootstrap araligi; Holm duzeltmesi
bir yontem ve hucredeki uc ozellik uzerinden.  Ayrica iklim grubu basina ortanca yanlilik.

Cikti: e7_yanlilik_korelasyon.csv, e7_yanlilik_ozet.csv
Kullanim:  python yanlilik_konum.py <cikti> <sonuclar klasoru>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import stats  # noqa: E402
from pvbench.experiment import read_metrics  # noqa: E402

import pandas as pd  # noqa: E402

MODELS = ["ridge", "lightgbm", "gru", "cnn_lstm_gru"]
COVARIATES = ["abs_lat", "mean_daytime_p", "log10_capacity_kw"]
FOLDERS = {"S1": "ana", "S2": "ana_icon"}


def main():
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    plants = pd.read_csv(target / "e7_santraller.csv")
    key = "plant" if "plant" in plants.columns else "no"
    plants = plants.rename(columns={key: "plant"}).set_index("plant")
    parts = []
    for info, folder in FOLDERS.items():
        df = read_metrics(out / folder)
        df = df[(df["stage"] == "main") & (df["status"] == "ok") & (df["info"] == info) & df["model"].isin(MODELS)]
        parts.append(df.groupby(["info", "horizon", "model", "plant"], as_index=False)["nmbe"].mean())
    bias = pd.concat(parts, ignore_index=True)
    assert (bias.groupby(["info", "horizon", "model"])["plant"].nunique() == 25).all()
    bias = bias.join(plants[COVARIATES + [c for c in ("climate_group",) if c in plants.columns]], on="plant")

    rows = []
    for (info, h, model), part in bias.groupby(["info", "horizon", "model"]):
        family = []
        for covariate in COVARIATES:
            r = stats.spearman_ci(part[covariate].to_numpy(), part["nmbe"].to_numpy())
            family.append({"info": info, "horizon": h, "model": model, "covariate": covariate, "n": r["n"],
                           "rho": r["rho"], "ci_lo": r["ci"][0], "ci_hi": r["ci"][1], "p": r["p"]})
        for row, p in zip(family, stats.holm([f["p"] for f in family])):
            row["p_holm"] = p
        rows += family
    corr = pd.DataFrame(rows)
    corr.to_csv(target / "e7_yanlilik_korelasyon.csv", index=False)

    summary = bias.groupby(["info", "horizon", "model"]).agg(
        plants=("plant", "nunique"), nmbe_mean=("nmbe", "mean"), nmbe_median=("nmbe", "median"),
        nmbe_min=("nmbe", "min"), nmbe_max=("nmbe", "max"),
        plants_negative=("nmbe", lambda v: int((v < 0).sum()))).reset_index()
    if "climate_group" in bias.columns:
        groups = bias.pivot_table(index=["info", "horizon", "model"], columns="climate_group", values="nmbe",
                                  aggfunc="median").add_prefix("median_").reset_index()
        summary = summary.merge(groups, on=["info", "horizon", "model"])
    summary.to_csv(target / "e7_yanlilik_ozet.csv", index=False)

    pd.set_option("display.width", 220)
    print(summary.round(2).to_string(index=False))
    print(corr.pivot_table(index=["model", "covariate"], columns=["info", "horizon"], values="rho").round(2).to_string())
    print("en kucuk Holm p:", corr["p_holm"].min(), "| anlamli (p_holm < 0.05):", int((corr["p_holm"] < 0.05).sum()), "/", len(corr))
    print(corr[corr["p_holm"] < 0.05].round(4).to_string(index=False))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
