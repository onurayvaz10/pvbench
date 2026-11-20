"""Ana kiyas (E1) sonuc tablolari: kosu ciktilarindan (metrics_w*.csv) dogrudan uretilir.

Kaynak klasorler (<cikti> = pvbench_cikti):
    <cikti>/ana       bilgi S1 -> "S1";  bilgi S2 -> "S2B" (Open-Meteo best_match girdisi; yalniz duyarlilik)
    <cikti>/ana_icon  bilgi S2 -> "S2"  (ICON girdisi; makaledeki S2)

Her (bilgi, ufuk, santral, yontem) hucresinde tohumlarin ortalamasi alinir; ozetler santraller uzerindendir.
Beceri = 1 - RMSE_yontem / RMSE_referans, ayni santralin ayni test saatlerinde (hucrede n esitligi denetlenir).

Kullanim:  python sonuc_tablolari.py <cikti> <hedef klasor>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench.experiment import REFERENCES, read_metrics  # noqa: E402

METRICS = ["nrmse", "nmae", "nmbe", "r2", "theil_u2", "mase", "nrmse_day", "nmae_day", "rmse", "mae"]


def load(out: Path) -> pd.DataFrame:
    a = read_metrics(out / "ana")
    a = a[a["stage"] == "main"].copy()
    a["info"] = a["info"].replace({"S2": "S2B"})
    b = read_metrics(out / "ana_icon")
    b = b[(b["stage"] == "main") & (b["info"] == "S2")].copy()
    df = pd.concat([a, b], ignore_index=True)
    bad = df[df["status"] != "ok"]
    if len(bad):
        print("UYARI: basarisiz kosular:", bad.groupby(["info", "status"]).size().to_dict())
    return df[df["status"] == "ok"]


def main():
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)
    df = load(out)
    keys = ["info", "horizon", "plant", "model"]
    cell = df.groupby(keys).agg(**{m: (m, "mean") for m in METRICS}, seeds=("seed", "nunique"),
                                seed_sd_nrmse=("nrmse", "std"), n=("n", "first"), n_min=("n", "min"),
                                n_max=("n", "max"), params=("params", "first"), fit_s=("fit_s", "mean"),
                                epochs=("epochs", "mean")).reset_index()
    assert (cell["n_min"] == cell["n_max"]).all(), "bir hucrede tohumlar farkli test saati sayisi kullanmis"
    same_n = cell.groupby(["info", "horizon", "plant"])["n"].nunique()
    assert (same_n == 1).all(), f"yontemler farkli test saatlerinde puanlanmis: {same_n[same_n > 1].index.tolist()[:5]}"
    wide = cell.pivot_table(index=["info", "horizon", "plant"], columns="model", values="rmse")
    for ref in ("persistence", "smart_persistence", "lightgbm"):
        skill = (1 - wide.div(wide[ref], axis=0)).stack().rename(f"skill_{ref}").reset_index()
        cell = cell.merge(skill, on=keys, how="left")
    cell["rank_all"] = cell.groupby(["info", "horizon", "plant"])["nrmse"].rank(method="average")
    dl = ~cell["model"].isin(REFERENCES)
    cell.loc[dl, "rank_dl"] = cell[dl].groupby(["info", "horizon", "plant"])["nrmse"].rank(method="average")
    cell["kind"] = np.where(dl, "deep", "reference")
    cell.to_csv(target / "e1_hucreler.csv", index=False)

    g = cell.groupby(["info", "horizon", "model"])
    summary = g.agg(kind=("kind", "first"), plants=("plant", "nunique"), nrmse_mean=("nrmse", "mean"),
                    nrmse_median=("nrmse", "median"), nrmse_sd_plants=("nrmse", "std"),
                    nmae_mean=("nmae", "mean"), nrmse_day_mean=("nrmse_day", "mean"),
                    skill_persistence=("skill_persistence", "mean"),
                    skill_smart_persistence=("skill_smart_persistence", "mean"),
                    skill_lightgbm=("skill_lightgbm", "mean"),
                    beats_lightgbm=("skill_lightgbm", lambda s: int((s > 0).sum())),
                    rank_all=("rank_all", "mean"), rank_dl=("rank_dl", "mean"),
                    wins_all=("rank_all", lambda s: int((s == 1).sum())),
                    seed_sd_nrmse=("seed_sd_nrmse", "mean"), params=("params", "median"),
                    fit_s=("fit_s", "mean")).reset_index()
    summary = summary.sort_values(["info", "horizon", "rank_all"])
    summary.to_csv(target / "e1_ozet.csv", index=False)
    cell.pivot_table(index=["info", "horizon", "model"], columns="plant", values="nrmse").round(3).to_csv(
        target / "e1_santral_nrmse.csv")
    cell.pivot_table(index=["info", "horizon", "model"], columns="plant", values="nmae").round(3).to_csv(
        target / "e1_santral_nmae.csv")

    pd.set_option("display.width", 250)
    for (info, h), part in summary.groupby(["info", "horizon"], sort=False):
        show = part[["model", "kind", "nrmse_mean", "nrmse_median", "skill_persistence", "skill_lightgbm",
                     "beats_lightgbm", "rank_all", "rank_dl", "wins_all", "seed_sd_nrmse"]]
        print(f"\n=== {info}  h={h}  ({part['plants'].iloc[0]} santral) ===")
        print(show.round(3).to_string(index=False))
    print(f"\nyazildi: {target}")


if __name__ == "__main__":
    main()
