"""E8: zaman cozunurlugu (5, 10, 20 ve 60 dakika), bir adim ileri tahmin, bilgi kumesi S0 (hava verisi yok).

Kaynak (<cikti> = pvbench_cikti):
    <cikti>/e8_ref, <cikti>/e8   5/10/20 dakika: sureklilik, acik gokyuzu surekliligi, Ridge, LightGBM; GRU, TCN
                                 [pvbench.resolution; bilgi etiketi S0r5 / S0r10 / S0r20]
    <cikti>/ana_e2               60 dakika: ana kiyasin S0, h = 1 kosusu (ayni santraller; aglarda E8'deki tohumlar)
Santraller: veri/islenmis/altsaatlik_envanter.csv dosyasinda `eligible` olanlar (5 dakikalik kayit, ince sayac adimi).
Her (cozunurluk, yontem, santral) hucresinde tohum ortalamasi; tum yontemlerin ayni test orneklerini kullandigi denetlenir.
Cozunurlukler arasinda nRMSE dogrudan karsilastirilmaz (ileri sure ve hedefin degiskenligi farkli); karsilastirilan,
her cozunurlukte yontemlerin siralamasi ve acik gokyuzu surekliligine gore beceridir (1 - RMSE / RMSE_aks).

Eslenmis karsilastirmalar (santral = blok, cift yonlu Wilcoxon, alfa 0.05; Holm ailesi = ayni cozunurlukteki ciftler):
ogrenen her yontem - acik gokyuzu surekliligi, LightGBM - Ridge, her ag - Ridge ve LightGBM.

Cikti: e8_hucreler.csv, e8_ozet.csv, e8_wilcoxon.csv
Kullanim:  python e8_tablolari.py <cikti> <sonuclar klasoru>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import stats  # noqa: E402
from pvbench.experiment import read_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ["persistence", "smart_persistence"]
LEARNED = ["ridge", "lightgbm", "gru", "tcn"]
NETWORKS = ["gru", "tcn"]
KEYS = ["minutes", "model", "plant"]
METRICS = ["nrmse", "nmae", "nrmse_day", "rmse", "mase"]


def main():
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    inventory = pd.read_csv(ROOT / "veri" / "islenmis" / "altsaatlik_envanter.csv", sep=";")
    plants = sorted(inventory.loc[inventory["eligible"].astype(str) == "True", "no"].astype(int))
    inventory[["no", "name", "source", "capacity_kw", "records", "spacing_median_s", "share_300s", "share_on_grid",
               "counter_step_wh", "counter_step_share", "eligible"]].to_csv(target / "e8_envanter.csv", index=False)

    fine = pd.concat([read_metrics(out / name) for name in ("e8_ref", "e8") if (out / name).is_dir()], ignore_index=True)
    fine = fine[fine["info"].astype(str).str.fullmatch(r"S0r\d+")]
    fine["minutes"] = fine["info"].str[3:].astype(int)
    hourly = read_metrics(out / "ana_e2")
    hourly = hourly[(hourly["stage"] == "main") & (hourly["info"] == "S0") & (hourly["horizon"] == 1)].assign(minutes=60)
    df = pd.concat([fine, hourly], ignore_index=True)
    df = df[df["plant"].isin(plants) & df["model"].isin(BASELINES + LEARNED)]
    bad = df[df["status"] != "ok"]
    if len(bad):
        print("UYARI: basarisiz kosular:", bad.groupby(["minutes", "model", "status"]).size().to_dict())
    df = df[df["status"] == "ok"]
    assert not df.duplicated(KEYS + ["seed"]).any(), "ayni kosu iki kez yazilmis"

    seeds = df[df["minutes"] < 60].groupby("model")["seed"].nunique()          # E8'de kosulan tohum sayisi
    df = df[df["model"].isin(seeds.index) & (df["seed"] < df["model"].map(seeds))]
    cell = df.groupby(KEYS).agg(**{m: (m, "mean") for m in METRICS}, name=("name", "first"), seeds=("seed", "nunique"),
                                n=("n", "first"), n_min=("n", "min"), n_max=("n", "max"),
                                n_train=("n_train", "first")).reset_index()
    assert (cell["n_min"] == cell["n_max"]).all(), "bir hucrede tohumlar farkli test ornegi sayisi kullanmis"
    cell = cell[cell["seeds"] == cell["model"].map(seeds)]                       # tohumlari tamamlanmis hucreler
    count = cell.groupby(["minutes", "model"])["plant"].nunique()
    partial = count[count < len(plants)]
    if len(partial):
        print(f"UYARI: {len(plants)} santrali tamamlanmamis (cozunurluk, yontem) ciftleri tabloya girmedi:", partial.to_dict())
    cell = cell.set_index(["minutes", "model"]).loc[count[count == len(plants)].index].reset_index()
    same = cell.groupby(["minutes", "plant"])["n"].nunique()
    assert (same == 1).all(), f"yontemler ayni test orneklerinde degil: {same[same > 1].index.tolist()[:5]}"

    wide = cell.pivot(index=["minutes", "plant"], columns="model", values="rmse")
    for reference in BASELINES:
        skill = (1 - wide.div(wide[reference], axis=0)).stack().rename(f"skill_{reference}").reset_index()
        cell = cell.merge(skill, on=KEYS, how="left")
    cell = cell.drop(columns=["n_min", "n_max"]).sort_values(KEYS)
    cell.to_csv(target / "e8_hucreler.csv", index=False)

    models = [m for m in BASELINES + LEARNED if m in set(cell["model"])]
    summary = cell.groupby(["minutes", "model"]).agg(
        plants=("plant", "nunique"), seeds=("seeds", "first"), n_test_median=("n", "median"),
        nrmse_mean=("nrmse", "mean"), nrmse_median=("nrmse", "median"), nmae_mean=("nmae", "mean"),
        nrmse_day_mean=("nrmse_day", "mean"), skill_persistence=("skill_persistence", "mean"),
        skill_smart_persistence=("skill_smart_persistence", "mean"),
        better_than_smart_persistence=("skill_smart_persistence", lambda s: int((s > 0).sum()))).reset_index()
    rank = cell[cell["model"].isin(LEARNED)].copy()
    rank["rank"] = rank.groupby(["minutes", "plant"])["nrmse"].rank()
    summary = summary.merge(rank.groupby(["minutes", "model"])["rank"].mean().rename("mean_rank_learned").reset_index(),
                            on=["minutes", "model"], how="left")
    summary.to_csv(target / "e8_ozet.csv", index=False)

    tests = []
    scores = cell.pivot(index=["minutes", "plant"], columns="model", values="nrmse")
    for minutes in sorted(cell["minutes"].unique()):
        part = scores.loc[minutes].dropna(axis=1, how="all")
        have = [m for m in LEARNED if m in part.columns]
        pairs = [(m, "smart_persistence") for m in have]
        pairs += [(a, b) for a, b in [("lightgbm", "ridge")] + [(n, r) for n in NETWORKS for r in ("ridge", "lightgbm")]
                  if a in have and b in have]
        family = [{"minutes": minutes, "a": a, "b": b, **stats.paired_wilcoxon(part[a].to_numpy(), part[b].to_numpy())}
                  for a, b in pairs]
        for row, p in zip(family, stats.holm([f["p"] for f in family])):
            row["p_holm"] = p
            row["mean_ci_lo"], row["mean_ci_hi"] = row.pop("mean_ci")
            row["median_ci_lo"], row["median_ci_hi"] = row.pop("median_ci")
        tests += family
    tests = pd.DataFrame(tests)
    tests.to_csv(target / "e8_wilcoxon.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"santraller ({len(plants)}): {plants}")
    for column, label in (("nrmse_mean", "ortalama test nRMSE (%)"), ("skill_smart_persistence", "acik gokyuzu surekliligine gore beceri"),
                          ("mean_rank_learned", "ogrenen yontemler arasinda ortalama sira")):
        table = summary.pivot(index="model", columns="minutes", values=column)
        print(f"\n=== {label} ===")
        print(table.loc[[m for m in models if m in table.index]].round(3 if "skill" in column else 2).to_string())
    print("\nfark a-b (nRMSE puani; eksi = a daha iyi), a'nin kazandigi santral / n, Holm p:")
    show = tests.assign(pair=tests["a"] + " - " + tests["b"],
                        cell=[f"{r.mean_diff:+.2f} ({r.wins_a}/{r.n}, p={r.p_holm:.3g})" for r in tests.itertuples()])
    print(show.pivot(index="pair", columns="minutes", values="cell").to_string())


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
