"""E6: gorulmemis santral. Yerel model <-> kuresel-gorulmemis <-> kuresel-gorulmus, ayni test ornekleri.

Kaynak (<cikti> = pvbench_cikti):
    <cikti>/e6_ref, <cikti>/e6   havuz (kuresel) modeller: Ridge, LightGBM; GRU, hibrit (varsa TCN)  [pvbench.unseen]
    <cikti>/ana (S1), <cikti>/ana_icon (S2)   ayni santralin kendi verisiyle egitilen yerel model (ana kiyas)
Her (bilgi, ufuk, yontem, santral) hucresinde uc puan (test nRMSE %, tohum ortalamasi):
    local    yerel model; aglarda yalniz kuresel kosuda da bulunan tohumlar (0..2)
    unseen   santralin egitimde hic bulunmadigi kuresel model (ortak protokollu 22 santral: disarida birakildigi
             katman; kisa kayitli 3 santral: 22 santralle kurulan katman 0)
    seen     22 santralin hepsiyle kurulan kuresel model (katman 0), santralin egitim donemi havuzdaydi (22 santral)
Uc puanin ayni test orneklerinden geldigi (n) denetlenir.

Eslenmis karsilastirmalar (santral = blok, cift yonlu Wilcoxon, alfa 0.05; Holm ailesi = ayni bilgi kumesi, ufuk ve
karsilastirmadaki yontemler):  unseen-local (25 santral), seen-local (22), unseen-seen (22).
Uzaklik: gorulmemis santralin, o modelin egitim santrallerinden en yakinina uzakligi (km) ile unseen-local ve
unseen-seen farki arasinda Spearman korelasyonu (kesifsel; duzeltme yok).

Cikti: e6_hucreler.csv, e6_ozet.csv, e6_wilcoxon.csv, e6_uzaklik.csv
Kullanim:  python e6_tablolari.py <cikti> <sonuclar klasoru>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import dataset, stats  # noqa: E402
from pvbench.experiment import read_metrics  # noqa: E402
from pvbench.unseen import ALL, distance_km  # noqa: E402

LOCAL = {"S1": "ana", "S2": "ana_icon"}
ORDER = ["ridge", "lightgbm", "gru", "tcn", "cnn_lstm_gru"]
PAIRS = [("unseen", "local"), ("seen", "local"), ("unseen", "seen")]
KEYS = ["info", "horizon", "model", "plant"]


def pooled_cells(out: Path) -> pd.DataFrame:
    parts = [read_metrics(out / name) for name in ("e6_ref", "e6") if (out / name).is_dir()]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    bad = df[df["status"] != "ok"]
    if len(bad):
        print("UYARI: basarisiz kuresel kosular:", bad.groupby(["model", "status"]).size().to_dict())
    df = df[df["status"] == "ok"]
    full = df.groupby(["info", "horizon", "model", "fold", "seed"])["plant"].nunique()
    whole = full[full == 25].reset_index()[["info", "horizon", "model", "fold", "seed"]]
    if len(whole) < len(full):
        print(f"UYARI: {len(full) - len(whole)} kuresel uyumun satirlari eksik (25 santral yok); kullanilmadi")
    df = df.merge(whole, on=["info", "horizon", "model", "fold", "seed"])
    dup = df.duplicated(["info", "horizon", "model", "fold", "seed", "plant"])
    assert not dup.any(), "ayni kuresel uyum iki kez yazilmis"

    seeds = df.groupby(["info", "horizon", "model", "fold"])["seed"].nunique().groupby(["info", "horizon", "model"])
    done = seeds.agg(["min", "max", "size"])
    ready = done[(done["min"] == done["max"]) & (done["size"] == df["fold"].nunique())].index
    skipped = sorted(set(done.index) - set(ready))
    if skipped:
        print("UYARI: katmanlari ya da tohumlari tamamlanmamis hucreler tabloya girmedi:", skipped)
    df = df.set_index(["info", "horizon", "model"]).loc[ready].reset_index()

    unseen = df[df["role"].eq("unseen") | (df["role"].eq("short") & df["fold"].eq(ALL))]
    seen = df[df["role"].eq("seen") & df["fold"].eq(ALL)]
    cell = unseen.groupby(KEYS).agg(name=("name", "first"), role=("role", "first"), fold=("fold", "first"),
                                    folds=("fold", "nunique"), seeds=("seed", "nunique"), n=("n", "first"),
                                    n_min=("n", "min"), n_max=("n", "max"), n_plants_train=("n_plants_train", "first"),
                                    unseen=("nrmse", "mean"), unseen_nmae=("nmae", "mean")).reset_index()
    assert (cell["folds"] == 1).all(), "bir santral birden cok katmanda gorulmemis"
    assert (cell["n_min"] == cell["n_max"]).all()
    cell = cell.merge(seen.groupby(KEYS).agg(seen=("nrmse", "mean"), seen_nmae=("nmae", "mean"), n_seen=("n", "first"))
                      .reset_index(), on=KEYS, how="left")
    assert (cell["n_seen"].isna() | (cell["n_seen"] == cell["n"])).all(), "gorulmus ve gorulmemis puan farkli orneklerde"

    plants = dataset.load_plants()
    members = df[df["role"].eq("seen")].groupby("fold")["plant"].unique().to_dict()   # egitim santralleri

    def nearest(row):
        here = plants.loc[row.plant]
        return min(distance_km(here["lat"], here["lon"], plants.loc[o, "lat"], plants.loc[o, "lon"])
                   for o in members[row.fold])

    cell["nearest_km"] = [nearest(r) for r in cell.itertuples()]
    return cell.drop(columns=["folds", "n_min", "n_max", "n_seen"])


def local_cells(out: Path, cell: pd.DataFrame) -> pd.DataFrame:
    seeds = cell.groupby(["info", "model"])["seeds"].first()
    parts = []
    for info in sorted(cell["info"].unique()):
        df = read_metrics(out / LOCAL[info])
        df = df[(df["stage"] == "main") & (df["info"] == info) & (df["status"] == "ok")]
        parts.append(df[df["model"] == "smart_persistence"])
        for model, n_seeds in seeds.loc[info].items():
            parts.append(df[(df["model"] == model) & (df["seed"] < n_seeds)])
    df = pd.concat(parts, ignore_index=True)
    local = df.groupby(KEYS).agg(local=("nrmse", "mean"), local_nmae=("nmae", "mean"), local_seeds=("seed", "nunique"),
                                 n_local=("n", "first")).reset_index()
    sp = local[local["model"] == "smart_persistence"].rename(columns={"local": "smart_persistence"})
    return local[local["model"] != "smart_persistence"], sp[["info", "horizon", "plant", "smart_persistence"]]


def main():
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    cell = pooled_cells(out)
    local, sp = local_cells(out, cell)
    cell = cell.merge(local, on=KEYS, how="left").merge(sp, on=["info", "horizon", "plant"], how="left")
    assert cell["local"].notna().all(), "yerel puani olmayan hucre var"
    assert (cell["n_local"] == cell["n"]).all(), "yerel ve kuresel model farkli test orneklerinde"
    assert (cell["local_seeds"] == cell["seeds"]).all(), "yerel ve kuresel modelde tohum sayisi farkli"
    count = cell.groupby(["info", "horizon", "model"])["plant"].nunique()
    assert (count == 25).all(), f"eksik santral: {count[count < 25].to_dict()}"
    cell["unseen_minus_local"] = cell["unseen"] - cell["local"]
    cell["seen_minus_local"] = cell["seen"] - cell["local"]
    cell["unseen_minus_seen"] = cell["unseen"] - cell["seen"]
    models = [m for m in ORDER if m in set(cell["model"])] + sorted(set(cell["model"]) - set(ORDER))
    cell = cell.drop(columns=["n_local", "local_seeds"]).sort_values(KEYS)
    cell.to_csv(target / "e6_hucreler.csv", index=False)

    cell["group"] = np.where(cell["role"] == "short", "short", "common")
    summary = cell.groupby(["info", "horizon", "model", "group"]).agg(
        plants=("plant", "nunique"), seeds=("seeds", "first"), smart_persistence=("smart_persistence", "mean"),
        local=("local", "mean"), unseen=("unseen", "mean"), seen=("seen", "mean"),
        unseen_minus_local=("unseen_minus_local", "mean"), seen_minus_local=("seen_minus_local", "mean"),
        unseen_minus_seen=("unseen_minus_seen", "mean"),
        unseen_better_than_local=("unseen_minus_local", lambda d: int((d < 0).sum())),
        nearest_km_median=("nearest_km", "median")).reset_index()
    summary.to_csv(target / "e6_ozet.csv", index=False)

    tests, distance = [], []
    for (info, h), part in cell.groupby(["info", "horizon"]):
        wide = {m: part[part["model"] == m].set_index("plant") for m in models}
        for a, b in PAIRS:
            family = []
            for m in models:
                rows = wide[m].dropna(subset=[a, b])
                family.append({"info": info, "horizon": h, "a": a, "b": b, "model": m,
                               **stats.paired_wilcoxon(rows[a].to_numpy(), rows[b].to_numpy())})
            for row, p in zip(family, stats.holm([f["p"] for f in family])):
                row["p_holm"] = p
                row["mean_ci_lo"], row["mean_ci_hi"] = row.pop("mean_ci")
                row["median_ci_lo"], row["median_ci_hi"] = row.pop("median_ci")
            tests += family
        for m in models:
            for column, rows in (("unseen_minus_local", wide[m]), ("unseen_minus_seen", wide[m][wide[m]["seen"].notna()])):
                r = stats.spearman_ci(rows["nearest_km"].to_numpy(), rows[column].to_numpy())
                distance.append({"info": info, "horizon": h, "model": m, "difference": column, "n": r["n"],
                                 "rho": r["rho"], "p": r["p"], "ci_lo": r["ci"][0], "ci_hi": r["ci"][1]})
    tests = pd.DataFrame(tests)
    tests.to_csv(target / "e6_wilcoxon.csv", index=False)
    distance = pd.DataFrame(distance)
    distance.to_csv(target / "e6_uzaklik.csv", index=False)

    pd.set_option("display.width", 250)
    for group, label in (("common", "ortak protokollu 22 santral"), ("short", "kisa kayitli 3 santral (yalniz gorulmemis)")):
        part = summary[summary["group"] == group].set_index(["info", "horizon", "model"])
        print(f"\n=== {label}: ortalama test nRMSE (%) ===")
        print(part[["smart_persistence", "local", "unseen", "seen", "unseen_minus_local", "unseen_better_than_local"]]
              .round(2).to_string())
    print("\nfark a-b (nRMSE puani; eksi = a daha iyi), a'nin kazandigi santral / n, Holm p:")
    show = tests.assign(pair=tests["a"] + "-" + tests["b"],
                        cell=[f"{r.mean_diff:+.2f} ({r.wins_a}/{r.n}, p={r.p_holm:.3g})" for r in tests.itertuples()])
    print(show.pivot(index=["info", "horizon", "model"], columns="pair", values="cell")[[f"{a}-{b}" for a, b in PAIRS]].to_string())
    print("\nSpearman: en yakin egitim santraline uzaklik (km) ile fark:")
    print(distance.round(3).to_string(index=False))
    near = cell.drop_duplicates("plant")[["plant", "name", "role", "fold", "nearest_km"]].sort_values("nearest_km")
    print("\nen yakin egitim santraline uzaklik (km):")
    print(near.round(0).to_string(index=False))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
