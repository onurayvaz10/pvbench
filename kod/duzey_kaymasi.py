"""Duzey kaymasi denetimi: bir santralin uretim duzeyi egitim donemi ile test donemi arasinda degismis mi?

PVOutput sistemlerinde kayitli kurulu guc tek bir sayidir; sistem buyutulur, bir dizisi devre disi kalir ya da
onarilirsa uretimin kurulu guce orani kalici olarak degisir.  Boyle bir degisiklik egitim ile test donemi arasina
duserse ogrenen yontemler eski duzeyi ogrenmis olur; son olculen degeri kullanan sureklilik yontemleri etkilenmez.

Olcut (tahmin sonuclari kullanilmaz): acik gokyuzu orani  k = p / (cs_ghi / 1000),  yalniz cs_ghi >= 200 W/m2 olan
gecerli saatlerde.  Karsilastirilan iki donem mevsim dengesi icin on ikiser aydir:
    egitim : dogrulama doneminden onceki son 12 ay         test : test donemi (12 ay)
    oran_k   = ortanca k (test) / ortanca k (egitim)        oran_p99 = p'nin 99. yuzdeligi (test) / (egitim)
Kisa kayitli santrallerde (egitim < 12 ay) egitim donemi eldeki kadardir ve mevsim dengesi yoktur (not dusulur).
Bayrak: oran_k ya da oran_p99, [1/ESIK, ESIK] araliginin disinda (ESIK = 1.5).
Test donemi icindeki kayma icin ayni oran test doneminin son 6 ayi ile bir yil onceki ayni 6 ay arasinda da verilir
(oran_k_yil; ayni takvim aylari).

Ikinci cikti: ana kiyasin (E1) ozetleri bayrakli santraller olmadan (e1_ozet_kaymasiz.csv).
Cikti: duzey_kaymasi.csv, e1_ozet_kaymasiz.csv
Kullanim:  python duzey_kaymasi.py <sonuclar klasoru>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import dataset, features  # noqa: E402

THRESHOLD = 1.5
MIN_CLEAR_SKY = 200.0
CELLS = [("S1", 1), ("S1", 24), ("S2", 1), ("S2", 24)]


def level(table: pd.DataFrame, lo, hi) -> dict:
    part = table[(table.index > lo) & (table.index <= hi) & table["p"].notna()]
    sunny = part[part["cs_ghi"] >= MIN_CLEAR_SKY]
    if len(sunny) < 100:
        return {"k": np.nan, "p99": np.nan, "hours": len(sunny)}
    return {"k": float((sunny["p"] / (sunny["cs_ghi"] / 1000.0)).median()), "p99": float(part["p"].quantile(0.99)),
            "hours": len(sunny)}


def main():
    target = Path(sys.argv[1])
    rows = []
    for no, plant in dataset.load_plants().iterrows():
        table = dataset.load_table(no)
        periods = features.protocol(table, no)
        year, half = pd.DateOffset(months=12), pd.DateOffset(months=6)
        first = table["p"].first_valid_index()
        train_lo = max(periods["val_start"] - year, first - pd.Timedelta(hours=1))
        train = level(table, train_lo, periods["val_start"])
        test = level(table, periods["test_start"], periods["test_end"])
        late = level(table, periods["test_end"] - half, periods["test_end"])
        year_before = level(table, periods["test_end"] - half - year, periods["test_end"] - year)
        months = (periods["val_start"] - train_lo).days / 30.44
        with np.errstate(divide="ignore", invalid="ignore"):          # a level of zero gives an infinite ratio
            ratio_k, ratio_p99 = np.float64(test["k"]) / train["k"], np.float64(test["p99"]) / train["p99"]
            ratio_year = np.float64(late["k"]) / year_before["k"]
            extreme = np.nanmax([ratio_k, 1 / ratio_k, ratio_p99, 1 / ratio_p99])
        rows.append({"no": no, "name": plant["name"], "protocol": periods["name"], "train_months": round(months, 1),
                     "hours_train": train["hours"], "hours_test": test["hours"],
                     "k_train": train["k"], "k_test": test["k"], "ratio_k": ratio_k,
                     "p99_train": train["p99"], "p99_test": test["p99"], "ratio_p99": ratio_p99,
                     "ratio_k_year": ratio_year, "flag": bool(extreme > THRESHOLD)})
    out = pd.DataFrame(rows)
    out.to_csv(target / "duzey_kaymasi.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(out.round(3).to_string(index=False))
    flagged = out.loc[out["flag"], "no"].tolist()
    print(f"\nbayrakli santraller (esik {THRESHOLD}): {flagged}  {out.loc[out['flag'], 'name'].tolist()}")

    cell = pd.read_csv(target / "e1_hucreler.csv")
    cell = cell[cell[["info", "horizon"]].apply(tuple, axis=1).isin(CELLS)]
    skill = cell[cell["model"].isin(["ridge", "lightgbm", "gru"])].pivot_table(
        index=["info", "horizon", "plant"], columns="model", values="skill_smart_persistence")
    negative = (skill < 0).all(axis=1)
    print("\nuc onde gelen yontemin (Ridge, LightGBM, GRU) ucunun de becerisi eksi olan (hucre, santral):")
    print(negative[negative].reset_index().groupby(["info", "horizon"])["plant"].apply(list).to_string())

    keep = cell[~cell["plant"].isin(flagged)].copy()
    keep["rank_all"] = keep.groupby(["info", "horizon", "plant"])["nrmse"].rank()
    summary = keep.groupby(["info", "horizon", "model"]).agg(
        plants=("plant", "nunique"), nrmse_mean=("nrmse", "mean"),
        skill_smart_persistence=("skill_smart_persistence", "mean"), rank_all=("rank_all", "mean")).reset_index()
    every = cell.groupby(["info", "horizon", "model"]).agg(nrmse_mean_all=("nrmse", "mean"),
                                                           skill_all=("skill_smart_persistence", "mean"),
                                                           rank_all_all=("rank_all", "mean")).reset_index()
    summary = summary.merge(every, on=["info", "horizon", "model"]).sort_values(["info", "horizon", "rank_all"])
    summary.to_csv(target / "e1_ozet_kaymasiz.csv", index=False)
    for (info, h), part in summary.groupby(["info", "horizon"], sort=False):
        print(f"\n=== {info}  h={h}: bayrakli santraller olmadan ({part['plants'].iloc[0]} santral) / hepsi ===")
        print(part.drop(columns=["info", "horizon", "plants"]).head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
