"""Sure olcumu: egitim ve tahmin sureleri, tek surecle (GPU'da baska is yokken calistirilir).

Ana kiyas 8 iscinin tek GPU'yu paylastigi kosularla yapildi; oradaki `fit_s` degerleri yontemler arasinda sure
karsilastirmasi icin kullanilamaz.  Bu betik ayni kodu (experiment.fit_ridge / fit_lightgbm / fit_neural) bir santralde,
tek surecle calistirir ve su degerleri yazar:
    fit_s         egitim suresi (Ridge: 6 alfa; LightGBM: 12 ayarlik arama; aglar: erken durdurmayla bir egitim)
    epochs        aglarda egitilen donem sayisi;  epoch_s = fit_s / epochs
    predict_ms    test doneminin butun ornekleri icin tahmin suresi (isinmadan sonra REPEATS olcumun ortancasi)
    per_1000_ms   1000 ornek basina tahmin suresi
    params        egitilebilir parametre sayisi (aglar)
    device        aglarda egitim ve tahminin yapildigi aygit (dense ve simplernn ana kiyasta da islemcide egitildi)
Kosul: tek surec, ana kiyasin iscileri gibi bir performans cekirdegine (iki mantiksal islemci) bagli, BLAS ve LightGBM
iki is parcacigi; GPU'da baska is yok.
Santral: ortak protokollu santraller icinde S1, h = 1 egitim ornegi sayisi ortanca olan (sonuclara bakmayan kural).
Hucreler: S1, h = 1 ve h = 24.  Boyutlar: ayarla secilen (tuned) ve yaklasik 50.000 parametre (matched; E3).  Tohum 0.
Ag yapilandirmalari <cikti>/ayar/selected.csv dosyasindan okunur (ana kiyasla ayni).

Cikti: <sonuclar>/sure_olcumu.csv
Kullanim:  python sure_olcumu.py <cikti> <sonuclar klasoru> [santral no]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# pvbench.experiment limits the BLAS threads to two; it must be imported before numpy, otherwise the thread pool of
# all cores is squeezed onto the one performance core the process is bound to and linear algebra stalls.
from pvbench import dataset, experiment, features  # noqa: E402, I001
from pvbench.experiment import read_metrics  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INFO, HORIZONS, SEED, REPEATS, MATCH = "S1", (1, 24), 0, 5, 50000


def median_plant(out: Path) -> int:
    df = read_metrics(out / "ana")
    df = df[(df["stage"] == "main") & (df["info"] == INFO) & (df["horizon"] == 1) & (df["model"] == "ridge")
            & (df["protocol"] == "common")]
    df = df.sort_values(["n_train", "plant"]).reset_index(drop=True)
    return int(df.loc[(len(df) - 1) // 2, "plant"])


def timed(fn):
    fn()                                                     # warm-up
    laps = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        fn()
        laps.append(time.perf_counter() - started)
    return 1000 * float(np.median(laps))


def main():
    from pvbench.models import MODEL_NAMES
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    plant = int(sys.argv[3]) if len(sys.argv) > 3 else median_plant(out)
    experiment.boost_process(0)
    selected = pd.read_csv(out / "ayar" / "selected.csv").set_index(["horizon", "info", "model"])
    table = dataset.load_table(plant)
    periods = features.protocol(table, plant)
    rows = []
    for h in HORIZONS:
        s = features.prepare(table, h, INFO, periods, experiment.LOOKBACK)
        X_test = np.ascontiguousarray(s.X[s.test])
        base = {"plant": plant, "info": INFO, "horizon": h, "n_features": s.X.shape[2], "n_train": int(s.train.sum()),
                "n_val": int(s.val.sum()), "n_test": len(X_test)}

        def add(model, size, fit_s, predict_ms, **extra):
            rows.append({**base, "model": model, "size": size, "fit_s": round(fit_s, 2), "predict_ms": round(predict_ms, 3),
                         "per_1000_ms": round(1000 * predict_ms / len(X_test), 4), **extra})
            print({k: rows[-1][k] for k in ("horizon", "model", "size", "fit_s", "predict_ms")}, extra, flush=True)
            pd.DataFrame(rows).to_csv(target / "sure_olcumu.csv", index=False)

        # Ridge: the search of the benchmark, then prediction with the selected alpha
        from sklearn.linear_model import Ridge
        started = time.perf_counter()
        _, cfg = experiment.fit_ridge(s)
        fit_s = time.perf_counter() - started
        ridge = Ridge(alpha=cfg["alpha"]).fit(experiment.flatten(s.X[s.train]), s.y[s.train])
        flat = experiment.flatten(X_test)
        add("ridge", "", fit_s, timed(lambda: ridge.predict(flat)), device="cpu")

        # LightGBM: the search of the benchmark, then prediction with the selected configuration
        import lightgbm as lgb
        started = time.perf_counter()
        _, cfg = experiment.fit_lightgbm(s)
        fit_s = time.perf_counter() - started
        trees = int(cfg.pop("trees"))
        booster = lgb.train({"objective": "regression", "verbosity": -1, "seed": 0, "num_threads": 2, **cfg},
                            lgb.Dataset(experiment.flatten(s.X[s.train]), s.y[s.train]), trees)
        add("lightgbm", "", fit_s, timed(lambda: booster.predict(flat)), device="cpu", trees=trees)

        for name in MODEL_NAMES:
            tuned = {k: selected.loc[(h, INFO, name), k] for k in experiment.DEFAULT}
            tuned["batch_size"] = int(tuned["batch_size"])
            matched = {**tuned, "width": experiment.matched_width(name, MATCH, s.X.shape[2])}
            for size, cfg in (("tuned", tuned), ("matched", matched)):
                _, info = experiment.fit_neural(name, s, cfg, SEED, probe=lambda predict: timed(lambda: predict(X_test)))
                add(name, size, info["fit_s"], info["probe"], device=info["device"], params=info["params"],
                    epochs=info["epochs"], epoch_s=round(info["fit_s"] / info["epochs"], 3), width=round(float(cfg["width"]), 4),
                    batch_size=cfg["batch_size"])
    print(f"yazildi: {target / 'sure_olcumu.csv'}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
