"""Belirlenim denemesi: ayni tohumla iki kez egitilen ag ayni tahmini veriyor mu?

Her mimari, bir santralde (ortak protokollu santraller icinde S1, h = 1 egitim ornegi sayisi ortanca olan; sure
olcumundeki santral), S1 girdileriyle, h = 1 ve h = 24 icin, ana kiyasta secilen yapilandirma ve tohum 0 ile ayni
surec icinde iki kez egitilir.  Yazilanlar: iki egitimin donem sayisi, test nRMSE degeri, test tahminleri arasindaki
en buyuk mutlak fark (kurulu gucun payi) ve tahminlerin bit duzeyinde ayni olup olmadigi.

Sonuc (20 Kasim 2026): evrisim katmani icermeyen aglar (dense, simplernn, lstm, gru, bilstm, transformer) birebir
yinelenir; evrisim katmanli aglar (cnn, cnn_lstm, cnn_gru, cnn_lstm_gru, tcn) yinelenmez (GPU'daki evrisim yordamlari
belirlenimci degildir).

Cikti: <sonuclar>/ek_belirlenim_denemesi.csv
Kullanim:  python belirlenim_denemesi.py <cikti> <sonuclar klasoru> [isci no] [mimariler ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# pvbench.experiment must be imported before numpy (it limits the BLAS threads; see sure_olcumu.py)
from pvbench import dataset, experiment, features, metrics  # noqa: E402, I001
from pvbench.experiment import read_metrics  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INFO, HORIZONS, SEED = "S1", (1, 24), 0


def median_plant(out: Path) -> int:
    df = read_metrics(out / "ana")
    df = df[(df["stage"] == "main") & (df["info"] == INFO) & (df["horizon"] == 1) & (df["model"] == "ridge")
            & (df["protocol"] == "common")].sort_values(["n_train", "plant"]).reset_index(drop=True)
    return int(df.loc[(len(df) - 1) // 2, "plant"])


def main():
    from pvbench.models import MODEL_NAMES
    out, target = Path(sys.argv[1]), Path(sys.argv[2])
    worker = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    names = sys.argv[4:] or list(MODEL_NAMES)
    experiment.boost_process(worker)
    plant = median_plant(out)
    selected = pd.read_csv(out / "ayar" / "selected.csv").set_index(["horizon", "info", "model"])
    table = dataset.load_table(plant)
    periods = features.protocol(table, plant)
    rows = []
    for h in HORIZONS:
        s = features.prepare(table, h, INFO, periods, experiment.LOOKBACK)
        for name in names:
            cfg = {k: selected.loc[(h, INFO, name), k] for k in experiment.DEFAULT}
            cfg["batch_size"] = int(cfg["batch_size"])
            fits = [experiment.fit_neural(name, s, cfg, SEED) for _ in range(2)]
            scores = [100 * metrics.rmse(s.y[s.test], features.postprocess(pred, s.aux["cs_lead"])[s.test]) for pred, _ in fits]
            a, b = fits[0][0][s.test], fits[1][0][s.test]
            rows.append({"plant": plant, "info": INFO, "horizon": h, "model": name, "device": fits[0][1]["device"],
                         "epochs_1": fits[0][1]["epochs"], "epochs_2": fits[1][1]["epochs"],
                         "nrmse_1": scores[0], "nrmse_2": scores[1], "abs_diff_nrmse": abs(scores[0] - scores[1]),
                         "max_abs_diff_forecast": float(np.abs(a - b).max()), "identical": bool(np.array_equal(a, b))})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(target / "ek_belirlenim_denemesi.csv", index=False)
    print(f"yazildi: {target / 'ek_belirlenim_denemesi.csv'}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
