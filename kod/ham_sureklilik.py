"""Ham sureklilik: referans tahminlerin son islemden (gece sifirlama ve [0, 1.15] kirpma) once ve sonra hatasi.

Makaledeki butun yontemlere ayni son islem uygulanir; bu, sureklilik yontemlerini de iyilestirir ve onlara gore
hesaplanan beceri puanlarini ihtiyatli yapar.  Bu betik etkinin buyuklugunu verir: her santral ve hucrede
(bilgi kumesi S1 ve S2, h = 1 ve 24) sureklilik, 24 saatlik mevsimsel sureklilik, acik gokyuzu surekliligi ve
klimatolojinin test nRMSE degeri (% kurulu guc), son islemsiz (raw) ve son islemli (post).

Cikti: <sonuclar>/ek_ham_sureklilik_hucreler.csv (santral duzeyi), ek_ham_sureklilik.csv (25 santral ortalamasi)
Kullanim:  python ham_sureklilik.py <sonuclar klasoru>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import dataset, experiment, features, metrics  # noqa: E402

MODELS = ["persistence", "seasonal_persistence", "smart_persistence", "climatology"]


def main():
    target = Path(sys.argv[1])
    rows = []
    for no, plant in dataset.load_plants().iterrows():
        table = dataset.load_table(no)
        periods = features.protocol(table, no)
        for info in ("S1", "S2"):
            for h in (1, 24):
                s = features.prepare(table, h, info, periods, experiment.LOOKBACK)
                y = s.y[s.test]
                for model in MODELS:
                    raw = s.aux[model]
                    post = features.postprocess(raw, s.aux["cs_lead"])
                    rows.append({"plant": no, "name": plant["name"], "info": info, "horizon": h, "model": model,
                                 "n": int(s.test.sum()), "nrmse_raw": 100 * metrics.rmse(y, raw[s.test]),
                                 "nrmse_post": 100 * metrics.rmse(y, post[s.test])})
        print(f"santral {no} tamam", flush=True)
    cells = pd.DataFrame(rows)
    cells["raw_minus_post"] = cells["nrmse_raw"] - cells["nrmse_post"]
    cells.to_csv(target / "ek_ham_sureklilik_hucreler.csv", index=False)
    summary = cells.groupby(["info", "horizon", "model"], sort=False).agg(
        plants=("plant", "nunique"), nrmse_raw=("nrmse_raw", "mean"), nrmse_post=("nrmse_post", "mean"),
        raw_minus_post=("raw_minus_post", "mean"), raw_minus_post_max=("raw_minus_post", "max")).reset_index()
    summary.to_csv(target / "ek_ham_sureklilik.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
