"""Zaman hizalamasi denetimi: uretim ve hava verisi, acik gokyuzu isinimina gore kac saat kaymis?

Her santralin saatlik tablosunda (saat sonu etiketli, UTC) su seriler, hesaplanan acik gokyuzu isinimi cs(t) ile
karsilastirilir:  p (hedef), era_shortwave_radiation (geriye donuk analiz), nwp_shortwave_radiation (arsivlenmis
ICON tahmini, 24 saat once uretilmis).  corr(x(t), cs(t + k)), k = -3 .. +3 saat hesaplanir; en yuksek korelasyonun
kaymasi k* tam sayi olarak ve komsu uc noktadan gecen parabolun tepe noktasi olarak (saat kesri) raporlanir.
k* > 0: seri gunese gore erken (ornegin doguya bakan dizi);  k* = 0: ayni saat araligi.
Bir saatlik etiket hatasi (saat basi / saat sonu, yaz saati, saat dilimi) |k*| = 1 olarak gorunur.

Cikti: hizalama_denetimi.csv
Kullanim:  python hizalama_denetimi.py <sonuclar klasoru>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import dataset  # noqa: E402

LAGS = range(-3, 4)
SERIES = {"p": "generation", "era_shortwave_radiation": "reanalysis_ghi", "nwp_shortwave_radiation": "forecast_ghi"}


def lag_of_maximum(x: pd.Series, cs: pd.Series) -> dict:
    corr = {k: x.corr(cs.shift(-k)) for k in LAGS}                 # cs.shift(-k)[t] = cs(t + k)
    best = max(corr, key=corr.get)
    peak = float(best)
    if min(LAGS) < best < max(LAGS):
        a, b, c = corr[best - 1], corr[best], corr[best + 1]
        peak = best + 0.5 * (a - c) / (a - 2 * b + c)
    return {"lag_h": int(best), "lag_interp_h": peak, "corr_at_0": corr[0], "corr_at_best": corr[best]}


def main():
    target = Path(sys.argv[1])
    rows = []
    for no, plant in dataset.load_plants().iterrows():
        table = dataset.load_table(no)
        table = table[table["p"].notna()]
        row = {"no": no, "name": plant["name"], "hours": len(table)}
        for column, label in SERIES.items():
            ok = table[column].notna()
            r = lag_of_maximum(table.loc[ok, column].reindex(table.index), table["cs_ghi"])
            row.update({f"{label}_{k}": v for k, v in r.items()})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(target / "hizalama_denetimi.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(out[["no", "name"] + [f"{label}_{k}" for label in SERIES.values() for k in ("lag_h", "lag_interp_h", "corr_at_0")]]
          .round(3).to_string(index=False))
    for label in SERIES.values():
        print(f"{label:<15} tam sayi kayma dagilimi: {out[f'{label}_lag_h'].value_counts().to_dict()};  "
              f"parabol tepe noktasi {out[f'{label}_lag_interp_h'].min():+.2f} .. {out[f'{label}_lag_interp_h'].max():+.2f} saat")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
