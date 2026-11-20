"""Saat siniri denetimi: PVOutput kayitlarindan saatlik ortalama guc hesaplanirken, saat sinirinda kayit yoksa siniri
iceren kayit araliginin enerjisi iki saate sureyle orantili paylastirilir (pvoutput.hourly_from_intervals).
Bu betik bunun ne siklikta oldugunu olcer.  Santral basina, gecerli (tam aciklanmis) ve uretimi olan saatler icinde:
    hours            gecerli ve enerjisi sifirdan buyuk saat sayisi
    share_straddled  bir sinirinda (basinda ya da sonunda) kayit bulunmayan, yani enerjisinin bir bolumu paylastirmayla
                     gelen saatlerin payi
    energy_share     bu saatlerde paylastirmayla gelen enerjinin toplam enerjiye orani
    longest_min      siniri asan araliklarin en uzunu (dakika; kural geregi en cok 30)

Cikti: <sonuclar>/ek_saat_siniri.csv
Kullanim:  python saat_siniri_denetimi.py <sonuclar klasoru>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import dataset, pvoutput  # noqa: E402


def main():
    target = Path(sys.argv[1])
    plants = dataset.load_plants()
    rows = []
    for no, plant in plants[plants["source"] == "pvoutput"].iterrows():
        folder = dataset.RAW / "pvoutput" / plant["sid"]
        raw = pvoutput.read_raw(folder)
        totals = pvoutput.read_daily_totals(folder)
        iv = pvoutput.build_intervals(raw, plant["tz"], plant["capacity_kw"], 30.0, totals if len(totals) else None)
        hourly = pvoutput.hourly_from_intervals(iv, plant["capacity_kw"])
        start, end, energy = iv["start"].to_numpy(), iv["end"].to_numpy(), iv["energy_wh"].to_numpy()
        first, last = np.floor(start / 3600.0), np.ceil(end / 3600.0) - 1      # first and last hour an interval touches
        crossing = (last > first) & (energy > 0) & iv["valid"].to_numpy(bool)
        shared = pd.Series(0.0, index=hourly.index)                           # apportioned energy per hour (Wh)
        for s, e, w, a, b in zip(start[crossing], end[crossing], energy[crossing], first[crossing], last[crossing]):
            for hour in np.arange(a, b + 1):
                lo, hi = max(s, hour * 3600.0), min(e, (hour + 1) * 3600.0)
                stamp = pd.Timestamp((hour + 1) * 3600.0, unit="s", tz="UTC")
                if stamp in shared.index:
                    shared[stamp] += w * (hi - lo) / (e - s)
        good = hourly["power_kw"].notna() & (hourly["power_kw"] > 0)
        good &= hourly.index > dataset.START
        total = float(hourly.loc[good, "power_kw"].sum() * 1000.0)
        rows.append({"no": no, "name": plant["name"], "hours": int(good.sum()),
                     "share_straddled": float((shared[good] > 0).mean()),
                     "energy_share": float(shared[good].sum() / total),
                     "longest_min": float(((end - start)[crossing]).max() / 60.0) if crossing.any() else 0.0})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(target / "ek_saat_siniri.csv", index=False)
    print(f"\nortanca pay {out['share_straddled'].median():.4f}, en buyuk {out['share_straddled'].max():.4f}; "
          f"enerji payi ortanca {out['energy_share'].median():.4f}, en buyuk {out['energy_share'].max():.4f}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
