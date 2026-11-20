"""EPIAS ham kaydinin denetimi (olculen santral): bos deger, eksik ve yinelenen saat, gun ortasinda sifir uretim.

Saatler Turkiye saatiyle (UTC+3) saat baslangicina gore etiketlidir.  "Gun ortasi": yerel 10:00-14:00 arasinda
baslayan bes saat.  Sifir uretimli gun ortasi saatleri kalite kontrolunun hicbir kuraliyla cikarilmaz ve gecerli hedef
olarak kalir; test doneminde kac tanesinin bulundugu ayrica verilir.

Cikti: <sonuclar>/ek_epias_denetimi.csv (tek satir) ve ek_epias_sifir_gunler.csv (gun basina sifir saat sayisi)
Kullanim:  python epias_denetimi.py <sonuclar klasoru>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvbench import dataset, features  # noqa: E402

import pandas as pd  # noqa: E402


def main():
    target = Path(sys.argv[1])
    plants = dataset.load_plants()
    rows, days = [], []
    for no, plant in plants[plants["source"] == "epias"].iterrows():
        raw = pd.read_csv(dataset.RAW / "epias" / "uevm" / f"{plant['sid']}.csv", dtype=str)
        time = pd.to_datetime(raw["zaman_tr"], utc=True)
        solar = pd.to_numeric(raw["gunes"], errors="coerce")
        blank = raw["gunes"].isna() | (raw["gunes"].str.strip() == "")
        grid = pd.date_range(time.min(), time.max(), freq="h")
        local_hour = (time + pd.Timedelta(hours=3)).dt.hour
        midday = local_hour.between(10, 14)
        zero = midday & (solar == 0)
        table = dataset.load_table(no)
        periods = features.protocol(table, no)
        label = time + pd.Timedelta(hours=1)                     # hour-end label of the hourly tables
        in_test = (label > periods["test_start"]) & (label <= periods["test_end"])   # split limits are UTC timestamps
        valid = table["p"].reindex(pd.DatetimeIndex(label[zero])).notna()
        local_day = (time + pd.Timedelta(hours=3)).dt.date
        per_day = zero.groupby(local_day).sum()
        per_day = per_day[per_day > 0]
        rows.append({"no": no, "name": plant["name"], "first_hour": str(time.min()), "last_hour": str(time.max()),
                     "hours": len(raw), "blank_values": int(blank.sum()), "missing_hours": len(grid) - time.nunique(),
                     "duplicated_hours": int(time.duplicated().sum()), "midday_hours": int(midday.sum()),
                     "midday_zero_hours": int(zero.sum()), "midday_zero_days": len(per_day),
                     "midday_zero_hours_kept_as_valid": int(valid.sum()),
                     "midday_zero_hours_in_test": int((zero & in_test).sum())})
        for day, count in per_day.items():
            days.append({"no": no, "local_day": str(day), "weekday": pd.Timestamp(day).day_name(), "zero_hours": int(count)})
    pd.DataFrame(rows).to_csv(target / "ek_epias_denetimi.csv", index=False)
    pd.DataFrame(days).to_csv(target / "ek_epias_sifir_gunler.csv", index=False)
    print(pd.DataFrame(rows).T.to_string())
    print(pd.DataFrame(days).to_string(index=False))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
