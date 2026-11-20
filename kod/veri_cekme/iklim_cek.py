"""Santral konumlarinin iklim normalleri ve Koppen-Geiger sinifi (katmanli analiz, E7).

Kaynak: Open-Meteo Historical Weather API (ERA5 tabanli), 1991-2020, gunluk ortalama sicaklik ve
gunluk toplam yagis; santralin bulundugu model hucresi. Ham JSON yanitlari degistirilmeden
veri/ham/openmeteo/iklim_<no>.json olarak saklanir; mevcut dosyalar yeniden cekilmez.
Sinif, 12 aylik normalden Peel, Finlayson ve McMahon (2007, Hydrol. Earth Syst. Sci. 11, 1633-1644)
Tablo 1 olcutleriyle hesaplanir (C/D ayrimi: en soguk ay 0 derece C).

Calistirma (kod klasorunden):  python veri_cekme/iklim_cek.py
Cikti: veri/islenmis/iklim.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

KOK = Path(__file__).resolve().parents[2]
HAM = KOK / "veri" / "ham" / "openmeteo"
ENVANTER = KOK / "veri" / "islenmis" / "envanter.csv"
CIKTI = KOK / "veri" / "islenmis" / "iklim.csv"
URL = "https://archive-api.open-meteo.com/v1/archive"
DONEM = ("1991-01-01", "2020-12-31")
ISTEMCI_ADI = "pv-benchmark-fetch/1.0"


def cek(no, enlem, boylam, bekleme=20):
    yol = HAM / f"iklim_{no}.json"
    if yol.exists():
        return json.loads(yol.read_text(encoding="utf-8"))
    for deneme in range(6):
        r = requests.get(URL, params={"latitude": enlem, "longitude": boylam, "start_date": DONEM[0],
                                      "end_date": DONEM[1], "daily": "temperature_2m_mean,precipitation_sum",
                                      "timezone": "GMT"}, headers={"User-Agent": ISTEMCI_ADI}, timeout=120)
        if r.status_code == 200:
            yol.write_text(r.text, encoding="utf-8")
            time.sleep(bekleme)                       # uzun donem istekleri agir sayilir; siniri zorlamamak icin
            return r.json()
        time.sleep(60 * (deneme + 1))
    raise RuntimeError(f"santral {no}: Open-Meteo yaniti {r.status_code}: {r.text[:200]}")


def aylik_normal(yanit):
    """12 aylik ortalama sicaklik (C) ve aylik toplam yagis (mm): once her yil-ay, sonra yillar uzerinden ortalama."""
    g = pd.DataFrame(yanit["daily"])
    g["time"] = pd.to_datetime(g["time"])
    ay = g.groupby([g["time"].dt.year, g["time"].dt.month]).agg(
        t=("temperature_2m_mean", "mean"), p=("precipitation_sum", "sum"))
    ay.index.names = ["yil", "ay"]
    n = ay.groupby("ay").mean()
    return n["t"].to_numpy(), n["p"].to_numpy()


def koppen(t, p, enlem):
    """Koppen-Geiger sinifi (Peel vd. 2007, Tablo 1). t: 12 aylik sicaklik, p: 12 aylik yagis (Ocak-Aralik)."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    mat, map_ = t.mean(), p.sum()
    thot, tcold, tmon10 = t.max(), t.min(), int((t > 10).sum())
    yaz = np.zeros(12, bool)
    yaz[3:9] = True                                   # Nisan-Eylul
    if enlem < 0:
        yaz = ~yaz
    kis = ~yaz
    psdry, pwdry, pswet, pwwet = p[yaz].min(), p[kis].min(), p[yaz].max(), p[kis].max()
    if p[kis].sum() >= 0.7 * map_:
        esik = 2 * mat
    elif p[yaz].sum() >= 0.7 * map_:
        esik = 2 * mat + 28
    else:
        esik = 2 * mat + 14
    if map_ < 10 * esik:
        return ("BW" if map_ < 5 * esik else "BS") + ("h" if mat >= 18 else "k")
    if tcold >= 18:
        if p.min() >= 60:
            return "Af"
        return "Am" if p.min() >= 100 - map_ / 25 else "Aw"
    if thot <= 10:
        return "ET" if thot > 0 else "EF"
    ana = "C" if tcold > 0 else "D"
    kuru_yaz = psdry < 40 and psdry < pwwet / 3
    kuru_kis = pwdry < pswet / 10
    if kuru_yaz and kuru_kis:                          # ikisi de saglanirsa yagisin az oldugu mevsim belirler
        kuru_yaz = p[yaz].sum() < p[kis].sum()
        kuru_kis = not kuru_yaz
    ikinci = "s" if kuru_yaz else "w" if kuru_kis else "f"
    if thot >= 22:
        ucuncu = "a"
    elif tmon10 >= 4:
        ucuncu = "b"
    elif ana == "D" and tcold < -38:
        ucuncu = "d"
    else:
        ucuncu = "c"
    return ana + ikinci + ucuncu


def main():
    HAM.mkdir(parents=True, exist_ok=True)
    env = pd.read_csv(ENVANTER, sep=";")
    satirlar = []
    for s in env.itertuples():
        yanit = cek(int(s.no), float(s.lat), float(s.lon))
        t, p = aylik_normal(yanit)
        sinif = koppen(t, p, float(s.lat))
        satirlar.append({"no": int(s.no), "name": s.name, "lat": s.lat, "lon": s.lon,
                         "grid_lat": yanit.get("latitude"), "grid_lon": yanit.get("longitude"),
                         "mat_c": round(float(t.mean()), 2), "map_mm": round(float(p.sum()), 1),
                         "t_cold_c": round(float(t.min()), 2), "t_hot_c": round(float(t.max()), 2),
                         "koppen": sinif, "koppen_main": sinif[0]})
        print(f"{int(s.no):>3} {s.name:<28} {sinif:<4} MAT {t.mean():5.1f} C  MAP {p.sum():6.0f} mm", flush=True)
    pd.DataFrame(satirlar).to_csv(CIKTI, index=False, encoding="utf-8")
    print(f"yazildi: {CIKTI}")


if __name__ == "__main__":
    for akis in (sys.stdout, sys.stderr):
        akis.reconfigure(encoding="utf-8", errors="replace")
    main()
