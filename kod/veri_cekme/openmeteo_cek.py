"""Open-Meteo meteoroloji verisi cekme araci (API anahtari gerekmez).

Uc veri turu, saatlik ve UTC:
    arsiv    ERA5 tabanli reanaliz (Historical Weather API); S1 ve S3 girdileri
    icon     Arsivlenmis hava tahminleri (Previous Runs API), tek kuresel model (DWD ICON): hedef saat icin
             24 ve 48 saat once uretilmis degerler (_previous_day1, _previous_day2); S2 ve S2d2 girdileri
    tahmin   Ayni arsivin varsayilan `best_match` secimi; yalnizca S2B duyarlilik karsilastirmasi icin

Koordinat onceligi: santraller.csv'deki enlem/boylam sutunlari, yoksa
pvoutput_cek.py'nin `sistem` komutuyla olusan sistemler.csv, yoksa aday arama dosyalari
(aday_santraller*.csv; sonradan eklenen santraller). Yalnizca etkin santraller (atla bos,
kaynak pvoutput ya da epias) cekilir. Koordinat denetimi: kod/veri_cekme/konum_dogrula.py.
Ham JSON yanitlari degistirilmeden saklanir; mevcut dosyalar yeniden cekilmez.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvoutput_cek import KOK, SANTRAL_DOSYASI, sayi, santralleri_oku  # noqa: E402

HAM_KLASOR = KOK / "veri" / "ham" / "openmeteo"
SISTEM_DOSYASI = KOK / "veri" / "ham" / "pvoutput" / "sistemler.csv"
ISTEMCI_ADI = "pv-benchmark-fetch/1.0"
DEGISKENLER = [
    "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "temperature_2m",
    "relative_humidity_2m", "wind_speed_10m", "cloud_cover",
]
TURLER = {
    "arsiv": {
        "url": "https://archive-api.open-meteo.com/v1/archive",
        "saatlik": DEGISKENLER,
        "en_erken": dt.date(1940, 1, 1),
    },
    "tahmin": {
        "url": "https://previous-runs-api.open-meteo.com/v1/forecast",
        "saatlik": [d + ek for d in DEGISKENLER for ek in ("", "_previous_day1", "_previous_day2")],
        "en_erken": dt.date(2024, 1, 1),
    },
    # Ayni arsiv, tek bir kuresel modelden (DWD ICON). Varsayilan `best_match` secimi bazi ulkelerde
    # 2024 yazinda baska bir modele gectigi icin (bkz. makale, Yontem) S2 girdisi
    # butun santrallerde bu turden alinir; `tahmin` yalnizca duyarlilik karsilastirmasi icindir.
    "icon": {
        "url": "https://previous-runs-api.open-meteo.com/v1/forecast",
        "saatlik": [d + ek for d in DEGISKENLER for ek in ("_previous_day1", "_previous_day2")],
        "en_erken": dt.date(2024, 1, 1),
        "model": "icon_global",
    },
}


def konumlari_oku(santral_yolu=None, sistem_yolu=None):
    """[(no, ad, enlem, boylam)] ve koordinati bulunamayan santral adlari."""
    sistemler = {}
    for aday in ("aday_santraller.csv", "aday_santraller_nl.csv", "aday_santraller_tr.csv"):
        a_yol = Path(sistem_yolu or SISTEM_DOSYASI).parent / aday
        if a_yol.exists():
            with open(a_yol, encoding="utf-8-sig", newline="") as f:
                for s in csv.DictReader(f, delimiter=";"):
                    if s.get("enlem"):
                        sistemler[s["sid"]] = (s["enlem"], s["boylam"])
    yol = Path(sistem_yolu or SISTEM_DOSYASI)
    if yol.exists():
        with open(yol, encoding="utf-8-sig", newline="") as f:
            for s in csv.DictReader(f, delimiter=";"):
                sistemler[s["sid"]] = (s.get("enlem", ""), s.get("boylam", ""))
    konumlar, eksik = [], []
    for s in santralleri_oku(santral_yolu or SANTRAL_DOSYASI):
        if s.get("atla") or s.get("kaynak") not in ("pvoutput", "epias"):
            continue                                    # yalnizca etkin santraller
        enlem, boylam = s.get("enlem", ""), s.get("boylam", "")
        if not (enlem and boylam) and s["sid"] in sistemler:
            enlem, boylam = sistemler[s["sid"]]
        try:
            konumlar.append((int(s["no"]), s["ad"], sayi(enlem), sayi(boylam)))
        except ValueError:
            eksik.append(s["ad"])
    return konumlar, eksik


def yil_parcalari(bas, bit):
    parcalar, imlec = [], bas
    while imlec <= bit:
        son = min(dt.date(imlec.year, 12, 31), bit)
        parcalar.append((imlec, son))
        imlec = son + dt.timedelta(days=1)
    return parcalar


def cek(url, parametreler, oturum, uyku=time.sleep):
    for deneme in range(6):
        try:
            yanit = oturum.get(url, params=parametreler, headers={"User-Agent": ISTEMCI_ADI}, timeout=120)
        except requests.RequestException:
            uyku(15 * (deneme + 1))
            continue
        if yanit.status_code == 200:
            return yanit.json()
        if yanit.status_code == 429 or yanit.status_code >= 500:
            uyku(60 * (deneme + 1))
            continue
        raise RuntimeError(f"Open-Meteo {yanit.status_code}: {yanit.text[:300]}")
    raise RuntimeError("Open-Meteo yanit vermedi (6 deneme).")


def kapsam(veri, anahtar):
    saatler, degerler = veri["hourly"]["time"], veri["hourly"].get(anahtar, [])
    dolu = [i for i, x in enumerate(degerler) if x is not None]
    return len(dolu), len(degerler), (saatler[dolu[0]] if dolu else "")


def calistir(args, oturum=None, uyku=time.sleep, yaz=print):
    oturum = oturum or requests.Session()
    konumlar, eksik = konumlari_oku(args.santraller, args.sistemler)
    if eksik:
        yaz(f"Koordinati olmayan santraller atlandi: {', '.join(eksik)}")
    ham = Path(args.ham)
    ozet = []
    for no, ad, enlem, boylam in konumlar:
        for tur in args.tur:
            ayar = TURLER[tur]
            bas = max(args.baslangic, ayar["en_erken"])
            for p_bas, p_bit in yil_parcalari(bas, args.bitis):
                hedef = ham / f"{no:02d}" / f"{tur}_{p_bas:%Y%m%d}_{p_bit:%Y%m%d}.json"
                if hedef.exists() and not args.yenile:
                    veri = json.loads(hedef.read_text(encoding="utf-8"))
                else:
                    veri = cek(ayar["url"], {
                        "latitude": enlem, "longitude": boylam, "start_date": p_bas.isoformat(),
                        "end_date": p_bit.isoformat(), "hourly": ",".join(ayar["saatlik"]),
                        "timezone": "GMT", "wind_speed_unit": "ms",
                        **({"models": ayar["model"]} if "model" in ayar else {})}, oturum, uyku)
                    hedef.parent.mkdir(parents=True, exist_ok=True)
                    hedef.write_text(json.dumps(veri), encoding="utf-8")
                    uyku(args.aralik)
                olcut = "shortwave_radiation" + ("" if tur == "arsiv" else "_previous_day1")
                dolu, toplam, ilk = kapsam(veri, olcut)
                ozet.append({"no": no, "ad": ad, "tur": tur, "baslangic": p_bas, "bitis": p_bit,
                             "olcut": olcut, "dolu_saat": dolu, "toplam_saat": toplam, "ilk_dolu": ilk,
                             "izgara_enlem": veri.get("latitude"), "izgara_boylam": veri.get("longitude"),
                             "rakim": veri.get("elevation")})
                yaz(f"  {no:>2} {ad:<28} {tur:<6} {p_bas}..{p_bit}  dolu {dolu}/{toplam}  ilk: {ilk}")
    if ozet:
        ham.mkdir(parents=True, exist_ok=True)
        with open(ham / "kapsam.csv", "w", encoding="utf-8-sig", newline="") as f:
            yazici = csv.DictWriter(f, fieldnames=list(ozet[0].keys()), delimiter=";")
            yazici.writeheader()
            yazici.writerows(ozet)
    return ozet


def ayristirici():
    p = argparse.ArgumentParser(description="Open-Meteo meteoroloji verisi cekme araci")
    p.add_argument("--tur", nargs="+", choices=list(TURLER), default=list(TURLER))
    p.add_argument("--santraller", default=str(SANTRAL_DOSYASI))
    p.add_argument("--sistemler", default=str(SISTEM_DOSYASI))
    p.add_argument("--ham", default=str(HAM_KLASOR))
    p.add_argument("--baslangic", type=dt.date.fromisoformat, default=dt.date(2022, 11, 1))
    p.add_argument("--bitis", type=dt.date.fromisoformat, default=dt.date.today() - dt.timedelta(days=1))
    p.add_argument("--aralik", type=float, default=1.0, help="Istekler arasi bekleme (sn)")
    p.add_argument("--yenile", action="store_true")
    return p


if __name__ == "__main__":
    for akis in (sys.stdout, sys.stderr):
        if hasattr(akis, "reconfigure"):
            akis.reconfigure(encoding="utf-8", errors="replace")
    calistir(ayristirici().parse_args())
