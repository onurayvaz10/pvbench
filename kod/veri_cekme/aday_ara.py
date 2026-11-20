"""Ek santral adaylarini onceden yazili olcutle arar (Ek Not 1).

Olcut: Belcika, 10-50 kW, son cikti <= 7 gun, >= 1460 gunluk cikti, saat dilimi ve
koordinat Belcika ile tutarli, alti ornek gunun en az besinde gun ici veri.
Adaylar cikti sayisina gore azalan siralanir; ilk ikisi secilir, sonraki ikisi yedektir.
Tum adaylar ve her olcutun sonucu aday_santraller.csv dosyasina yazilir.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pvoutput_cek as pc  # noqa: E402

MERKEZLER = ["1000", "2000", "9000", "4000", "3000", "6000", "8000", "5000", "3500"]   # posta kodlari
ORNEK_GUNLER = ["20261115", "20260615", "20251215", "20250615", "20240615", "20230615"]
BELCIKA = {"enlem": (49.45, 51.55), "boylam": (2.5, 6.45)}
SAAT_DILIMLERI = {"Europe/Brussels", "Europe/Amsterdam", "Europe/Paris", "Europe/Berlin", "Europe/Luxembourg"}
ASGARI_CIKTI, GUC_ARALIGI, ASGARI_ORNEK = 1460, (10_000.0, 50_000.0), 5


def son_cikti_gun(metin):
    metin = metin.strip().lower()
    if metin in ("today", "yesterday"):
        return 0 if metin == "today" else 1
    bulunan = re.search(r"(\d+)\s+day", metin)
    if bulunan:
        return int(bulunan.group(1))
    hafta = re.search(r"(\d+)\s+week", metin)
    return int(hafta.group(1)) * 7 if hafta else 9999


def uretimli_kayit_sayisi(metin):
    """Gun ici yanitta uretim gucu (alan 4) ya da kumulatif enerji (alan 2) sifirdan buyuk kayit sayisi.
    Yalnizca kayit saymak yetmez: bazi sistemler bos (0/NaN) kayit gonderir."""
    sayac = 0
    for kayit in metin.split(";"):
        alan = kayit.split(",")
        try:
            if len(alan) > 4 and (float(alan[4]) > 0 or float(alan[2]) > 0):
                sayac += 1
        except ValueError:
            continue
    return sayac


def ara(istemci, yaricap="25km"):
    adaylar = {}
    for merkez in MERKEZLER:
        kod, metin = istemci.istek("search.jsp", {"q": f"{merkez} {yaricap}", "country_code": "be"})
        if kod != 200:
            istemci.yaz(f"  arama {merkez}: {kod} {metin[:80]}")
            continue
        for satir in metin.splitlines():
            alan = dict(zip(pc.ARAMA_ALANLARI, pc.csv_satiri(satir)))
            if alan.get("sid"):
                adaylar[alan["sid"]] = alan
        istemci.yaz(f"  arama {merkez}: {len(metin.splitlines())} sonuc")
    return list(adaylar.values())


def on_ele(adaylar, mevcut_sidler):
    uygun = []
    for a in adaylar:
        try:
            guc, cikti = float(a["guc_w"]), int(a["cikti_sayisi"])
        except ValueError:
            continue
        a["on_eleme"] = (GUC_ARALIGI[0] <= guc <= GUC_ARALIGI[1] and cikti >= ASGARI_CIKTI
                         and son_cikti_gun(a["son_cikti"]) <= 7 and a["sid"] not in mevcut_sidler)
        if a["on_eleme"]:
            uygun.append(a)
    return sorted(uygun, key=lambda a: (-int(a["cikti_sayisi"]), int(a["sid"])))


def dogrula(istemci, aday):
    """Sistem kaydi ve ornek gunlerle 4. ve 5. olcutleri denetler; adayi yerinde gunceller."""
    kod, metin = istemci.istek("getsystem.jsp", {"sid1": aday["sid"], "tz": 1})
    alanlar = pc.csv_satiri(metin.split(";")[0]) if kod == 200 else []
    aday["saat_dilimi"] = alanlar[-1] if len(alanlar) > 16 else ""
    aday["aralik_dk"] = alanlar[15] if len(alanlar) > 15 else ""
    try:
        enlem, boylam = float(alanlar[13]), float(alanlar[14])
    except (ValueError, IndexError):
        enlem = boylam = float("nan")
    aday["sistem_enlem"], aday["sistem_boylam"] = enlem, boylam
    aday["konum_uygun"] = (BELCIKA["enlem"][0] <= enlem <= BELCIKA["enlem"][1]
                           and BELCIKA["boylam"][0] <= boylam <= BELCIKA["boylam"][1]
                           and aday["saat_dilimi"] in SAAT_DILIMLERI)
    aday["ornek_dolu"] = 0
    if aday["konum_uygun"]:
        for gun in ORNEK_GUNLER:
            kod, metin = istemci.istek("getstatus.jsp", {"d": gun, "h": 1, "limit": 288, "asc": 1, "sid1": aday["sid"]})
            if kod == 200 and uretimli_kayit_sayisi(metin) >= 20:
                aday["ornek_dolu"] += 1
    aday["uygun"] = bool(aday["konum_uygun"] and aday["ornek_dolu"] >= ASGARI_ORNEK)
    return aday


def main(hedef=4, azami_dogrulama=12):
    args = pc.ayristirici().parse_args(["kimlik"] + sys.argv[1:])
    anahtar, kimlik = pc.kimlik_bilgileri(args.kimlik)
    if not anahtar:
        print("API anahtari bulunamadi.")
        return 2
    istemci = pc.Istemci(anahtar, kimlik, asgari_aralik=0.5)
    mevcut = {s["sid"] for s in pc.santralleri_oku(args.santraller) if s["sid"]}
    adaylar = ara(istemci)
    siralı = on_ele(adaylar, mevcut)
    print(f"Toplam aday: {len(adaylar)} | on elemeyi gecen: {len(siralı)}")
    uygunlar = []
    for aday in siralı[:azami_dogrulama]:
        dogrula(istemci, aday)
        print(f"  {aday['sistem_adi'][:30]:<31} sid={aday['sid']:<7} {float(aday['guc_w']) / 1000:>6.2f} kW  "
              f"cikti={aday['cikti_sayisi']:>5}  {aday['saat_dilimi']:<17} ({aday['sistem_enlem']:.2f}, {aday['sistem_boylam']:.2f})  "
              f"ornek={aday['ornek_dolu']}/6  {'UYGUN' if aday['uygun'] else 'elendi'}")
        if aday["uygun"]:
            uygunlar.append(aday)
            if len(uygunlar) >= hedef:
                break
    for sira, aday in enumerate(uygunlar, 1):
        aday["karar"] = "secildi" if sira <= 2 else "yedek"
    sutunlar = pc.ARAMA_ALANLARI + ["on_eleme", "saat_dilimi", "aralik_dk", "sistem_enlem", "sistem_boylam",
                                    "konum_uygun", "ornek_dolu", "uygun", "karar"]
    hedef_dosya = Path(args.ham) / "aday_santraller.csv"
    with open(hedef_dosya, "w", encoding="utf-8-sig", newline="") as f:
        yazici = csv.DictWriter(f, fieldnames=sutunlar, delimiter=";", extrasaction="ignore")
        yazici.writeheader()
        yazici.writerows(sorted(adaylar, key=lambda a: (not a.get("on_eleme", False), -int(a.get("cikti_sayisi") or 0))))
    print(f"Kaydedildi: {hedef_dosya}")
    for aday in uygunlar:
        print(f"  {aday['karar'].upper():<8} {aday['sistem_adi']} (sid {aday['sid']}, {float(aday['guc_w']) / 1000:.2f} kW)")
    return 0


if __name__ == "__main__":
    for akis in (sys.stdout, sys.stderr):
        if hasattr(akis, "reconfigure"):
            akis.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
