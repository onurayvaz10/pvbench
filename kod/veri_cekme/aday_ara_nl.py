"""Hollanda'daki PVOutput adaylarini onceden yazili olcutle arar
(Ek Not 1; Belcika kuralinin kosullari, Hollanda).

Kosullar olcut 1 ile aynidir: 10-50 kW, son cikti <= 7 gun, >= 1460 gunluk cikti, koordinat ve saat dilimi
Hollanda ile tutarli, alti ornek gunun (olcut 1'deki gunler) en az besinde gun ici veri; cikti sayisina gore
azalan siralama. Arama posta kodu + 25 km ile yapilir; 30 sonuc sinirinda aktif sistemlerin elenmemesi icin
`seen=7` suzgeci eklenir (son cikti kosuluyla ayni sey). Turkiye'den 1 santral ciktigi icin 4 santral secilir,
sonraki 2 yedektir. Sonuclar aday_santraller_nl.csv dosyasina yazilir.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aday_ara as ab  # noqa: E402
import pvoutput_cek as pc  # noqa: E402

MERKEZLER = ["1011", "2511", "3011", "3511", "5611", "9711", "8011", "6811", "6211", "8911", "4811", "7511"]
ab.BELCIKA = {"enlem": (50.70, 53.60), "boylam": (3.30, 7.25)}       # dogrula() bu sinirlari kullanir
# Kaba sinir kutusu Belcika Limburg'unu da kapsiyor. Hollanda ile tutarli saat dilimi
# yalnizca Europe/Amsterdam'dir; bu kosul sinir bolgesindeki Belcika sistemlerini ayirir.
ab.SAAT_DILIMLERI = {"Europe/Amsterdam"}
SECILECEK, YEDEK = 4, 2


def ara(istemci, yaricap="25km"):
    adaylar = {}
    for merkez in MERKEZLER:
        kod, metin = istemci.istek("search.jsp", {"q": f"{merkez} {yaricap}", "country_code": "nl", "seen": 7})
        satirlar = [s for s in metin.splitlines() if s.strip()] if kod == 200 else []
        for satir in satirlar:
            alan = dict(zip(pc.ARAMA_ALANLARI, pc.csv_satiri(satir)))
            if alan.get("sid"):
                adaylar.setdefault(alan["sid"], alan)
        istemci.yaz(f"  arama {merkez}: kod {kod}, {len(satirlar)} sonuc")
    return list(adaylar.values())


def main(azami_dogrulama=20):
    args = pc.ayristirici().parse_args(["kimlik"] + sys.argv[1:])
    anahtar, kimlik = pc.kimlik_bilgileri(args.kimlik)
    if not anahtar:
        print("API anahtari bulunamadi.")
        return 2
    istemci = pc.Istemci(anahtar, kimlik, asgari_aralik=0.5)
    mevcut = {s["sid"] for s in pc.santralleri_oku(args.santraller) if s["sid"]}
    adaylar = ara(istemci)
    sirali = ab.on_ele(adaylar, mevcut)
    print(f"Toplam aday: {len(adaylar)} | on elemeyi gecen: {len(sirali)}")
    uygunlar = []
    for aday in sirali[:azami_dogrulama]:
        ab.dogrula(istemci, aday)
        print(f"  {aday['sistem_adi'][:30]:<31} sid={aday['sid']:<7} {float(aday['guc_w']) / 1000:>6.2f} kW  "
              f"cikti={aday['cikti_sayisi']:>5}  {aday['saat_dilimi']:<17} ({aday['sistem_enlem']:.2f}, "
              f"{aday['sistem_boylam']:.2f})  ornek={aday['ornek_dolu']}/6  {'UYGUN' if aday['uygun'] else 'elendi'}")
        if aday["uygun"]:
            uygunlar.append(aday)
            if len(uygunlar) >= SECILECEK + YEDEK:
                break
    for sira, aday in enumerate(uygunlar, 1):
        aday["karar"] = "secildi" if sira <= SECILECEK else "yedek"
    sutunlar = pc.ARAMA_ALANLARI + ["on_eleme", "saat_dilimi", "aralik_dk", "sistem_enlem", "sistem_boylam",
                                    "konum_uygun", "ornek_dolu", "uygun", "karar"]
    hedef_dosya = Path(args.ham) / "aday_santraller_nl.csv"
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
