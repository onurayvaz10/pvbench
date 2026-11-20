"""Turkiye'deki PVOutput adaylarini onceden yazili olcutle arar
(Ek Not 1; Turkiye icin kural ve duzeltmesi).

Olcut: 81 il merkezinden 50 km mesafe aramasi (aktiflik suzgeci yok), 1-100 kW, >= 730 gunluk cikti,
koordinat Turkiye sinirlari icinde ve saat dilimi Istanbul, sistemin kendi kaydinin son iki yilina
yayilmis alti ornek gunun en az besinde gun ici veri. Adaylar cikti sayisina gore azalan siralanir;
ilk besi secilir, sonraki ikisi yedek. Tum adaylar aday_santraller_tr.csv dosyasina yazilir.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aday_ara as ab  # noqa: E402
import pvoutput_cek as pc  # noqa: E402

ILLER = [
    "Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Amasya", "Ankara", "Antalya", "Artvin", "Aydın", "Balıkesir",
    "Bilecik", "Bingöl", "Bitlis", "Bolu", "Burdur", "Bursa", "Çanakkale", "Çankırı", "Çorum", "Denizli",
    "Diyarbakır", "Edirne", "Elazığ", "Erzincan", "Erzurum", "Eskişehir", "Gaziantep", "Giresun", "Gümüşhane",
    "Hakkari", "Hatay", "Isparta", "Mersin", "İstanbul", "İzmir", "Kars", "Kastamonu", "Kayseri", "Kırklareli",
    "Kırşehir", "Kocaeli", "Konya", "Kütahya", "Malatya", "Manisa", "Kahramanmaraş", "Mardin", "Muğla", "Muş",
    "Nevşehir", "Niğde", "Ordu", "Rize", "Sakarya", "Samsun", "Siirt", "Sinop", "Sivas", "Tekirdağ", "Tokat",
    "Trabzon", "Tunceli", "Şanlıurfa", "Uşak", "Van", "Yozgat", "Zonguldak", "Aksaray", "Bayburt", "Karaman",
    "Kırıkkale", "Batman", "Şırnak", "Bartın", "Ardahan", "Iğdır", "Yalova", "Karabük", "Kilis", "Osmaniye", "Düzce",
]
YARICAP = "50km"
TURKIYE = {"enlem": (35.8, 42.1), "boylam": (25.6, 44.8)}
SAAT_DILIMLERI = {"Europe/Istanbul", "Asia/Istanbul", "Turkey"}
ASGARI_CIKTI, GUC_ARALIGI, ASGARI_ORNEK = 730, (1_000.0, 100_000.0), 5
SECILECEK, YEDEK = 5, 2


def il_merkezi(il):
    """Open-Meteo coğrafi kodlamasından il merkezinin koordinatı (önce il merkezi yerleşimi)."""
    url = ("https://geocoding-api.open-meteo.com/v1/search?"
           + urllib.parse.urlencode({"name": il, "count": 10, "language": "tr", "countryCode": "TR"}))
    with urllib.request.urlopen(url, timeout=30) as yanit:
        sonuclar = json.load(yanit).get("results", [])
    for kod in ("PPLC", "PPLA"):
        for s in sonuclar:
            if s.get("feature_code") == kod:
                return s["latitude"], s["longitude"]
    return (sonuclar[0]["latitude"], sonuclar[0]["longitude"]) if sonuclar else None


def turkiye_icinde(enlem, boylam):
    return TURKIYE["enlem"][0] <= enlem <= TURKIYE["enlem"][1] and TURKIYE["boylam"][0] <= boylam <= TURKIYE["boylam"][1]


def ara(istemci):
    adaylar = {}
    for il in ILLER:
        merkez = il_merkezi(il)
        time.sleep(0.3)
        if not merkez:
            istemci.yaz(f"  {il}: koordinat bulunamadi")
            continue
        kod, metin = istemci.istek("search.jsp", {"q": YARICAP, "ll": f"{merkez[0]},{merkez[1]}"})
        satirlar = [s for s in metin.splitlines() if s.strip()] if kod == 200 else []
        for satir in satirlar:
            alan = dict(zip(pc.ARAMA_ALANLARI, pc.csv_satiri(satir)))
            if alan.get("sid"):
                alan.setdefault("arama_merkezi", il)
                adaylar.setdefault(alan["sid"], alan)
        istemci.yaz(f"  {il} ({merkez[0]:.2f}, {merkez[1]:.2f}): kod {kod}, {len(satirlar)} sonuc"
                    f"{' (30 sinirinda)' if len(satirlar) >= 30 else ''}")
    return list(adaylar.values())


def on_ele(adaylar, mevcut_sidler):
    uygun = []
    for a in adaylar:
        try:
            guc, cikti = float(a["guc_w"]), int(a["cikti_sayisi"])
            enlem, boylam = float(a["enlem"]), float(a["boylam"])
        except (ValueError, KeyError, TypeError):
            a["on_eleme"] = False
            continue
        a["on_eleme"] = (GUC_ARALIGI[0] <= guc <= GUC_ARALIGI[1] and cikti >= ASGARI_CIKTI
                         and turkiye_icinde(enlem, boylam) and a["sid"] not in mevcut_sidler)
        if a["on_eleme"]:
            uygun.append(a)
    return sorted(uygun, key=lambda a: (-int(a["cikti_sayisi"]), int(a["sid"])))


def ornek_gunler(son_gun: dt.date):
    """Kaydın son iki yılına eşit aralıklı altı gün (ayın 15'i)."""
    gunler = []
    for i in range(6):
        aday = son_gun - dt.timedelta(days=round(i * 730 / 5))
        gunler.append(aday.replace(day=15) if aday.day >= 15 else (aday.replace(day=1) - dt.timedelta(days=1)).replace(day=15))
    return [g.strftime("%Y%m%d") for g in gunler]


def dogrula(istemci, aday):
    kod, metin = istemci.istek("getsystem.jsp", {"sid1": aday["sid"], "tz": 1})
    alanlar = pc.csv_satiri(metin.split(";")[0]) if kod == 200 else []
    aday["saat_dilimi"] = alanlar[-1] if len(alanlar) > 16 else ""
    aday["aralik_dk"] = alanlar[15] if len(alanlar) > 15 else ""
    try:
        enlem, boylam = float(alanlar[13]), float(alanlar[14])
    except (ValueError, IndexError):
        enlem = boylam = float("nan")
    aday["sistem_enlem"], aday["sistem_boylam"] = enlem, boylam
    aday["konum_uygun"] = turkiye_icinde(enlem, boylam) and aday["saat_dilimi"] in SAAT_DILIMLERI
    aday["ornek_dolu"], aday["ilk_gun"], aday["son_gun"], aday["ornek_gunler"] = 0, "", "", ""
    if aday["konum_uygun"]:
        kod, metin = istemci.istek("getstatistic.jsp", {"sid1": aday["sid"]})
        ist = pc.csv_satiri(metin.strip()) if kod == 200 else []
        if len(ist) >= 9:
            aday["ilk_gun"], aday["son_gun"] = ist[7], ist[8]
            gunler = ornek_gunler(dt.datetime.strptime(ist[8], "%Y%m%d").date())
            aday["ornek_gunler"] = ",".join(gunler)
            for gun in gunler:
                kod, metin = istemci.istek("getstatus.jsp", {"d": gun, "h": 1, "limit": 288, "asc": 1, "sid1": aday["sid"]})
                if kod == 200 and ab.uretimli_kayit_sayisi(metin) >= 20:
                    aday["ornek_dolu"] += 1
    aday["uygun"] = bool(aday["konum_uygun"] and aday["ornek_dolu"] >= ASGARI_ORNEK)
    return aday


def main(azami_dogrulama=25):
    args = pc.ayristirici().parse_args(["kimlik"] + sys.argv[1:])
    anahtar, kimlik = pc.kimlik_bilgileri(args.kimlik)
    if not anahtar:
        print("API anahtari bulunamadi.")
        return 2
    istemci = pc.Istemci(anahtar, kimlik, asgari_aralik=0.5)
    mevcut = {s["sid"] for s in pc.santralleri_oku(args.santraller) if s["sid"]}
    adaylar = ara(istemci)
    sirali = on_ele(adaylar, mevcut)
    print(f"Toplam aday: {len(adaylar)} | on elemeyi gecen: {len(sirali)}")
    uygunlar = []
    for aday in sirali[:azami_dogrulama]:
        dogrula(istemci, aday)
        print(f"  {aday['sistem_adi'][:30]:<31} sid={aday['sid']:<7} {float(aday['guc_w']) / 1000:>6.2f} kW  "
              f"cikti={aday['cikti_sayisi']:>5}  {aday['saat_dilimi']:<16} ({aday['sistem_enlem']:.2f}, "
              f"{aday['sistem_boylam']:.2f})  kayit {aday['ilk_gun']}-{aday['son_gun']}  "
              f"ornek={aday['ornek_dolu']}/6  {'UYGUN' if aday['uygun'] else 'elendi'}")
        if aday["uygun"]:
            uygunlar.append(aday)
            if len(uygunlar) >= SECILECEK + YEDEK:
                break
    for sira, aday in enumerate(uygunlar, 1):
        aday["karar"] = "secildi" if sira <= SECILECEK else "yedek"
    sutunlar = pc.ARAMA_ALANLARI + ["arama_merkezi", "on_eleme", "saat_dilimi", "aralik_dk", "sistem_enlem",
                                    "sistem_boylam", "konum_uygun", "ilk_gun", "son_gun", "ornek_gunler",
                                    "ornek_dolu", "uygun", "karar"]
    hedef_dosya = Path(args.ham) / "aday_santraller_tr.csv"
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
