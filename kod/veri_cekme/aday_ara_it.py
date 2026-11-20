"""Ek santral secim olcutu (Ek Not 1): Italya, dislanan Italya sisteminin yerine.

Arama: 20 bolge baskenti + Prepotto, 25 km, son 7 gunde cikti gonderenler.
Kosullar: 10-50 kW; >= 1460 cikti; koordinat Italya sinir kutusunda ve saat dilimi Europe/Rome;
gunluk takvimde 2022-12-01..2025-11-30 gunlerinin ve test donemi gunlerinin en az %90'inda uretim > 0;
alti ornek gunun en az besinde gun ici uretim; yalniz PV kosulu.
Siralama: cikti sayisina gore azalan; ilk uygun secilir, sonraki ikisi yedek. 10-50 kW'ta uygun yoksa
guc kosulu kaldirilir. Sonuc: veri/ham/pvoutput/aday_santraller_it.csv
"""

from __future__ import annotations

import csv
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aday_ara as ab  # noqa: E402
import aday_ara_lu as lu  # noqa: E402
import pvoutput_cek as pc  # noqa: E402

MERKEZLER = {
    "Roma": (41.90, 12.50), "Milano": (45.46, 9.19), "Napoli": (40.85, 14.27), "Torino": (45.07, 7.69),
    "Palermo": (38.12, 13.36), "Genova": (44.41, 8.93), "Bologna": (44.49, 11.34), "Firenze": (43.77, 11.26),
    "Bari": (41.12, 16.87), "Catanzaro": (38.91, 16.59), "Venezia": (45.44, 12.33), "Trieste": (45.65, 13.78),
    "Perugia": (43.11, 12.39), "Cagliari": (39.22, 9.12), "Trento": (46.07, 11.12), "Ancona": (43.62, 13.52),
    "L'Aquila": (42.35, 13.40), "Potenza": (40.64, 15.80), "Campobasso": (41.56, 14.66), "Aosta": (45.74, 7.32),
    "Prepotto": (46.05, 13.48),
}
SINIR = {"enlem": (35.4, 47.1), "boylam": (6.6, 18.6)}
SAAT_DILIMI = {"Europe/Rome"}
DONEM = (dt.date(2022, 12, 1), dt.date(2025, 11, 30))
TEST = (dt.date(2024, 12, 1), dt.date(2025, 11, 30))
ASGARI_GUN_ORANI = 0.90


def ara(istemci, yaricap="25km"):
    adaylar, ulke_kodu = {}, True
    for ad, (enlem, boylam) in MERKEZLER.items():
        parametre = {"q": yaricap, "ll": f"{enlem},{boylam}", "seen": 7}
        if ulke_kodu:
            parametre["country_code"] = "it"
        kod, metin = istemci.istek("search.jsp", parametre)
        if kod == 400 and ulke_kodu:                      # ulke kodu desteklenmiyorsa koordinat + saat dilimi
            istemci.yaz(f"  ulke kodu kabul edilmedi ({metin[:60]}); ulke kodsuz aranıyor")
            ulke_kodu = False
            parametre.pop("country_code")
            kod, metin = istemci.istek("search.jsp", parametre)
        satirlar = [s for s in metin.splitlines() if s.strip()] if kod == 200 else []
        for satir in satirlar:
            alan = dict(zip(pc.ARAMA_ALANLARI, pc.csv_satiri(satir)))
            if alan.get("sid"):
                alan.setdefault("merkez", ad)
                adaylar.setdefault(alan["sid"], alan)
        istemci.yaz(f"  arama {ad:<11}: kod {kod}, {len(satirlar)} sonuc")
    return list(adaylar.values())


def gunluk_kapsam(istemci, sid):
    """Gunluk takvimden (Get Output) uretimi > 0 olan gunlerin orani: (butun donem, test donemi)."""
    uretimli = set()
    imlec = DONEM[1]
    while imlec >= DONEM[0]:
        pencere_basi = max(DONEM[0], imlec - dt.timedelta(days=149))
        kod, metin = istemci.istek("getoutput.jsp", {"df": f"{pencere_basi:%Y%m%d}", "dt": f"{imlec:%Y%m%d}",
                                                     "limit": 150, "sid1": sid})
        if kod == 200 and metin:
            for kayit in metin.split(";"):
                alan = kayit.strip().split(",")
                try:
                    if len(alan) > 1 and float(alan[1]) > 0:
                        uretimli.add(alan[0])
                except ValueError:
                    continue
        imlec = pencere_basi - dt.timedelta(days=1)

    def oran(bas, bit):
        gunler = [(bas + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range((bit - bas).days + 1)]
        return sum(g in uretimli for g in gunler) / len(gunler)

    return oran(*DONEM), oran(*TEST)


def sec(istemci, adaylar, mevcut, guc_kosulu=True, hedef=3):
    ab.GUC_ARALIGI = (10_000.0, 50_000.0) if guc_kosulu else (0.0, float("inf"))
    sirali = ab.on_ele(adaylar, mevcut)
    istemci.yaz(f"On elemeyi gecen ({'10-50 kW' if guc_kosulu else 'guc kosulu yok'}): {len(sirali)}")
    uygunlar = []
    for aday in sirali:
        if "uygun" in aday:                                  # onceki turda denetlendi
            if aday["uygun"]:
                uygunlar.append(aday)
            continue
        lu.sistem_bilgisi(istemci, aday)
        aday["konum_uygun"] = (SINIR["enlem"][0] <= aday["sistem_enlem"] <= SINIR["enlem"][1]
                               and SINIR["boylam"][0] <= aday["sistem_boylam"] <= SINIR["boylam"][1]
                               and aday["saat_dilimi"] in SAAT_DILIMI)
        aday.update({"gun_orani": "", "test_gun_orani": "", "takvim_uygun": "", "ornek_dolu": "", "gece_kayit": "",
                     "gece_guc_orani": "", "gece_enerji_orani": "", "yalniz_pv": ""})
        if aday["konum_uygun"]:
            butun, test = gunluk_kapsam(istemci, aday["sid"])
            aday.update({"gun_orani": round(butun, 4), "test_gun_orani": round(test, 4),
                         "takvim_uygun": butun >= ASGARI_GUN_ORANI and test >= ASGARI_GUN_ORANI})
            if aday["takvim_uygun"]:
                aday["ornek_dolu"], metinler = lu.ornek_gunler(istemci, aday["sid"])
                aday.update(lu.yalniz_pv(metinler))
        aday["uygun"] = bool(aday["konum_uygun"] and aday["takvim_uygun"] is True
                             and aday["ornek_dolu"] >= ab.ASGARI_ORNEK and aday["yalniz_pv"] is True)
        istemci.yaz(f"  {aday['sistem_adi'][:26]:<27} sid={aday['sid']:<7} {float(aday['guc_w']) / 1000:>6.2f} kW "
                    f"cikti={aday['cikti_sayisi']:>5} {aday['saat_dilimi']:<13} ({aday['sistem_enlem']:.2f}, "
                    f"{aday['sistem_boylam']:.2f}) gun {aday['gun_orani']} test {aday['test_gun_orani']} "
                    f"ornek={aday['ornek_dolu']}/6 gece guc>0 {aday['gece_guc_orani']} "
                    f"{'UYGUN' if aday['uygun'] else 'elendi'}")
        if aday["uygun"]:
            uygunlar.append(aday)
        if len(uygunlar) >= hedef:
            break
    return uygunlar[:hedef]


def main():
    args = pc.ayristirici().parse_args(["kimlik"] + sys.argv[1:])
    anahtar, kimlik = pc.kimlik_bilgileri(args.kimlik)
    if not anahtar:
        print("API anahtari bulunamadi.")
        return 2
    istemci = pc.Istemci(anahtar, kimlik, asgari_aralik=0.5)
    mevcut = {s["sid"] for s in pc.santralleri_oku(args.santraller) if s["sid"]}
    adaylar = ara(istemci)
    print(f"Toplam aday: {len(adaylar)}")
    uygunlar = sec(istemci, adaylar, mevcut, guc_kosulu=True)
    guc_kosulu = True
    if not uygunlar:
        guc_kosulu = False
        uygunlar = sec(istemci, adaylar, mevcut, guc_kosulu=False)
    for sira, aday in enumerate(uygunlar, 1):
        aday["karar"] = "secildi" if sira == 1 else "yedek"
    sutunlar = pc.ARAMA_ALANLARI + ["merkez", "on_eleme", "saat_dilimi", "aralik_dk", "sistem_enlem", "sistem_boylam",
                                    "konum_uygun", "gun_orani", "test_gun_orani", "takvim_uygun", "ornek_dolu",
                                    "gece_kayit", "gece_guc_orani", "gece_enerji_orani", "yalniz_pv", "uygun", "karar"]
    with open(Path(args.ham) / "aday_santraller_it.csv", "w", encoding="utf-8-sig", newline="") as f:
        yazici = csv.DictWriter(f, fieldnames=sutunlar, delimiter=";", extrasaction="ignore")
        yazici.writeheader()
        yazici.writerows(sorted(adaylar, key=lambda a: (not a.get("on_eleme", False), -int(a.get("cikti_sayisi") or 0))))
    print(f"Guc kosulu: {'10-50 kW' if guc_kosulu else 'kaldirildi'}")
    for aday in uygunlar:
        print(f"  {aday['karar'].upper():<8} {aday['sistem_adi']} (sid {aday['sid']}, {float(aday['guc_w']) / 1000:.2f} kW, "
              f"{aday['sistem_enlem']:.4f} K {aday['sistem_boylam']:.4f} D, aralik {aday['aralik_dk']} dk)")
    if not uygunlar:
        print("  Italya'da uygun aday bulunamadi.")
    return 0


if __name__ == "__main__":
    for akis in (sys.stdout, sys.stderr):
        if hasattr(akis, "reconfigure"):
            akis.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
