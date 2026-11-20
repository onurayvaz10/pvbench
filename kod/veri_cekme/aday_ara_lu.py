"""Ek santral secim olcutu (Ek Not 1): yalniz PV sinyali ve degistirme.

1. Hollanda: olcut 3'un yedekleri sirayla yalniz PV kosuluyla denenir; ilk saglayan alinir.
2. Luksemburg: olcut 1'in kosullari Luksemburg icin + yalniz PV kosulu; cikti sayisina gore azalan,
   ilk uygun secilir, sonraki ikisi yedek. 10-50 kW'ta uygun yoksa guc kosulu kaldirilir.

Yalniz PV kosulu: ornek gunlerdeki yerel 00:00-04:00 kayitlarinin en fazla %5'inde guc > 0 ve bu saatlerde
enerji sayaci gunun son degerinin %1'inden fazla artmiyor (gece kaydi yoksa saglanmis sayilir).
Sonuclar: veri/ham/pvoutput/aday_santraller_lu.csv ve yedek_denetimi_nl.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aday_ara as ab  # noqa: E402
import pvoutput_cek as pc  # noqa: E402

LU_MERKEZLER = {"Luxembourg": (49.61, 6.13), "Esch-sur-Alzette": (49.50, 5.98), "Ettelbruck": (49.85, 6.10),
                "Clervaux": (50.05, 6.03), "Grevenmacher": (49.68, 6.44)}
LU_SINIR = {"enlem": (49.44, 50.19), "boylam": (5.73, 6.53)}
LU_SAAT_DILIMI = {"Europe/Luxembourg"}
NL_YEDEKLER = ["76871", "87811", "94064"]     # olcut 3'un yedekleri, sirayla (PVOutput sistem kimlikleri)
GECE_GUC_ORANI, GECE_ENERJI_ORANI = 0.05, 0.01


def gece_olcumu(metin):
    """Bir gunun getstatus yanitindan (gece kayit sayisi, guc>0 olan gece kaydi, gece enerji artisi, gun sonu enerji)."""
    n = pozitif = 0
    gece_enerji, tum_enerji = [], []
    for kayit in metin.split(";"):
        alan = kayit.strip().split(",")
        if len(alan) < 5:
            continue
        try:
            e = float(alan[2])
            if e == e:
                tum_enerji.append(e)
        except ValueError:
            e = None
        if alan[1] < "04:00":
            try:
                g = float(alan[4])
            except ValueError:
                g = float("nan")
            if g == g:
                n += 1
                pozitif += g > 0
            if e is not None and e == e:
                gece_enerji.append(e)
    artis = (max(gece_enerji) - min(gece_enerji)) if len(gece_enerji) > 1 else 0.0
    return n, pozitif, artis, (max(tum_enerji) if tum_enerji else 0.0)


def yalniz_pv(metinler):
    n = pozitif = 0
    enerji_orani = 0.0
    for metin in metinler:
        a, b, artis, son = gece_olcumu(metin)
        n, pozitif = n + a, pozitif + b
        if son > 0:
            enerji_orani = max(enerji_orani, artis / son)
    guc_orani = pozitif / n if n else 0.0
    return {"gece_kayit": n, "gece_guc_orani": round(guc_orani, 4), "gece_enerji_orani": round(enerji_orani, 4),
            "yalniz_pv": guc_orani <= GECE_GUC_ORANI and enerji_orani <= GECE_ENERJI_ORANI}


def ornek_gunler(istemci, sid):
    metinler, dolu = [], 0
    for gun in ab.ORNEK_GUNLER:
        kod, metin = istemci.istek("getstatus.jsp", {"d": gun, "h": 1, "limit": 288, "asc": 1, "sid1": sid})
        if kod == 200:
            metinler.append(metin)
            dolu += ab.uretimli_kayit_sayisi(metin) >= 20
    return dolu, metinler


def sistem_bilgisi(istemci, aday):
    kod, metin = istemci.istek("getsystem.jsp", {"sid1": aday["sid"], "tz": 1})
    alanlar = pc.csv_satiri(metin.split(";")[0]) if kod == 200 else []
    aday["saat_dilimi"] = alanlar[-1] if len(alanlar) > 16 else ""
    aday["aralik_dk"] = alanlar[15] if len(alanlar) > 15 else ""
    try:
        aday["sistem_enlem"], aday["sistem_boylam"] = float(alanlar[13]), float(alanlar[14])
    except (ValueError, IndexError):
        aday["sistem_enlem"] = aday["sistem_boylam"] = float("nan")


def nl_yedekleri(istemci, ham):
    adlar = {}                                    # sistem adlari, varsa Hollanda aday listesinden
    liste = Path(ham) / "aday_santraller_nl.csv"
    if liste.exists():
        with open(liste, encoding="utf-8-sig", newline="") as f:
            adlar = {r["sid"]: r.get("sistem_adi", "") for r in csv.DictReader(f, delimiter=";")}
    sonuc = []
    for sid in NL_YEDEKLER:
        ad = adlar.get(sid, "")
        dolu, metinler = ornek_gunler(istemci, sid)
        g = yalniz_pv(metinler)
        satir = {"ad": ad, "sid": sid, "ornek_dolu": dolu, **g,
                 "karar": "secildi" if g["yalniz_pv"] and dolu >= ab.ASGARI_ORNEK and not any(
                     s["karar"] == "secildi" for s in sonuc) else ""}
        sonuc.append(satir)
        istemci.yaz(f"  NL yedek {ad:<18} sid={sid:<6} ornek={dolu}/6 gece kayit={g['gece_kayit']:>4} "
                    f"guc>0 {g['gece_guc_orani']:.3f} enerji artisi {g['gece_enerji_orani']:.4f}  "
                    f"{'YALNIZ PV' if g['yalniz_pv'] else 'gece uretimi var'}  {satir['karar']}")
        if satir["karar"] == "secildi":
            break
    with open(Path(ham) / "yedek_denetimi_nl.csv", "w", encoding="utf-8-sig", newline="") as f:
        yazici = csv.DictWriter(f, fieldnames=list(sonuc[0].keys()), delimiter=";")
        yazici.writeheader()
        yazici.writerows(sonuc)
    return sonuc


def lu_ara(istemci, yaricap="25km"):
    adaylar = {}
    for bolge, (enlem, boylam) in LU_MERKEZLER.items():
        # API 'lu' ulke kodunu desteklemiyor (HTTP 400 "Unsupported country_code [lu]"); ulke kosulu
        # dogrulamada koordinat siniri ve saat dilimiyle saglanir.
        kod, metin = istemci.istek("search.jsp", {"q": yaricap, "ll": f"{enlem},{boylam}", "seen": 7})
        satirlar = [s for s in metin.splitlines() if s.strip()] if kod == 200 else []
        for satir in satirlar:
            alan = dict(zip(pc.ARAMA_ALANLARI, pc.csv_satiri(satir)))
            if alan.get("sid"):
                adaylar.setdefault(alan["sid"], alan)
        istemci.yaz(f"  LU arama {bolge}: kod {kod}, {len(satirlar)} sonuc")
    return list(adaylar.values())


def lu_sec(istemci, adaylar, mevcut, guc_kosulu=True, hedef=3, azami=25):
    ab.GUC_ARALIGI = (10_000.0, 50_000.0) if guc_kosulu else (0.0, float("inf"))
    sirali = ab.on_ele(adaylar, mevcut)
    istemci.yaz(f"LU on elemeyi gecen ({'10-50 kW' if guc_kosulu else 'guc kosulu yok'}): {len(sirali)}")
    uygunlar = []
    for aday in sirali[:azami]:
        sistem_bilgisi(istemci, aday)
        aday["konum_uygun"] = (LU_SINIR["enlem"][0] <= aday["sistem_enlem"] <= LU_SINIR["enlem"][1]
                               and LU_SINIR["boylam"][0] <= aday["sistem_boylam"] <= LU_SINIR["boylam"][1]
                               and aday["saat_dilimi"] in LU_SAAT_DILIMI)
        aday["ornek_dolu"], g = 0, {"gece_kayit": "", "gece_guc_orani": "", "gece_enerji_orani": "", "yalniz_pv": ""}
        if aday["konum_uygun"]:
            aday["ornek_dolu"], metinler = ornek_gunler(istemci, aday["sid"])
            g = yalniz_pv(metinler)
        aday.update(g)
        aday["uygun"] = bool(aday["konum_uygun"] and aday["ornek_dolu"] >= ab.ASGARI_ORNEK and g["yalniz_pv"] is True)
        istemci.yaz(f"  {aday['sistem_adi'][:28]:<29} sid={aday['sid']:<7} {float(aday['guc_w']) / 1000:>7.2f} kW "
                    f"cikti={aday['cikti_sayisi']:>5} {aday['saat_dilimi']:<17} ({aday['sistem_enlem']:.2f}, "
                    f"{aday['sistem_boylam']:.2f}) ornek={aday['ornek_dolu']}/6 gece guc>0 {g['gece_guc_orani']} "
                    f"{'UYGUN' if aday['uygun'] else 'elendi'}")
        if aday["uygun"]:
            uygunlar.append(aday)
            if len(uygunlar) >= hedef:
                break
    return uygunlar


TEST_ORNEK_GUNLER = ["20241215", "20250315", "20250615", "20250915", "20251115"]


def lu_ara_genis(istemci, yaricap="25km"):
    """Gevsetilmis kosullarla arama (Ek Not 1): son cikti suzgeci olmadan."""
    adaylar = {}
    for bolge, (enlem, boylam) in LU_MERKEZLER.items():
        kod, metin = istemci.istek("search.jsp", {"q": yaricap, "ll": f"{enlem},{boylam}"})
        satirlar = [s for s in metin.splitlines() if s.strip()] if kod == 200 else []
        for satir in satirlar:
            alan = dict(zip(pc.ARAMA_ALANLARI, pc.csv_satiri(satir)))
            if alan.get("sid"):
                adaylar.setdefault(alan["sid"], alan)
        istemci.yaz(f"  LU genis arama {bolge}: kod {kod}, {len(satirlar)} sonuc")
    return list(adaylar.values())


def lu_sec_genis(istemci, adaylar, mevcut, guc_kosulu=True, hedef=3):
    alt, ust = (10_000.0, 50_000.0) if guc_kosulu else (0.0, float("inf"))
    on = []
    for a in adaylar:
        try:
            guc, cikti = float(a["guc_w"]), int(a["cikti_sayisi"])
        except ValueError:
            continue
        a["on_eleme"] = (alt <= guc <= ust and cikti >= 365 and ab.son_cikti_gun(a["son_cikti"]) <= 350
                         and a["sid"] not in mevcut)
        if a["on_eleme"]:
            on.append(a)
    on.sort(key=lambda a: (-int(a["cikti_sayisi"]), int(a["sid"])))
    istemci.yaz(f"LU genis on eleme ({'10-50 kW' if guc_kosulu else 'guc kosulu yok'}): {len(on)}")
    uygunlar = []
    for aday in on:
        sistem_bilgisi(istemci, aday)
        aday["konum_uygun"] = (LU_SINIR["enlem"][0] <= aday["sistem_enlem"] <= LU_SINIR["enlem"][1]
                               and LU_SINIR["boylam"][0] <= aday["sistem_boylam"] <= LU_SINIR["boylam"][1]
                               and aday["saat_dilimi"] in LU_SAAT_DILIMI)
        aday["ornek_dolu"], g = 0, {"gece_kayit": "", "gece_guc_orani": "", "gece_enerji_orani": "", "yalniz_pv": ""}
        if aday["konum_uygun"]:
            metinler = []
            for gun in TEST_ORNEK_GUNLER:
                kod, metin = istemci.istek("getstatus.jsp", {"d": gun, "h": 1, "limit": 288, "asc": 1, "sid1": aday["sid"]})
                if kod == 200:
                    metinler.append(metin)
                    aday["ornek_dolu"] += ab.uretimli_kayit_sayisi(metin) >= 20
            g = yalniz_pv(metinler)
        aday.update(g)
        aday["uygun"] = bool(aday["konum_uygun"] and aday["ornek_dolu"] >= 4 and g["yalniz_pv"] is True)
        istemci.yaz(f"  {aday['sistem_adi'][:28]:<29} sid={aday['sid']:<7} {float(aday['guc_w']) / 1000:>7.2f} kW "
                    f"cikti={aday['cikti_sayisi']:>5} son={aday['son_cikti']:<12} {aday['saat_dilimi']:<17} "
                    f"({aday['sistem_enlem']:.2f}, {aday['sistem_boylam']:.2f}) ornek={aday['ornek_dolu']}/5 "
                    f"gece guc>0 {g['gece_guc_orani']} {'UYGUN' if aday['uygun'] else 'elendi'}")
        if aday["uygun"]:
            uygunlar.append(aday)
            if len(uygunlar) >= hedef:
                break
    return uygunlar


def main():
    args = pc.ayristirici().parse_args(["kimlik"] + [a for a in sys.argv[1:] if a != "--nl-atla"])
    anahtar, kimlik = pc.kimlik_bilgileri(args.kimlik)
    if not anahtar:
        print("API anahtari bulunamadi.")
        return 2
    istemci = pc.Istemci(anahtar, kimlik, asgari_aralik=0.5)
    if "--nl-atla" not in sys.argv:
        nl_yedekleri(istemci, args.ham)
    mevcut = {s["sid"] for s in pc.santralleri_oku(args.santraller) if s["sid"]}
    adaylar = lu_ara(istemci)
    uygunlar = lu_sec(istemci, adaylar, mevcut, guc_kosulu=True)
    if not uygunlar:
        uygunlar = lu_sec(istemci, adaylar, mevcut, guc_kosulu=False)
    if not uygunlar:                                    # gevsetilmis kosullar (Ek Not 1): >= 365 gun, ornek gunlerin 4/5'i
        adaylar = lu_ara_genis(istemci)
        uygunlar = lu_sec_genis(istemci, adaylar, mevcut, guc_kosulu=True)
        if not uygunlar:
            uygunlar = lu_sec_genis(istemci, adaylar, mevcut, guc_kosulu=False)
    for sira, aday in enumerate(uygunlar, 1):
        aday["karar"] = "secildi" if sira == 1 else "yedek"
    sutunlar = pc.ARAMA_ALANLARI + ["on_eleme", "saat_dilimi", "aralik_dk", "sistem_enlem", "sistem_boylam", "konum_uygun",
                                    "ornek_dolu", "gece_kayit", "gece_guc_orani", "gece_enerji_orani", "yalniz_pv",
                                    "uygun", "karar"]
    with open(Path(args.ham) / "aday_santraller_lu.csv", "w", encoding="utf-8-sig", newline="") as f:
        yazici = csv.DictWriter(f, fieldnames=sutunlar, delimiter=";", extrasaction="ignore")
        yazici.writeheader()
        yazici.writerows(sorted(adaylar, key=lambda a: (not a.get("on_eleme", False), -int(a.get("cikti_sayisi") or 0))))
    for aday in uygunlar:
        print(f"  {aday['karar'].upper():<8} {aday['sistem_adi']} (sid {aday['sid']}, {float(aday['guc_w']) / 1000:.2f} kW)")
    if not uygunlar:
        print("  Luksemburg'da uygun aday bulunamadi.")
    return 0


if __name__ == "__main__":
    for akis in (sys.stdout, sys.stderr):
        if hasattr(akis, "reconfigure"):
            akis.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
