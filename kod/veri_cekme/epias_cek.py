"""EPİAŞ Şeffaflık Platformu'ndan Türkiye santrali seçimi ve verisi
(Ek Not 1, EPİAŞ kuralı).

Komutlar (sırayla):
  liste    UEVM santral listesini çeker; adında GES veya GÜNEŞ geçenleri aday sayar.
  tara     her aday için 01.12.2022 – 30.11.2025 saatlik UEVM'yi çeker ve 1–5. koşulları değerlendirir.
  capraz   1–5'i sağlayanlar için test döneminde UEVM ile TEİAŞ gerçek zamanlı üretimini karşılaştırır (koşul 6).
  cek      --santral ile verilen santralin bütün verisini (01.12.2022'den dünden önceki son güne) çeker.

Sonuçlar veri/ham/epias/ altına yazılır: uevm_santral_listesi.csv, uevm/<id>.csv, gercek_zamanli/<id>.csv,
aday_santraller_epias.csv, cekim_gunlugu.txt.

Kimlik: EPİAŞ Şeffaflık hesabı. Kullanıcı adı ve şifre EPIAS_KULLANICI / EPIAS_SIFRE ortam değişkenlerinden,
yoksa --kimlik ile verilen dosyadan okunur (iki satır: EPIAS_KULLANICI=... ve EPIAS_SIFRE=...).
Şifre ve oturum bileti (TGT) hiçbir dosyaya ya da ekrana yazılmaz.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import requests

KOK = Path(__file__).resolve().parents[2]
HAM = KOK / "veri" / "ham" / "epias"
GIRIS_URL = "https://giris.epias.com.tr/cas/v1/tickets"
SERVIS = "https://seffaflik.epias.com.tr/electricity-service"
UEVM_LISTE = "/v1/generation/data/injection-quantity-powerplant-list"
UEVM = "/v1/generation/data/injection-quantity"
GZ_LISTE = "/v1/generation/data/powerplant-list"
GZ = "/v1/generation/data/realtime-generation"
TR = dt.timezone(dt.timedelta(hours=3))           # Türkiye saati, 2016'dan beri yaz saati yok
DONEM = (dt.date(2022, 12, 1), dt.date(2025, 11, 30))
TEST = (dt.date(2024, 12, 1), dt.date(2025, 11, 30))
YILLAR = [(dt.date(2022, 12, 1), dt.date(2023, 11, 30)), (dt.date(2023, 12, 1), dt.date(2024, 11, 30)),
          (dt.date(2024, 12, 1), dt.date(2025, 11, 30))]
GECE_SAATLERI = {22, 23, 0, 1, 2, 3}             # saat başı etiketi; 22:00–04:00
KAYNAK_ALANLARI = ["asphaltite", "biomass", "dam", "fueloil", "geothermal", "importedCoal", "internationalExport",
                   "internationalImport", "lignite", "lng", "naphtha", "naturalGas", "other", "river", "stoneCoal",
                   "sun", "wind"]
ESIK = {"kapsam": 0.99, "yillik_sapma": 0.10, "gece_oran": 0.995, "gece_pay": 0.01, "dusuk_gun_pay": 0.05,
        "dusuk_gun_oran": 0.02, "korelasyon": 0.98}
ISTEMCI_ADI = "pv-benchmark-fetch/1.0"


# ---------------------------------------------------------------- kimlik ve istemci

def kimlik_bilgileri(dosya=None, ortam=None):
    """(kullanici, sifre): önce ortam değişkenleri, yoksa KEY=VALUE satırlı dosya."""
    ortam = os.environ if ortam is None else ortam
    degerler = {ad: (ortam.get(ad) or "").strip() for ad in ("EPIAS_KULLANICI", "EPIAS_SIFRE")}
    if not all(degerler.values()) and dosya and Path(dosya).exists():
        for satir in Path(dosya).read_text(encoding="utf-8-sig").splitlines():
            if "=" in satir and not satir.lstrip().startswith("#"):
                ad, deger = satir.split("=", 1)
                ad = ad.strip().upper()
                if ad in degerler and not degerler[ad]:
                    degerler[ad] = deger.strip()
    return degerler["EPIAS_KULLANICI"], degerler["EPIAS_SIFRE"]


class Istemci:
    """EPİAŞ istemcisi: TGT alır (2 saat geçerli; 100 dakikada bir ya da 401'de yenilenir), istekler arasında bekler."""

    def __init__(self, kullanici, sifre, oturum=None, uyku=time.sleep, simdi=time.time, asgari_aralik=0.5,
                 yaz=print, yedek=3):
        self._kullanici, self._sifre = kullanici, sifre
        self.oturum = oturum or requests.Session()
        self.uyku, self.simdi, self.yaz = uyku, simdi, yaz
        self.asgari_aralik, self.yedek = asgari_aralik, yedek
        self._tgt, self._tgt_zamani, self._son = None, 0.0, 0.0

    def _bilet(self, zorla=False):
        if zorla or not self._tgt or self.simdi() - self._tgt_zamani > 100 * 60:
            yanit = self.oturum.post(GIRIS_URL, data={"username": self._kullanici, "password": self._sifre},
                                     headers={"Accept": "text/plain", "User-Agent": ISTEMCI_ADI}, timeout=60)
            if yanit.status_code not in (200, 201) or not yanit.text.strip().startswith("TGT-"):
                raise RuntimeError(f"EPİAŞ girişi başarısız (HTTP {yanit.status_code}). Kullanıcı adı/şifreyi "
                                   "ve hesabın Şeffaflık Platformu'nda etkin olduğunu kontrol edin.")
            self._tgt, self._tgt_zamani = yanit.text.strip(), self.simdi()
        return self._tgt

    def istek(self, yol, govde=None):
        """(http_kodu, json_veya_None). GET (gövde yoksa) ya da POST."""
        kod = None
        for deneme in range(self.yedek + 1):
            bekle = self.asgari_aralik - (self.simdi() - self._son)
            if bekle > 0:
                self.uyku(bekle)
            basliklar = {"TGT": self._bilet(zorla=kod == 401), "Accept": "application/json",
                         "User-Agent": ISTEMCI_ADI}
            try:
                if govde is None:
                    yanit = self.oturum.get(SERVIS + yol, headers=basliklar, timeout=120)
                else:
                    yanit = self.oturum.post(SERVIS + yol, json=govde, headers=basliklar, timeout=120)
                kod = yanit.status_code
            except requests.RequestException as hata:
                kod, yanit = None, None
                self.yaz(f"  ag hatasi: {type(hata).__name__}")
            self._son = self.simdi()
            if kod == 200:
                try:
                    return 200, yanit.json()
                except ValueError:
                    return 200, None
            if kod in (400, 404):
                return kod, None
            if kod != 401:
                self.uyku(min(300, 10 * 2 ** deneme))
        return kod, None


# ---------------------------------------------------------------- yardımcılar

def tarih_metni(gun: dt.date):
    """EPİAŞ tarih biçimi; servisler bitiş gününü bütün olarak döndürür (ilk çekimde denetlenir)."""
    return dt.datetime.combine(gun, dt.time(0, 0), TR).isoformat()


def aylar(bas: dt.date, bit: dt.date):
    """[bas, bit] aralığını takvim aylarına böler."""
    parcalar, gun = [], bas
    while gun <= bit:
        sonraki = (gun.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        parcalar.append((gun, min(bit, sonraki - dt.timedelta(days=1))))
        gun = sonraki
    return parcalar


def gunes_santrali_mi(ad: str) -> bool:
    ad = (ad or "").upper().replace("İ", "I")
    return bool(re.search(r"(?<![A-ZÇĞÖŞÜ])GES(?![A-ZÇĞÖŞÜ])", ad) or "GÜNEŞ" in ad or "GUNES" in ad)


def kayit_zamani(kayit):
    """EPİAŞ kaydının saat başı (Türkiye saati) zamanı."""
    zaman = dt.datetime.fromisoformat(kayit["date"])
    if zaman.tzinfo is None:
        zaman = zaman.replace(tzinfo=TR)
    saat = kayit.get("hour")
    if isinstance(saat, str) and ":" in saat:
        saat = int(saat.split(":")[0])
    if saat is not None and zaman.hour == 0 and int(saat) != 0:
        zaman = zaman.replace(hour=int(saat))
    return zaman.astimezone(TR)


def sayi(deger):
    try:
        return float(deger)
    except (TypeError, ValueError):
        return 0.0


def sayfali_cek(istemci, yol, govde, alan="items"):
    """Sayfalı yanıtın bütün öğeleri; hata kodunda None."""
    kod, veri = istemci.istek(yol, govde)
    if kod != 200 or veri is None:
        return None
    ogeler = list(veri.get(alan) or [])
    sayfa = veri.get("page") or {}
    toplam, boyut = sayfa.get("total"), len(ogeler)
    numara = 1
    while toplam and boyut and len(ogeler) < toplam:
        numara += 1
        kod, veri = istemci.istek(yol, dict(govde, page={"number": numara, "size": boyut}))
        yeni = (veri or {}).get(alan) or []
        if kod != 200 or not yeni:
            break
        ogeler.extend(yeni)
    return ogeler


# ---------------------------------------------------------------- veri çekme

def uevm_cek(istemci, santral_id, bas, bit, hedef: Path):
    """Aylık parçalarla saatlik UEVM; hedef CSV'ye (zaman_tr, gunes, toplam, diger) yazar. Var olan aylar atlanır."""
    mevcut = {}
    if hedef.exists():
        with open(hedef, encoding="utf-8") as f:
            for satir in csv.DictReader(f):
                mevcut[satir["zaman_tr"]] = satir
    eksik_aylar = 0
    for a_bas, a_bit in aylar(bas, bit):
        ay = f"{a_bas:%Y-%m}"
        if sum(1 for z in mevcut if z.startswith(ay)) >= 24 * ((a_bit - a_bas).days + 1):
            continue                                            # ay tam: yeniden istenmez
        ogeler = sayfali_cek(istemci, UEVM, {"startDate": tarih_metni(a_bas), "endDate": tarih_metni(a_bit),
                                             "powerplantId": int(santral_id)})
        if ogeler is None:
            eksik_aylar += 1
            continue
        for k in ogeler:
            zaman = kayit_zamani(k).isoformat()
            diger = sum(abs(sayi(k.get(a))) for a in KAYNAK_ALANLARI if a != "sun")
            mevcut[zaman] = {"zaman_tr": zaman, "gunes": sayi(k.get("sun")), "toplam": sayi(k.get("total")),
                             "diger": diger}
    hedef.parent.mkdir(parents=True, exist_ok=True)
    with open(hedef, "w", encoding="utf-8", newline="") as f:
        yazici = csv.DictWriter(f, fieldnames=["zaman_tr", "gunes", "toplam", "diger"])
        yazici.writeheader()
        yazici.writerows(mevcut[z] for z in sorted(mevcut))
    return eksik_aylar


def gercek_zamanli_cek(istemci, santral_id, bas, bit, hedef: Path):
    kayitlar = {}
    for a_bas, a_bit in aylar(bas, bit):
        ogeler = sayfali_cek(istemci, GZ, {"startDate": tarih_metni(a_bas), "endDate": tarih_metni(a_bit),
                                           "powerPlantId": int(santral_id)})
        for k in ogeler or []:
            kayitlar[kayit_zamani(k).isoformat()] = sayi(k.get("sun"))
    hedef.parent.mkdir(parents=True, exist_ok=True)
    with open(hedef, "w", encoding="utf-8", newline="") as f:
        yazici = csv.writer(f)
        yazici.writerow(["zaman_tr", "gunes"])
        yazici.writerows((z, kayitlar[z]) for z in sorted(kayitlar))
    return kayitlar


def seri_oku(yol: Path, alan="gunes"):
    seri = {}
    with open(yol, encoding="utf-8") as f:
        for satir in csv.DictReader(f):
            seri[dt.datetime.fromisoformat(satir["zaman_tr"])] = {k: float(v) for k, v in satir.items() if k != "zaman_tr"}
    return seri


# ---------------------------------------------------------------- koşullar

def yuzdelik(degerler, p):
    if not degerler:
        return 0.0
    sirali = sorted(degerler)
    konum = (len(sirali) - 1) * p
    alt = int(konum)
    ust = min(alt + 1, len(sirali) - 1)
    return sirali[alt] + (sirali[ust] - sirali[alt]) * (konum - alt)


def kosullari_degerlendir(seri, donem=DONEM, yillar=YILLAR):
    """seri: {saat_basi_zamani_tr: {"gunes":..., "diger":...}} -> koşul sonuçları sözlüğü (MWh/saat birimi)."""
    bas = dt.datetime.combine(donem[0], dt.time(), TR)
    bit = dt.datetime.combine(donem[1] + dt.timedelta(days=1), dt.time(), TR)
    icerde = {z: d for z, d in seri.items() if bas <= z < bit}
    beklenen = int((bit - bas).total_seconds() // 3600)
    gunes = [d["gunes"] for d in icerde.values()]
    p99 = yuzdelik(gunes, 0.99)
    sonuc = {"kayit_saat": len(icerde), "beklenen_saat": beklenen,
             "kapsam": len(icerde) / beklenen if beklenen else 0.0,
             "diger_toplam": sum(d.get("diger", 0.0) for d in icerde.values()), "p99_mwh": p99}
    sonuc["k1_yalniz_gunes"] = sonuc["diger_toplam"] == 0.0 and p99 > 0
    sonuc["k2_kapsam"] = sonuc["kapsam"] >= ESIK["kapsam"]

    yillik = []
    for y_bas, y_bit in yillar:
        yb = dt.datetime.combine(y_bas, dt.time(), TR)
        ys = dt.datetime.combine(y_bit + dt.timedelta(days=1), dt.time(), TR)
        yillik.append(yuzdelik([d["gunes"] for z, d in icerde.items() if yb <= z < ys], 0.99))
    ortanca = statistics.median(yillik) if yillik else 0.0
    sapma = max(abs(y - ortanca) / ortanca for y in yillik) if ortanca > 0 else float("inf")
    sonuc["yillik_p99"] = ";".join(f"{y:.3f}" for y in yillik)
    sonuc["yillik_sapma"] = sapma
    sonuc["k3_sabit_guc"] = sapma <= ESIK["yillik_sapma"]

    gece = [d["gunes"] for z, d in icerde.items() if z.hour in GECE_SAATLERI]
    sifir = sum(1 for g in gece if g < ESIK["gece_pay"] * p99)
    sonuc["gece_sifir_orani"] = sifir / len(gece) if gece else 0.0
    sonuc["k4_gece_sifir"] = bool(gece) and sonuc["gece_sifir_orani"] >= ESIK["gece_oran"]

    gunluk = {}
    for z, d in icerde.items():
        gunluk[z.date()] = gunluk.get(z.date(), 0.0) + d["gunes"]
    aylik = {}
    for g, e in gunluk.items():
        aylik.setdefault((g.year, g.month), []).append(e)
    ortancalar = {a: statistics.median(v) for a, v in aylik.items()}
    dusuk = sum(1 for g, e in gunluk.items() if e < ESIK["dusuk_gun_pay"] * ortancalar[(g.year, g.month)])
    gun_sayisi = (donem[1] - donem[0]).days + 1
    eksik_gun = gun_sayisi - len(gunluk)
    sonuc["dusuk_gun_orani"] = (dusuk + eksik_gun) / gun_sayisi
    sonuc["k5_kesinti_az"] = sonuc["dusuk_gun_orani"] <= ESIK["dusuk_gun_oran"]
    sonuc["k1_5"] = all(sonuc[k] for k in ("k1_yalniz_gunes", "k2_kapsam", "k3_sabit_guc", "k4_gece_sifir",
                                           "k5_kesinti_az"))
    return sonuc


def korelasyon(a, b):
    ortak = [z for z in a if z in b]
    if len(ortak) < 100:
        return float("nan"), len(ortak)
    x, y = [a[z] for z in ortak], [b[z] for z in ortak]
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxy = sum((i - mx) * (j - my) for i, j in zip(x, y))
    sx = sum((i - mx) ** 2 for i in x) ** 0.5
    sy = sum((j - my) ** 2 for j in y) ** 0.5
    return (sxy / (sx * sy) if sx and sy else float("nan")), len(ortak)


# ---------------------------------------------------------------- komutlar

ADAY_SUTUNLARI = ["id", "ad", "kisa_ad", "eic", "kayit_saat", "beklenen_saat", "kapsam", "diger_toplam", "p99_mwh",
                  "yillik_p99", "yillik_sapma", "gece_sifir_orani", "dusuk_gun_orani", "k1_yalniz_gunes", "k2_kapsam",
                  "k3_sabit_guc", "k4_gece_sifir", "k5_kesinti_az", "k1_5", "gz_id", "korelasyon", "ortak_saat",
                  "k6_capraz", "sira", "karar", "not"]


def adaylari_oku(ham: Path):
    yol = ham / "aday_santraller_epias.csv"
    if yol.exists():
        with open(yol, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f, delimiter=";"))
    with open(ham / "uevm_santral_listesi.csv", encoding="utf-8-sig") as f:
        return [s for s in csv.DictReader(f, delimiter=";") if s["gunes"] == "True"]


def adaylari_yaz(ham: Path, adaylar):
    with open(ham / "aday_santraller_epias.csv", "w", encoding="utf-8-sig", newline="") as f:
        yazici = csv.DictWriter(f, fieldnames=ADAY_SUTUNLARI, delimiter=";", extrasaction="ignore")
        yazici.writeheader()
        yazici.writerows(adaylar)


def komut_liste(istemci, args):
    kod, veri = istemci.istek(UEVM_LISTE)
    if kod != 200 or not veri:
        istemci.yaz(f"UEVM santral listesi alınamadı (HTTP {kod}).")
        return 1
    ogeler = veri.get("items") or []
    args.ham.mkdir(parents=True, exist_ok=True)
    with open(args.ham / "uevm_santral_listesi.csv", "w", encoding="utf-8-sig", newline="") as f:
        yazici = csv.writer(f, delimiter=";")
        yazici.writerow(["id", "ad", "kisa_ad", "eic", "gunes"])
        for o in ogeler:
            yazici.writerow([o.get("id"), o.get("name"), o.get("shortName"), o.get("eic"),
                             gunes_santrali_mi(f"{o.get('name')} {o.get('shortName')}")])
    adet = sum(gunes_santrali_mi(f"{o.get('name')} {o.get('shortName')}") for o in ogeler)
    istemci.yaz(f"UEVM listesinde {len(ogeler)} santral; adında GES/GÜNEŞ geçen aday: {adet}")
    return 0


def komut_tara(istemci, args):
    adaylar = adaylari_oku(args.ham)
    for i, a in enumerate(adaylar, 1):
        if a.get("k1_5") in ("True", "False") and not args.yenile:
            continue
        yol = args.ham / "uevm" / f"{a['id']}.csv"
        eksik = uevm_cek(istemci, a["id"], DONEM[0], DONEM[1], yol)
        sonuc = kosullari_degerlendir(seri_oku(yol))
        a.update({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in sonuc.items()})
        a["not"] = f"{eksik} ay alinamadi" if eksik else ""
        istemci.yaz(f"  [{i}/{len(adaylar)}] {a['ad'][:40]:<41} kapsam={sonuc['kapsam']:.3f} p99={sonuc['p99_mwh']:.2f} "
                    f"MWh  k1-5={'UYGUN' if sonuc['k1_5'] else 'elendi'}")
        adaylari_yaz(args.ham, adaylar)        # her adaydan sonra kaydet: kesilirse kaldığı yerden sürer
    adaylari_yaz(args.ham, adaylar)
    return 0


def komut_capraz(istemci, args):
    adaylar = adaylari_oku(args.ham)
    kod, veri = istemci.istek(GZ_LISTE)
    gz_eic = {str(o.get("eic")): o.get("id") for o in (veri or {}).get("items") or [] if o.get("eic")}
    for a in adaylar:
        if a.get("k1_5") != "True":
            continue
        a["gz_id"] = gz_eic.get(a.get("eic") or "", "")
        if not a["gz_id"]:
            a["korelasyon"], a["ortak_saat"], a["k6_capraz"] = "", "0", "uygulanmadi"
            continue
        gz = gercek_zamanli_cek(istemci, a["gz_id"], TEST[0], TEST[1], args.ham / "gercek_zamanli" / f"{a['id']}.csv")
        uevm = {z.isoformat(): d["gunes"] for z, d in seri_oku(args.ham / "uevm" / f"{a['id']}.csv").items()}
        r, n = korelasyon({z: v for z, v in uevm.items() if z in gz}, gz)
        a["korelasyon"], a["ortak_saat"] = f"{r:.4f}", str(n)
        a["k6_capraz"] = str(r == r and r >= ESIK["korelasyon"])
        istemci.yaz(f"  {a['ad'][:40]:<41} r={r:.4f} (n={n})")
    gecenler = [a for a in adaylar if a.get("k1_5") == "True" and a.get("k6_capraz") in ("True", "uygulanmadi")]
    gecenler.sort(key=lambda a: (float(a["p99_mwh"]), int(a["id"])))
    for a in adaylar:
        a["sira"], a["karar"] = "", ""
    for sira, a in enumerate(gecenler, 1):
        a["sira"] = str(sira)
        a["karar"] = "konum denetimi bekliyor" if sira <= 3 else ""
    adaylari_yaz(args.ham, adaylar)
    istemci.yaz(f"1-6. kosullari saglayan: {len(gecenler)}")
    for a in gecenler[:5]:
        istemci.yaz(f"  {a['sira']}. {a['ad']} (id {a['id']}, p99 {float(a['p99_mwh']):.2f} MWh/saat)")
    return 0


def komut_cek(istemci, args):
    if not args.santral:
        istemci.yaz("--santral gerekli")
        return 2
    bit = dt.date.today() - dt.timedelta(days=2)
    eksik = uevm_cek(istemci, args.santral, DONEM[0], bit, args.ham / "uevm" / f"{args.santral}.csv")
    istemci.yaz(f"  {args.santral}: {DONEM[0]} .. {bit}, alinamayan ay: {eksik}")
    return 0


def ayristirici():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("komut", choices=["liste", "tara", "capraz", "cek"])
    p.add_argument("--kimlik", default=os.environ.get("EPIAS_KIMLIK_DOSYASI", ""), help="kimlik dosyasının yolu")
    p.add_argument("--ham", type=Path, default=HAM)
    p.add_argument("--santral", help="cek: UEVM santral kimliği")
    p.add_argument("--yenile", action="store_true", help="tara: değerlendirilmiş adayları yeniden değerlendir")
    return p


def main(argv=None):
    args = ayristirici().parse_args(argv)
    kullanici, sifre = kimlik_bilgileri(args.kimlik)
    if not (kullanici and sifre):
        print("EPİAŞ kullanıcı adı/şifre bulunamadı (EPIAS_KULLANICI / EPIAS_SIFRE ya da --kimlik dosyası).")
        return 2
    args.ham.mkdir(parents=True, exist_ok=True)
    gunluk = open(args.ham / "cekim_gunlugu.txt", "a", encoding="utf-8")

    def yaz(metin):
        satir = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {metin}"
        print(satir, flush=True)
        gunluk.write(satir + "\n")
        gunluk.flush()

    istemci = Istemci(kullanici, sifre, yaz=yaz, asgari_aralik=1.0)
    try:
        return {"liste": komut_liste, "tara": komut_tara, "capraz": komut_capraz, "cek": komut_cek}[args.komut](istemci, args)
    finally:
        gunluk.close()


if __name__ == "__main__":
    for akis in (sys.stdout, sys.stderr):
        if hasattr(akis, "reconfigure"):
            akis.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
