"""PVOutput ham veri cekme araci.

Kimlik bilgileri proje klasorune yazilmaz. Ortam degiskenlerinden ya da ev
klasorundeki pvoutput_kimlik.txt dosyasindan okunur:
    PVOUTPUT_API_KEY     API anahtari
    PVOUTPUT_SYSTEM_ID   Bagis modundaki kendi sistem kimliginiz

Alt komutlar (sirayla calistirilir):
    kimlik   santraller.csv'de sid'i bos olan santralleri isimle arar
    sistem   Her santralin sistem bilgisini (konum, saat dilimi, aralik) kaydeder
    gunluk   Gunluk uretim toplamlarini ceker (takvim ve kalite kontrol icin)
    cek      5 dakikalik durum gecmisini gun gun ceker; kaldigi yerden devam eder
    durum    Ilerleme ozetini yazar

Ham yanitlar degistirilmeden saklanir; ayristirma ayri bir adimda yapilir.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import os
import sys
import time
from pathlib import Path

import requests

TABAN = "https://pvoutput.org/service/r2/"
KOK = Path(__file__).resolve().parents[2]
SANTRAL_DOSYASI = KOK / "veri" / "santraller.csv"
HAM_KLASOR = KOK / "veri" / "ham" / "pvoutput"
ISTEMCI_ADI = "pv-benchmark-fetch/1.0"

SISTEM_ALANLARI = [
    "sistem_adi", "guc_w", "posta_kodu", "panel_sayisi", "panel_gucu_w", "panel_marka",
    "evirici_sayisi", "evirici_gucu_w", "evirici_marka", "yonelim", "egim", "golge",
    "kurulum_tarihi", "enlem", "boylam", "durum_araligi_dk", "saat_dilimi",
]
ARAMA_ALANLARI = [
    "sistem_adi", "guc_w", "posta_kodu", "yonelim", "cikti_sayisi", "son_cikti",
    "sid", "panel", "evirici", "mesafe", "enlem", "boylam",
]


class KaliciHata(Exception):
    """Yeniden denemenin anlamsiz oldugu hata (kimlik, yetki)."""


class GeciciHata(Exception):
    """Ag hatasi; istek daha sonra yeniden denenebilir."""


class Istemci:
    """PVOutput API istemcisi: kimlik basliklari, hiz siniri, yeniden deneme."""

    def __init__(self, api_key, system_id, oturum=None, uyku=time.sleep, simdi=time.time,
                 yedek=3, asgari_aralik=1.0, yaz=print):
        self.oturum = oturum or requests.Session()
        self.basliklar = {
            "X-Pvoutput-Apikey": api_key,
            "X-Pvoutput-SystemId": str(system_id),
            "X-Rate-Limit": "1",
            "User-Agent": ISTEMCI_ADI,
        }
        self.uyku, self.simdi, self.yaz = uyku, simdi, yaz
        self.yedek, self.asgari_aralik = yedek, asgari_aralik
        self.kalan = None
        self.sifirlanma = None
        self.istek_sayisi = 0
        self._son_istek = 0.0

    def _sinir_bekle(self):
        """Saatlik hak bitmek uzereyse sifirlanma anina kadar bekler."""
        if self.kalan is not None and self.kalan <= self.yedek:
            self._sifirlanmaya_kadar_bekle()

    def _sifirlanmaya_kadar_bekle(self):
        simdi = self.simdi()
        hedef = self.sifirlanma if self.sifirlanma and self.sifirlanma > simdi else simdi + 3600 - simdi % 3600
        sure = max(5.0, hedef - simdi + 5)
        bitis = dt.datetime.now() + dt.timedelta(seconds=sure)
        self.yaz(f"  Saatlik istek siniri doldu; {sure / 60:.0f} dk bekleniyor (devam: {bitis:%H:%M}).")
        self.uyku(sure)
        self.kalan = None

    def istek(self, servis, parametreler):
        """(http_kodu, metin) dondurur. 400 yanitlari cagirana birakilir."""
        ag_denemesi = 0
        while True:
            self._sinir_bekle()
            gecen = self.simdi() - self._son_istek
            if gecen < self.asgari_aralik:
                self.uyku(self.asgari_aralik - gecen)
            try:
                yanit = self.oturum.get(TABAN + servis, params=parametreler,
                                        headers=self.basliklar, timeout=60)
            except requests.RequestException as hata:
                ag_denemesi += 1
                if ag_denemesi > 5:
                    raise GeciciHata(str(hata)) from hata
                bekle = [10, 30, 60, 120, 300][ag_denemesi - 1]
                self.yaz(f"  Ag hatasi ({hata.__class__.__name__}); {bekle} sn sonra yeniden denenecek.")
                self.uyku(bekle)
                continue
            self._son_istek = self.simdi()
            self.istek_sayisi += 1
            self._basliklari_oku(yanit)
            kod, metin = yanit.status_code, yanit.text.strip()
            if kod == 403 and "exceeded" in metin.lower():
                self._sifirlanmaya_kadar_bekle()
                continue
            if kod in (401, 403, 405):
                raise KaliciHata(f"{kod}: {metin}")
            if kod >= 500:
                ag_denemesi += 1
                if ag_denemesi > 5:
                    raise GeciciHata(f"{kod}: {metin[:200]}")
                self.uyku([10, 30, 60, 120, 300][ag_denemesi - 1])
                continue
            return kod, metin

    def _basliklari_oku(self, yanit):
        try:
            if "X-Rate-Limit-Remaining" in yanit.headers:
                self.kalan = int(yanit.headers["X-Rate-Limit-Remaining"])
            if "X-Rate-Limit-Reset" in yanit.headers:
                self.sifirlanma = int(yanit.headers["X-Rate-Limit-Reset"])
        except (TypeError, ValueError):
            pass


# --------------------------------------------------------------------------- dosyalar

def santralleri_oku(yol=None):
    with open(yol or SANTRAL_DOSYASI, encoding="utf-8-sig", newline="") as f:
        ornek = f.readline()
        f.seek(0)
        ayrac = ";" if ornek.count(";") >= ornek.count(",") else ","
        satirlar = list(csv.DictReader(f, delimiter=ayrac))
    for s in satirlar:
        for anahtar in s:
            s[anahtar] = (s[anahtar] or "").strip()
    return satirlar


def santralleri_yaz(satirlar, yol=None):
    with open(yol or SANTRAL_DOSYASI, "w", encoding="utf-8-sig", newline="") as f:
        yazici = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()), delimiter=";")
        yazici.writeheader()
        yazici.writerows(satirlar)


def sayi(metin):
    return float(str(metin).replace(",", "."))


def ay_basi(metin):
    yil, ay = metin.split("-")
    return dt.date(int(yil), int(ay), 1)


def pvoutput_santralleri(satirlar):
    """PVOutput kaynakli ve `atla` sutunu bos olan santraller."""
    return [s for s in satirlar if s["kaynak"] == "pvoutput" and not s.get("atla")]


def csv_satiri(metin):
    return next(csv.reader(io.StringIO(metin)))


class Ilerleme:
    """Yalnizca sona eklenen gunluk: sid;tarih;durum;kayit;zaman."""

    def __init__(self, klasor):
        self.yol = Path(klasor) / "ilerleme.csv"
        self.durum = {}
        self.bozuk = 0          # ani kapanmada yarim kalan ya da NUL dolu satirlar atlanir (bkz. pvoutput_onar.py)
        if self.yol.exists():
            metin = self.yol.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="replace")
            for alan in csv.reader(metin.splitlines(), delimiter=";"):
                if len(alan) == 5 and alan[3].isdigit():
                    self.durum[(alan[0], alan[1])] = (alan[2], int(alan[3]))
                elif any(a.strip() for a in alan):
                    self.bozuk += 1

    def ekle(self, sid, tarih, durum, kayit):
        self.durum[(sid, tarih)] = (durum, kayit)
        self.yol.parent.mkdir(parents=True, exist_ok=True)
        satir = io.StringIO()
        csv.writer(satir, delimiter=";").writerow(
            [sid, tarih, durum, kayit, dt.datetime.now().isoformat(timespec="seconds")])
        dosyaya_ekle(self.yol, satir.getvalue())

    def tamam_mi(self, sid, tarih):
        return self.durum.get((sid, tarih), ("", 0))[0] in ("ok", "bos")


def dosyaya_ekle(yol, metin, deneme=40, uyku=time.sleep):
    """Dosyanin sonuna ekler. Proje OneDrive'da oldugu icin esitleme dosyayi kisa sure kilitleyebilir
    (PermissionError); bu durumda artan aralikla yeniden dener (toplam ~7 dk)."""
    for i in range(deneme):
        try:
            with open(yol, "a", encoding="utf-8", newline="") as f:
                f.write(metin)
            return
        except PermissionError:
            if i == deneme - 1:
                raise
            uyku(min(0.5 * (i + 1), 15.0))


# --------------------------------------------------------------------------- komutlar

def komut_kimlik(istemci, args):
    satirlar = santralleri_oku(args.santraller)
    adaylar = []
    for s in pvoutput_santralleri(satirlar):
        if s["sid"]:
            continue
        hedef_w = sayi(s["kapasite_kw"]) * 1000
        sorgular = [s["pvoutput_adi"], s["pvoutput_adi"].split()[0]]
        bulunan = []
        for sorgu in dict.fromkeys(sorgular):
            kod, metin = istemci.istek("search.jsp", {"q": sorgu})
            if kod == 200 and metin:
                for satir in metin.splitlines():
                    alan = dict(zip(ARAMA_ALANLARI, csv_satiri(satir)))
                    if alan.get("sid"):
                        bulunan.append(alan)
            if bulunan:
                break
        for alan in bulunan:
            adaylar.append({"santral": s["ad"], **alan})

        def guc_uyar(alan):
            try:
                return abs(sayi(alan["guc_w"]) - hedef_w) <= 0.02 * hedef_w
            except ValueError:
                return False

        uyan = [a for a in bulunan if guc_uyar(a)]
        ad_uyan = [a for a in uyan if a["sistem_adi"].strip().lower() == s["pvoutput_adi"].lower()]
        secim = ad_uyan if len(ad_uyan) == 1 else uyan
        if len(secim) == 1:
            s["sid"] = secim[0]["sid"]
            istemci.yaz(f"  BULUNDU   {s['ad']:<28} sid={s['sid']:<8} ({secim[0]['sistem_adi']}, {secim[0]['guc_w']} W)")
        else:
            istemci.yaz(f"  BELIRSIZ  {s['ad']:<28} {len(bulunan)} aday, {len(uyan)} guc eslesmesi "
                        f"-> kimlik_adaylari.csv dosyasina bakin")
    HAM = Path(args.ham)
    HAM.mkdir(parents=True, exist_ok=True)
    if adaylar:
        with open(HAM / "kimlik_adaylari.csv", "w", encoding="utf-8-sig", newline="") as f:
            yazici = csv.DictWriter(f, fieldnames=["santral"] + ARAMA_ALANLARI, delimiter=";",
                                    extrasaction="ignore")
            yazici.writeheader()
            yazici.writerows(adaylar)
    santralleri_yaz(satirlar, args.santraller)
    eksik = [s["ad"] for s in pvoutput_santralleri(satirlar) if not s["sid"]]
    istemci.yaz(f"Kimligi eksik santral: {len(eksik)}" + (f" -> {', '.join(eksik)}" if eksik else ""))


def komut_sistem(istemci, args):
    satirlar = santralleri_oku(args.santraller)
    HAM = Path(args.ham)
    HAM.mkdir(parents=True, exist_ok=True)
    cikti = []
    for s in pvoutput_santralleri(satirlar):
        if not s["sid"]:
            continue
        kod, metin = istemci.istek("getsystem.jsp", {"sid1": s["sid"], "tz": 1})
        if kod != 200:
            istemci.yaz(f"  HATA  {s['ad']}: {kod} {metin[:100]}")
            continue
        alanlar = csv_satiri(metin.split(";")[0])
        kayit = dict(zip(SISTEM_ALANLARI[:16], alanlar[:16]))
        kayit["saat_dilimi"] = alanlar[-1] if len(alanlar) > 16 else ""
        cikti.append({"ad": s["ad"], "sid": s["sid"], **kayit, "ham_yanit": metin})
        istemci.yaz(f"  {s['ad']:<28} {kayit.get('guc_w', '?'):>8} W  aralik={kayit.get('durum_araligi_dk', '?')} dk  "
                    f"saat dilimi={kayit['saat_dilimi'] or '?'}")
    with open(HAM / "sistemler.csv", "w", encoding="utf-8-sig", newline="") as f:
        yazici = csv.DictWriter(f, fieldnames=["ad", "sid"] + SISTEM_ALANLARI + ["ham_yanit"], delimiter=";")
        yazici.writeheader()
        yazici.writerows(cikti)
    istemci.yaz(f"{len(cikti)} sistem kaydedildi: {HAM / 'sistemler.csv'}")


def santral_araligi(s, args):
    """Cekim araligi. `cekim_baslangic` / `cekim_bitis` (YYYY-AA-GG) varsa genel araligin yerine gecer;
    kaydi erken biten santraller icin kullanilir."""
    if s.get("cekim_baslangic"):
        bas = dt.date.fromisoformat(s["cekim_baslangic"])
    elif s["tez_veri_baslangic"]:
        bas = max(args.baslangic, ay_basi(s["tez_veri_baslangic"]))
    else:
        bas = args.baslangic
    bit = min(args.bitis, dt.date.fromisoformat(s["cekim_bitis"])) if s.get("cekim_bitis") else args.bitis
    return bas, bit


def komut_gunluk(istemci, args):
    satirlar = santralleri_oku(args.santraller)
    yalniz = set(getattr(args, "sid", None) or [])
    for s in pvoutput_santralleri(satirlar):
        if not s["sid"] or (yalniz and s["sid"] not in yalniz):
            continue
        hedef = Path(args.ham) / s["sid"] / "gunluk.csv"
        if hedef.exists() and not args.yenile:
            continue
        bas, bit = santral_araligi(s, args)
        kayitlar = {}
        imlec = bit
        while imlec >= bas:
            pencere_basi = max(bas, imlec - dt.timedelta(days=149))
            kod, metin = istemci.istek("getoutput.jsp", {
                "df": f"{pencere_basi:%Y%m%d}", "dt": f"{imlec:%Y%m%d}", "limit": 150, "sid1": s["sid"]})
            if kod == 200 and metin:
                for kayit in metin.split(";"):
                    if kayit.strip():
                        kayitlar[kayit.split(",")[0]] = kayit.strip()
            imlec = pencere_basi - dt.timedelta(days=1)
        hedef.parent.mkdir(parents=True, exist_ok=True)
        hedef.write_text("\n".join(kayitlar[k] for k in sorted(kayitlar)) + "\n", encoding="utf-8")
        istemci.yaz(f"  {s['ad']:<28} {len(kayitlar):>5} gun ({bas} .. {bit})")


def uretimli_gunler(ham, sid):
    """gunluk.csv varsa uretimi sifirdan buyuk gunlerin kumesi; yoksa None."""
    yol = Path(ham) / sid / "gunluk.csv"
    if not yol.exists():
        return None
    gunler = set()
    for satir in yol.read_text(encoding="utf-8").splitlines():
        alan = satir.split(",")
        try:
            if len(alan) > 1 and float(alan[1]) > 0:
                gunler.add(alan[0])
        except ValueError:
            continue
    return gunler


def gorevleri_kur(satirlar, ilerleme, args):
    gorevler = []
    yalniz = set(getattr(args, "sid", None) or [])
    for s in pvoutput_santralleri(satirlar):
        if not s["sid"] or (yalniz and s["sid"] not in yalniz):
            continue
        bas, bit = santral_araligi(s, args)
        takvim = uretimli_gunler(args.ham, s["sid"])
        gun = bit
        while gun >= bas:
            tarih = f"{gun:%Y%m%d}"
            if (takvim is None or tarih in takvim) and not ilerleme.tamam_mi(s["sid"], tarih):
                gorevler.append((tarih, s["sid"], s["ad"]))
            gun -= dt.timedelta(days=1)
    gorevler.sort(key=lambda g: (g[0], g[1]), reverse=True)  # en yeni gunler once
    return gorevler


def komut_cek(istemci, args):
    satirlar = santralleri_oku(args.santraller)
    ilerleme = Ilerleme(args.ham)
    gorevler = gorevleri_kur(satirlar, ilerleme, args)
    if args.azami:
        gorevler = gorevler[:args.azami]
    istemci.yaz(f"Cekilecek santral-gun: {len(gorevler)} (saatte 300 istekle yaklasik {len(gorevler) / 300:.0f} saat)")
    for sira, (tarih, sid, ad) in enumerate(gorevler, 1):
        parametreler = {"d": tarih, "h": 1, "limit": 288, "asc": 1}
        if sid != istemci.basliklar["X-Pvoutput-SystemId"]:
            parametreler["sid1"] = sid
        try:
            kod, metin = istemci.istek("getstatus.jsp", parametreler)
        except GeciciHata as hata:
            ilerleme.ekle(sid, tarih, "hata", 0)
            istemci.yaz(f"  HATA {ad} {tarih}: {hata}")
            continue
        if kod == 200 and metin:
            kayitlar = [k.strip() for k in metin.split(";") if k.strip()]
            hedef = Path(args.ham) / sid / f"{tarih[:4]}-{tarih[4:6]}.csv"
            hedef.parent.mkdir(parents=True, exist_ok=True)
            with open(hedef, "a", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(kayitlar) + "\n")
            ilerleme.ekle(sid, tarih, "ok", len(kayitlar))
        elif kod == 400 and "no status" in metin.lower():
            ilerleme.ekle(sid, tarih, "bos", 0)
        else:
            ilerleme.ekle(sid, tarih, "hata", 0)
            istemci.yaz(f"  HATA {ad} {tarih}: {kod} {metin[:100]}")
        if sira % 50 == 0 or sira == len(gorevler):
            istemci.yaz(f"  {sira}/{len(gorevler)}  son: {ad} {tarih}  kalan hak: {istemci.kalan}")
    istemci.yaz("Tamamlandi.")


def komut_durum(istemci, args):
    satirlar = santralleri_oku(args.santraller)
    ilerleme = Ilerleme(args.ham)
    kalan = {}
    for _, sid, _ in gorevleri_kur(satirlar, ilerleme, args):
        kalan[sid] = kalan.get(sid, 0) + 1
    print(f"{'santral':<28} {'sid':<8} {'dolu':>6} {'bos':>6} {'hata':>6} {'kalan':>6} {'kayit':>9}")
    for s in pvoutput_santralleri(satirlar):
        sid = s["sid"]
        d = [v for (k_sid, _), v in ilerleme.durum.items() if k_sid == sid] if sid else []
        print(f"{s['ad']:<28} {sid or '-':<8} {sum(x[0] == 'ok' for x in d):>6} {sum(x[0] == 'bos' for x in d):>6} "
              f"{sum(x[0] == 'hata' for x in d):>6} {kalan.get(sid, 0):>6} {sum(x[1] for x in d):>9}")
    print(f"Toplam kalan santral-gun: {sum(kalan.values())}")


def ayristirici():
    p = argparse.ArgumentParser(description="PVOutput ham veri cekme araci")
    p.add_argument("komut", choices=["kimlik", "sistem", "gunluk", "cek", "durum"])
    p.add_argument("--santraller", default=str(SANTRAL_DOSYASI))
    p.add_argument("--ham", default=str(HAM_KLASOR))
    p.add_argument("--baslangic", type=dt.date.fromisoformat, default=dt.date(2022, 12, 1),
                   help="Cekilecek ilk gun (YYYY-AA-GG)")
    p.add_argument("--bitis", type=dt.date.fromisoformat, default=dt.date.today() - dt.timedelta(days=1),
                   help="Cekilecek son gun (varsayilan: dun)")
    p.add_argument("--azami", type=int, default=0, help="Bu calistirmada en fazla kac santral-gun cekilsin")
    p.add_argument("--yenile", action="store_true", help="gunluk: mevcut dosyalari yeniden cek")
    p.add_argument("--sid", nargs="*", default=[], help="gunluk/cek/durum: yalnizca bu sistem kimlikleri")
    p.add_argument("--kimlik", default=os.environ.get("PVOUTPUT_KIMLIK_DOSYASI", str(KIMLIK_DOSYASI)),
                   help="Kimlik bilgilerinin okunacagi dosya")
    return p


KIMLIK_DOSYASI = Path.home() / "pvoutput_kimlik.txt"


def kimlik_bilgileri(dosya=None, ortam=None):
    """(api_anahtari, sistem_kimligi): once ortam degiskenleri, yoksa kimlik dosyasi.

    Dosya proje ve OneDrive disinda, kullanicinin ev klasorundedir; iki satir icerir:
        PVOUTPUT_API_KEY=...
        PVOUTPUT_SYSTEM_ID=...
    """
    ortam = os.environ if ortam is None else ortam
    degerler = {ad: (ortam.get(ad) or "").strip() for ad in ("PVOUTPUT_API_KEY", "PVOUTPUT_SYSTEM_ID")}
    dosya = Path(dosya or KIMLIK_DOSYASI)
    if not all(degerler.values()) and dosya.exists():
        for satir in dosya.read_text(encoding="utf-8-sig").splitlines():
            if "=" in satir and not satir.lstrip().startswith("#"):
                ad, deger = satir.split("=", 1)
                ad = ad.strip().upper()
                if ad in degerler and not degerler[ad]:
                    degerler[ad] = deger.strip().strip("\"'")
    return degerler["PVOUTPUT_API_KEY"], degerler["PVOUTPUT_SYSTEM_ID"]


def main(argv=None):
    for akis in (sys.stdout, sys.stderr):
        if hasattr(akis, "reconfigure"):
            akis.reconfigure(encoding="utf-8", errors="replace")
    args = ayristirici().parse_args(argv)
    if args.komut == "durum":
        komut_durum(None, args)
        return 0
    anahtar, kimlik = kimlik_bilgileri(args.kimlik)
    if not anahtar or not kimlik:
        print("API anahtari bulunamadi. Su dosyayi olusturup iki satiri yazin:")
        print(f"    {args.kimlik}")
        print("    PVOUTPUT_API_KEY=anahtariniz")
        print("    PVOUTPUT_SYSTEM_ID=sistem_kimliginiz")
        return 2
    print(f"Ham veri klasoru: {args.ham}")
    istemci = Istemci(anahtar, kimlik)
    try:
        {"kimlik": komut_kimlik, "sistem": komut_sistem, "gunluk": komut_gunluk, "cek": komut_cek}[args.komut](istemci, args)
    except KaliciHata as hata:
        print(f"Durduruldu - PVOutput istegi reddetti: {hata}")
        return 1
    except KeyboardInterrupt:
        print("\nKullanici durdurdu. Ayni komutu yeniden calistirinca kaldigi yerden devam eder.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
