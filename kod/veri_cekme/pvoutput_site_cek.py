"""PVOutput web sitesinden 5 dakikalik durum gecmisini gun gun ceker.

Neden: API'nin saatlik istek siniri. Site sayfalari (intraday.jsp) API ile ayni degerleri verir;
calistir() bunu her calismada api_ile_karsilastir() ile denetler.

Cikti API ile AYNI ham bicimdedir ve ayni yerlere yazilir:
    veri/ham/pvoutput/<sid>/<YYYY-AA>.csv   (date,time,energy_Wh,efficiency,inst_W,avg_W,
                                             normalised,cons_energy_Wh,cons_power_W,temp_C,voltage_V)
    veri/ham/pvoutput/ilerleme.csv          (API ile ayni ilerleme gunlugu; `pvoutput_cek.py durum` calisir)
Sitede `normalised` sutunu yoktur; NaN yazilir. Kaynagi siteden olan gunler ayrica
veri/ham/pvoutput/site_kaynakli_gunler.csv dosyasina kaydedilir (veri kaynagi izi).

Kimlik: PVOutput kullanici adi ve sifresi calisma aninda sorulur (ya da PVOUTPUT_KULLANICI /
PVOUTPUT_SIFRE ortam degiskenleri). Sifre hicbir yere yazilmaz ve ekrana basilmaz.

Kullanim (proje kokunden):
    python -u kod/veri_cekme/pvoutput_site_cek.py            # kalan tum gunler, eskiden yeniye
    python -u kod/veri_cekme/pvoutput_site_cek.py --azami 20 # deneme
Kesilirse yeniden calistirmak yeterlidir; tamamlanan gunler atlanir.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import getpass
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pvoutput_cek as pc  # noqa: E402

SITE = "https://pvoutput.org/"
ISTEMCI_ADI = "pv-benchmark-fetch/1.0 (site)"

SATIR_RE = re.compile(r"<tr class='[eo]'>(.*?)</tr>", re.S)
HUCRE_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
ETIKET_RE = re.compile(r"<[^>]+>")
SAYI_RE = re.compile(r"(-?[\d,]*\.?\d+)\s*([A-Za-z/]*)")
SAHIP_RE = re.compile(r"intraday\.jsp\?id=(\d+)&(?:amp;)?sid=(\d+)")

ENERJI = {"Wh": 1.0, "kWh": 1e3, "MWh": 1e6, "GWh": 1e9}
GUC = {"W": 1.0, "kW": 1e3, "MW": 1e6, "GW": 1e9}


class Durdur(Exception):
    """Devam etmenin anlamsiz ya da sakincali oldugu durum (engelleme, captcha, giris basarisiz)."""


class GirisHatasi(Durdur):
    """Site kullanici adi/sifreyi kabul etmedi."""


IPUCU_RE = re.compile(r"(invalid|incorrect|wrong|locked|too many|failed|not found)[^<]{0,80}", re.I)


def deger(metin, birimler=None):
    """'3,383.600kWh' -> 3383600.0; '11.1C' -> 11.1; '-' -> None. Birim taniniyorsa temel birime cevirir."""
    metin = ETIKET_RE.sub("", metin).strip()
    m = SAYI_RE.fullmatch(metin)
    if not m:
        return None
    sayi = float(m.group(1).replace(",", ""))
    if birimler is None:
        return sayi
    birim = m.group(2)
    if birim not in birimler:
        raise ValueError(f"beklenmeyen birim: {metin!r}")
    return sayi * birimler[birim]


def _tam(x):
    return "NaN" if x is None else str(int(round(x)))


def _ondalik(x, basamak):
    return "NaN" if x is None else f"{x:.{basamak}f}"


def satirlari_ayristir(html, tarih):
    """intraday.jsp HTML'inden `tarih` (YYYYAAGG) gunune ait satirlari API biciminde, artan saat sirasiyla dondurur."""
    hedef = f"{tarih[6:8]}/{tarih[4:6]}/{tarih[2:4]}"
    cikti = []
    for satir in SATIR_RE.findall(html):
        h = [ETIKET_RE.sub("", c).strip() for c in HUCRE_RE.findall(satir)]
        if len(h) < 10 or h[0] != hedef:
            continue
        enerji = deger(h[2], ENERJI)
        verim = deger(h[3])
        guc = deger(h[4], GUC)
        ortalama = deger(h[5], GUC)
        sicaklik = deger(h[6])
        gerilim = deger(h[7])
        tuketim_enerji = deger(h[8], ENERJI)
        tuketim_guc = deger(h[9], GUC)
        cikti.append(",".join([
            tarih, h[1], _tam(enerji), _ondalik(verim, 3), _tam(guc), _tam(ortalama), "NaN",
            _tam(tuketim_enerji), _tam(tuketim_guc), _ondalik(sicaklik, 1), _ondalik(gerilim, 1)]))
    cikti.sort(key=lambda s: s.split(",")[1])
    # Site, API'de bos (NaN) olan enerji degerini "0.000kWh" gosterir. Gun icinde sayac bir kez arttiktan
    # sonra gorulen 0 fiziksel olarak imkansizdir (birikimli sayac); API'deki gibi NaN yazilir.
    # Sayac baslamadan onceki sabah sifirlari 0 kalir (o saatlerde uretim yoktur).
    tepe = 0.0
    for i, s in enumerate(cikti):
        alan = s.split(",")
        if alan[2] != "NaN":
            e = float(alan[2])
            if e == 0.0 and tepe > 0.0:
                alan[2] = "NaN"
                cikti[i] = ",".join(alan)
            tepe = max(tepe, e)
    return cikti


class Site:
    def __init__(self, kullanici, sifre, aralik=1.0, yaz=print, oturum=None, uyku=time.sleep, simdi=time.monotonic):
        self.kullanici, self._sifre = kullanici, sifre
        self.aralik, self.yaz, self.uyku, self.simdi = aralik, yaz, uyku, simdi
        self.oturum = oturum or requests.Session()
        self.oturum.headers["User-Agent"] = ISTEMCI_ADI
        self._son = 0.0

    def _bekle(self):
        fark = self.aralik - (self.simdi() - self._son)
        if fark > 0:
            self.uyku(fark)
        self._son = self.simdi()

    def giris(self):
        """Tarayicidaki formun yaptigini yapar: login.html'i acar (oturum cerezi), index.jsp'ye POST eder.
        Basari olcutu: donen sayfada 'Logout' baglantisi (girisli sayfalarin hepsinde bulunur)."""
        self._bekle()
        self.oturum.get(SITE + "login.html", timeout=60)
        self._bekle()
        r = self.oturum.post(SITE + "index.jsp", timeout=60,
                             data={"login": self.kullanici, "password": self._sifre, "remember": "on"},
                             headers={"Referer": SITE + "login.html", "Origin": SITE.rstrip("/")})
        if "Logout" not in r.text:
            self._bekle()
            r2 = self.oturum.get(SITE + "index.jsp", timeout=60)
            if "Logout" not in r2.text:
                ipucu = IPUCU_RE.search(ETIKET_RE.sub(" ", r.text))
                raise GirisHatasi(f"HTTP {r.status_code}, sayfa {r.url.replace(SITE, '')}"
                                  + (f", site mesaji: {ipucu.group(0).strip()!r}" if ipucu else ", site mesaj vermedi"))
        self.yaz("PVOutput sitesine giris yapildi.")

    def sayfa(self, yol):
        ardisik = 0
        while True:
            self._bekle()
            try:
                r = self.oturum.get(SITE + yol, timeout=60)
            except requests.RequestException as hata:
                r, ag_hatasi = None, hata
            if r is not None and r.status_code == 200 and "login.html" in r.url:
                self.yaz("Oturum dusmus; yeniden giris yapiliyor.")
                self.giris()
                continue
            if r is not None and "captcha" in r.text.lower():
                raise Durdur("Site dogrulama (captcha) istedi; durduruldu.")
            if r is not None and r.status_code in (200, 404):
                return r.status_code, r.text
            ardisik += 1
            neden = f"HTTP {r.status_code}" if r is not None else f"ag hatasi: {ag_hatasi}"
            if ardisik >= 5:
                raise Durdur(f"Ust uste 5 hata ({neden}); site engelliyor olabilir, durduruldu.")
            bekle = 60 * ardisik
            self.yaz(f"  {neden}; {bekle} sn bekleniyor ({ardisik}/5)")
            self.uyku(bekle)

    def sahip_kimligi(self, sid):
        _, metin = self.sayfa(f"list.jsp?sid={sid}")
        for kimlik, s in SAHIP_RE.findall(metin):
            if s == str(sid):
                return kimlik
        return None

    def gun(self, sid, sahip, tarih):
        kod, metin = self.sayfa(f"intraday.jsp?id={sahip}&sid={sid}&dt={tarih}")
        return satirlari_ayristir(metin, tarih) if kod == 200 else []


def sahipleri_getir(site, sidler, ham):
    yol = Path(ham) / "site_sahip_kimlikleri.csv"
    sahipler = {}
    if yol.exists():
        for satir in yol.read_text(encoding="utf-8").splitlines()[1:]:
            sid, kimlik = satir.split(";")
            sahipler[sid] = kimlik
    for sid in sidler:
        if sid not in sahipler:
            kimlik = site.sahip_kimligi(sid)
            if not kimlik:
                raise Durdur(f"sid {sid} icin sahip kimligi bulunamadi")
            sahipler[sid] = kimlik
    yol.write_text("sid;sahip_id\n" + "".join(f"{s};{k}\n" for s, k in sorted(sahipler.items())), encoding="utf-8")
    return sahipler


def api_ile_karsilastir(site, sahipler, ham, ilerleme, adet=3, rastgele=random.Random(17)):
    """API'den zaten cekilmis birkac gunu siteden de ceker; enerji ve anlik guc birebir ayni olmali."""
    adaylar = [(sid, tarih) for (sid, tarih), (durum, kayit) in ilerleme.durum.items()
               if durum == "ok" and kayit > 50 and sid in sahipler]
    def esit(x, y):
        return x == y or ("NaN" not in (x, y) and float(x) == float(y))

    ayni = toplam = bos_sifir = 0
    for sid, tarih in rastgele.sample(adaylar, min(adet, len(adaylar))):
        dosya = Path(ham) / sid / f"{tarih[:4]}-{tarih[4:6]}.csv"
        api = {}
        for alan in csv.reader(dosya.read_text(encoding="utf-8").splitlines()):
            if alan and alan[0] == tarih:
                api[alan[1]] = (alan[2], alan[4])
        for satir in site.gun(sid, sahipler[sid], tarih):
            alan = satir.split(",")
            if alan[1] in api:
                a = api[alan[1]]
                if a[0] == "NaN" and alan[2] in ("0", "NaN"):     # API'de bos, sitede 0: bilinen gosterim farki
                    bos_sifir += 1
                    continue
                toplam += 1
                ayni += esit(alan[2], a[0]) and esit(alan[4], a[1])
    oran = ayni / toplam if toplam else 0.0
    site.yaz(f"API ile karsilastirma: {ayni}/{toplam} satir ayni (enerji ve anlik guc), oran {oran:.4f}; "
             f"API'de bos / sitede 0 enerji: {bos_sifir} satir (ayri sayildi)")
    if oran < 0.99:
        raise Durdur("Site verisi API verisiyle uyusmuyor; cekim baslatilmadi.")


def api_cekimini_durdur(pid, yaz):
    """Calisan API cekimini (uzun_cekim.cmd sureci ve alt surecleri) durdurur; ayni dosyalara iki surec yazmasin."""
    sonuc = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
    yaz(f"API cekimi durduruldu (PID {pid}): " + " | ".join(
        s.strip() for s in (sonuc.stdout + sonuc.stderr).splitlines() if s.strip()))
    time.sleep(2)


def calistir(args, site):
    satirlar = pc.santralleri_oku(args.santraller)
    sahipler = sahipleri_getir(site, sorted(s["sid"] for s in pc.pvoutput_santralleri(satirlar) if s["sid"]), args.ham)
    api_ile_karsilastir(site, sahipler, args.ham, pc.Ilerleme(args.ham))
    if args.api_durdur:
        api_cekimini_durdur(args.api_durdur, site.yaz)
    ilerleme = pc.Ilerleme(args.ham)                                    # API durduktan sonraki guncel durum
    gorevler = sorted(pc.gorevleri_kur(satirlar, ilerleme, args))      # eskiden yeniye
    if args.azami:
        gorevler = gorevler[:args.azami]
    site.yaz(f"Cekilecek santral-gun: {len(gorevler)} (sayfa basina {site.aralik:.1f} sn ile yaklasik "
             f"{len(gorevler) * site.aralik / 3600:.1f} saat)")
    iz = Path(args.ham) / "site_kaynakli_gunler.csv"
    baslangic = time.time()
    ardisik_bos = ardisik_hata = 0
    for sira, (tarih, sid, ad) in enumerate(gorevler, 1):
        try:
            kayitlar = site.gun(sid, sahipler[sid], tarih)
            ardisik_bos = 0 if kayitlar else ardisik_bos + 1
            if ardisik_bos >= 50:                           # gorevler yalnizca uretimli gunler; bu kadar bos olagan disi
                raise Durdur("Ust uste 50 bos sayfa; oturum ya da sayfa bicimi bozulmus olabilir, durduruldu.")
            if kayitlar:
                hedef = Path(args.ham) / sid / f"{tarih[:4]}-{tarih[4:6]}.csv"
                hedef.parent.mkdir(parents=True, exist_ok=True)
                pc.dosyaya_ekle(hedef, "\n".join(kayitlar) + "\n")
                ilerleme.ekle(sid, tarih, "ok", len(kayitlar))
            else:
                ilerleme.ekle(sid, tarih, "bos", 0)
            pc.dosyaya_ekle(iz, f"{sid};{tarih};{len(kayitlar)};{dt.datetime.now().isoformat(timespec='seconds')}\n")
            ardisik_hata = 0
        except Durdur:
            raise
        except Exception as hata:                           # tek gunluk beklenmeyen hata: kaydet, bekle, devam et
            ardisik_hata += 1
            site.yaz(f"  HATA {ad} {tarih}: {type(hata).__name__}: {hata}  ({ardisik_hata}/20)")
            if ardisik_hata >= 20:
                raise Durdur("Ust uste 20 beklenmeyen hata; durduruldu.")
            site.uyku(30)
            continue
        if sira % 100 == 0 or sira == len(gorevler):
            gecen = time.time() - baslangic
            kalan_sn = gecen / sira * (len(gorevler) - sira)
            site.yaz(f"  {sira}/{len(gorevler)}  son: {ad} {tarih} ({len(kayitlar)} kayit)  "
                     f"tahmini kalan: {kalan_sn / 3600:.1f} saat")
    site.yaz("Tamamlandi.")


def main(argv=None):
    for akis in (sys.stdout, sys.stderr):
        if hasattr(akis, "reconfigure"):
            akis.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="PVOutput sitesinden 5 dakikalik veri cekme")
    p.add_argument("--santraller", default=str(pc.SANTRAL_DOSYASI))
    p.add_argument("--ham", default=str(pc.HAM_KLASOR))
    p.add_argument("--baslangic", type=dt.date.fromisoformat, default=dt.date(2022, 12, 1))
    p.add_argument("--bitis", type=dt.date.fromisoformat, default=dt.date.today() - dt.timedelta(days=1))
    p.add_argument("--azami", type=int, default=0)
    p.add_argument("--aralik", type=float, default=1.0, help="Iki sayfa istegi arasindaki en kisa sure (sn)")
    p.add_argument("--api-durdur", type=int, default=0,
                   help="Giris ve API karsilastirmasi basarili olunca bu PID'deki API cekimini durdur")
    args = p.parse_args(argv)

    gunluk = Path(args.ham) / "site_cekim_gunlugu.txt"

    def yaz(mesaj):
        satir = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {mesaj}"
        print(satir, flush=True)
        with open(gunluk, "a", encoding="utf-8") as f:
            f.write(satir + "\n")

    kullanici = os.environ.get("PVOUTPUT_KULLANICI") or input("PVOutput kullanici adi: ").strip()
    sifre = os.environ.get("PVOUTPUT_SIFRE") or getpass.getpass("PVOutput sifresi (ekranda gorunmez): ")
    site = Site(kullanici, sifre, aralik=args.aralik, yaz=yaz)
    for deneme in range(1, 4):
        try:
            site.giris()
            break
        except GirisHatasi as hata:
            yaz(f"Giris basarisiz ({deneme}/3): {hata}")
            if os.environ.get("PVOUTPUT_SIFRE") or deneme == 3:
                yaz("DURDU: giris yapilamadi.")
                return 1
            site._sifre = getpass.getpass("Sifreyi yeniden yazin (ekranda gorunmez): ")
        except requests.RequestException as hata:
            yaz(f"DURDU: siteye ulasilamadi: {hata}")
            return 1
    try:
        calistir(args, site)
    except Durdur as hata:
        yaz(f"DURDU: {hata}")
        return 1
    except KeyboardInterrupt:
        yaz("Kullanici durdurdu. Ayni komutu yeniden calistirinca kaldigi yerden devam eder.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
