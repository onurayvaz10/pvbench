"""pvoutput_cek icin sahte sunucuyla testler (ag erisimi gerekmez)."""

import argparse
import datetime as dt
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "veri_cekme"))
import pvoutput_cek as pc  # noqa: E402

SANTRALLER = """no;ad;pvoutput_adi;kaynak;sid;kapasite_kw;ulke;tez_veri_baslangic;tez_veri_bitis
1;Alfa;Alfa Solar;pvoutput;;7.7;X;2019-03;2019-08
2;Beta;Beta #1;pvoutput;222;6.6;X;2019-02;2019-07
3;Firma;Firma GES;firma;;99.9;X;2019-02;2019-07
"""


class SahteYanit:
    def __init__(self, kod, metin, basliklar=None):
        self.status_code, self.text, self.headers = kod, metin, basliklar or {}


class SahteSaat:
    def __init__(self):
        self.t = 1_800_000_000.0
        self.uykular = []

    def simdi(self):
        return self.t

    def uyku(self, sure):
        self.uykular.append(sure)
        self.t += sure


class SahteOturum:
    """Saatlik `sinir` istekten sonra 403 dondurur; saat sifirlaninca hak yenilenir."""

    def __init__(self, saat, sinir=300, bos_gunler=()):
        self.saat, self.sinir, self.bos_gunler = saat, sinir, set(bos_gunler)
        self.pencere_basi, self.kullanilan = saat.t, 0
        self.cagrilar = []

    def get(self, url, params=None, headers=None, timeout=None):
        assert headers["X-Pvoutput-Apikey"] == "ANAHTAR"
        if self.saat.t - self.pencere_basi >= 3600:
            self.pencere_basi, self.kullanilan = self.saat.t, 0
        sifir = int(self.pencere_basi + 3600)
        if self.kullanilan >= self.sinir:
            return SahteYanit(403, "Forbidden 403: Exceeded 300 requests per hour",
                              {"X-Rate-Limit-Remaining": "0", "X-Rate-Limit-Reset": str(sifir)})
        self.kullanilan += 1
        servis = url.rsplit("/", 1)[1]
        self.cagrilar.append((servis, dict(params)))
        b = {"X-Rate-Limit-Remaining": str(self.sinir - self.kullanilan), "X-Rate-Limit-Reset": str(sifir)}
        if servis == "search.jsp":
            return SahteYanit(200, "Alfa Solar,7700,1000,S,50,Today,111,P,I,NaN,1.0,2.0\n"
                                   "Alfa Solar Two,3000,1000,S,50,Today,112,P,I,NaN,1.0,2.0", b)
        if servis == "getsystem.jsp":
            return SahteYanit(200, "Alfa Solar,7700,1000,40,250,Marka,1,7700,Ev,S,30.0,No,20240101,"
                                   "50.1,4.2,5,Europe/Brussels;;0", b)
        if servis == "getoutput.jsp":
            bas = dt.datetime.strptime(params["df"], "%Y%m%d").date()
            bit = dt.datetime.strptime(params["dt"], "%Y%m%d").date()
            assert (bit - bas).days < 150
            gunler = [bas + dt.timedelta(days=i) for i in range((bit - bas).days + 1)]
            return SahteYanit(200, ";".join(f"{g:%Y%m%d},{0 if g.day == 2 else 1500},0.5,0,0,900,12:00,Fine,5,15,0,0,0,0"
                                            for g in reversed(gunler)), b)
        if servis == "getstatus.jsp":
            if params["d"] in self.bos_gunler:
                return SahteYanit(400, "Bad request 400: No status found", b)
            return SahteYanit(200, f"{params['d']},10:35,298,0.067,710,708,0.159,NaN,NaN;"
                                   f"{params['d']},10:40,359,0.080,731,732,0.164,NaN,NaN", b)
        raise AssertionError(servis)


class Testler(unittest.TestCase):
    def setUp(self):
        self.klasor = Path(tempfile.mkdtemp())
        self.santraller = self.klasor / "santraller.csv"
        self.santraller.write_text(SANTRALLER, encoding="utf-8")
        self.saat = SahteSaat()
        self.mesajlar = []

    def tearDown(self):
        shutil.rmtree(self.klasor, ignore_errors=True)

    def istemci(self, **kw):
        self.oturum = SahteOturum(self.saat, **kw)
        return pc.Istemci("ANAHTAR", "999", oturum=self.oturum, uyku=self.saat.uyku,
                          simdi=self.saat.simdi, yaz=self.mesajlar.append)

    def args(self, **kw):
        temel = dict(santraller=str(self.santraller), ham=str(self.klasor / "ham"),
                     baslangic=dt.date(2025, 11, 1), bitis=dt.date(2025, 11, 10), azami=0, yenile=False)
        temel.update(kw)
        return argparse.Namespace(**temel)

    def test_kimlik_guc_ile_eslesir_ve_dosyaya_yazar(self):
        pc.komut_kimlik(self.istemci(), self.args())
        satirlar = pc.santralleri_oku(self.santraller)
        self.assertEqual(satirlar[0]["sid"], "111")   # 7,7 kW olan aday secildi
        self.assertEqual(satirlar[1]["sid"], "222")   # mevcut kimlige dokunulmadi
        self.assertEqual(satirlar[2]["sid"], "")      # firma verisi aranmadi
        self.assertEqual(len([c for c in self.oturum.cagrilar if c[0] == "search.jsp"]), 1)

    def test_sistem_bilgisi_ve_saat_dilimi(self):
        pc.komut_sistem(self.istemci(), self.args())
        metin = (self.klasor / "ham" / "sistemler.csv").read_text(encoding="utf-8-sig")
        self.assertIn("Europe/Brussels", metin)
        self.assertIn(";50.1;4.2;5;", metin)

    def test_cek_takvimi_kullanir_bos_gunu_isaretler_ve_devam_eder(self):
        args = self.args()
        pc.komut_gunluk(self.istemci(), args)
        pc.komut_cek(self.istemci(bos_gunler={"20251105"}), args)
        durum = [c for c in self.oturum.cagrilar if c[0] == "getstatus.jsp"]
        self.assertEqual(len(durum), 9)                       # 10 gun - uretimsiz 2 Kasim
        self.assertEqual(durum[0][1]["d"], "20251110")        # en yeni gun once
        self.assertEqual(durum[0][1]["sid1"], "222")
        ilerleme = pc.Ilerleme(args.ham)
        self.assertEqual(ilerleme.durum[("222", "20251105")], ("bos", 0))
        self.assertEqual(ilerleme.durum[("222", "20251110")], ("ok", 2))
        satirlar = (self.klasor / "ham" / "222" / "2025-11.csv").read_text().splitlines()
        self.assertEqual(len(satirlar), 16)                   # 8 dolu gun x 2 kayit
        self.assertTrue(satirlar[0].startswith("20251110,10:35"))
        pc.komut_cek(self.istemci(), args)                    # ikinci calistirma: is kalmadi
        self.assertEqual([c for c in self.oturum.cagrilar if c[0] == "getstatus.jsp"], [])

    def test_hiz_siniri_dolunca_bekler_ve_tamamlar(self):
        args = self.args(baslangic=dt.date(2025, 10, 1), bitis=dt.date(2025, 10, 30))
        pc.komut_cek(self.istemci(sinir=10), args)            # takvim yok: 30 gunun hepsi
        ilerleme = pc.Ilerleme(args.ham)
        self.assertEqual(sum(v[0] == "ok" for v in ilerleme.durum.values()), 30)
        uzun = [u for u in self.saat.uykular if u > 600]
        self.assertGreaterEqual(len(uzun), 2)                 # saat basini bekledi
        self.assertTrue(all(u <= 3700 for u in uzun))

    def test_kimlik_bilgileri_dosyadan_ve_ortamdan(self):
        dosya = self.klasor / "kimlik.txt"
        dosya.write_text('﻿# not\npvoutput_api_key = "DOSYA-ANAHTAR" \nPVOUTPUT_SYSTEM_ID=12345\n', encoding="utf-8")
        self.assertEqual(pc.kimlik_bilgileri(dosya, ortam={}), ("DOSYA-ANAHTAR", "12345"))
        ortam = {"PVOUTPUT_API_KEY": "ORTAM-ANAHTAR", "PVOUTPUT_SYSTEM_ID": "9"}
        self.assertEqual(pc.kimlik_bilgileri(dosya, ortam=ortam), ("ORTAM-ANAHTAR", "9"))   # ortam oncelikli
        self.assertEqual(pc.kimlik_bilgileri(self.klasor / "yok.txt", ortam={}), ("", ""))

    def test_santral_araligi_ozel_cekim_tarihleri(self):
        args = argparse.Namespace(baslangic=dt.date(2022, 12, 1), bitis=dt.date(2026, 11, 16))
        genel = {"tez_veri_baslangic": "", "cekim_baslangic": "", "cekim_bitis": ""}
        self.assertEqual(pc.santral_araligi(genel, args), (dt.date(2022, 12, 1), dt.date(2026, 11, 16)))
        gec = {**genel, "tez_veri_baslangic": "2023-05"}
        self.assertEqual(pc.santral_araligi(gec, args), (dt.date(2023, 5, 1), dt.date(2026, 11, 16)))
        erken_biten = {**genel, "cekim_baslangic": "2021-02-14", "cekim_bitis": "2023-02-13"}
        self.assertEqual(pc.santral_araligi(erken_biten, args), (dt.date(2021, 2, 14), dt.date(2023, 2, 13)))
        self.assertEqual(pc.santral_araligi({"tez_veri_baslangic": ""}, args)[0], dt.date(2022, 12, 1))  # sutun yok

    def test_yetki_hatasi_kalici_hatadir(self):
        istemci = self.istemci()
        self.oturum.get = lambda *a, **k: SahteYanit(401, "Unauthorized 401: Inaccessible System ID")
        with self.assertRaises(pc.KaliciHata):
            pc.komut_cek(istemci, self.args())


if __name__ == "__main__":
    unittest.main(verbosity=2)
