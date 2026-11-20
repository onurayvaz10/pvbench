"""pvoutput_site_cek: sayfa ayristirma ve birim donusumu (ag erisimi gerekmez)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "veri_cekme"))
import pvoutput_site_cek as ps  # noqa: E402

TD = "<td align='right' style='padding-right:20px' nowrap>{}</td>"
BOS = "<small class='comment' style='padding-right:15px'>-</small>"


def satir(sinif, *hucreler):
    return f"<tr class='{sinif}'>" + "".join(TD.format(h) for h in hucreler) + "</tr>"


SAYFA = ("<table><tr><th>Date</th><th>Time</th></tr>"
         + satir("e", "15/06/23", "23:55", "2,150.400kWh", "2.150kWh/kW", "0W", "0W", "12.5C", "230.0V", "30.000kWh", "250W")
         + satir("o", "15/06/23", "12:05", "1,000.200kWh", "1.000kWh/kW", "300,000W", "305,000W", "-", "-", "-", "-")
         + satir("e", "15/06/23", "05:05", "0.000kWh", "0.000kWh/kW", "0W", BOS, "10.0C", "229.0V", "-", "-")
         + satir("o", "14/06/23", "23:55", "9.000kWh", "1.000kWh/kW", "0W", "0W", "-", "-", "-", "-")
         + "</table>")


class Testler(unittest.TestCase):
    def test_deger_birimleri(self):
        self.assertEqual(ps.deger("2,150.400kWh", ps.ENERJI), 2150400.0)
        self.assertEqual(ps.deger("1.5MWh", ps.ENERJI), 1500000.0)
        self.assertEqual(ps.deger("300,000W", ps.GUC), 300000.0)
        self.assertEqual(ps.deger("2.5kW", ps.GUC), 2500.0)
        self.assertEqual(ps.deger("12.5C"), 12.5)
        self.assertIsNone(ps.deger("-"))
        self.assertIsNone(ps.deger(BOS, ps.GUC))
        with self.assertRaises(ValueError):
            ps.deger("5Wh/kW", ps.GUC)

    def test_api_bicimi_ve_sira(self):
        cikti = ps.satirlari_ayristir(SAYFA, "20230615")
        self.assertEqual(cikti, [
            "20230615,05:05,0,0.000,0,NaN,NaN,NaN,NaN,10.0,229.0",
            "20230615,12:05,1000200,1.000,300000,305000,NaN,NaN,NaN,NaN,NaN",
            "20230615,23:55,2150400,2.150,0,0,NaN,30000,250,12.5,230.0",
        ])                                                       # baska gunun satiri alinmadi, saat artan

    def test_gun_ortasi_sifir_enerji_nan_olur(self):
        sayfa = (satir("e", "24/08/26", "06:05", "0.000kWh", "0.000kWh/kW", "0W", BOS, BOS, BOS, BOS, BOS)
                 + satir("o", "24/08/26", "07:05", "0.003kWh", "0.000kWh/kW", "40W", BOS, BOS, BOS, BOS, BOS)
                 + satir("e", "24/08/26", "12:00", "0.000kWh", "0.000kWh/kW", "0W", BOS, BOS, BOS, BOS, BOS)
                 + satir("o", "24/08/26", "12:05", "7.500kWh", "0.500kWh/kW", "5,000W", BOS, BOS, BOS, BOS, BOS))
        enerji = [s.split(",")[2] for s in ps.satirlari_ayristir(sayfa, "20260824")]
        self.assertEqual(enerji, ["0", "3", "NaN", "7500"])     # sabah sifiri kalir, gun ortasi sifir bos sayilir

    def test_bos_sayfa(self):
        self.assertEqual(ps.satirlari_ayristir("<html>no data</html>", "20230615"), [])

    def test_oturum_dusunce_yeniden_giris(self):
        class Yanit:
            def __init__(self, url, metin, kod=200):
                self.url, self.text, self.status_code = url, metin, kod

        class Oturum:
            def __init__(self):
                self.headers, self.postlar = {}, []
                self.getler = [Yanit(ps.SITE + "login.html", "giris"),          # oturum dusmus
                               Yanit(ps.SITE + "login.html", "form"),           # giris(): cerez icin
                               Yanit(ps.SITE + "intraday.jsp", SAYFA)]

            def post(self, url, data=None, headers=None, timeout=None):
                self.postlar.append(data)
                return Yanit(ps.SITE + "intraday.jsp", "... Logout ...")

            def get(self, url, timeout=None):
                return self.getler.pop(0)

        o = Oturum()
        site = ps.Site("k", "gizli", aralik=0, yaz=lambda m: None, oturum=o, uyku=lambda s: None)
        self.assertEqual(len(site.gun("11111", "22222", "20230615")), 3)
        self.assertEqual(o.postlar, [{"login": "k", "password": "gizli", "remember": "on"}])

    def test_giris_hatasi_mesaji(self):
        class Yanit:
            def __init__(self, url, metin):
                self.url, self.text, self.status_code = url, metin, 200

        class Oturum:
            headers = {}

            def post(self, url, data=None, headers=None, timeout=None):
                return Yanit(ps.SITE + "login.html", "<div>Invalid login or password</div>")

            def get(self, url, timeout=None):
                return Yanit(ps.SITE + "login.html", "Login or Email")

        site = ps.Site("k", "gizli", aralik=0, yaz=lambda m: None, oturum=Oturum(), uyku=lambda s: None)
        with self.assertRaises(ps.GirisHatasi) as baglam:
            site.giris()
        self.assertIn("Invalid login or password", str(baglam.exception))
        self.assertNotIn("gizli", str(baglam.exception))

    def test_ust_uste_hata_durdurur(self):
        class Yanit:
            url, text, status_code = ps.SITE + "intraday.jsp", "", 503

        class Oturum:
            headers = {}

            def get(self, url, timeout=None):
                return Yanit()

        site = ps.Site("k", "s", aralik=0, yaz=lambda m: None, oturum=Oturum(), uyku=lambda s: None)
        with self.assertRaises(ps.Durdur):
            site.gun("1", "2", "20230615")


if __name__ == "__main__":
    unittest.main()
