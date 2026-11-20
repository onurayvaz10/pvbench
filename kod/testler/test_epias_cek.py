"""epias_cek icin sahte sunucuyla testler (ag erisimi gerekmez)."""

import datetime as dt
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "veri_cekme"))
import epias_cek as ec  # noqa: E402

TR = ec.TR


def sentetik_seri(bas=ec.DONEM[0], bit=ec.DONEM[1], tepe=10.0, degistir=None):
    """Saat başı etiketli, 06–18 arası yarım sinüs üretimi olan tam seri."""
    seri, gun = {}, bas
    while gun <= bit:
        for saat in range(24):
            z = dt.datetime.combine(gun, dt.time(saat), TR)
            g = tepe * max(0.0, math.sin(math.pi * (saat - 6) / 12)) if 6 <= saat <= 18 else 0.0
            seri[z] = {"gunes": g, "diger": 0.0}
        gun += dt.timedelta(days=1)
    if degistir:
        degistir(seri)
    return seri


class SahteYanit:
    def __init__(self, kod, govde=None, metin=""):
        self.status_code, self._govde, self.text = kod, govde, metin

    def json(self):
        if self._govde is None:
            raise ValueError
        return self._govde


class SahteOturum:
    def __init__(self, servis_yanitlari):
        self.servis_yanitlari = list(servis_yanitlari)
        self.girisler, self.istekler = [], []

    def post(self, url, data=None, json=None, headers=None, timeout=None):
        if url == ec.GIRIS_URL:
            self.girisler.append(data)
            return SahteYanit(201, metin=f"TGT-{len(self.girisler)}")
        self.istekler.append((url, json, dict(headers)))
        return self.servis_yanitlari.pop(0)

    def get(self, url, headers=None, timeout=None):
        self.istekler.append((url, None, dict(headers)))
        return self.servis_yanitlari.pop(0)


class SahteSaat:
    def __init__(self):
        self.t = 1_000_000.0

    def simdi(self):
        return self.t

    def uyku(self, s):
        self.t += s


class Testler(unittest.TestCase):
    def test_gunes_santrali_adi(self):
        for ad in ["KALE GES", "ABC GES-2", "XYZ GÜNEŞ ENERJİSİ", "Karapınar Ges", "DEF GUNES"]:
            self.assertTrue(ec.gunes_santrali_mi(ad), ad)
        for ad in ["GESTAŞ RES", "DOĞALGAZ KÇS", "AGES HES", ""]:
            self.assertFalse(ec.gunes_santrali_mi(ad), ad)

    def test_aylar(self):
        parcalar = ec.aylar(dt.date(2022, 12, 1), dt.date(2023, 2, 15))
        self.assertEqual(parcalar, [(dt.date(2022, 12, 1), dt.date(2022, 12, 31)),
                                    (dt.date(2023, 1, 1), dt.date(2023, 1, 31)),
                                    (dt.date(2023, 2, 1), dt.date(2023, 2, 15))])

    def test_kayit_zamani(self):
        self.assertEqual(ec.kayit_zamani({"date": "2024-01-01T13:00:00+03:00", "hour": "13:00"}),
                         dt.datetime(2024, 1, 1, 13, tzinfo=TR))
        self.assertEqual(ec.kayit_zamani({"date": "2024-01-01T00:00:00+03:00", "hour": 5}),
                         dt.datetime(2024, 1, 1, 5, tzinfo=TR))

    def test_kimlik_dosyadan_ve_ortamdan(self):
        with tempfile.TemporaryDirectory() as d:
            yol = Path(d) / "kimlik.txt"
            yol.write_text("# yorum\nEPIAS_KULLANICI=deneme@ornek.test\nEPIAS_SIFRE=sahte-sifre\n", encoding="utf-8")
            self.assertEqual(ec.kimlik_bilgileri(yol, ortam={}), ("deneme@ornek.test", "sahte-sifre"))
            self.assertEqual(ec.kimlik_bilgileri(yol, ortam={"EPIAS_KULLANICI": "o", "EPIAS_SIFRE": "s"}), ("o", "s"))
        self.assertEqual(ec.kimlik_bilgileri(None, ortam={}), ("", ""))

    def test_istemci_bilet_ve_401_yenileme(self):
        saat = SahteSaat()
        oturum = SahteOturum([SahteYanit(200, {"items": [1]}), SahteYanit(401), SahteYanit(200, {"items": [2]})])
        mesajlar = []
        ist = ec.Istemci("k", "gizli-sifre", oturum=oturum, uyku=saat.uyku, simdi=saat.simdi, yaz=mesajlar.append)
        self.assertEqual(ist.istek("/a", {"x": 1}), (200, {"items": [1]}))
        self.assertEqual(len(oturum.girisler), 1)
        self.assertEqual(ist.istek("/b", {"x": 1}), (200, {"items": [2]}))
        self.assertEqual(len(oturum.girisler), 2)                       # 401'de yeni bilet
        self.assertEqual(oturum.istekler[-1][2]["TGT"], "TGT-2")
        self.assertFalse(any("gizli-sifre" in m for m in mesajlar))

    def test_istemci_bilet_suresi(self):
        saat = SahteSaat()
        oturum = SahteOturum([SahteYanit(200, {}), SahteYanit(200, {})])
        ist = ec.Istemci("k", "s", oturum=oturum, uyku=saat.uyku, simdi=saat.simdi)
        ist.istek("/a", {})
        saat.t += 101 * 60
        ist.istek("/a", {})
        self.assertEqual(len(oturum.girisler), 2)

    def test_uevm_cek_yazar_ve_tam_aylari_atlar(self):
        class SahteIstemci:
            def __init__(self):
                self.govdeler = []

            def istek(self, yol, govde=None):
                self.govdeler.append(govde)
                bas = dt.date.fromisoformat(govde["startDate"][:10])
                bit = dt.date.fromisoformat(govde["endDate"][:10])
                ogeler, gun = [], bas
                while gun <= bit:
                    for h in range(24):
                        ogeler.append({"date": dt.datetime.combine(gun, dt.time(h), TR).isoformat(), "hour": f"{h:02d}:00",
                                       "sun": 1.0 if 8 <= h <= 16 else 0.0, "wind": 0.0, "total": 1.0})
                    gun += dt.timedelta(days=1)
                return 200, {"items": ogeler, "page": {"total": len(ogeler)}}

        ist = SahteIstemci()
        with tempfile.TemporaryDirectory() as d:
            hedef = Path(d) / "uevm" / "5.csv"
            eksik = ec.uevm_cek(ist, "5", dt.date(2023, 1, 1), dt.date(2023, 2, 28), hedef)
            self.assertEqual(eksik, 0)
            self.assertEqual(len(ist.govdeler), 2)
            seri = ec.seri_oku(hedef)
            self.assertEqual(len(seri), (31 + 28) * 24)
            self.assertEqual(seri[dt.datetime(2023, 1, 1, 9, tzinfo=TR)]["gunes"], 1.0)
            ec.uevm_cek(ist, "5", dt.date(2023, 1, 1), dt.date(2023, 2, 28), hedef)
            self.assertEqual(len(ist.govdeler), 2)                       # tam aylar yeniden istenmedi

    def test_kosullar_temiz_seri(self):
        s = ec.kosullari_degerlendir(sentetik_seri())
        self.assertTrue(s["k1_5"], s)
        self.assertAlmostEqual(s["kapsam"], 1.0)
        self.assertGreater(s["p99_mwh"], 9.0)

    def test_kosul1_hibrit(self):
        def ruzgar(seri):
            next(iter(seri.values()))["diger"] = 0.5
        s = ec.kosullari_degerlendir(sentetik_seri(degistir=ruzgar))
        self.assertFalse(s["k1_yalniz_gunes"])
        self.assertFalse(s["k1_5"])

    def test_kosul2_kapsam(self):
        def bosluk(seri):
            for z in list(seri)[: int(len(seri) * 0.02)]:
                del seri[z]
        s = ec.kosullari_degerlendir(sentetik_seri(degistir=bosluk))
        self.assertFalse(s["k2_kapsam"])

    def test_kosul3_kapasite_artisi(self):
        def artis(seri):
            for z, d in seri.items():
                if z.date() >= dt.date(2024, 12, 1):
                    d["gunes"] *= 1.5
        s = ec.kosullari_degerlendir(sentetik_seri(degistir=artis))
        self.assertFalse(s["k3_sabit_guc"])
        self.assertTrue(s["k2_kapsam"])

    def test_kosul4_gece_uretimi(self):
        def gece(seri):
            for z, d in seri.items():
                if z.hour == 23 and z.day % 2 == 0:
                    d["gunes"] = 1.0
        s = ec.kosullari_degerlendir(sentetik_seri(degistir=gece))
        self.assertFalse(s["k4_gece_sifir"])

    def test_kosul5_kesinti_gunleri(self):
        def kesinti(seri):
            for z, d in seri.items():
                if z.day in (3, 17):                                     # ayda iki gün sıfır ≈ %6,6
                    d["gunes"] = 0.0
        s = ec.kosullari_degerlendir(sentetik_seri(degistir=kesinti))
        self.assertFalse(s["k5_kesinti_az"])
        self.assertTrue(s["k3_sabit_guc"])

    def test_korelasyon(self):
        a = {i: math.sin(i / 5) for i in range(500)}
        r, n = ec.korelasyon(a, {i: 2 * v + 1 for i, v in a.items()})
        self.assertAlmostEqual(r, 1.0)
        self.assertEqual(n, 500)


if __name__ == "__main__":
    unittest.main()
