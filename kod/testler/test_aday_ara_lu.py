"""aday_ara_lu: yalniz PV (gece) kosulu."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "veri_cekme"))
import aday_ara_lu as lu  # noqa: E402


def gun(gece_guc, gece_enerji_artisi=0, son=20000):
    kayitlar = []
    for i, saat in enumerate(["00:00", "01:00", "02:00", "03:00", "03:55"]):
        kayitlar.append(f"20250615,{saat},{i * gece_enerji_artisi // 4},0.0,{gece_guc},NaN,NaN,NaN,NaN,NaN,NaN")
    kayitlar.append(f"20250615,12:00,{son // 2},0.5,5000,5000,NaN,NaN,NaN,NaN,NaN")
    kayitlar.append(f"20250615,20:00,{son},1.0,0,0,NaN,NaN,NaN,NaN,NaN")
    return ";".join(kayitlar)


class Testler(unittest.TestCase):
    def test_temiz_pv(self):
        self.assertTrue(lu.yalniz_pv([gun(0), gun(0)])["yalniz_pv"])

    def test_gece_gucu(self):
        s = lu.yalniz_pv([gun(400), gun(0)])
        self.assertFalse(s["yalniz_pv"])
        self.assertAlmostEqual(s["gece_guc_orani"], 0.5)

    def test_gece_enerji_artisi(self):
        s = lu.yalniz_pv([gun(0, gece_enerji_artisi=800)])          # %4 artis, guc 0 bildirilse de
        self.assertFalse(s["yalniz_pv"])

    def test_sifirlanmayan_sayac_gecer(self):
        sabit = gun(0).replace("20250615,00:00,0,", "20250615,00:00,0,")
        self.assertTrue(lu.yalniz_pv([sabit])["yalniz_pv"])

    def test_gece_kaydi_yok(self):
        self.assertTrue(lu.yalniz_pv(["20250615,12:00,100,0.1,500,NaN,NaN,NaN,NaN,NaN,NaN"])["yalniz_pv"])


if __name__ == "__main__":
    unittest.main()
