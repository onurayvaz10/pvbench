"""pvbench.epias: UEVM CSV -> PVOutput ile ayni bicimde saatlik UTC tablo."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pvbench import epias as E  # noqa: E402


def yaz(klasor, ad, satirlar):
    yol = Path(klasor) / ad
    yol.write_text("zaman_tr,gunes,toplam,diger\n" + "".join(f"{z},{g},{g},0.0\n" for z, g in satirlar),
                   encoding="utf-8")
    return yol


class Testler(unittest.TestCase):
    def test_bicim_birim_ve_zaman(self):
        with tempfile.TemporaryDirectory() as d:
            yol = yaz(d, "1.csv", [("2024-06-01T12:00:00+03:00", 2.5), ("2024-06-01T13:00:00+03:00", 3.0),
                                   ("2024-06-01T15:00:00+03:00", 1.0)])
            t = E.hourly_power(yol, capacity_kw=5000.0)
        self.assertEqual(list(t.columns), ["power_kw", "valid_fraction", "power_inst_kw"])
        self.assertEqual(t.index.name, "time_utc_end")
        self.assertEqual(str(t.index.tz), "UTC")
        # 12:00-13:00 TR = 09:00-10:00 UTC -> etiket 10:00 UTC; 2,5 MWh -> 2500 kW
        self.assertEqual(t.loc[pd.Timestamp("2024-06-01 10:00", tz="UTC"), "power_kw"], 2500.0)
        eksik = t.loc[pd.Timestamp("2024-06-01 12:00", tz="UTC")]           # 14:00 TR kaydı yok
        self.assertTrue(np.isnan(eksik["power_kw"]))
        self.assertEqual(eksik["valid_fraction"], 0.0)
        self.assertEqual(len(t), 4)

    def test_kapasite_ustu_nan_ve_gercek_zamanli(self):
        with tempfile.TemporaryDirectory() as d:
            yol = yaz(d, "1.csv", [("2024-06-01T12:00:00+03:00", 9.0), ("2024-06-01T13:00:00+03:00", 3.0)])
            gz = Path(d) / "gz.csv"
            gz.write_text("zaman_tr,gunes\n2024-06-01T13:00:00+03:00,2.9\n", encoding="utf-8")
            t = E.hourly_power(yol, capacity_kw=5000.0, realtime_path=gz)
        self.assertTrue(np.isnan(t["power_kw"].iloc[0]))
        self.assertEqual(t["power_inst_kw"].iloc[1], 2900.0)


if __name__ == "__main__":
    unittest.main()
