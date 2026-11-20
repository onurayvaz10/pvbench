"""Tests for pvbench.pvoutput: raw records -> hourly UTC interval-averaged power."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pvbench import pvoutput as P  # noqa: E402


def constant_day(day="2025-06-15", start="06:00", end="18:00", kw=6.0, step_min=5):
    """Records with constant power `kw` between start and end (cumulative Wh counter)."""
    local = pd.date_range(f"{day} {start}", f"{day} {end}", freq=f"{step_min}min")
    energy = np.arange(len(local)) * kw * 1000.0 * step_min / 60.0
    return pd.DataFrame({"local": local, "energy_wh": energy, "power_w": kw * 1000.0})


def hourly(raw, tz, capacity=10.0, totals=None):
    return P.hourly_from_intervals(P.build_intervals(raw, tz, capacity, daily_totals=totals), capacity)["power_kw"]


def at(series, stamp):
    return series[pd.Timestamp(stamp, tz="UTC")]


class HourlyTests(unittest.TestCase):
    def test_half_hour_offset_zone(self):
        # Asia/Kolkata is UTC+5:30: local 06:00-18:00 is 00:30-12:30 UTC
        h = hourly(constant_day(), "Asia/Kolkata")
        self.assertAlmostEqual(at(h, "2025-06-15 02:00"), 6.0)       # 01:00-02:00 UTC, full production
        self.assertAlmostEqual(at(h, "2025-06-15 01:00"), 3.0)       # production starts at 00:30 UTC
        self.assertAlmostEqual(at(h, "2025-06-15 13:00"), 3.0)       # production ends at 12:30 UTC
        self.assertAlmostEqual(at(h, "2025-06-15 00:00"), 0.0)       # before first record: zero, valid
        self.assertAlmostEqual(at(h, "2025-06-15 16:00"), 0.0)       # after last record: zero, valid
        self.assertTrue(np.isnan(at(h, "2025-06-14 19:00")))         # half of this hour is the previous, empty day
        self.assertAlmostEqual(h.sum(), 72.0)                        # 12 h x 6 kW, energy conserved

    def test_dst_zone_and_hour_end_label(self):
        h = hourly(constant_day(), "Europe/Berlin")                  # CEST = UTC+2
        self.assertAlmostEqual(at(h, "2025-06-15 11:00"), 6.0)       # local 12:00-13:00
        self.assertAlmostEqual(at(h, "2025-06-15 04:00"), 0.0)       # local 05:00-06:00
        self.assertAlmostEqual(at(h, "2025-06-15 05:00"), 6.0)       # local 06:00-07:00
        winter = hourly(constant_day("2025-12-15", "09:00", "15:00"), "Europe/Berlin")   # CET = UTC+1
        self.assertAlmostEqual(at(winter, "2025-12-15 09:00"), 6.0)  # local 09:00-10:00

    def test_gap_with_energy_is_missing_not_interpolated(self):
        raw = constant_day()
        raw = raw[(raw["local"] <= "2025-06-15 10:00") | (raw["local"] >= "2025-06-15 11:00")]
        h = hourly(raw, "Europe/Berlin")
        self.assertTrue(np.isnan(at(h, "2025-06-15 09:00")))         # local 10:00-11:00 lies in the gap
        self.assertAlmostEqual(at(h, "2025-06-15 08:00"), 6.0)
        self.assertAlmostEqual(at(h, "2025-06-15 10:00"), 6.0)

    def test_long_gap_without_energy_is_zero(self):
        raw = pd.DataFrame({"local": pd.to_datetime(["2025-06-15 06:00", "2025-06-15 09:00", "2025-06-15 09:05"]),
                            "energy_wh": [0.0, 0.0, 500.0], "power_w": [0.0, 0.0, 6000.0]})
        h = hourly(raw, "Europe/Berlin")
        self.assertAlmostEqual(at(h, "2025-06-15 06:00"), 0.0)       # local 07:00-08:00 inside the zero gap
        self.assertAlmostEqual(at(h, "2025-06-15 08:00"), 0.5)       # 500 Wh in local 09:00-10:00

    def test_records_missing_at_day_start(self):
        raw = constant_day(start="12:00")
        raw["energy_wh"] += 20000.0                                  # counter already at 20 kWh at noon
        h = hourly(raw, "Europe/Berlin")
        self.assertTrue(np.isnan(at(h, "2025-06-15 08:00")))         # morning unknown, not zero
        self.assertTrue(np.isnan(at(h, "2025-06-15 10:00")))         # hour containing the first record
        self.assertAlmostEqual(at(h, "2025-06-15 12:00"), 6.0)

    def test_records_missing_at_day_end_detected_with_daily_total(self):
        raw = constant_day(end="12:00")                              # counter ends at 36 kWh
        day = pd.Timestamp("2025-06-15").date()
        without = hourly(raw, "Europe/Berlin")
        self.assertAlmostEqual(at(without, "2025-06-15 13:00"), 0.0)
        with_total = hourly(raw, "Europe/Berlin", totals=pd.Series({day: 72000.0}))
        self.assertTrue(np.isnan(at(with_total, "2025-06-15 13:00")))
        self.assertAlmostEqual(at(with_total, "2025-06-15 09:00"), 6.0)
        complete = hourly(raw, "Europe/Berlin", totals=pd.Series({day: 36000.0}))
        self.assertAlmostEqual(at(complete, "2025-06-15 13:00"), 0.0)

    def test_implausible_increment_and_over_capacity(self):
        raw = constant_day()
        raw.loc[raw["local"] >= "2025-06-15 12:00", "energy_wh"] += 50000.0     # counter jump at noon
        h = hourly(raw, "Europe/Berlin")
        self.assertTrue(np.isnan(at(h, "2025-06-15 10:00")))         # hour containing the jump
        self.assertAlmostEqual(at(h, "2025-06-15 12:00"), 6.0)
        self.assertLessEqual(np.nanmax(h), 10.0 * 1.15)

    def test_counter_decrease_is_invalid(self):
        raw = constant_day()
        raw.loc[raw["local"] == "2025-06-15 12:00", "energy_wh"] = 0.0
        h = hourly(raw, "Europe/Berlin")
        self.assertTrue(np.isnan(at(h, "2025-06-15 10:00")) and np.isnan(at(h, "2025-06-15 11:00")))


class ReadTests(unittest.TestCase):
    def test_read_raw_and_full_pipeline(self):
        folder = Path(tempfile.mkdtemp())
        lines = ["20261115,12:00,12000,0.400,4800,4800,0.170,NaN,NaN,4.0,230.0",
                 "20261115,12:05,12400,0.410,5000,5200,0.180,NaN,NaN,4.5,231.0",
                 "20261115,12:05,12400,0.410,5000,5200,0.180,NaN,NaN,4.5,231.0",      # duplicate from a re-fetch
                 "20261115,12:10,12800,0.420,5200,5100,0.180,NaN,NaN,4.5,232.0"]
        (folder / "2026-11.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (folder / "gunluk.csv").write_text("20261115,12800,0.420,0,0,5200,12:10,Fine,1,5,0,0,0,0\n", encoding="utf-8")
        raw = P.read_raw(folder)
        self.assertEqual(len(raw), 3)
        self.assertEqual(raw["energy_wh"].tolist(), [12000.0, 12400.0, 12800.0])
        self.assertEqual(P.read_daily_totals(folder).iloc[0], 12800.0)
        table = P.hourly_power(folder, "Europe/Berlin", 30.0)
        self.assertIn("power_inst_kw", table.columns)
        # the 12:00 local sample (11:00 UTC) closes the hour ending 11:00; the other two fall in the next hour
        self.assertAlmostEqual(at(table["power_inst_kw"], "2026-11-15 11:00"), 4.800, places=3)
        self.assertAlmostEqual(at(table["power_inst_kw"], "2026-11-15 12:00"), (5.000 + 5.200) / 2, places=3)
        # counter starts at 12.0 kWh at noon: the morning is unknown, the afternoon is zero
        self.assertTrue(np.isnan(at(table["power_kw"], "2026-11-15 09:00")))
        self.assertAlmostEqual(at(table["power_kw"], "2026-11-15 15:00"), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
