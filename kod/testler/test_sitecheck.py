"""Tests for pvbench.sitecheck using synthetic clear-sky generation at known sites."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pvbench import sitecheck as S  # noqa: E402


def synthetic_raw(lat, lon, tz, days):
    """5-minute records whose power follows clear-sky GHI at (lat, lon), stamped in local time `tz`."""
    frames = []
    for day in days:
        local = pd.date_range(day, periods=288, freq="5min", tz=tz)
        ghi = pvlib.location.Location(lat, lon).get_clearsky(local.tz_convert("UTC"), model="haurwitz")["ghi"].to_numpy()
        frames.append(pd.DataFrame({"local": local.tz_localize(None), "power_w": ghi * 5.0,
                                    "energy_wh": np.cumsum(ghi * 5.0) / 12.0}))
    return pd.concat(frames, ignore_index=True)


DAYS = [f"2025-{m:02d}-{d:02d}" for m in (5, 6, 7, 11, 12) for d in (5, 15, 25)] + ["2026-01-15"]


class LocationTests(unittest.TestCase):
    def check(self, lat, lon, tz, lon_tol=1.5, lat_tol=6.0):
        est = S.estimate_location(synthetic_raw(lat, lon, tz, DAYS), tz)
        self.assertEqual(est["n_days"], len(DAYS))
        self.assertLess(abs(est["longitude"] - lon), lon_tol, est)
        self.assertLess(abs(est["latitude"] - lat), lat_tol, est)
        return est

    def test_northern_europe(self):
        self.check(52.0, 5.0, "Europe/Amsterdam")

    def test_southern_hemisphere(self):
        est = self.check(-27.6, 153.2, "Australia/Brisbane")
        self.assertLess(est["latitude"], 0)

    def test_tropics_and_half_hour_zone(self):
        self.check(9.7, 76.3, "Asia/Kolkata")

    def test_wrong_time_zone_shifts_longitude(self):
        # records stamped in UK local time but declared as Brisbane time: longitude comes out ~150 degrees off
        raw = synthetic_raw(52.0, -1.0, "Europe/London", DAYS)
        est = S.estimate_location(raw, "Australia/Brisbane")
        self.assertGreater(abs(est["longitude"] - (-1.0)), 100.0)

    def test_equation_of_time_and_declination(self):
        self.assertAlmostEqual(float(S.equation_of_time_minutes(307)), 16.4, delta=0.6)    # early November maximum
        self.assertAlmostEqual(float(S.declination_deg(172)), 23.45, delta=0.05)           # June solstice

    def test_no_usable_days(self):
        raw = pd.DataFrame({"local": pd.date_range("2025-06-15", periods=5, freq="5min"),
                            "power_w": 0.0, "energy_wh": 0.0})
        self.assertEqual(S.estimate_location(raw, "UTC")["n_days"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
