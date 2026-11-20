"""Tests for pvbench.metrics and pvbench.stats against hand-computed values."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pvbench import metrics as M  # noqa: E402
from pvbench import stats as S  # noqa: E402


class MetricTests(unittest.TestCase):
    y = np.array([0.0, 0.0, 2.0, 4.0, 6.0, np.nan])
    yhat = np.array([0.0, 1.0, 3.0, 4.0, 4.0, 5.0])

    def test_basic_errors_ignore_nan(self):
        self.assertAlmostEqual(M.rmse(self.y, self.yhat), np.sqrt(6 / 5))
        self.assertAlmostEqual(M.mae(self.y, self.yhat), 4 / 5)
        self.assertAlmostEqual(M.mbe(self.y, self.yhat), 0.0)

    def test_r2(self):
        self.assertAlmostEqual(M.r2(self.y, self.yhat), 1 - 6 / 27.2)
        self.assertTrue(np.isnan(M.r2([1.0, 1.0], [1.0, 2.0])))     # zero-variance target

    def test_mape_threshold_excludes_small_targets(self):
        # capacity 10, threshold 30 % -> only targets 4 and 6 count: (0 + 2/6) / 2
        self.assertAlmostEqual(M.mape(self.y, self.yhat, 10, 0.30), 100 * (2 / 6) / 2)
        # threshold 5 % -> targets 2, 4, 6: (1/2 + 0 + 2/6) / 3
        self.assertAlmostEqual(M.mape(self.y, self.yhat, 10, 0.05), 100 * (0.5 + 2 / 6) / 3)

    def test_theil_u2_and_skill(self):
        naive = np.array([0.0, 0.0, 0.0, 2.0, 4.0, 6.0])              # RMSE = sqrt(12/5)
        self.assertAlmostEqual(M.theil_u2(self.y, self.yhat, naive), np.sqrt(6 / 12))
        self.assertAlmostEqual(M.skill_score(1.0, 2.0), 0.5)

    def test_mase_scale(self):
        self.assertAlmostEqual(M.mase_scale([1, 2, 3, 5, 2, 4], season=2), (2 + 3 + 1 + 1) / 4)

    def test_evaluate_daytime_split(self):
        day = np.array([False, False, True, True, True, True])
        out = M.evaluate(self.y, self.yhat, capacity=10, daytime=day, naive=self.yhat, scale=2.0)
        self.assertEqual((out["n"], out["n_day"]), (5, 3))
        self.assertAlmostEqual(out["nrmse_day"], 100 * np.sqrt(5 / 3) / 10)
        self.assertAlmostEqual(out["mase"], 0.4)
        self.assertAlmostEqual(out["theil_u2"], 1.0)
        self.assertIn("mape_05", out)


class StatsTests(unittest.TestCase):
    def test_holm(self):
        np.testing.assert_allclose(S.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(S.holm([0.5, 0.9]), [1.0, 1.0])

    def test_wilcoxon_matches_hand_computation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        b = a + np.array([1.0, 2.0, 3.0, 4.0, 5.0, -6.0])            # d = a - b: five negative, one positive
        r = S.paired_wilcoxon(a, b)
        self.assertEqual((r["wins_a"], r["wins_b"], r["n_nonzero"]), (5, 1, 6))
        self.assertAlmostEqual(r["W"], 6.0)                           # W+ = rank of |6| = 6, W- = 15
        self.assertAlmostEqual(r["rank_biserial"], (6 - 15) / 21)
        self.assertAlmostEqual(r["p"], 0.4375)                        # exact two-sided, n = 6
        self.assertLess(r["mean_ci"][0], r["mean_diff"])
        self.assertGreater(r["mean_ci"][1], r["mean_diff"])

    def test_wilcoxon_all_ties(self):
        r = S.paired_wilcoxon([1.0, 2.0], [1.0, 2.0])
        self.assertTrue(np.isnan(r["p"]))

    def test_nemenyi_critical_values_match_demsar_table(self):
        # Demsar (2006), Table 5a, alpha = 0.05: q = 2.343 (k=3), 2.728 (k=5), 3.164 (k=10)
        for k, q in ((3, 2.343), (5, 2.728), (10, 3.164)):
            cd = S.nemenyi_critical_difference(k, n=6)
            self.assertAlmostEqual(cd / np.sqrt(k * (k + 1) / 36.0), q, delta=0.002)  # table has 3 decimals

    def test_friedman_and_reference_comparison(self):
        rng = np.random.default_rng(1)
        base = rng.uniform(5, 15, size=20)
        scores = pd.DataFrame({"good": base - 1.0 + rng.normal(0, 0.1, 20),
                               "mid": base + rng.normal(0, 0.1, 20),
                               "bad": base + 1.0 + rng.normal(0, 0.1, 20)})
        f = S.friedman_nemenyi(scores)
        self.assertEqual(list(f["mean_ranks"].index), ["good", "mid", "bad"])
        self.assertLess(f["p"], 1e-6)
        table = S.compare_to_reference(scores, "good")
        self.assertTrue((table["p_holm"] < 0.001).all())
        self.assertTrue((table["rank_biserial"] == -1.0).all())      # reference better on every plant
        self.assertTrue((table["p_holm"] >= table["p"]).all())

    def test_diebold_mariano(self):
        rng = np.random.default_rng(2)
        days = pd.date_range("2025-01-01", periods=300, freq="D")           # daily losses, as the function expects
        small = pd.Series(rng.normal(0, 1.0, 300) ** 2, index=days)
        large = pd.Series(rng.normal(0, 2.0, 300) ** 2, index=days)
        r = S.diebold_mariano(small, large)
        self.assertLess(r["stat"], 0)
        self.assertLess(r["p"], 0.001)
        same = S.diebold_mariano(small, pd.Series(rng.normal(0, 1.0, 300) ** 2, index=days))
        self.assertGreater(same["p"], 0.01)
        self.assertTrue(np.isnan(S.diebold_mariano(small, small)["stat"]))

    def test_spearman_ci(self):
        x = np.arange(30.0)
        r = S.spearman_ci(x, -x + np.random.default_rng(3).normal(0, 3, 30), n_boot=500)
        self.assertLess(r["rho"], -0.8)
        self.assertLess(r["ci"][1], 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
