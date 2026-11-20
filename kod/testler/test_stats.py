"""Checks of pvbench.stats on cases with known answers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pvbench import stats  # noqa: E402


def test_holm():
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    out = stats.holm([0.01, np.nan])                       # an undefined test must stay undefined
    assert np.isclose(out[0], 0.01) and np.isnan(out[1])
    assert np.isnan(stats.holm([np.nan, np.nan])).all()


def test_wilcoxon():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 25), rng.normal(0.8, 1, 25)
    r = stats.paired_wilcoxon(a, b)
    ref = sps.wilcoxon(a - b)
    assert np.isclose(r["p"], ref.pvalue) and np.isclose(r["W"], ref.statistic)
    assert r["rank_biserial"] < 0 and r["wins_a"] > r["wins_b"]          # a is lower (better)
    tied = stats.paired_wilcoxon(np.ones(10), np.ones(10))
    assert np.isnan(tied["p"]) and tied["mean_ci"] == (0.0, 0.0) and tied["median_ci"] == (0.0, 0.0)
    table = stats.compare_to_reference(pd.DataFrame({"x": a, "same": a, "y": b}), "x")
    assert np.isnan(table.set_index("model").loc["same", "p_holm"])
    assert np.isclose(table.set_index("model").loc["y", "p_holm"], ref.pvalue)   # family of one defined test


def test_friedman():
    rng = np.random.default_rng(2)
    scores = pd.DataFrame(rng.normal(0, 1, (25, 5)) + np.arange(5) * 0.7, columns=list("abcde"))
    r = stats.friedman_nemenyi(scores)
    assert r["mean_ranks"].index[0] == "a" and r["p"] < 0.001
    q = sps.studentized_range.ppf(0.95, 5, np.inf) / np.sqrt(2)
    assert np.isclose(r["critical_difference"], q * np.sqrt(5 * 6 / (6 * 25)))


def test_diebold_mariano():
    days = pd.date_range("2025-01-01", periods=300, freq="D")
    rng = np.random.default_rng(3)
    base = pd.Series(rng.gamma(2.0, 1.0, 300), index=days)
    same = stats.diebold_mariano(base, base)
    assert np.isnan(same["stat"])                                        # identical forecasts: no test
    alt = pd.Series(np.where(np.arange(50) % 2 == 0, 1.0, -1.0), index=days[:50])
    r = stats.diebold_mariano(alt, alt * 0)                              # alternating differential
    assert np.isfinite(r["stat"]) and r["p"] > 0.5                       # variance stays positive, no effect
    worse = base + 0.5 + rng.normal(0, 0.3, 300)
    r = stats.diebold_mariano(base, worse)
    assert r["stat"] < -5 and r["p"] < 1e-6 and r["n"] == 300            # a clearly better
    # white-noise differential, bandwidth 0: equals the one-sample t statistic
    d = pd.Series(rng.normal(0.1, 1, 300), index=days)
    r0 = stats.diebold_mariano(d, d * 0, bandwidth=0)
    t = d.mean() / (d.std(ddof=0) / np.sqrt(300))
    assert np.isclose(r0["stat"], t)
    # a gap in the calendar must not turn distant days into neighbours
    gap = d.drop(days[100:200])
    rg = stats.diebold_mariano(gap, gap * 0)
    assert rg["n"] == 200 and np.isfinite(rg["stat"])
    # daily aggregation in local time
    times = pd.date_range("2025-06-01 01:00", periods=48, freq="h", tz="UTC")
    loss = stats.daily_loss(np.r_[np.ones(24), 2 * np.ones(24)], times, utc_offset_hours=0)
    assert len(loss) == 3 and np.isclose(loss.iloc[0], 1.0)              # hour-end labels: 23 h + 1 h spill
    shifted = stats.daily_loss(np.r_[np.ones(24), 2 * np.ones(24)], times, utc_offset_hours=-1)
    assert len(shifted) == 2 and np.allclose(shifted.to_numpy(), [1.0, 4.0])


def test_spearman_ci():
    rng = np.random.default_rng(3)
    x = np.round(rng.random(22), 1)                                      # ties on purpose
    y = np.round(x + rng.normal(0, 0.4, 22), 1)
    r = stats.spearman_ci(x, y, n_boot=2000, seed=0)
    rho, p = sps.spearmanr(x, y)
    assert np.isclose(r["rho"], rho) and np.isclose(r["p"], p) and r["n"] == 22
    boots = []                                                           # the same resamples, one scipy call each
    for idx in np.random.default_rng(0).integers(0, 22, size=(2000, 22)):
        if np.unique(x[idx]).size > 1 and np.unique(y[idx]).size > 1:
            boots.append(sps.spearmanr(x[idx], y[idx])[0])
    assert np.allclose(r["ci"], np.quantile(boots, [0.025, 0.975]))
    assert r["ci"][0] <= r["rho"] <= r["ci"][1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all statistics tests passed")
