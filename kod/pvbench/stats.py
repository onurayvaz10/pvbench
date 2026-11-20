"""Statistical comparison of forecasting models across plants.

Conventions: score matrices have one row per plant (block) and one column
per model, and lower scores are better (e.g. nRMSE).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvalues):
    """Holm step-down adjusted p-values (same order as the input).

    Undefined p-values (NaN, e.g. a comparison in which every pair is tied) stay undefined
    and do not count as members of the family.
    """
    p = np.asarray(pvalues, float)
    adjusted = np.full(p.size, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    m = finite.size
    running = 0.0
    for rank, idx in enumerate(finite[np.argsort(p[finite], kind="stable")]):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def bootstrap_ci(values, statistic=np.mean, n_boot=10000, level=0.95, seed=0):
    """Percentile bootstrap confidence interval for a statistic of paired differences."""
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    samples = statistic(x[rng.integers(0, x.size, size=(n_boot, x.size))], axis=1)
    lo, hi = np.quantile(samples, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def paired_wilcoxon(a, b, seed=0):
    """Two-sided Wilcoxon signed-rank test on paired plant-level scores a - b.

    Returns the test statistic W = min(W+, W-), the p-value, the matched-pairs
    rank-biserial correlation r = (W+ - W-) / (W+ + W-) and summary statistics.
    With lower-is-better scores, negative differences and negative r favour `a`.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    nz = d[d != 0]
    out = {"n": int(d.size), "n_nonzero": int(nz.size), "wins_a": int((d < 0).sum()),
           "ties": int((d == 0).sum()), "wins_b": int((d > 0).sum()),
           "mean_diff": float(d.mean()) if d.size else np.nan,
           "median_diff": float(np.median(d)) if d.size else np.nan}
    ci = {"mean_ci": bootstrap_ci(d, np.mean, seed=seed) if d.size else (np.nan, np.nan),
          "median_ci": bootstrap_ci(d, np.median, seed=seed) if d.size else (np.nan, np.nan)}
    if nz.size < 1:                                   # every pair tied: the test is undefined
        return {**out, "W": np.nan, "p": np.nan, "rank_biserial": np.nan, **ci}
    ranks = stats.rankdata(np.abs(nz))
    w_plus, w_minus = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    res = stats.wilcoxon(nz, alternative="two-sided")
    out.update({"W": float(min(w_plus, w_minus)), "p": float(res.pvalue),
                "rank_biserial": float((w_plus - w_minus) / (w_plus + w_minus)), **ci})
    return out


def compare_to_reference(scores: pd.DataFrame, reference: str, seed=0) -> pd.DataFrame:
    """Wilcoxon tests of `reference` against every other model, Holm-adjusted."""
    rows = []
    for model in scores.columns:
        if model == reference:
            continue
        r = paired_wilcoxon(scores[reference].to_numpy(), scores[model].to_numpy(), seed=seed)
        rows.append({"reference": reference, "model": model, **r})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].to_numpy())
    return table.sort_values("p").reset_index(drop=True)


def nemenyi_critical_difference(k, n, alpha=0.05):
    """Critical difference of mean ranks for the Nemenyi post-hoc test (Demsar, 2006)."""
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2.0)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def friedman_nemenyi(scores: pd.DataFrame, alpha=0.05):
    """Friedman test with Iman-Davenport correction and Nemenyi critical difference.

    Rows with any missing value are dropped. Rank 1 is the best (lowest) score.
    """
    clean = scores.dropna()
    n, k = clean.shape
    ranks = clean.rank(axis=1, method="average")
    chi2, p = stats.friedmanchisquare(*[clean[c].to_numpy() for c in clean.columns])
    f_stat = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) != chi2 else np.inf
    p_f = float(stats.f.sf(f_stat, k - 1, (k - 1) * (n - 1)))
    return {"n_plants": int(n), "n_models": int(k), "chi2": float(chi2), "p": float(p),
            "iman_davenport_F": float(f_stat), "p_iman_davenport": p_f,
            "mean_ranks": ranks.mean().sort_values(),
            "critical_difference": nemenyi_critical_difference(k, n, alpha)}


def daily_loss(errors, times, utc_offset_hours=0, power=2) -> pd.Series:
    """Mean of |error|^power over the valid target hours of each local calendar day.

    `times` are UTC hour-end labels of the targets; the day is taken in local solar time
    (UTC + utc_offset_hours) so that one daylight period falls into one day.
    """
    e = np.asarray(errors, float)
    day = (pd.DatetimeIndex(times) + pd.Timedelta(hours=utc_offset_hours)).tz_localize(None).floor("D")
    loss = pd.Series(np.abs(e) ** power, index=day)
    return loss[np.isfinite(e)].groupby(level=0).mean()


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, bandwidth=None):
    """Diebold-Mariano test of equal predictive accuracy on one plant's test period.

    `loss_*` are per-day losses (see `daily_loss`) indexed by calendar day.  Working on daily
    losses removes the diurnal cycle of hourly PV errors and makes the lag structure
    well defined when quality control has removed single hours.  The long-run variance of
    the loss differential uses the Bartlett kernel (Newey-West), which cannot be negative;
    days missing from the calendar are treated as missing observations (their centred
    differential is set to zero and sums are divided by the number of observed days), so a
    lag always means that number of calendar days.  Default bandwidth: the Newey-West rule
    floor(4 (n/100)^(2/9)).  Two-sided p-value from Student's t with n-1 degrees of freedom.
    Negative statistics favour model a.
    """
    d = (loss_a - loss_b).dropna()
    n = int(d.size)
    if n < 10:
        return {"stat": np.nan, "p": np.nan, "n": n, "mean_diff": np.nan, "bandwidth": np.nan}
    grid = pd.date_range(d.index.min(), d.index.max(), freq="D")
    centred = (d - d.mean()).reindex(grid).fillna(0.0).to_numpy()
    m = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))) if bandwidth is None else int(bandwidth)
    m = max(0, min(m, len(centred) - 1))
    lrv = np.sum(centred ** 2) / n
    for lag in range(1, m + 1):
        lrv += 2.0 * (1.0 - lag / (m + 1.0)) * np.sum(centred[lag:] * centred[:-lag]) / n
    if not lrv > 0:
        return {"stat": np.nan, "p": np.nan, "n": n, "mean_diff": float(d.mean()), "bandwidth": m}
    stat = d.mean() / np.sqrt(lrv / n)
    return {"stat": float(stat), "p": float(2 * stats.t.sf(abs(stat), df=n - 1)), "n": n,
            "mean_diff": float(d.mean()), "bandwidth": m}


def spearman_ci(x, y, n_boot=10000, level=0.95, seed=0):
    """Spearman correlation with a percentile bootstrap confidence interval."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    # Spearman = Pearson correlation of the average ranks; all resamples are ranked in one call
    # (resamples in which x or y is constant have no correlation and are left out)
    rx, ry = stats.rankdata(x[idx], axis=1), stats.rankdata(y[idx], axis=1)
    rx, ry = rx - rx.mean(axis=1, keepdims=True), ry - ry.mean(axis=1, keepdims=True)
    scale = np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    boots = (rx * ry).sum(axis=1)[scale > 0] / scale[scale > 0]
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"rho": float(rho), "p": float(p), "ci": (float(lo), float(hi)), "n": int(x.size)}
