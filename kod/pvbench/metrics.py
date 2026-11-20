"""Forecast accuracy metrics for PV power (inputs in kW unless stated otherwise).

All functions ignore pairs where either value is NaN. Capacity-normalised
metrics are returned in percent of rated capacity.
"""

from __future__ import annotations

import numpy as np


def _clean(y, yhat, mask=None):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    keep = np.isfinite(y) & np.isfinite(yhat)
    if mask is not None:
        keep &= np.asarray(mask, bool)
    return y[keep], yhat[keep]


def rmse(y, yhat, mask=None):
    y, yhat = _clean(y, yhat, mask)
    return float(np.sqrt(np.mean((yhat - y) ** 2))) if y.size else np.nan


def mae(y, yhat, mask=None):
    y, yhat = _clean(y, yhat, mask)
    return float(np.mean(np.abs(yhat - y))) if y.size else np.nan


def mbe(y, yhat, mask=None):
    """Mean bias error; positive values mean over-forecasting."""
    y, yhat = _clean(y, yhat, mask)
    return float(np.mean(yhat - y)) if y.size else np.nan


def r2(y, yhat, mask=None):
    """Coefficient of determination. Algebraically identical to the
    Nash-Sutcliffe efficiency (NSE) when computed on forecasts."""
    y, yhat = _clean(y, yhat, mask)
    if y.size < 2:
        return np.nan
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - yhat) ** 2) / ss_tot) if ss_tot > 0 else np.nan


def mape(y, yhat, capacity, threshold=0.05, mask=None):
    """MAPE (%) restricted to targets >= threshold * capacity, which removes
    night-time zeros and unstable near-zero denominators."""
    y, yhat = _clean(y, yhat, mask)
    keep = y >= threshold * capacity
    return float(100.0 * np.mean(np.abs((yhat[keep] - y[keep]) / y[keep]))) if keep.any() else np.nan


def theil_u2(y, yhat, naive, mask=None):
    """Theil's U2: RMSE of the model divided by the RMSE of the no-change
    (persistence) forecast issued at the same origin for the same horizon.
    U2 < 1 means the model beats persistence."""
    y, yhat, naive = np.asarray(y, float), np.asarray(yhat, float), np.asarray(naive, float)
    keep = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(naive)
    if mask is not None:
        keep &= np.asarray(mask, bool)
    if not keep.any():
        return np.nan
    denom = np.sqrt(np.mean((naive[keep] - y[keep]) ** 2))
    return float(np.sqrt(np.mean((yhat[keep] - y[keep]) ** 2)) / denom) if denom > 0 else np.nan


def skill_score(model_rmse, reference_rmse):
    """Forecast skill, 1 - RMSE_model / RMSE_reference (positive is better)."""
    return float(1.0 - model_rmse / reference_rmse) if reference_rmse and np.isfinite(reference_rmse) else np.nan


def mase_scale(train_series, season=24):
    """In-sample MAE of the seasonal-naive forecast, used as the MASE denominator."""
    x = np.asarray(train_series, float)
    d = np.abs(x[season:] - x[:-season])
    d = d[np.isfinite(d)]
    return float(d.mean()) if d.size else np.nan


def evaluate(y, yhat, capacity, daytime=None, naive=None, scale=None, mape_thresholds=(0.02, 0.05, 0.10)):
    """Full metric set for one model on one plant.

    Returns a flat dict. Keys without a suffix use all test hours; keys with
    the ``_day`` suffix use only daytime hours (``daytime`` boolean mask).
    """
    out = {}
    for suffix, mask in (("", None), ("_day", daytime)):
        if suffix and daytime is None:
            continue
        e_rmse, e_mae = rmse(y, yhat, mask), mae(y, yhat, mask)
        out[f"n{suffix}"] = int(_clean(y, yhat, mask)[0].size)
        out[f"rmse{suffix}"], out[f"mae{suffix}"] = e_rmse, e_mae
        out[f"nrmse{suffix}"], out[f"nmae{suffix}"] = 100 * e_rmse / capacity, 100 * e_mae / capacity
        out[f"nmbe{suffix}"] = 100 * mbe(y, yhat, mask) / capacity
        out[f"r2{suffix}"] = r2(y, yhat, mask)
        if naive is not None:
            out[f"theil_u2{suffix}"] = theil_u2(y, yhat, naive, mask)
        if scale:
            out[f"mase{suffix}"] = e_mae / scale
    for thr in mape_thresholds:
        out[f"mape_{int(round(thr * 100)):02d}"] = mape(y, yhat, capacity, thr)
    return out
