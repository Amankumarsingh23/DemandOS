"""Pinball, inclusive calibration and proper interval scoring with explicit counts."""

import numpy as np
import pandas as pd
from .contracts import LEVELS, Q, levels, crossing


def pinball(actual, prediction, tau):
    t = np.asarray(tau, float)
    if not np.isfinite(t).all() or not ((t > 0) & (t < 1)).all():
        raise ValueError("Quantile level must be in (0,1)")
    y, q = np.asarray(actual, float), np.asarray(prediction, float)
    if not np.isfinite(y).all() or not np.isfinite(q).all() or (y < 0).any():
        raise ValueError("Finite predictions and nonnegative observed targets required")
    error = y-q
    return np.maximum(t*error, (t-1)*error)


def interval_score(actual, lower, upper, alpha=.2):
    y, lo, hi = [np.asarray(a, float) for a in (actual, lower, upper)]
    if (not 0 < alpha < 1 or y.shape != lo.shape or y.shape != hi.shape
            or not np.isfinite([y, lo, hi]).all() or (y < 0).any() or (lo > hi).any()):
        raise ValueError("Finite aligned ordered interval and nonnegative target required")
    return hi-lo+(2/alpha)*np.maximum(lo-y, 0)+(2/alpha)*np.maximum(y-hi, 0)


def wilson(successes, n, z=1.959963984540054):
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Positive binomial count and valid success count required")
    p = successes/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0., center-radius), min(1., center+radius)


def row_losses(frame):
    result = frame.copy()
    y, q = result.actual.to_numpy(float), result[Q].to_numpy(float)
    p = pinball(y[:, None], q, levels())
    c = crossing(q)
    for i, name in enumerate(Q):
        result["pinball_"+name], result["coverage_"+name] = p[:, i], y <= q[:, i]
        result["ties_"+name] = y == q[:, i]
    for i, name in enumerate(["q10_gt_q50", "q50_gt_q90", "q10_gt_q90"]):
        result[name] = c[:, i]
    result["any_crossing"] = c.any(axis=1)
    result["negative_quantile"] = (q < 0).any(axis=1)
    result["pinball_mean"] = p.mean(axis=1)
    valid = q[:, 0] <= q[:, 2]
    result["valid_interval"] = valid
    result["interval_width"] = np.where(valid, q[:, 2]-q[:, 0], np.nan)
    result["interval_coverage"] = np.where(valid, (y >= q[:, 0]) & (y <= q[:, 2]), np.nan)
    score = np.full(len(result), np.nan)
    score[valid] = interval_score(y[valid], q[valid, 0], q[valid, 2])
    result["interval_score"] = score
    result["median_absolute_error"] = np.abs(y-q[:, 1])
    result["zero_width"] = valid & (q[:, 0] == q[:, 2])
    result["multiple_zero_quantiles"] = (q == 0).sum(axis=1) >= 2
    result["positive_target"] = y > 0
    return result


def summarize(frame, groups):
    source = row_losses(frame)
    records = []
    for key, group in source.groupby(groups, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        n = len(group)
        row = dict(zip(groups, key), observations=n, series=group[["store_id", "item_id"]].drop_duplicates().shape[0])
        for name in ["pinball_q10", "pinball_q50", "pinball_q90", "pinball_mean", "coverage_q10", "coverage_q50", "coverage_q90",
                     "interval_coverage", "interval_score", "median_absolute_error"]:
            row[name] = float(group[name].mean())
        for name in ["any_crossing", "q10_gt_q50", "q50_gt_q90", "q10_gt_q90", "negative_quantile", "zero_width", "multiple_zero_quantiles"]:
            row[name+"_count"] = int(group[name].sum())
            row[name+"_rate"] = float(group[name].mean())
        row.update(interval_rows=int(group.valid_interval.sum()), invalid_interval_rows=int((~group.valid_interval).sum()),
                   interval_width_mean=float(group.interval_width.mean()), interval_width_median=float(group.interval_width.median()),
                   actual_units=float(group.actual.sum()), positive_targets=int(group.positive_target.sum()))
        row["median_wape"] = float(group.median_absolute_error.sum()/group.actual.sum()) if group.actual.sum() > 0 else None
        for name, tau in zip(Q, LEVELS):
            count = int(group["coverage_"+name].sum())
            lo, hi = wilson(count, n)
            row.update({name+"_coverage_count": count, name+"_tie_count": int(group["ties_"+name].sum()),
                        name+"_nominal": tau, name+"_calibration_deviation": count/n-tau,
                        name+"_wilson_low": lo, name+"_wilson_high": hi})
        records.append(row)
    return pd.DataFrame(records)
