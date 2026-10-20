"""Corrected order statistics and target-free support/empty-set handling."""

from decimal import Decimal, ROUND_CEILING
import numpy as np


def scores(actual, lower, upper, point=None):
    y, lo, hi = [np.asarray(a, float) for a in (actual, lower, upper)]
    if (y.shape != lo.shape or y.shape != hi.shape or not np.isfinite([y, lo, hi]).all()
            or (y < 0).any() or (lo > hi).any()):
        raise ValueError("Finite aligned nonnegative observations and ordered bases required")
    signed = np.maximum(lo-y, y-hi)
    result = {"cqr_signed": signed, "cqr_nonnegative": np.maximum(signed, 0)}
    if point is not None:
        point = np.asarray(point, float)
        if point.shape != y.shape or not np.isfinite(point).all():
            raise ValueError("Finite aligned point predictions required")
        result["absolute"] = np.abs(y-point)
    return result


def corrected(values, alpha=.2):
    a = np.asarray(values, float)
    if a.ndim != 1 or not np.isfinite(a).all() or not 0 < alpha < 1:
        raise ValueError("Finite one-dimensional scores and alpha in (0,1) required")
    n = len(a)
    k = int(((Decimal(n)+1)*(Decimal(1)-Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))
    if not 1 <= k <= n+1:
        raise ValueError("Invalid finite-sample conformal rank")
    threshold = float(np.partition(a, k-1)[k-1]) if k <= n else float("inf")
    return threshold, {"n": n, "rank": k, "alpha": float(alpha),
                       "status": "finite" if k <= n else "extended_infinite", "heuristic": False}


def weighted(values, weights, alpha=.2):
    a, w = np.asarray(values, float), np.asarray(weights, float)
    if (a.ndim != 1 or not len(a) or a.shape != w.shape or not np.isfinite(a).all()
            or not np.isfinite(w).all() or (w < 0).any() or not w.sum() > 0 or not 0 < alpha < 1):
        raise ValueError("Finite scores and nonnegative aligned nonzero weights required")
    order = np.argsort(a, kind="stable")
    normalized = w[order]/w.sum()
    position = min(int(np.searchsorted(np.cumsum(normalized), 1-alpha, side="left")), len(a)-1)
    return float(a[order[position]]), {"n": len(a), "rank": None, "alpha": float(alpha),
        "weight_sum": float(w.sum()), "kish_effective_n": float(w.sum()**2/np.square(w).sum()),
        "status": "finite", "heuristic": True, "finite_sample_rank_correction": False}


def intervals(lower, upper, correction):
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != hi.shape or not np.isfinite([lo, hi]).all() or (lo < 0).any() or (lo > hi).any():
        raise ValueError("Finite ordered nonnegative base endpoints required")
    c = np.broadcast_to(np.asarray(correction, float), lo.shape)
    if np.isnan(c).any() or np.isneginf(c).any():
        raise ValueError("Finite or positive-infinite conformal correction required")
    # A signed CQR empty set is enlarged to the midpoint singleton, without targets.
    effective = np.maximum(c, -(hi-lo)/2)
    left, right = np.maximum(lo-effective, 0), hi+effective
    collapsed = c <= -(hi-lo)/2
    midpoint = lo+(hi-lo)/2
    left, right = np.where(collapsed, midpoint, left), np.where(collapsed, midpoint, right)
    if (left > right+1e-10).any() or np.isnan(left).any() or np.isnan(right).any():
        raise ValueError("Invalid calibrated interval")
    return left, right, effective


def update_alpha(current, miss, target=.2, gamma=.001, bounds=(.02, .5)):
    if (not 0 < target < 1 or not 0 < gamma < 1 or not 0 < bounds[0] < bounds[1] < 1
            or not bounds[0] <= current <= bounds[1] or miss not in [0, 1, False, True]):
        raise ValueError("Invalid bounded ACI state/update")
    proposed = current+gamma*(target-int(miss))
    result = float(np.clip(proposed, *bounds))
    return result, proposed != result
