"""Daily marginal metrics and complete 28-day trajectory coverage stay distinct."""

import numpy as np
import pandas as pd
from demandos.quantiles.metrics import wilson


def losses(frame, alpha=.2):
    result = frame.copy()
    y, lo, hi = [result[c].to_numpy(float) for c in ["actual", "lower", "upper"]]
    if (not 0 < alpha < 1 or not np.isfinite(y).all() or (y < 0).any() or not np.isfinite(lo).all()
            or np.isnan(hi).any() or np.isneginf(lo).any() or (lo < 0).any() or (lo > hi+1e-10).any()):
        raise ValueError("Valid nonnegative-support intervals and observed targets required")
    result["covered"] = (y >= lo) & (y <= hi)
    result["lower_miss"], result["upper_miss"] = y < lo, y > hi
    result["width"] = hi-lo
    # Positive infinity is the explicit extended conformal interval policy.
    result["interval_score"] = hi-lo+(2/alpha)*np.maximum(lo-y, 0)+(2/alpha)*np.maximum(y-hi, 0)
    result["zero_included"] = lo == 0
    result["is_zero_target"] = y == 0
    result["finite_interval"] = np.isfinite(lo) & np.isfinite(hi)
    return result


def summarize(frame, groups, alpha=.2):
    rows = []
    for key, group in losses(frame, alpha).groupby(groups, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        count, covered = len(group), int(group.covered.sum())
        low, high = wilson(covered, count)
        zeros = group[group.is_zero_target]
        rows.append(dict(zip(groups, key), observations=count, covered=covered, coverage=covered/count,
                         coverage_deviation=covered/count-(1-alpha), width_mean=float(group.width.mean()),
                         width_median=float(group.width.median()), interval_score=float(group.interval_score.mean()),
                         lower_miss_rate=float(group.lower_miss.mean()), upper_miss_rate=float(group.upper_miss.mean()),
                         zero_targets=len(zeros), zero_sales_coverage=float(zeros.covered.mean()) if len(zeros) else None,
                         zero_inclusion_rate=float(group.zero_included.mean()), infinite_intervals=int((~group.finite_interval).sum()),
                         wilson_low=low, wilson_high=high))
    return pd.DataFrame(rows)


def trajectory_coverage(frame):
    source = losses(frame)
    keys = ["cohort", "method", "store_id", "item_id", "origin_date"]
    records = []
    for key, group in source.groupby(keys, sort=True):
        if len(group) != 28 or sorted(group.horizon.tolist()) != list(range(1, 29)):
            raise ValueError("Whole-trajectory coverage requires all 28 exact horizons")
        records.append(dict(zip(keys, key), all_28_covered=bool(group.covered.all())))
    cases = pd.DataFrame(records)
    return cases, cases.groupby(["cohort", "method"], sort=True).agg(trajectories=("all_28_covered", "size"),
        simultaneous_coverage=("all_28_covered", "mean")).reset_index()
