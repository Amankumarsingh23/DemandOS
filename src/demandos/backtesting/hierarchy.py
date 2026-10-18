"""Small-fixture hierarchy and explicitly as-of revenue-weight utilities.

These are reusable building blocks, NOT an official full M5 WRMSSE scorer.
"""

import json
import numpy as np
import pandas as pd


LEVELS = ((), ("state_id",), ("store_id",), ("cat_id",), ("dept_id",),
          ("state_id", "cat_id"), ("state_id", "dept_id"), ("store_id", "cat_id"),
          ("store_id", "dept_id"), ("item_id",), ("state_id", "item_id"), ("store_id", "item_id"))


def aggregate_hierarchy(bottom):
    required = {"date", "units_sold", "state_id", "store_id", "cat_id", "dept_id", "item_id"}
    if not required.issubset(bottom) or bottom[list(required)].isna().any().any():
        raise ValueError("Complete bottom-grain sales/metadata required")
    if bottom.duplicated(["store_id", "item_id", "date"]).any():
        raise ValueError("Duplicate bottom-grain keys")
    if not np.isfinite(bottom.units_sold.to_numpy(float)).all() or bottom.units_sold.lt(0).any():
        raise ValueError("Invalid bottom-grain sales")
    for key, attributes in (("store_id", ["state_id"]), ("item_id", ["cat_id", "dept_id"]), ("dept_id", ["cat_id"])):
        if bottom.groupby(key)[attributes].nunique().gt(1).any().any():
            raise ValueError("Inconsistent hierarchy metadata")
    frames = []
    for level, keys in enumerate(LEVELS, 1):
        grouped = bottom.groupby([*keys, "date"], as_index=False).units_sold.sum()
        grouped["series_id"] = [json.dumps([str(row[k]) for k in keys], separators=(",", ":")) for _, row in grouped.iterrows()]
        grouped["level"] = level
        frames.append(grouped[["level", "series_id", "date", "units_sold"]])
    return pd.concat(frames, ignore_index=True)


def historical_revenue_weights(observations, origin, days=28):
    """Require verified price availability metadata, never infer known-ahead prices."""
    required = ["series_id", "date", "units_sold", "sell_price", "price_available_at"]
    if not set(required).issubset(observations) or type(days) is not int or days < 1:
        raise ValueError("Verified as-of price fields and positive window required")
    origin = pd.Timestamp(origin)
    dates = pd.to_datetime(observations.date)
    prefix = observations.loc[dates.between(origin - pd.Timedelta(days=days - 1), origin), required].copy()
    if prefix.empty or prefix.duplicated(["series_id", "date"]).any():
        raise ValueError("Empty or duplicate historical price observations")
    if not prefix.groupby("series_id").date.nunique().eq(days).all():
        raise ValueError("Complete daily historical weighting window required")
    if prefix[["series_id", "date", "units_sold"]].isna().any().any() or not np.isfinite(prefix.units_sold.to_numpy(float)).all() or prefix.units_sold.lt(0).any():
        raise ValueError("Invalid historical observations")
    positive = prefix.units_sold.gt(0)
    available = pd.to_datetime(prefix.price_available_at)
    prices = pd.to_numeric(prefix.sell_price, errors="coerce")
    valid = prices.notna() & np.isfinite(prices) & prices.ge(0) & available.notna() & available.le(origin)
    if (positive & ~valid).any():
        raise ValueError("Positive sales require verified origin-known historical prices")
    prefix["revenue"] = np.where(positive, prefix.units_sold * prices, 0.)
    result = prefix.groupby("series_id", as_index=False).revenue.sum()
    total = result.revenue.sum()
    result["weight"] = result.revenue / total if total > 0 else np.nan
    result["weight_status"] = "ok" if total > 0 else "zero_revenue_total"
    return result


def cohort_weighted_rmsse(scores, weights):
    if scores.series_id.duplicated().any() or weights.series_id.duplicated().any() or set(scores.series_id) != set(weights.series_id):
        raise ValueError("Weights and scores must have identical unique series")
    joined = scores.merge(weights, on="series_id", validate="one_to_one")
    if not np.isfinite(joined.weight).all() or joined.weight.lt(0).any() or not np.isclose(joined.weight.sum(), 1):
        raise ValueError("Finite nonnegative weights summing to one required")
    included = joined[joined.weight.gt(0)].copy()
    included["rmsse"] = pd.to_numeric(included.rmsse, errors="coerce")
    if not np.isfinite(included.rmsse).all() or included.rmsse.lt(0).any():
        return {"cohort_weighted_rmsse": None, "status": "undefined_positive_weight_scale"}
    return {"cohort_weighted_rmsse": float((included.weight * included.rmsse).sum()), "status": "ok"}
