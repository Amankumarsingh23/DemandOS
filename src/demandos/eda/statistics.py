"""Explicit population statistics, sales-occurrence heuristics, and safe joins."""

from decimal import Decimal
import math

import numpy as np
import pandas as pd

from .config import EDAConfig


def sales_statistics(values):
    """Missing observations are counted separately and never replaced with zero."""
    data = pd.to_numeric(pd.Series(values), errors="raise").astype(float)
    observed = data.dropna()
    if not np.isfinite(observed).all() or observed.lt(0).any():
        raise ValueError("Sales must be finite and nonnegative when observed")
    positive = observed[observed > 0]
    n, positives = len(observed), len(positive)
    mean = float(observed.mean()) if n else None
    std = float(observed.std(ddof=0)) if n else None
    positive_cv2 = float((positive.std(ddof=0) / positive.mean()) ** 2) if positives >= 2 else None
    return {
        "observed_periods": n, "missing_periods": int(data.isna().sum()),
        "positive_periods": positives, "zero_fraction": float(observed.eq(0).mean()) if n else None,
        "mean": mean, "median": float(observed.median()) if n else None,
        "variance": std ** 2 if n else None, "std": std,
        "cv": std / mean if mean and mean > 0 else None,
        "adi": n / positives if positives else None, "positive_cv2": positive_cv2,
    }


def intermittent_class(periods, positives, cv2, config=EDAConfig()):
    if periods < 0 or positives < 0 or positives > periods:
        raise ValueError("Invalid observation counts")
    if periods <= 0:
        return "unclassified", "no_observations"
    if positives == 0:
        return "unclassified", "all_zero"
    if positives < config.min_positive_days or cv2 is None or pd.isna(cv2):
        return "unclassified", "insufficient_positive_history"
    if not math.isfinite(cv2) or cv2 < 0 or not 0 < positives <= periods:
        raise ValueError("Invalid intermittency inputs")
    sparse = periods / positives >= config.adi_boundary
    variable = cv2 >= config.positive_cv2_boundary
    return ("lumpy" if variable else "intermittent") if sparse else ("erratic" if variable else "smooth"), "classified"


def xyz_class(periods, positives, mean, std, config=EDAConfig()):
    if periods < 0 or positives < 0 or positives > periods:
        raise ValueError("Invalid weekly observation counts")
    if periods == 0 or mean is None or pd.isna(mean):
        return "unclassified", "no_complete_weeks"
    if not math.isfinite(mean) or mean < 0 or std is None or not math.isfinite(std) or std < 0:
        raise ValueError("Invalid weekly statistics")
    if mean == 0:
        return "unclassified", "all_zero"
    if periods < config.xyz_min_weeks:
        return "unclassified", "insufficient_weeks"
    if positives < config.xyz_min_positive_weeks:
        return "unclassified", "insufficient_positive_weeks"
    if mean < config.xyz_min_mean:
        return "unclassified", "low_mean"
    cv = std / mean
    return ("X" if cv <= config.xyz_x else "Y" if cv <= config.xyz_y else "Z"), "classified"


def abc_classification(products, config=EDAConfig()):
    """Product grain. Crossing item stays in the earlier class; ties use item_id."""
    result = products.copy(deep=True)
    if result.item_id.isna().any() or result.item_id.duplicated().any():
        raise ValueError("ABC requires one non-null row per item")
    result["known_revenue"] = result.known_revenue.map(lambda value: None if pd.isna(value) else Decimal(str(value)))
    if any(value is not None and (not value.is_finite() or value < 0) for value in result.known_revenue):
        raise ValueError("ABC revenue must be finite and nonnegative")
    result = result.sort_values(["known_revenue", "item_id"], ascending=[False, True], na_position="last").reset_index(drop=True)
    eligible = result.unknown_revenue_count.eq(0) & result.known_revenue.notna()
    total = sum(result.loc[eligible, "known_revenue"], Decimal(0))
    running = Decimal(0)
    records = []
    for index, row in result.iterrows():
        revenue = row.known_revenue
        if not eligible.iloc[index]:
            records.append(("unclassified", "incomplete_revenue", None, None))
        elif total == 0:
            records.append(("unclassified", "no_positive_total", None, None))
        else:
            before = running / total
            label = "C" if revenue == 0 else "A" if before < Decimal(str(config.abc_a)) else "B" if before < Decimal(str(config.abc_b)) else "C"
            running += revenue
            records.append((label, "classified", float(revenue / total), float(running / total)))
    result[["abc_class", "abc_reason", "revenue_share", "cumulative_revenue_share"]] = pd.DataFrame(records, index=result.index)
    return result


def combine_segments(abc, xyz):
    for frame in (abc, xyz):
        if frame.item_id.isna().any() or frame.item_id.duplicated().any():
            raise ValueError("ABC/XYZ join requires unique non-null item keys")
    if set(abc.item_id) != set(xyz.item_id):
        raise ValueError("ABC and XYZ must cover the same product universe")
    result = abc.merge(xyz, on="item_id", how="left", validate="one_to_one")
    result["abc_xyz"] = [a + x if a in "ABC" and x in ("X", "Y", "Z") else "unclassified"
                         for a, x in zip(result.abc_class, result.xyz_class)]
    return result


def prior_average(values, window=28):
    """Prior rows only; a one-step descriptive baseline, not a multi-horizon feature."""
    return pd.Series(values).shift(1).rolling(window, min_periods=1).mean()


def outlier_flags(frame, config=EDAConfig()):
    """Screen a selected daily series without removing or altering observations."""
    result = frame.sort_values("date").copy(deep=True).reset_index(drop=True)
    values = result.units_sold.astype(float)
    q1, q3 = values.quantile([0.25, 0.75])
    result["iqr_lower"] = q1 - config.iqr_multiplier * (q3 - q1)
    result["iqr_upper"] = q3 + config.iqr_multiplier * (q3 - q1)
    result["iqr_flag"] = values.lt(result.iqr_lower) | values.gt(result.iqr_upper)
    result["prior_average"] = prior_average(values, config.moving_average_days)
    weekdays = pd.to_datetime(result.date).dt.dayofweek
    result["robust_baseline"] = np.nan
    result["robust_mad"] = np.nan
    for _, indices in weekdays.groupby(weekdays).groups.items():
        history = values.loc[indices].shift(1).rolling(config.robust_same_weekday_weeks, min_periods=config.robust_min_history)
        result.loc[indices, "robust_baseline"] = history.median()
        result.loc[indices, "robust_mad"] = history.apply(lambda a: np.median(np.abs(a - np.median(a))), raw=True)
    result["robust_evaluable"] = result.robust_mad.gt(0) & result.robust_baseline.notna()
    result["robust_flag"] = result.robust_evaluable & ((values - result.robust_baseline).abs() > config.robust_z * 1.4826 * result.robust_mad)
    return result
