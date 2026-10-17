"""Fixed-origin point metrics. Comparable panels use intersections, never fallbacks."""

import numpy as np
import pandas as pd


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or not len(actual) or actual.shape != predicted.shape:
        raise ValueError("Metrics require nonempty aligned vectors")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or (actual < 0).any():
        raise ValueError("Invalid metric observations")
    absolute = np.abs(actual - predicted)
    denominator = float(actual.sum())
    return {"observations": len(actual), "mae": float(absolute.mean()),
            "wape": float(absolute.sum() / denominator) if denominator > 0 else None,
            "actual_units": denominator, "absolute_error": float(absolute.sum())}


def summarize(frame, groups):
    records = []
    for key, group in frame.groupby(groups, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        records.append(dict(zip(groups, key), series=group.series_id.nunique(),
                            **metrics(group.actual, group.prediction)))
    return pd.DataFrame(records)


def comparisons(forecasts, models):
    """Require every requested model on every compared series/date/horizon."""
    keys = ["series_id", "target_date", "horizon"]
    selected = forecasts[forecasts.model.isin(models)].copy()
    if selected.duplicated(keys + ["model"]).any():
        raise ValueError("Duplicate forecast keys")
    counts = selected.groupby(keys).model.nunique()
    common = counts[counts.eq(len(models))].reset_index()[keys]
    return selected.merge(common, on=keys, how="inner", validate="many_to_one")
