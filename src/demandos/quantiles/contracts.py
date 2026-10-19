"""Exact daily keys; raw outputs remain separate from operational quantiles."""

import numpy as np
import pandas as pd

from demandos.features.contracts import KEYS

LEVELS = (.1, .5, .9)
Q = ["q10", "q50", "q90"]
CASE = [*KEYS, "target_date"]


def levels(values=LEVELS):
    a = np.asarray(values, float)
    if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all() or not ((a > 0) & (a < 1)).all() or (np.diff(a) <= 0).any():
        raise ValueError("Strictly ordered unique quantile levels in (0,1) required")
    return a


def crossing(values):
    a = np.asarray(values, float)
    if a.ndim < 2 or a.shape[-1] != 3 or not np.isfinite(a).all():
        raise ValueError("Finite three-quantile predictions required")
    return np.stack([a[..., 0] > a[..., 1], a[..., 1] > a[..., 2], a[..., 0] > a[..., 2]], axis=-1)


def repair(values):
    crossing(values)
    return np.sort(np.maximum(np.asarray(values, float), 0), axis=-1)


def validate(frame, *, operational=True):
    required = CASE+Q+["model", "fit_cutoff", "fit_target_max"]
    if frame.empty or not set(required).issubset(frame) or frame[required].isna().any().any():
        raise ValueError("Complete forecast keys, quantiles and fit provenance required")
    if frame.duplicated([*CASE, "model"]).any():
        raise ValueError("Duplicate quantile forecast grain")
    origin, target = pd.to_datetime(frame.origin_date), pd.to_datetime(frame.target_date)
    fitted, label_max = pd.to_datetime(frame.fit_cutoff), pd.to_datetime(frame.fit_target_max)
    if (not origin.eq(origin.dt.normalize()).all() or not target.eq(target.dt.normalize()).all()
            or not frame.horizon.isin(range(1, 29)).all()
            or not target.eq(origin+pd.to_timedelta(frame.horizon, unit="D")).all()
            or not fitted.le(origin).all() or not label_max.le(fitted).all()):
        raise ValueError("Invalid daily horizon or future-fitted model provenance")
    if frame.groupby(["store_id", "item_id", "origin_date", "model"]).horizon.nunique().ne(28).any():
        raise ValueError("Every successful series-origin needs 28 horizons")
    values = frame[Q].to_numpy(float)
    bad = crossing(values)
    if operational and (bad.any() or (values < 0).any()):
        raise ValueError("Operational quantiles require 0 <= P10 <= P50 <= P90")
    return frame


def forecast_rows(rows, predictions, model, fit_cutoff, fit_target_max):
    a = np.asarray(predictions, float)
    if a.shape != (len(rows), 28, 3) or not np.isfinite(a).all():
        raise ValueError("Expected aligned finite [B,28,3] forecast tensor")
    frame = rows.loc[rows.index.repeat(28), ["store_id", "item_id", "origin_date"]].reset_index(drop=True)
    frame["horizon"] = np.tile(np.arange(1, 29), len(rows))
    frame["target_date"] = frame.origin_date+pd.to_timedelta(frame.horizon, unit="D")
    frame["model"], frame["fit_cutoff"], frame["fit_target_max"] = model, pd.Timestamp(fit_cutoff), pd.Timestamp(fit_target_max)
    frame[Q] = a.reshape(-1, 3)
    return validate(frame, operational=False)


def matched(frame, models):
    """Hard complete coverage, not a silent intersection of successful rows."""
    if not models or len(set(models)) != len(models) or set(frame.model) != set(models):
        raise ValueError("Exact distinct comparison models required")
    validate(frame, operational=False)
    if frame.groupby(CASE).model.nunique().ne(len(models)).any():
        raise ValueError("Unmatched forecast cases between required methods")
    if frame.groupby(CASE).actual.nunique(dropna=False).ne(1).any():
        raise ValueError("Compared methods have different actuals")
    return frame
