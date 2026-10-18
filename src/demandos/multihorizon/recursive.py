"""Closed-loop one-step predictions; future actuals are never accepted as history."""

import numpy as np
import pandas as pd

from demandos.boosting.models import validate_rows
from demandos.features.contracts import KEYS
from .vector import ORIGIN_KEYS


def history_prefix(daily, keys, origin, lookback):
    required = ["store_id", "item_id", "date", "units_sold"]
    if not set(required).issubset(daily):
        raise ValueError("Dated observed series required")
    origin = pd.Timestamp(origin)
    selected = daily.loc[pd.to_datetime(daily.date).between(origin - pd.Timedelta(days=lookback), origin), required].copy()
    selected = selected.merge(keys[["store_id", "item_id"]].drop_duplicates(), on=["store_id", "item_id"], validate="many_to_one")
    if selected.empty or selected.duplicated(["store_id", "item_id", "date"]).any():
        raise ValueError("Unique observed daily prefix required")
    values = selected.units_sold.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Observed prefix contains invalid sales")
    return selected.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)


def update_history_features(template, histories, dates, simulated_origin, config):
    """Recompute every sales-history predictor in float, retaining fractional feedback."""
    frame = template.copy()
    for name in frame:
        if name == "sales_at_origin" or ((name.startswith("rolling_") or name.startswith("origin_lag_")) and not name.endswith("_source_date")):
            frame[name] = frame[name].astype("float64")
    origin = pd.Timestamp(simulated_origin)
    for idx, row in frame.iterrows():
        key = (row.store_id, row.item_id)
        values, when = np.asarray(histories[key], float), pd.DatetimeIndex(dates[key])
        expected_start = max(pd.Timestamp(row.first_observed_date), origin - pd.Timedelta(days=config.lookback))
        expected = pd.date_range(expected_start, origin)
        if not when.equals(expected) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Simulated history must be complete, finite, isolated and end at its virtual origin")
        changes = {"origin_date": origin, "horizon": 1, "target_date": origin + pd.Timedelta(days=1),
                   "bounded_history_start": when[0], "max_sales_source_date": origin,
                   "history_observations": len(values), "history_distinct_dates": len(values),
                   "history_invalid_rows": 0, "sales_at_origin": values[-1]}
        for lag in config.lags:
            known = len(values) > lag
            changes[f"origin_lag_{lag}"] = values[-lag-1] if known else np.nan
            changes[f"origin_lag_{lag}_source_date"] = origin - pd.Timedelta(days=lag) if known else pd.NaT
        for window in config.windows:
            tail = values[-window:]
            complete = len(tail) == window
            stats = {"mean": np.mean(tail), "sum": np.sum(tail), "std": np.std(tail, ddof=0),
                     "max": np.max(tail), "zero_fraction": np.mean(tail == 0), "positive_count": np.sum(tail > 0)}
            changes.update({f"rolling_{name}_{window}_at_origin": float(v) if complete else np.nan for name, v in stats.items()})
            changes[f"rolling_observations_{window}"] = len(tail)
        previous = np.mean(values[-14:-7]) if len(values) >= 14 else np.nan
        changes["mean_previous_7_at_origin"] = previous
        changes["level_change_7"] = changes["rolling_mean_7_at_origin"] - previous
        changes["relative_level_change_7"] = changes["level_change_7"] / previous if previous > 0 else np.nan
        for name, value in changes.items():
            frame.loc[idx, name] = value
    validate_rows(frame)
    return frame


def recursive_predict(model, inference, observed_prefix, feature_config):
    validate_rows(inference)
    if inference.origin_date.nunique() != 1:
        raise ValueError("Recursive call requires one issuance origin")
    origin = pd.Timestamp(inference.origin_date.iloc[0])
    if pd.to_datetime(observed_prefix.date).gt(origin).any():
        raise ValueError("Recursive inference accepts a pre-origin prefix only, never holdout actuals")
    if inference.groupby(ORIGIN_KEYS).horizon.size().ne(28).any():
        raise ValueError("Recursive output requires all 28 target calendars")
    prefix = history_prefix(observed_prefix, inference, origin, feature_config.lookback)
    histories, dates = {}, {}
    for key, group in prefix.groupby(["store_id", "item_id"], sort=True):
        histories[key] = group.units_sold.to_numpy(float).tolist()
        dates[key] = list(pd.to_datetime(group.date))
    results, trace = [], []
    for h in range(1, 29):
        # Target calendar is always the original issuance's scheduled day h.
        template = inference[inference.horizon.eq(h)].sort_values(["store_id", "item_id"]).reset_index(drop=True).copy()
        virtual = origin + pd.Timedelta(days=h-1)
        step = update_history_features(template, histories, dates, virtual, feature_config)
        if h == 1:
            np.testing.assert_allclose(model.matrix(step).select_dtypes('number'), model.matrix(template).select_dtypes('number'), rtol=1e-6, atol=1e-6, equal_nan=True)
        output = model.predict(step)
        pd.testing.assert_frame_equal(step[KEYS+["target_date"]].reset_index(drop=True),
                                      output[KEYS+["target_date"]].reset_index(drop=True))
        if not np.isfinite(output[["raw_prediction", "prediction"]].to_numpy(float)).all():
            raise ValueError("Nonfinite recursive forecast; no invalid feedback or partial vector")
        if not np.array_equal(output.prediction.to_numpy(), np.maximum(output.raw_prediction.to_numpy(), 0)):
            raise ValueError("Recursive feedback must follow the nonnegative raw-forecast policy")
        output["origin_date"], output["horizon"] = origin, h
        results.append(output)
        for i, point in output.iterrows():
            key = (point.store_id, point.item_id)
            trace.append({"store_id": point.store_id, "item_id": point.item_id, "origin_date": origin, "horizon": h,
                          "simulated_origin": virtual, "actual_source_max": origin,
                          "prior_predictions": h-1, "feedback_last_value": histories[key][-1],
                          "rolling_mean_7": step.loc[i, "rolling_mean_7_at_origin"],
                          "rolling_sum_7": step.loc[i, "rolling_sum_7_at_origin"],
                          "origin_lag_1": step.loc[i, "origin_lag_1"]})
            histories[key].append(float(point.prediction))
            dates[key].append(pd.Timestamp(point.target_date))
            histories[key] = histories[key][-(feature_config.lookback+1):]
            dates[key] = dates[key][-(feature_config.lookback+1):]
    return pd.concat(results, ignore_index=True).sort_values(KEYS).reset_index(drop=True), pd.DataFrame(trace)
