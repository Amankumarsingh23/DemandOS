"""Train-only M5-style RMSSE scales, explicit undefined values and paired losses."""

import numpy as np
import pandas as pd

from demandos.forecasting.evaluation import metrics


CASE = ["window", "series_id", "origin", "target_date", "horizon"]


def rmsse_scale(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Scale requires finite nonnegative training history")
    positive = np.flatnonzero(values > 0)
    if not len(positive):
        return {"scale": None, "scale_status": "no_positive_history", "scale_pairs": 0, "first_positive_index": None}
    first = int(positive[0])
    trimmed = values[first:]
    scale = float(np.mean(np.diff(trimmed) ** 2)) if len(trimmed) > 1 else None
    status = "insufficient_after_first_positive" if scale is None else "zero_scale" if scale == 0 else "ok"
    return {"scale": scale, "scale_status": status, "scale_pairs": max(0, len(trimmed) - 1), "first_positive_index": first}


def point_metrics(actual, prediction, scale):
    result = metrics(actual, prediction)
    mse = float(np.mean((np.asarray(actual) - np.asarray(prediction)) ** 2))
    valid = scale is not None and np.isfinite(scale) and scale > 0
    result.update(mse=mse, rmsse=float(np.sqrt(mse / scale)) if valid else None,
                  rmsse_status="ok" if valid else "undefined_training_scale",
                  wape_status="ok" if result["wape"] is not None else "zero_actual_total")
    return result


def score(frame):
    """Pool rows for micro losses; macro units are series-origin cases within group.

    Mean case RMSSE and sqrt(mean scaled point loss) are intentionally distinct.
    All sample counts remain visible when zero denominators exclude scaled losses.
    """
    if frame.empty:
        raise ValueError("Cannot score an empty comparison")
    row = summarize(frame.assign(_summary_group="all"), ["_summary_group"]).iloc[0].drop("_summary_group")
    return {key: None if pd.isna(value) else value for key, value in row.items()}


def summarize(frame, groups):
    if frame.empty:
        return pd.DataFrame(columns=groups)
    # Aggregate all strata together, rather than repeatedly creating per-case frames.
    metrics(frame.actual.to_numpy(float), frame.prediction.to_numpy(float))
    source = frame.copy()
    source["_abs"] = np.abs(source.actual - source.prediction)
    source["_sq"] = (source.actual - source.prediction) ** 2
    source["scale"] = pd.to_numeric(source.scale, errors="raise")
    source["_scaled"] = source._sq / source.scale.where(source.scale.gt(0))
    case_keys = list(dict.fromkeys([*groups, "window", "series_id", "origin"]))
    grouped = source.groupby(case_keys, dropna=False, sort=True)
    if grouped.scale.nunique(dropna=False).gt(1).any():
        raise ValueError("One train-only scale required per case")
    cases = grouped.agg(_n=("actual", "size"), _actual=("actual", "sum"), _abs=("_abs", "sum"),
                        _sq=("_sq", "sum"), _scale=("scale", "first")).reset_index()
    cases["_mae"] = cases._abs / cases._n
    cases["_wape"] = cases._abs / cases._actual.where(cases._actual.gt(0))
    cases["_rmsse"] = np.sqrt(cases._sq / cases._n / cases._scale.where(cases._scale.gt(0)))
    macro = cases.groupby(groups, dropna=False, sort=True).agg(
        cases=("_n", "size"), macro_mae=("_mae", "mean"), macro_wape=("_wape", "mean"),
        macro_wape_cases=("_wape", "count"), macro_rmsse=("_rmsse", "mean"), rmsse_cases=("_rmsse", "count")).reset_index()
    pooled = source.groupby(groups, dropna=False, sort=True).agg(
        observations=("actual", "size"), mae=("_abs", "mean"), actual_units=("actual", "sum"),
        absolute_error=("_abs", "sum"), series=("series_id", "nunique"), origins=("origin", "nunique"),
        _scaled_sum=("_scaled", "sum"), rmsse_observations=("_scaled", "count")).reset_index()
    result = pooled.merge(macro, on=groups, validate="one_to_one")
    result["wape"] = result.absolute_error / result.actual_units.where(result.actual_units.gt(0))
    result["wape_status"] = np.where(result.actual_units.gt(0), "ok", "zero_actual_total")
    result["undefined_rmsse_cases"] = result.cases - result.rmsse_cases
    result["rmsse_status"] = np.where(result.rmsse_cases.eq(0), "undefined_training_scale",
                                      np.where(result.undefined_rmsse_cases.gt(0), "partial_undefined_scales", "ok"))
    result["micro_scaled_rmse"] = np.sqrt(result._scaled_sum / result.rmsse_observations.where(result.rmsse_observations.gt(0)))
    return result.drop(columns="_scaled_sum")


def common_cases(frame, models):
    if not models or len(set(models)) != len(models):
        raise ValueError("Nonempty unique comparison model list required")
    selected = frame[frame.model.isin(models) & frame.status.eq("ok")].copy()
    if selected.duplicated(CASE + ["model"]).any():
        raise ValueError("Duplicate forecast comparison keys")
    invariant = ["actual"] + (["scale"] if "scale" in selected else [])
    if selected.groupby(CASE)[invariant].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Comparison models must share actual targets and training scales")
    counts = selected.groupby(CASE).model.nunique()
    common = counts[counts.eq(len(models))].reset_index()[CASE]
    return selected.merge(common, on=CASE, how="inner", validate="many_to_one")


def paired_losses(frame, baseline="seasonal_naive"):
    """Signed losses = candidate minus baseline; negative favors candidate."""
    records = []
    for model in sorted(set(frame.model) - {baseline}):
        common = common_cases(frame, [baseline, model])
        candidate = common[common.model.eq(model)]
        reference = common[common.model.eq(baseline)][CASE + ["actual", "prediction"]]
        joined = candidate.merge(reference, on=CASE, suffixes=("", "_baseline"), validate="one_to_one")
        if not joined.actual.eq(joined.actual_baseline).all():
            raise ValueError("Paired models have different actual targets")
        if joined.empty:
            continue
        joined["absolute_loss_difference"] = np.abs(joined.actual - joined.prediction) - np.abs(joined.actual - joined.prediction_baseline)
        joined["squared_loss_difference"] = (joined.actual - joined.prediction) ** 2 - (joined.actual - joined.prediction_baseline) ** 2
        joined["baseline"] = baseline
        records.append(joined)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=CASE + ["kind", "model", "baseline", "absolute_loss_difference", "squared_loss_difference"])


def horizon_band(horizons):
    values = np.asarray(horizons)
    if not np.isin(values, np.arange(1, 29)).all():
        raise ValueError("Horizon bands require days 1..28")
    return np.where(values <= 7, "01-07", np.where(values <= 14, "08-14", "15-28"))
