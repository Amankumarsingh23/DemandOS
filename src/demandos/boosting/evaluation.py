"""Exact-grain statistical baselines, shared-case scores and validation-only probes."""

from time import perf_counter
import numpy as np
import pandas as pd

from demandos.backtesting.metrics import rmsse_scale, summarize, common_cases, paired_losses, horizon_band, CASE
from demandos.forecasting.models import forecast, at_origin
from demandos.features.encoding import CATEGORICAL
from .models import TARGET


BASELINES = ("naive", "seasonal_naive", "moving_average", "tsb")
EVENT_FEATURES = ("target_event_count", "target_has_event", "target_event_cultural", "target_event_national", "target_event_religious", "target_event_sporting")


def statistical_predictions(data):
    predictions, fits, coverage = [], [], []
    for origin in data["origins"]:
        for row in data["evaluation_cohort"].itertuples():
            history = at_origin(data["frames"][row.series_id], origin)
            for model in BASELINES:
                result = forecast(history, 28, model)
                fit_id = f"{model}/{origin:%Y%m%d}/{row.series_id}"
                fits.append({"fit_id": fit_id, "model": model, "origin": origin, "phase": "baseline",
                             "status": result.status, "reason": result.reason, "runtime_seconds": result.runtime_seconds,
                             "warning_count": len(result.warnings), "warnings": str(result.warnings), "series": 1})
                coverage.append({"fit_id": fit_id, "model": model, "origin": origin, "series_id": row.series_id,
                                 "status": result.status, "reason": result.reason})
                if result.status == "ok":
                    for h, (raw, prediction) in enumerate(zip(result.raw_predictions, result.predictions), 1):
                        predictions.append({"store_id": row.store_id, "item_id": row.item_id, "origin_date": origin,
                                            "horizon": h, "target_date": origin + pd.Timedelta(days=h), "model": model,
                                            "raw_prediction": raw, "prediction": prediction})
    return pd.DataFrame(predictions), fits, coverage


def attach_targets(predictions, coverage, data, models):
    plan, scales = [], []
    for origin in data["origins"]:
        for row in data["evaluation_cohort"].itertuples():
            history = at_origin(data["frames"][row.series_id], origin)
            scale = rmsse_scale(history.to_numpy(float))
            scales.append({"series_id": row.series_id, "origin": origin, **scale})
            observed = data["frames"][row.series_id].set_index("date").units_sold
            for model in models:
                for h in range(1, 29):
                    target = origin + pd.Timedelta(days=h)
                    if target not in observed.index:
                        raise ValueError("Missing planned evaluation target")
                    plan.append({"store_id": row.store_id, "item_id": row.item_id, "series_id": row.series_id,
                                 "origin": origin, "target_date": target, "horizon": h, "model": model,
                                 "actual": observed.loc[target], "scale": scale["scale"], "scale_status": scale["scale_status"],
                                 "regime": row.regime, "abc_xyz": row.abc_xyz, "kind": "store_item", "window": "expanding"})
    frame = pd.DataFrame(plan)
    point_keys = ["store_id", "item_id", "origin", "target_date", "horizon", "model"]
    values = predictions.rename(columns={"origin_date": "origin"})
    if values.duplicated(point_keys).any():
        raise ValueError("Duplicate model prediction keys")
    planned = frame[point_keys]
    membership = values[point_keys].merge(planned, on=point_keys, how="left", indicator=True, validate="one_to_one")
    if membership._merge.ne("both").any():
        raise ValueError("Prediction contains an unplanned model/series/origin/target case")
    frame = frame.merge(values, on=point_keys, how="left", validate="one_to_one")
    status = pd.DataFrame(coverage)
    if status.duplicated(["series_id", "origin", "model"]).any():
        raise ValueError("Duplicate prediction-case status")
    if len(status) != len(frame[["series_id", "origin", "model"]].drop_duplicates()):
        raise ValueError("Fit coverage does not match the planned model-case grid")
    frame = frame.merge(status, on=["series_id", "origin", "model"], how="left", validate="many_to_one")
    if len(frame) != len(plan) or frame.status.isna().any():
        raise ValueError("Planned grid/status join does not reconcile")
    ok = frame.status.eq("ok")
    if not np.isfinite(frame.loc[ok, ["actual", "prediction", "raw_prediction"]].to_numpy(float)).all() or frame.loc[~ok, "prediction"].notna().any():
        raise ValueError("Invalid fit coverage or nonfinite forecasts")
    np.testing.assert_array_equal(frame.loc[ok, "prediction"], np.maximum(frame.loc[ok, "raw_prediction"], 0))
    frame["absolute_error"] = np.abs(frame.actual - frame.prediction)
    frame["squared_error"] = (frame.actual - frame.prediction) ** 2
    frame["horizon_band"] = horizon_band(frame.horizon)
    return frame, pd.DataFrame(scales)


def comparison_tables(frame, models):
    available = frame[frame.status.eq("ok")]
    common = common_cases(frame, models)
    if common.empty:
        raise ValueError("No shared successful evaluation cases; no ranking permitted")
    panels = {
        "metrics_common": ["model"], "metrics_origin": ["model", "origin"],
        "metrics_series_origin": ["model", "series_id", "origin"], "metrics_horizon": ["model", "horizon"],
        "metrics_band": ["model", "horizon_band"], "metrics_regime": ["model", "regime"],
        "metrics_abc_xyz": ["model", "abc_xyz"]}
    tables = {name: summarize(common, groups) for name, groups in panels.items()}
    tables["comparison_membership"] = common[CASE].drop_duplicates()
    tables["metrics_available"] = summarize(available, ["model"])
    tables["paired_losses"] = paired_losses(frame)
    pair = tables["paired_losses"]
    tables["paired_origin"] = pair.groupby(["model", "baseline", "origin"], as_index=False).agg(
        observations=("horizon", "size"), series=("series_id", "nunique"),
        mean_absolute_difference=("absolute_loss_difference", "mean"),
        mean_squared_difference=("squared_loss_difference", "mean"))
    for measure in ("mae", "wape", "macro_rmsse"):
        tables["metrics_common"][f"rank_{measure}"] = tables["metrics_common"][measure].rank(method="min")
    tables["coverage"] = frame.groupby(["model", "origin", "status", "reason"], dropna=False).agg(
        target_rows=("horizon", "size"), series=("series_id", "nunique")).reset_index()
    return tables


def permutation_importance(model, validation, config):
    eligible = model.importance()
    excluded = {*CATEGORICAL, "horizon"}
    names = eligible.loc[~eligible.feature.isin(excluded), "feature"].head(config.permutation_features)
    rng = np.random.default_rng(config.seed)
    positions = np.sort(rng.choice(len(validation), min(config.permutation_rows, len(validation)), replace=False))
    selected = validation.iloc[positions].copy()
    X, y = selected.drop(columns=TARGET), selected[TARGET].to_numpy(float)
    baseline = np.mean(np.abs(y - model.predict(X).prediction.to_numpy()))
    records = []
    for name in names:
        changed = X.copy()
        changed[name] = changed[name].iloc[rng.permutation(len(changed))].to_numpy()
        start = perf_counter()
        score = np.mean(np.abs(y - model.predict(changed).prediction.to_numpy()))
        records.append({"model": model.library, "feature": name, "rows": len(selected),
                        "baseline_mae": baseline, "permuted_mae": score, "mae_increase": score - baseline,
                        "runtime_seconds": perf_counter() - start})
    return pd.DataFrame(records)
