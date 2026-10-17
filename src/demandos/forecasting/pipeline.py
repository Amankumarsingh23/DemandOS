"""One bounded fixed-origin experiment over the existing Season 5 cohort."""

from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import tomllib

import numpy as np
import pandas as pd

from demandos.data.pipeline import fingerprint
from demandos.eda.pipeline import json_ready
from demandos.eda.queries import fetch_frame
from demandos.timeseries.diagnostics import DiagnosticsConfig
from demandos.timeseries.pipeline import extract
from demandos.warehouse.cli import settings, connection, build_artifacts
from .models import MODELS, forecast, at_origin, ForecastResult
from .evaluation import summarize, comparisons


@dataclass(frozen=True)
class ForecastConfig:
    horizon: int = 28
    seasonal_period: int = 7
    lookback: int = 730
    moving_window: int = 28
    alpha: float = .1
    beta: float = .1
    maxiter: int = 100
    sarima_series: int = 4
    arima_order: tuple = (1, 0, 0)
    sarima_order: tuple = (1, 0, 0)
    sarima_seasonal_order: tuple = (0, 1, 1, 7)

    def __post_init__(self):
        for name in ("horizon", "seasonal_period", "lookback", "moving_window", "maxiter", "sarima_series"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Positive integer required: {name}")
        if self.horizon != 28 or not 2 <= self.seasonal_period <= 31:
            raise ValueError("This experiment requires horizon=28 and seasonal period in 2..31")
        if not 2 * self.seasonal_period + 1 <= self.lookback <= 1857 or self.maxiter > 500 or self.sarima_series > 8:
            raise ValueError("Exceeded bounded history/iteration/SARIMA cohort limits")
        if self.moving_window > self.lookback or not 0 < self.alpha <= 1 or not 0 < self.beta <= 1:
            raise ValueError("Invalid smoothing/window settings")
        for order in (self.arima_order, self.sarima_order):
            if len(order) != 3 or any(type(n) is not int or not 0 <= n <= 2 for n in order):
                raise ValueError("ARIMA orders must have three integers in 0..2")
        order = self.sarima_seasonal_order
        if len(order) != 4 or any(type(n) is not int or not 0 <= n <= 2 for n in order[:3]) or order[3] != self.seasonal_period:
            raise ValueError("Invalid seasonal order")


def load_config(path):
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    if set(data) != {"forecasting"}:
        raise ValueError("Expected [forecasting] configuration")
    return ForecastConfig(**data["forecasting"])


def sarima_cohort(cohort, limit):
    """Training metadata only: smooth, intermittent, then aggregates, then ID order."""
    selected = []
    for regime in ("smooth", "intermittent"):
        candidates = cohort[cohort.regime.eq(regime)].sort_values("series_id")
        if not candidates.empty:
            selected.append(candidates.iloc[0].series_id)
    selected += cohort[cohort.kind.ne("store_item")].sort_values("series_id").series_id.tolist()
    selected += cohort.sort_values("series_id").series_id.tolist()
    return list(dict.fromkeys(selected))[:limit]


def parameters(model, config):
    if model == "moving_average":
        return {"window": config.moving_window}
    if model in ("croston", "sba", "tsb"):
        return {"alpha": config.alpha, **({"beta": config.beta} if model == "tsb" else {})}
    if model == "arima":
        return {"maxiter": config.maxiter, "order": config.arima_order}
    if model == "sarima":
        return {"maxiter": config.maxiter, "order": config.sarima_order, "seasonal_order": config.sarima_seasonal_order}
    return {"maxiter": config.maxiter} if model in ("ses", "holt", "holt_damped", "holt_winters") else {}


def fit_cohort(frames, cohort, origin, config):
    """Fitting loop cannot access holdout targets: at_origin clips each input first."""
    allowed = sarima_cohort(cohort, config.sarima_series)
    predictions, statuses, training = [], [], []
    for row in cohort.itertuples():
        if pd.Timestamp(row.segment_fit_end) > origin:
            raise ValueError("Segment label fit extends beyond forecast origin")
        history = at_origin(frames[row.series_id], origin, config.lookback)
        training.append(pd.DataFrame({"series_id": row.series_id, "date": history.index, "units_sold": history.to_numpy()}))
        for model in MODELS:
            params = parameters(model, config)
            result = (ForecastResult(model, "skipped", "outside_training_selected_sarima_cap")
                      if model == "sarima" and row.series_id not in allowed else
                      forecast(history, config.horizon, model, seasonal_period=config.seasonal_period, params=params))
            statuses.append({"series_id": row.series_id, "model": model, "kind": row.kind,
                             "regime": row.regime, "abc_xyz": row.abc_xyz,
                             "status": result.status, "reason": result.reason,
                             "runtime_seconds": result.runtime_seconds, "warning_count": len(result.warnings),
                             "warnings": json.dumps(result.warnings),
                             "metadata": json.dumps(json_ready(result.metadata), default=str, allow_nan=False),
                             "requested_parameters": json.dumps(params), "fit_start": history.index[0],
                             "fit_end": history.index[-1], "observations": len(history),
                             "clipped_count": result.metadata.get("clipped_count", 0)})
            if result.status == "ok":
                for h, (raw, prediction) in enumerate(zip(result.raw_predictions, result.predictions), 1):
                    predictions.append({"series_id": row.series_id, "model": model, "kind": row.kind,
                                        "regime": row.regime, "abc_xyz": row.abc_xyz, "origin": origin,
                                        "horizon": h, "target_date": origin + pd.Timedelta(days=h),
                                        "raw_prediction": raw, "prediction": prediction})
        print(f"Forecasted {row.series_id}", flush=True)
    return pd.DataFrame(predictions), pd.DataFrame(statuses), pd.concat(training, ignore_index=True), allowed


def attach_actuals(predictions, frames, origin, horizon):
    targets = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon)
    actuals = []
    for key, frame in frames.items():
        observed = frame.loc[pd.to_datetime(frame.date).isin(targets), ["date", "units_sold"]].copy()
        if observed.date.duplicated().any() or set(pd.to_datetime(observed.date)) != set(targets):
            raise ValueError(f"Incomplete or duplicated holdout: {key}")
        observed["series_id"] = key
        actuals.append(observed.rename(columns={"date": "target_date", "units_sold": "actual"}))
    joined = predictions.merge(pd.concat(actuals), on=["series_id", "target_date"], how="left", validate="many_to_one")
    if len(joined) != len(predictions) or joined.actual.isna().any():
        raise ValueError("Holdout join lost forecast observations")
    if not np.isfinite(joined.actual.to_numpy(float)).all() or joined.actual.lt(0).any():
        raise ValueError("Invalid actual sales; actuals are never clipped")
    joined["absolute_error"] = np.abs(joined.actual - joined.prediction)
    return joined


def validate_outputs(forecasts, statuses, cohort, origin, horizon):
    keys = ["series_id", "model"]
    if statuses.duplicated(keys).any() or len(statuses) != len(cohort) * len(MODELS):
        raise ValueError("Missing or duplicate fit statuses")
    if forecasts.duplicated(keys + ["horizon"]).any():
        raise ValueError("Duplicate forecast keys")
    if not statuses.fit_end.eq(origin).all() or not cohort.segment_fit_end.le(origin).all():
        raise ValueError("Invalid fit/segment cutoff")
    if not forecasts.target_date.eq(origin + pd.to_timedelta(forecasts.horizon, unit="D")).all():
        raise ValueError("Target/horizon mismatch")
    if not forecasts.origin.eq(origin).all():
        raise ValueError("Mismatched forecast origin")
    for _, group in forecasts.groupby(keys):
        if sorted(group.horizon) != list(range(1, horizon + 1)):
            raise ValueError("Incomplete forecast vector")
    if not np.isfinite(forecasts[["raw_prediction", "prediction", "actual", "absolute_error"]].to_numpy(float)).all():
        raise ValueError("Nonfinite forecast artifact")
    np.testing.assert_array_equal(forecasts.prediction, np.maximum(forecasts.raw_prediction, 0))
    succeeded = set(map(tuple, statuses.loc[statuses.status.eq("ok"), keys].to_numpy()))
    emitted = set(map(tuple, forecasts[keys].drop_duplicates().to_numpy()))
    if succeeded != emitted:
        raise ValueError("Forecasts/statuses do not reconcile")


def evaluation_tables(forecasts, statuses):
    tables = {"metrics_available": summarize(forecasts, ["kind", "model"]),
              "metrics_segments": summarize(forecasts, ["kind", "regime", "abc_xyz", "model"]),
              "metrics_horizons": summarize(forecasts[forecasts.horizon.isin([1, 7, 14, 28])], ["kind", "model", "horizon"])}
    panels, segments, membership = [], [], []
    for name, models in (("simple_all_series", list(MODELS[:3])),
                         ("broad_without_sarima", [m for m in MODELS if m != "sarima"]),
                         ("all_models_capped", list(MODELS))):
        common = comparisons(forecasts, models)
        if not common.empty:
            aggregate = summarize(common, ["kind", "model"])
            aggregate["comparison"] = name
            panels.append(aggregate)
            segments.append(summarize(common, ["kind", "regime", "model"]).assign(comparison=name))
            membership.append(common[["series_id", "target_date", "horizon", "kind"]].drop_duplicates().assign(comparison=name))
    tables["metrics_common"] = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    tables["comparison_membership"] = pd.concat(membership, ignore_index=True) if membership else pd.DataFrame()
    tables["metrics_common_segments"] = pd.concat(segments, ignore_index=True) if segments else pd.DataFrame()
    tables["coverage"] = statuses.groupby(["kind", "model", "status", "reason"], dropna=False).agg(
        series=("series_id", "nunique"), runtime_seconds=("runtime_seconds", "sum"), warnings=("warning_count", "sum")).reset_index()
    return tables


def run(root, config):
    started = perf_counter()
    root, source, warehouse, env = settings(root)
    build = build_artifacts(root, source, warehouse)
    inputs = [source, root / "data/interim/sales_long.parquet", *sorted((root / "data/raw").glob("*.csv"))]
    before = {str(path.relative_to(root)): fingerprint(path) for path in inputs}
    with connection(warehouse, env) as con:
        histories, cohort, panel, labels, products, segment_config, split, dates = extract(con, root, DiagnosticsConfig())
        origin = split["train_end"]
        if origin != pd.Timestamp("2016-02-28") or split["validation_end"] != origin + pd.Timedelta(days=config.horizon):
            raise ValueError("Source does not match the documented fixed-origin design")
        frames = {}
        for row in cohort.itertuples():
            if row.kind == "store_item":
                frame = panel[panel.store_id.eq(row.store_id) & panel.item_id.eq(row.item_id)][["date", "units_sold"]].copy()
            else:
                key, identifier = ("store_id", row.store_id) if row.kind == "store" else ("cat_id", row.series_id.split("/", 1)[1])
                frame = fetch_frame(con, f"select date, units_sold from main.mart_{row.kind}_daily where {key}=? order by date", [identifier])
            frame["date"] = pd.to_datetime(frame.date)
            frames[row.series_id] = frame
        source_rows = con.execute("select count(*) from main.fact_daily_sales").fetchone()[0]
    predictions, statuses, training, allowed = fit_cohort(frames, cohort, origin, config)
    forecasts = attach_actuals(predictions, frames, origin, config.horizon)
    validate_outputs(forecasts, statuses, cohort, origin, config.horizon)
    tables = evaluation_tables(forecasts, statuses)
    tables.update(forecasts=forecasts, fit_status=statuses, training=training, cohort=cohort,
                  training_store_item_labels=labels, training_product_labels=products)
    after = {str(path.relative_to(root)): fingerprint(path) for path in inputs}
    if before != after:
        raise ValueError("Raw/processed input immutability check failed")
    target = root / "data/forecasts" / f"origin_{origin:%Y%m%d}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".forecasts-", dir=root / "data") as temporary:
        temporary = Path(temporary)
        (temporary / "tables").mkdir()
        for name, frame in tables.items():
            frame.to_parquet(temporary / "tables" / f"{name}.parquet", index=False)
        from .report import figures, write_report
        paths = figures(training, forecasts, cohort, temporary / "figures", origin)
        summary = {"status": "passed", "config": asdict(config), "source_rows": source_rows,
                   "input_fingerprints": before, "inputs_unchanged": before == after, "origin": origin,
                   "holdout_start": origin + pd.Timedelta(days=1), "holdout_end": split["validation_end"],
                   "untouched_test_start": split["test_start"], "sarima_cohort": allowed,
                   "cohort": cohort.to_dict("records"), "segmentation_config": asdict(segment_config),
                   "models": list(MODELS), "fit_counts": statuses.status.value_counts().to_dict(),
                   "warning_count": int(statuses.warning_count.sum()), "clipped_predictions": int(statuses.clipped_count.sum()),
                   "table_rows": {k: len(v) for k, v in tables.items()}, "figures": paths,
                   "metrics_common": tables["metrics_common"].to_dict("records"),
                   "metrics_available": tables["metrics_available"].to_dict("records"),
                   "metrics_common_segments": tables["metrics_common_segments"].to_dict("records"),
                   "warehouse_build_tests_passed": sum(r["unique_id"].startswith("test.") and r["status"] == "pass" for r in build["results"]),
                   "versions": {name: version(name) for name in ("numpy", "pandas", "statsmodels", "scipy", "duckdb", "matplotlib")},
                   "runtime_seconds": perf_counter() - started}
        write_report(summary, tables, temporary / "forecast_report.md")
        (temporary / "summary.json").write_text(json.dumps(json_ready(summary), indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
        target.mkdir(parents=True, exist_ok=True)
        for path in temporary.rglob("*"):
            if path.is_file():
                final = target / path.relative_to(temporary)
                final.parent.mkdir(parents=True, exist_ok=True)
                path.replace(final)
    return summary, target
