"""Readable measured comparisons and bounded static diagnostic charts."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from demandos.eda.report import markdown_table as _markdown_table


def markdown_table(frame, columns):
    return _markdown_table(frame, columns) if not frame.empty else "No shared successful observations for this configuration."


def selected(frame, query):
    return frame.query(query) if not frame.empty else frame


def measured_findings(tables):
    common = selected(tables["metrics_common"], "comparison == 'simple' and window == 'expanding' and kind == 'store_item'")
    if common.empty:
        return "No simple expanding store-item comparison is available in this configuration."
    best = common.sort_values(["mae", "model"]).iloc[0]
    lines = [f"Among the three simple methods on {int(best.series)} store-item series / {int(best.cases)} origin cases, {best.model} has the lowest observed pooled MAE ({best.mae:.3f}) and WAPE {best.wape:.2%}. This is a within-panel observation, not a universal winner."]
    origin = selected(tables["metrics_common_origin"], "comparison == 'simple' and window == 'expanding' and kind == 'store_item'")
    for when, group in origin.groupby("origin"):
        row = group.sort_values(["mae", "model"]).iloc[0]
        lines.append(f"Origin {when:%Y-%m-%d}: lowest simple-method MAE is {row.model}, {row.mae:.3f} (WAPE {row.wape:.2%}).")
    segments = selected(tables["metrics_common_segments"], "comparison == 'simple' and window == 'expanding' and kind == 'store_item'")
    for regime, group in segments.groupby("regime"):
        row = group.sort_values(["mae", "model"]).iloc[0]
        lines.append(f"Earliest-origin {regime} stratum ({int(row.series)} series): lowest simple-method MAE {row.model}, {row.mae:.3f}; conclusions are limited to this small selected cell.")
    return "\n\n".join(lines)


def figures(tables, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    curve = tables["metrics_common_horizon"]
    window = "expanding" if "expanding" in set(tables["forecasts"].window) else "sliding"
    selected = (curve[curve.comparison.eq("simple") & curve.kind.eq("store_item") & curve.window.eq(window)]
                if not curve.empty else pd.DataFrame(columns=["model", "horizon", "mae"]))
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for model, group in selected.groupby("model"):
        ax.plot(group.horizon, group.mae, label=model)
    ax.set(xlabel="Days ahead", ylabel="MAE (observed units)", title="Common-case store-item errors across origins")
    if not selected.empty:
        ax.legend()
    path = output / "horizon_mae.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(f"figures/{path.name}")
    frame = tables["forecasts"]
    cohort = tables["cohort"]
    for regime in ("smooth", "intermittent"):
        candidates = cohort[cohort.regime.eq(regime)].sort_values("series_id")
        if candidates.empty:
            continue
        series_id = candidates.iloc[0].series_id
        fig, axes = plt.subplots(1, frame.origin.nunique(), figsize=(15, 4), squeeze=False, layout="constrained")
        for ax, origin in zip(axes[0], sorted(frame.origin.unique())):
            future = frame[frame.window.eq(window) & frame.series_id.eq(series_id) & frame.origin.eq(origin)]
            actual = future[["target_date", "actual"]].drop_duplicates().sort_values("target_date")
            ax.plot(actual.target_date, actual.actual, color="black", marker=".", label="Actual")
            for model in ("naive", "seasonal_naive", "holt_winters", "tsb"):
                predictions = future[future.model.eq(model) & future.status.eq("ok")].sort_values("horizon")
                if not predictions.empty:
                    ax.plot(predictions.target_date, predictions.prediction, label=model, alpha=.8)
            ax.axvline(pd.Timestamp(origin), color="0.5", linestyle="--")
            ax.set(title=f"Origin {pd.Timestamp(origin):%Y-%m-%d}", ylabel="Observed units / point forecast")
            ax.tick_params(axis="x", rotation=35)
            ax.legend(fontsize=7)
        fig.suptitle(f"{regime.capitalize()}: {series_id} ({window})")
        path = output / f"{regime}_origins.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(f"figures/{path.name}")
    return paths


def write_report(summary, tables, target):
    metrics_columns = ["comparison", "window", "kind", "model", "series", "cases", "observations", "mae", "wape", "macro_rmsse", "rmsse_cases", "undefined_rmsse_cases", "rank_mae"]
    status = tables["fit_status"]
    outcomes = status.groupby(["window", "model", "status", "reason"], dropna=False).agg(
        fits=("series_id", "count"), warnings=("warning_count", "sum"), runtime_seconds=("runtime_seconds", "sum")).reset_index()
    lines = ["# Season 7: rolling-origin observed-sales backtesting", "",
             f"Runtime {summary['runtime_seconds']:.2f}s; {len(summary['cohort'])} fixed series; {summary['fit_counts']}; {summary['warnings']} captured warnings.",
             "", "## Predeclared design", "",
             markdown_table(tables["origin_calendar"], ["window", "history_start", "origin", "target_start", "target_end", "horizon", "training_days"]),
             "Cohort labels/selection are fit once through the earliest origin and frozen. They are descriptive strata, not model features. All methods refit independently per origin/window. Targets and cohort are fixed before fitting. No target-driven tuning or future prices.",
             "Each origin-horizon pair is one evaluated event. If stride<horizon, repeated target dates count once per originating forecast, not once per unique calendar date; origin errors are dependent. Default stride=28 yields non-overlapping target blocks, but shared training histories still create dependence.",
             "", "## Measured observations", "", measured_findings(tables),
             "", "## Model by origin (simple expanding store-item panel)", "",
             markdown_table(selected(tables["metrics_common_origin"], "comparison == 'simple' and window == 'expanding' and kind == 'store_item'"),
                            ["origin", "model", "cases", "observations", "mae", "wape", "macro_rmsse", "rmsse_cases"]),
             "Other origin/model/grain panels are retained in metrics_common_origin.parquet and available coverage in metrics_origin.parquet.",
             "", "## Common-case metrics", "",
             "WAPE is a fraction: 0.40 means 40%. MAE is in observed units. Macro RMSSE is the equal-weight mean of series-origin RMSSE over valid training scales, with undefined case counts shown. Micro scaled RMSE is a separate statistic in the Parquet tables. Available-coverage metrics are not rankings.",
             "configured: all requested models; simple: three inexpensive baselines; core_10: requested methods except ARIMA/SARIMA (ten by default); capped panels: naive/seasonal naive versus the named capped model. Each panel intersects successful series/origin/target/horizon keys. No forecasts are substituted for failures. A successful-only intersection can favor easier cases; see explicit membership and outcomes.",
             markdown_table(tables["metrics_common"], metrics_columns),
             "", "## Origin variability (paired against seasonal naive)", "",
             "Negative candidate-minus-baseline loss differences favor the candidate. Origin-level differences retain their own common case counts in paired_origin.parquet. The range and population SD below are descriptive, not independent replications, confidence intervals or significance tests.",
             markdown_table(tables["paired_variability"], ["window", "kind", "model", "origins", "mean_difference", "minimum_difference", "maximum_difference", "std_difference", "candidate_better_origins"]),
             "", "## Horizon bands on shared successful cases", "",
             markdown_table(selected(tables["metrics_common_band"], "comparison == 'simple'"), ["window", "kind", "model", "horizon_band", "observations", "mae", "wape", "macro_rmsse"]),
             "", "## Origin-safe intermittency segments", "",
             markdown_table(selected(tables["metrics_common_segments"], "comparison == 'simple' and kind == 'store_item'"), ["window", "regime", "model", "series", "cases", "mae", "wape", "macro_rmsse", "undefined_rmsse_cases"]),
             "ABC/XYZ summaries use the same earliest-origin product-level definitions and are in metrics_common_abc_xyz.parquet. Small cells and zero actuals limit conclusions; undefined WAPE is not zero error.",
             "", "## Expanding/sliding sensitivity", "",
             "Each method compares the same successful series/origin/target cases across both windows. Model-specific intersections can differ, so this table supports within-method window sensitivity, not a new cross-model ranking. Sliding RMSSE uses its sliding training scale; changes can reflect both prediction changes and scale changes. MAE/WAPE compare unchanged actual targets directly.",
             markdown_table(tables.get("window_sensitivity", pd.DataFrame()), ["window", "kind", "model", "cases", "mae", "wape", "macro_rmsse"]),
             "", "## Coverage, warnings and failures", "",
             markdown_table(tables["coverage_summary"], ["window", "kind", "model", "planned_fits", "successful_fits", "failed_fits", "skipped_fits", "success_fraction", "predicted_targets", "planned_targets"]),
             markdown_table(outcomes, ["window", "model", "status", "reason", "fits", "warnings", "runtime_seconds"]),
             "All planned target rows are retained, even for failed/skipped fits (prediction=null). fit_status.parquet contains exact reasons, warning messages, fitted metadata, runtimes and configuration. No origin or sparse case disappears from planned coverage.",
             "", "## M5 conventions and limitations", "",
             "Training scales are mean nonseasonal squared first differences from the first positive training observation onward (excluding the transition from pre-introduction zeros). All-zero, one-last-positive and zero-scale histories yield explicit undefined RMSSE; no epsilon is added. Aggregate scales come from aggregate daily histories, never averages of bottom-level scales.",
             "These selected-cohort MAE/WAPE/RMSSE results are NOT official M5 WRMSSE. Official scoring covers 42,840 series at 12 hierarchy levels with trailing 28-day training sales-value weights normalized within levels and averaged across levels. This implementation provides tested hierarchy/strict as-of weighting utilities, not the complete competition scorer. No real-data revenue weighting is performed because recorded M5 price availability is unverified.",
             "Sources: [competition organizers' guide](https://github.com/Mcompetitions/M5-methods/blob/master/M5-Competitors-Guide.pdf), [M5 accuracy results paper](https://doi.org/10.1016/j.ijforecast.2021.11.013).",
             "The cohort is illustrative, not representative of all 30,490 store-item series. Three nearby origins and shared training histories do not establish a universal winner. Observed sales are not fully uncensored demand. All three blocks are now evaluation data, including the former Season 6 test block; none is an untouched final test after this study.",
             f"DuckDB resource settings: {summary['resources']}. Python fitting is sequential on a bounded cohort. Iteration/cohort caps are workload limits, not hard process-memory or wall-clock limits. Target vectors and train-only scales are reused in memory; model states are never shared across origins.",
             "", "## Figures", ""]
    lines.extend(f"![{path}]({path})" for path in summary["figures"])
    target.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
