"""Actual experiment tables and a small pair of fixed-cutoff comparison charts."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from demandos.eda.report import markdown_table


def figures(training, forecasts, cohort, output, origin):
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for regime in ("smooth", "intermittent"):
        selected = cohort[cohort.regime.eq(regime)].sort_values("series_id")
        if selected.empty:
            continue
        series_id = selected.iloc[0].series_id
        history = training[training.series_id.eq(series_id)].tail(56)
        future = forecasts[forecasts.series_id.eq(series_id)]
        actual = future[["target_date", "actual"]].drop_duplicates().sort_values("target_date")
        fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
        ax.plot(history.date, history.units_sold, color="0.4", label="Training observations")
        ax.plot(actual.target_date, actual.actual, color="black", marker=".", label="Holdout actual sales")
        for model in ("naive", "seasonal_naive", "holt_winters", "sba", "tsb", "sarima"):
            fit = future[future.model.eq(model)].sort_values("horizon")
            if not fit.empty:
                ax.plot(fit.target_date, fit.prediction, label=model, alpha=.8)
        ax.axvline(origin, linestyle="--", color="black", label="Fixed origin")
        ax.axvspan(actual.target_date.min(), actual.target_date.max(), color="steelblue", alpha=.07)
        ax.set(title=f"{regime.capitalize()}: {series_id}", ylabel="Observed units / point forecast", xlabel="Date")
        ax.legend(fontsize=8, ncol=3)
        fig.autofmt_xdate()
        path = output / f"{regime}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(f"figures/{path.name}")
    return paths


def write_report(summary, tables, target):
    common = tables["metrics_common"].copy()
    if not common.empty:
        common["wape_percent"] = common.wape * 100
    available = tables["metrics_available"].copy()
    available["wape_percent"] = available.wape * 100
    status = tables["fit_status"]
    failures = status[status.status.ne("ok")][["series_id", "model", "status", "reason"]]
    segments = tables["metrics_segments"].copy()
    segments["wape_percent"] = segments.wape * 100
    common_segments = tables["metrics_common_segments"].copy()
    if not common_segments.empty:
        common_segments["wape_percent"] = common_segments.wape * 100
    lines = ["# Season 6: fixed-origin observed-sales forecasts", "",
             f"Origin: {summary['origin']:%Y-%m-%d}. Holdout: {summary['holdout_start']:%Y-%m-%d} through {summary['holdout_end']:%Y-%m-%d} (28 days).",
             f"Runtime: {summary['runtime_seconds']:.2f}s; {len(summary['cohort'])} series; fit outcomes: {summary['fit_counts']}; warnings: {summary['warning_count']}.",
             "", "## Fair comparisons", "",
             "MAE = mean absolute error; WAPE = sum absolute error / sum actual sales. WAPE is undefined (null) for zero actual totals. Fractions are converted to percentages below.",
             "Each panel intersects successful series/date/horizon keys across its named models, separately by aggregation grain. Simple includes naive/seasonal naive/moving average; broad excludes capped SARIMA; capped includes all 12 models. No unavailable fits or alternative-model fallbacks enter any comparison.", "",
             markdown_table(common, ["comparison", "kind", "model", "series", "observations", "mae", "wape_percent"]) if not common.empty else "No common successful observations.",
             "", "## Available coverage (unequal cohorts; not a ranking)", "",
             markdown_table(available, ["kind", "model", "series", "observations", "mae", "wape_percent"]),
             "", "## Segments on common samples", "",
             markdown_table(common_segments, ["comparison", "kind", "regime", "model", "series", "mae", "wape_percent"]) if not common_segments.empty else "No common successful observations.",
             "", "## Segments (available coverage; selected series only)", "",
             markdown_table(segments, ["kind", "regime", "abc_xyz", "model", "series", "mae", "wape_percent"]),
             "", "## Coverage and fitting", "",
             markdown_table(tables["coverage"], ["kind", "model", "status", "reason", "series", "runtime_seconds", "warnings"]),
             "", "## Failed and skipped fits", "",
             markdown_table(failures, list(failures.columns)), "", "## Interpretation and limits", "",
             "This is one illustrative fixed-origin holdout, not rolling-origin backtesting or a representative M5 sample. Store-item rates and aggregate unit counts have different scales; they are never pooled into a common ranking. Segment summaries can contain unequal successful cohorts. ABC/XYZ and regimes are recomputed through the origin, not full-history labels.",
             "All methods see the same trailing 730 training days by default. No holdout tuning, price regressors, or target-dependent cohort selection. Training-history constants are explicitly skipped by optimized models; no fallback silently boosts coverage. Full vectors retain raw predictions and deterministic max(0, raw) scoring predictions. Actuals are unchanged.",
             "Croston/SBA/TSB estimate observed sales rates, not uncensored customer demand. Weekly seasonality is an explicit candidate assumption. ARIMA(1,0,0) captures short persistence; SARIMA(1,0,0)(0,1,1)[7] allows weekly differencing and seasonal shocks. Neither order is selected using this holdout. Iteration/history/cohort caps bound experiment size; they are not hard wall-clock deadlines.",
             "Use fit_status.parquet for every warning, initialization, fitted parameters, training bounds, runtime and explicit reason. See comparison_membership.parquet for exact common samples, metrics_horizons.parquet for h=1/7/14/28 and forecasts.parquet for every daily absolute error.",
             "Rolling-origin evaluation, RMSSE, formal model selection and probabilistic forecasts remain deferred.", "", "## Figures", ""]
    lines.extend(f"![{path}]({path})" for path in summary["figures"])
    target.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
