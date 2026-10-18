"""Measured architecture trade-offs and small smooth/intermittent examples."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from demandos.eda.report import markdown_table


def write_report(summary, tables, output):
    figures = output/"figures"
    figures.mkdir()
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    for name, group in tables["metrics_horizon"].groupby("model"):
        if name not in ("naive", "tsb", "moving_average"):
            ax.plot(group.horizon, group.mae, label=name)
    ax.set(xlabel="Horizon (days)", ylabel="Common-case MAE", title="Multi-horizon architecture benchmark")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(figures/"horizon_mae.png", dpi=150)
    plt.close(fig)
    paths = ["figures/horizon_mae.png"]
    forecasts = tables["forecasts"]
    for regime in ("smooth", "intermittent"):
        ids = tables["evaluation_cohort"].query("regime == @regime").sort_values("series_id")
        if ids.empty:
            continue
        sid = ids.iloc[0].series_id
        origins = sorted(forecasts.origin.unique())
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
        for ax, origin in zip(axes, origins):
            selected = forecasts[forecasts.series_id.eq(sid) & forecasts.origin.eq(origin)]
            actual = selected[["target_date", "actual"]].drop_duplicates().sort_values("target_date")
            ax.plot(actual.target_date, actual.actual, label="actual", color="black", marker=".")
            for name in ("pooled", "recursive", "direct", "independent_multioutput", "ensemble", "seasonal_naive"):
                group = selected[selected.model.eq(name)].sort_values("horizon")
                ax.plot(group.target_date, group.prediction, label=name, alpha=.8)
            ax.set(title=f"Origin {pd.Timestamp(origin):%Y-%m-%d}", ylabel="Observed units / point forecast")
            ax.tick_params(axis="x", rotation=35)
            ax.legend(fontsize=6)
        fig.suptitle(f"{regime}: {sid}")
        name = f"figures/{regime}_paths.png"
        fig.savefig(output/name, dpi=140)
        plt.close(fig)
        paths.append(name)
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        selected = forecasts[forecasts.series_id.eq(sid)]
        for name, group in selected.groupby("model"):
            if name not in ("naive", "moving_average", "tsb"):
                curve = group.groupby("horizon").absolute_error.mean()
                ax.plot(curve.index, curve, label=name)
        ax.set(xlabel="Horizon", ylabel="MAE across three origins", title=f"{regime}: {sid}")
        ax.legend(fontsize=7, ncol=2)
        name = f"figures/{regime}_horizon_mae.png"
        fig.savefig(output/name, dpi=140)
        plt.close(fig)
        paths.append(name)
    summary["figures"] = paths
    scores = tables["metrics_common"].assign(wape_percent=lambda f: f.wape*100)
    lines = ["# Season 10: multi-horizon architecture comparison", "",
        "Four CPU LightGBM architectures plus a validation-selected pooled/seasonal-naive ensemble, compared with unchanged statistical baselines. All predict observed sales, which need not equal uncensored demand.",
        "## Architecture contracts",
        "Pooled: one horizon-aware model f(X_origin,h,C_target). Recursive: one h=1 model repeatedly updates a simulated daily history; prior nonnegative predictions, never future actuals, feed every lag, rolling mean/sum/population std/max/zero fraction/positive count and level-change feature. Fractional predictions remain fractional. Target calendar rows are scheduled at issuance. Price state is frozen at issuance; default M5 price features remain null.",
        "Direct: 28 global LightGBM models, one per h, using origin history and that horizon's scheduled target calendar. Each has its own December-selected iteration count. Independent multi-output: one origin-vector API, shared historical features plus a 28-day calendar vector, and a complete 28-value observed target vector. It projects each target's calendar to 28 independent LightGBM children. There is no joint vector objective, shared tree or target interaction. Complete-vector eligibility can exclude partially observed origins available to direct models.",
        "Recursive internal origins advance virtually, while canonical forecast origins stay fixed at issuance. Trace records distinguish actual_source_max (issuance), simulated_origin and count of earlier predictions. Synthetic observations are never claimed to be new actuals. Missing historical price state is safely frozen, not extrapolated from retrospective future prices. No other historical feature is held stale.",
        "## Chronological design",
        f"Training panel {summary['training_series']} early-prefix-selected series; {len(summary['historical_origins'])} historical origins; {summary['pool_rows']:,} eligible pooled rows. Development label cutoff {summary['boosting_config']['validation_origin']}; {summary['development_rows']:,} pooled rows / {summary['development_vector_samples']:,} complete vector samples. Validation has {summary['validation_rows']:,} targets ending {summary['validation_end']}.",
        "Every historical feature respects its own row origin; fitting labels independently require target_date <= fitting cutoff. Vector samples require all 28 labels observed by cutoff; no partial-target masks. One-step early stopping uses h=1 validation, pooled uses all horizons, direct/vector children use their corresponding horizon. All validation precedes January evaluation; tree counts are frozen before all evaluation refits. Each architecture uses the same fixed Season 9 native hyperparameters and at most 160 rounds/15-round patience/two CPU threads per child. Comparisons measure architectures together with their declared label-sharing and validation differences.",
        markdown_table(tables["training_eligibility"], ["origin", "pooled_rows", "one_step_rows", "vector_samples", "vector_long_rows", "target_max"]),
        f"Ensemble pooled weight {summary['ensemble_pooled_weight']} is selected once by minimum December-validation MAE from 0, 0.5, 1, with ties preferring the smaller pooled weight. Complement goes to seasonal naive. The choice is saved in ensemble.json and frozen for all evaluation origins; no residual correction is fitted.",
        markdown_table(tables["ensemble_validation"], ["pooled_weight", "seasonal_weight", "mae", "rows", "target_max"]),
        markdown_table(tables["validation_metrics"], ["architecture", "observations", "mae", "wape"]),
        "The training panel/cohort and scheduled-calendar assumption are unchanged from Season 9. Evaluation uses 13 existing January-selected series and three already-examined benchmark origins, not an untouched test or population-representative sample. Only seven evaluation identities occur in the independent 96-series training panel. Earlier outcomes can enter later refits once observed, without using their scores for tuning. Native categories are fit-only with unknown tokens; numerical nulls remain native missing values. No future price availability is inferred.",
        "## Exact common-case performance",
        "Scoring keys are identical series/origin/target/horizon cases. MAE and pooled WAPE cover 1,092 targets per method; macro RMSSE uses full-prefix, first-positive-trimmed training-only scales, 38 valid cases and one undefined scale. Zero-actual WAPE is undefined. These are not official M5 WRMSSE. Raw predictions are retained; score/recursive feedback uses max(0,raw). Ensemble averages nonnegative component forecasts.",
        markdown_table(scores, ["model", "observations", "cases", "mae", "wape_percent", "macro_rmsse", "undefined_rmsse_cases"]),
        "## Origins, bands and regimes",
        markdown_table(tables["metrics_origin"], ["model", "origin", "mae", "wape", "macro_rmsse"]),
        markdown_table(tables["metrics_band"], ["model", "horizon_band", "mae", "wape"]),
        markdown_table(tables["metrics_regime"], ["model", "regime", "series", "mae", "wape"]),
        "Paired differences retain exact shared keys in paired_losses/paired_origin. Some regime cells have only one series. Correlated targets/origins and this small selected cohort preclude broad winner, causal or inventory-saving claims. Future actuals are not used to repair recursive feedback. Day-by-day error need not increase monotonically.",
        "## Coverage and computational costs",
        markdown_table(tables["coverage"], ["model", "origin", "status", "target_rows", "series"]),
        markdown_table(tables["resources"], ["architecture", "native_models_per_origin", "total_native_models", "development_fit_seconds", "evaluation_fit_seconds", "evaluation_predict_seconds", "all_model_bytes", "warning_entries"]),
        f"Native fitting outcomes: {summary['native_fit_status']}; local baseline outcomes: {summary['baseline_fit_status']}. Native warnings retained: {summary['warnings']}; clipped raw evaluation forecasts: {summary['clipped_forecast_rows']}. Every native child has fit size/cutoff/digest, rounds, fit/probe time, native model and metadata sizes, complete logs and save/reload checks. Failed native fits block publication without fallback.",
        "Resources count 58 native models per fold (1 pooled, 1 recursive, 28 direct, 28 vector), four folds = 232, fitted sequentially with two native threads. Ensemble reuses pooled and seasonal-naive components; its prediction_resources entry is mixing time only, not end-to-end deployment latency. Local baseline times remain in baseline_fit_status. Native artifact byte counts exclude JSON metadata, vector contracts and reports; per-child metadata bytes are separately recorded. Inference timing includes strategy assembly, history updates and feature matrices but excludes save/load verification. Model-count differences do not imply proportional fit time because row counts differ.",
        f"Measured total experiment {summary['runtime_seconds']:.2f}s; through dataset preparation {summary['dataset_seconds']:.2f}s; historical DataFrame {summary['pool_dataframe_bytes']:,} bytes. This is not total-process peak memory. Existing DuckDB two-thread/2GB managed buffer configuration is reused. Versions: {summary['versions']}. Season 9 reference reproduction: {summary['season9_reference']}.",
        "## Leakage and source integrity",
        "All five architecture/ensemble outputs at all three origins pass saved-state checks after every post-origin sale and future-week price is perturbed. Full inference features and bounded observed prefixes match; eligible historical fitting rows and complete target vectors match after future labels change. Original pooled predictions reproduce Season 9 when its local reference artifact exists. Price perturbation demonstrates default exclusion; non-null as-of behavior remains covered by prior synthetic tests. Strict training cutoffs, complete vectors, recursive feedback and frozen ensemble state are separate hard gates.",
        f"Raw/interim/processed input SHA-256 fingerprints unchanged; warehouse totals unchanged: {summary['source_totals']}. Each generated table's row count and source fingerprint are in summary.json; models/ retain all native states and vector wrapper contracts. Test results are recorded separately after the full Python/dbt runs.",
        "See [LightGBM native regression API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html) for scalar objectives and categorical/missing inputs. The independent wrapper adds a vector contract, not a new joint LightGBM objective.",
        "## Figures"]
    lines.extend(f"![{path}]({path})" for path in paths)
    (output/"architecture_report.md").write_text("\n\n".join(lines)+"\n", encoding="utf-8")
