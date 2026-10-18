"""Measured global-model comparisons, validation-only importance and small plots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from demandos.eda.report import markdown_table
from .config import LIBRARIES


def figures(tables, output):
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for model, group in tables["metrics_horizon"].groupby("model"):
        if model in (*LIBRARIES, "seasonal_naive", "moving_average"):
            ax.plot(group.horizon, group.mae, label=model)
    ax.set(xlabel="Forecast horizon (days)", ylabel="Common-case MAE (observed units)", title="Global models and statistical baselines")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(output / "horizon_mae.png", dpi=150)
    plt.close(fig)
    forecasts = tables["forecasts"]
    candidates = tables["evaluation_cohort"].query("regime == 'smooth'").sort_values("series_id")
    chosen = candidates.iloc[0].series_id if not candidates.empty else forecasts.series_id.iloc[0]
    origins = sorted(forecasts.origin.unique())
    fig, axes = plt.subplots(1, len(origins), figsize=(15, 4), squeeze=False, layout="constrained")
    for ax, origin in zip(axes[0], origins):
        selected = forecasts[forecasts.series_id.eq(chosen) & forecasts.origin.eq(origin)]
        actual = selected[["target_date", "actual"]].drop_duplicates().sort_values("target_date")
        ax.plot(actual.target_date, actual.actual, label="Actual", color="black", marker=".")
        for model in (*LIBRARIES, "seasonal_naive"):
            points = selected[selected.model.eq(model) & selected.status.eq("ok")].sort_values("horizon")
            ax.plot(points.target_date, points.prediction, label=model, alpha=.8)
        ax.set(title=f"Origin {pd.Timestamp(origin):%Y-%m-%d}", ylabel="Observed units / point forecast")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=7)
    fig.suptitle(chosen)
    fig.savefig(output / "forecast_origins.png", dpi=140)
    plt.close(fig)
    return ["figures/horizon_mae.png", "figures/forecast_origins.png"]


def write_report(summary, tables, path):
    scores = tables["metrics_common"].copy()
    scores["wape_percent"] = scores.wape * 100
    fits = tables["fit_status"]
    paired = tables["paired_origin"].query("model in @LIBRARIES")
    top = tables["feature_importance"].groupby("model", sort=True).head(6)
    lines = ["# Season 9: global gradient-boosted observed-sales forecasts", "",
             f"Runtime {summary['runtime_seconds']:.2f}s (dataset {summary['dataset_seconds']:.2f}s). "
             f"Training panel {summary['training_series']} series / {len(summary['historical_origins'])} historical origins / {summary['pool_rows']:,} eligible labeled rows; "
             f"evaluation {summary['evaluation_series']} store-item series / {len(summary['evaluation_origins'])} origins / 28 days.",
             "", "## Predeclared chronological protocol", "",
             f"Selection/mapping cutoff {summary['config']['selection_cutoff']}; historical start {summary['config']['training_start']}, "
             f"stride {summary['config']['training_stride']} days. Development fit targets end {summary['config']['validation_origin']}; "
             f"validation is the subsequent 28 days ending {summary['validation_end']}. Fit rows {summary['development_rows']:,}, validation rows {summary['validation_rows']:,}.",
             f"Frozen boosting rounds from internal validation: {summary['selected_rounds']}. One fixed parameter set per library, no uncontrolled search. "
             "Each evaluation origin refits on all and only labels with target_date <= origin, with those frozen rounds and no evaluation-driven early stopping. Earlier benchmark labels may enter a later origin's fit after they become observed; this is prequential updating, not tuning on their scores.",
             markdown_table(tables["training_eligibility"], ["origin", "rows", "series", "origins", "target_min", "target_max"]),
             "Training examples overlap in target dates: each series-origin-horizon is an intentional separate task. No random horizon-row split is used. Chronological development splits by target availability, so shared target days cannot cross the fit/validation boundary.",
             f"The independent historical training sample is selected from its early prefix; {summary['evaluation_series_seen_in_training']} of {summary['evaluation_series']} "
             "legacy benchmark series occur in it. Others are evaluated using shared signals and native unknown-category handling. The evaluation cohort is reused from Season 7 (selected through January 31); its earlier scores are already known. This is a retrospective known benchmark, not a pristine or population-representative final test. March 27 outcomes never choose hyperparameters, features or iterations.",
             "", "## Same-case evaluation", "",
             "All rows below use the successful intersection across all seven model names on identical series/origin/target/horizon keys. WAPE is pooled absolute error / actual units, with null for zero actual totals. Macro RMSSE averages valid series-origin scaled losses; denominators use the full historical prefix after first-positive trimming. Undefined counts remain explicit. Cohort metrics are NOT official M5 WRMSSE; restricted SARIMA scores are not included.",
             markdown_table(scores, ["model", "series", "cases", "observations", "mae", "wape_percent", "macro_rmsse", "rmsse_cases", "undefined_rmsse_cases"]),
             "", "## Paired changes against seasonal naive by origin", "",
             "Candidate-minus-baseline loss differences: negative favors the candidate. These are descriptive correlated-origin observations, not independent replications or significance tests.",
             markdown_table(paired, ["model", "origin", "series", "observations", "mean_absolute_difference", "mean_squared_difference"]),
             "", "## Horizon bands and regimes", "",
             markdown_table(tables["metrics_band"], ["model", "horizon_band", "observations", "mae", "wape", "macro_rmsse"]),
             markdown_table(tables["metrics_regime"], ["model", "regime", "series", "cases", "mae", "wape", "macro_rmsse", "undefined_rmsse_cases"]),
             "The frozen regime/ABCXYZ labels describe the January-first evaluation sample and never enter model inputs. Some regimes have one series; zero actuals and invalid RMSSE scales constrain interpretation. Origin/day/ABCXYZ details and exact comparison keys remain in Parquet.",
             "", "## Development importance and controlled ablation", "",
             "Native importance is library-specific (LightGBM gain/split, XGBoost total gain/weight, CatBoost PredictionValuesChange), not directly comparable across implementations and not causal. Only pre-evaluation development models supply these tables.",
             markdown_table(top, ["model", "feature", "importance", "splits", "kind"]),
             "Permutation uses one seeded marginal shuffle for each of five top native-importance non-key numerical predictors on at most 1,024 internal-validation rows. Identity/horizon keys stay unchanged. Correlated predictors and one repetition limit interpretation; no final evaluation outcomes select features.",
             markdown_table(tables["permutation_importance"], ["model", "feature", "rows", "baseline_mae", "permuted_mae", "mae_increase"]),
             "The sole ablation removes six event predictors from LightGBM, using identical development fit/validation keys, parameters and the primary model's selected fixed tree count. Its result does not change primary features or final configurations.",
             markdown_table(tables["validation_metrics"], ["model", "experiment", "rounds", "observations", "mae", "wape"]),
             "", "## Coverage, failures and serialization", "",
             markdown_table(tables["coverage"], ["model", "origin", "status", "reason", "target_rows", "series"]),
             f"Fit status counts (including development, ablation and local baselines): {summary['fit_status_counts']}; "
             f"case status counts: {summary['case_status_counts']}; warning entries: {summary['warnings']}; clipped negative forecast rows: {summary['clipped_forecast_rows']}.",
             "Native model files and JSON metadata retain predictor order, category vocabularies, fold cutoffs, selected rounds, versions, parameters and complete fit logs. Python warnings and distinct native warning lines are reported, not silently dropped. Every saved global fit is checked against reload predictions. Failed required global fits block publication rather than substitute a baseline.",
             "", "## Availability and resources", "",
             "Season 8 features are reused. Per-row history never passes its row origin; category state for source features precedes all historical origins. Per-model native vocabularies use fitting rows only, not validation/test categories. LightGBM/XGBoost receive fixed pandas categorical vocabularies; CatBoost receives explicit categorical strings. Unknown/missing categories use a reserved token. Duplicate integer ID codes are excluded. Numerical nulls remain native missing values; no future-fitted imputation or scaling is used.",
             "CatBoost uses CPU Plain boosting, has_time=True and rows sorted by target-date availability. Its internal categorical statistics stay inside the eligible fitting fold; this does not independently prove time causality. Future-price covariates remain unavailable and scheduled calendar/event/SNAP availability remains the Season 8 assumption. Raw predictions are retained, scoring uses max(0,raw), and actuals are unchanged. Sales need not equal uncensored demand.",
             f"Future checks: {len(tables['leakage_checks'])} model-origin cases pass complete predictor/forecast invariance after future sales and future-week prices change; eligible fit inputs also remain identical when future labels change. Default price exclusion is an exclusion check, with non-null as-of price behavior tested in Season 8 fixtures. Resources: {summary['resources']}. DataFrame bytes are not total-process peak memory; thread/iteration/panel/row caps bound workload, not a hard wall-clock deadline. Multithreaded training is not claimed universally bit-identical.",
             "Sources for library contracts: [LightGBM Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html), [XGBoost categorical IO](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html), [CatBoost parameters](https://catboost.ai/docs/en/references/training-parameters/common).",
             "Source and processed fingerprints and warehouse totals remain unchanged. No inventory savings, causality or official competition score is claimed. Horizon-specific models, pooling strategies and model-family ensembles remain Season 10 work.", "", "## Figures", ""]
    lines.extend(f"![{figure}]({figure})" for figure in summary["figures"])
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
