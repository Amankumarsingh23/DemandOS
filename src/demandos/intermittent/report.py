"""Measured sparse-sales findings with explicit simulation/censoring limitations."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from demandos.eda.report import markdown_table


def write_report(summary,tables,output):
    (output/"figures").mkdir()
    frame=tables["forecasts"]
    paths=[]
    for cohort in ("matched","sparse_stress","cold_simulation"):
        candidates=frame[frame.cohort.eq(cohort)]
        if candidates.empty: continue
        chosen=candidates[["series_id","regime"]].drop_duplicates().sort_values(["regime","series_id"]).iloc[0].series_id
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout="constrained")
        for ax,origin in zip(axes,sorted(frame.origin.unique())):
            group=candidates[candidates.series_id.eq(chosen)&candidates.origin.eq(origin)]
            actual=group[["target_date","actual"]].drop_duplicates().sort_values("target_date")
            ax.plot(actual.target_date,actual.actual,color="black",marker=".",label="actual observed sales")
            for model in ("pooled","hurdle","tsb","cold_prior","router","ensemble"):
                points=group[group.model.eq(model)&group.status.eq("ok")].sort_values("horizon")
                if not points.empty: ax.plot(points.target_date,points.prediction,label=model,alpha=.8)
            ax.set(title=f"Origin {pd.Timestamp(origin):%Y-%m-%d}",ylabel="Observed units / prediction")
            ax.tick_params(axis="x",rotation=35);ax.legend(fontsize=6)
        fig.suptitle(f"{cohort}: {chosen}")
        path=f"figures/{cohort}.png";fig.savefig(output/path,dpi=140);plt.close(fig);paths.append(path)
    calibration=tables["calibration"]
    fig,ax=plt.subplots(figsize=(6,5),layout="constrained")
    ax.plot([0,1],[0,1],linestyle="--",color="gray",label="identity")
    for cohort,group in calibration.groupby("cohort"):
        group=group.sort_values("mean_probability");ax.plot(group.mean_probability,group.positive_fraction,marker="o",label=cohort)
    ax.set(xlabel="Mean predicted positive-sales probability",ylabel="Observed positive fraction",xlim=(0,1),ylim=(0,1))
    ax.legend();fig.savefig(output/"figures/calibration.png",dpi=150);plt.close(fig);paths.append("figures/calibration.png")
    summary["figures"]=paths
    scores=tables["metrics_common"].assign(wape_percent=lambda f:f.wape*100)
    rules=pd.DataFrame(summary["policy"]["rules"])
    lines=["# Season 11: intermittent observed sales and cold-start simulation",
        "## Scope and identifiability",
        "All target observations are actual M5 recorded sales. Cold experiments deliberately withhold existing history and, in one scenario, category metadata; these are historical simulations on real products, not genuine observed introduction dates. No synthetic latent-demand values, stockout labels, inventory levels or lost-sales labels are added. Observed sales may be constrained by availability; zero sales, missing prices, intermittency and level changes do not establish a stockout. Exact latent demand/lost sales cannot be recovered here.",
        "## Fixed chronological design",
        f"Training {summary['training_series']} series, {len(summary['historical_origins'])} historical origins, {summary['pool_rows']:,} pooled labels; development {summary['development_rows']:,} labels through {summary['boosting_config']['validation_origin']}. December 7–January 3 internal validation has {summary['validation_rows']:,} target rows. Selected rounds {summary['selected_rounds']}. Every historical feature respects its row origin and every fit label has target_date <= allowed cutoff. Each later origin refits with frozen iterations.",
        f"Evaluation uses origins {summary['evaluation_origins']}, 28 days each. Series counts: {summary['cohort_series']}. Stress selection through the first evaluation origin: {summary['cohort_selection_counts']}. The 13-series continuity benchmark has already been examined; it is retrospective, not an untouched final test. Stress sampling uses only pre-origin information and does not establish population accuracy.",
        "## Historical regimes and cold priors",
        "ADI=N/K, positive-size CV²=population variance / squared positive mean, using the configured trailing historical window. Equality at ADI 1.32 is high-ADI; equality at CV² 0.49 is high-variation. Quadrants are smooth/erratic at low ADI and intermittent/lumpy at high ADI. All-zero ADI/CV² are undefined; one-positive CV² is undefined and the regime is insufficient-positive-history. No-observation histories differ from observed zeros. Separate trailing-window indicators, positive occurrence frequency and time since positive sale are exported.",
        "M5 records zeros across its calendar and does not verify item introduction. Cold simulation uses no visible history, existing-item/new-store metadata, 3/7/28 visible days, or absent category metadata. Cold item IDs are entirely absent from pooled/hurdle training, including other stores. Observed zeros inside visible windows remain observations; withheld dates are never fabricated as zeros. The standard observed-history feature contract cannot represent zero-history ML rows, so pooled/hurdle explicitly skip them. Short observed histories use the existing missing-window/lag policies.",
        "Each origin's prior uses only its last configured 56 observed days: store-category first, category second, always excluding the target item in every store. Require minimum prior support; there is no silent global mean. Forecast=(n*visible_observed_mean+kappa*prior_mean)/(n+kappa), kappa=28 by default, n=all visible observed days. With no prior but visible observations, use their explicitly labeled mean; with neither, report unavailable. This is a simple observed-sales fallback, not latent-demand recovery.",
        markdown_table(tables["diagnostics"].groupby(["cohort","origin_date","regime","cold_status"],dropna=False).size().reset_index(name="series"),["cohort","origin_date","regime","cold_status","series"]),
        "## Global hurdle interpretation",
        "Binary LightGBM occurrence uses target 1(y>0), binary log loss and fit-only categories. Positive-size LightGBM uses only y>0 labels with untransformed squared error, estimating a conditional mean rather than a median or uncorrected log estimate. Its score projects to the positive-integer support's mean lower bound 1; raw positive-size estimates remain available. Expected observed sales=p_positive*conditional_positive_mean. This two-part mean model does not fit a full truncated-count distribution. Degenerate occurrence partitions use their observed class probability; fewer than 20 positive labels use their observed positive mean; zero positives imply an explicit zero-rate fallback without a learned size. Native warnings and states are saved.",
        "A distributional count hurdle would use a zero-truncated positive count distribution. A zero-inflated Poisson mixes an extra-zero component with an ordinary Poisson component that can also emit zero. Neither mixture component proves a causal stockout. The optional ZIP comparator is deliberately omitted; overdispersion and sparse histories do not justify fitting it indiscriminately. Croston/SBA/TSB reuse existing observed-sales implementations and their visible-history requirements.",
        markdown_table(tables["occurrence_metrics"],["cohort","rows","brier_score","observed_occurrence"]),
        markdown_table(calibration,["cohort","probability_bin","rows","mean_probability","positive_fraction"]),
        markdown_table(tables["positive_size_metrics"],["cohort","actual_positive_rows","conditional_size_mae"]),
        "Positive-size errors are conditional on realized positive target observations among supported hurdle forecasts; they are not unconditional accuracy, an inventory claim, or a separate rank against all-observation point errors. Calibration uses five predeclared probability bins and no evaluation-fitted probability transform.",
        "## Frozen segment router and conservative ensemble",
        "Policy candidates: pooled, hurdle, TSB, seasonal naive. Regimes use each issuance's observed prefix; rules use only earlier December common-key validation losses. Specialists require at least five distinct validation series and 140 target rows by default. Otherwise route/default blend is pooled. Routing selects minimum MAE; blending selects pooled alone or 50/50 pooled with one candidate. Weights are nonnegative and normalized. Cold histories below the configured 28-day threshold override to the explicit prior. Missing requested candidates trigger logged pooled/prior defaults or unavailable status; no retrospective per-series winner selection occurs.",
        markdown_table(rules,["regime","validation_series","validation_rows","rule_status","route","weights","candidate_mae","blend_mae"]),
        f"Policy version {summary['policy']['version']}, validation end {summary['policy']['validation_end']}, evidence SHA-256 {summary['policy']['evidence_sha256']}. Every forecast route/effective weight is exported. Rules and weights are frozen before evaluation; labels/priors can change only as new pre-origin observations become available.",
        "## Same-case comparisons",
        "Rank only the seven primary methods on identical successful keys, separately by cohort. Cold zero-history and insufficient-history cases often lack a seven-method intersection; their coverage and available/prior results are separately reported, without pretending a complete head-to-head comparison. Croston/SBA/moving-average available metrics and paired pooled comparisons remain separate. WAPE zero denominators and RMSSE zero/absent visible-history scales remain undefined. Cold scales use only the visible history, never hidden observations. These are not official M5 WRMSSE.",
        markdown_table(scores,["cohort","model","observations","cases","mae","wape_percent","macro_rmsse","undefined_rmsse_cases"]),
        markdown_table(tables["metrics_available"],["cohort","model","observations","mae","wape","macro_rmsse"]),
        markdown_table(tables["paired_origin"],["cohort","model","origin","targets","mean_absolute_difference"]),
        "Origin/horizon/band/segment and series-origin WAPE/undefined-denominator details are in the Parquet tables; different cohorts and conditional populations must not be pooled into misleading rankings.",
        "## Coverage, resources and integrity",
        markdown_table(tables["coverage"],["cohort","model","origin","status","reason","targets","series"]),
        f"Global fit statuses {summary['fit_status']}; local statuses {summary['baseline_fit_status']}; forecast row statuses {summary['forecast_status']}. {summary['native_models']} native models, {summary['native_model_bytes']:,} native model bytes; {summary['warnings']} retained native warning entries. Configurations cap model rows/series/origins, rounds and CPU threads; no expensive full-panel fitting. Only {summary['cached_daily_rows']:,} selected daily observations are brought into Python. Prior tables reduce full warehouse facts in SQL before Python access.",
        f"Measured runtime {summary['runtime_seconds']:.2f}s. Versions {summary['versions']}. Pooled reference: {summary['season9_reference']}. Source/processed SHA-256 and warehouse totals unchanged: {summary['source_totals']}. Models, policy evidence, fit timings/warnings, probability/size forecasts, cold priors, route choices and manifests are saved. Tests are run separately and their reviewed evidence is recorded afterward.",
        "At each real evaluation origin, all post-origin sales and future-week prices are perturbed, permitted features/diagnostics/priors are regenerated and frozen loaded models/policy must retain their predictions and routes. Altering future training labels must leave eligible fitting rows unchanged. Default price invariance establishes exclusion, not otherwise unverified price announcement availability.",
        "References: [LightGBM objectives](https://lightgbm.readthedocs.io/en/latest/Parameters.html), [hurdle versus truncated count models](https://www.statsmodels.org/stable/examples/notebooks/generated/count_hurdle.html). No neural forecasting, supervised stockout detector or inventory policy is implemented.",
        "## Figures"]
    lines.extend(f"![{p}]({p})" for p in paths)
    (output/"experiment_report.md").write_text("\n\n".join(lines)+"\n",encoding="utf-8")
