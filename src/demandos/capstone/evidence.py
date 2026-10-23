"""Recalculate selected evidence from saved records, never from a passed flag."""
import json
import numpy as np
import pandas as pd

from demandos.risk.metrics import tail
from demandos.serving.core import digest, file_hash, read_json, require
from .config import LABELS


def metrics(frame):
    y = frame.actual.to_numpy(float)
    q = frame[["q10", "q50", "q90"]].to_numpy(float)
    error = y[:, None] - q
    levels = np.array([.1, .5, .9])
    return dict(n=len(y), mae=float(np.abs(y-q[:, 1]).mean()),
                wape=float(np.abs(y-q[:, 1]).sum()/y.sum()) if y.sum() else None,
                pinball=float(np.maximum(levels*error, (levels-1)*error).mean()),
                coverage=float(((y >= q[:, 0]) & (y <= q[:, 2])).mean()),
                width=float((q[:, 2]-q[:, 0]).mean()))


def collect(root, source_fingerprint):
    records = []

    def table(relative):
        return pd.read_parquet(root / relative)

    def add(season, module, experiment, sources, frame, values, label, *, checkpoint=None,
            fit=None, validation=None, grain="store-item-origin-horizon", assumptions="", comparison=None, horizon=28):
        origins = sorted(str(pd.Timestamp(x).date()) for x in frame.origin_date.unique()) if "origin_date" in frame else []
        count = len(frame[["store_id", "item_id"]].drop_duplicates()) if {"store_id", "item_id"}.issubset(frame) else 25
        records.append(dict(season=season, module=module, experiment_id=experiment,
            sources={p: file_hash(root/p) for p in sources}, source_data_fingerprint=source_fingerprint,
            model_checkpoint_id=checkpoint, model_fit_cutoff=fit, validation_cutoff=validation,
            origins=origins, horizon=horizon, series_count=count, cohort=sorted(frame.cohort.unique().tolist()) if "cohort" in frame else "selected 25-series pool",
            evaluation_grain=grain, metrics=values, metric_units="units/day for MAE; fraction for WAPE/coverage/fill; assumed currency for costs/rewards",
            assumptions=assumptions, outcome_label=label,
            comparison_key=comparison or digest(dict(origins=origins, grain=grain, experiment=experiment)),
            verification_status="RECOMPUTED_FROM_SAVED_RECORDS; no claim of full experiment rerun"))

    foundation_path = "data/warehouse/analytics_quality_report.json"
    foundation = read_json(root/foundation_path)["fact"]
    require(foundation["daily_rows"] == 30490*1913 and foundation["series"] == 30490, "EVIDENCE_SOURCE_DIMENSIONS")
    add(2, "data/warehouse", "m5-validation-foundation", [foundation_path], pd.DataFrame(),
        {k:foundation[k] for k in ["daily_rows","series","products","stores","date_min","date_max","zero_sales_count","missing_price_count"]},
        LABELS["source"], horizon=None, grain="store-item-date; full source fact",
        assumptions="Source fingerprints and warehouse total are independently checked by the current demo; prices may be missing; recorded sales may be censored.")
    records[-1]["series_count"] = foundation["series"]
    records[-1]["cohort"] = "full M5 validation history"

    # Native same-case comparison. Maturity is rechecked against dates, not its boolean alone.
    pointer = read_json(root/"data/serving/m5_h28/latest_verified.json")
    base = pointer["path"]
    relative = base + "/tables/maturity_ledger.parquet"
    ledger = table(relative)
    matched = []
    for checkpoint, fit, val in [("s16-lgb-20160131", "2016-01-31", "2016-01-03"),
                                 ("s17-precalibration-lgb-20151011", "2015-10-11", "2015-10-11")]:
        f = ledger[ledger.model_version.eq(checkpoint)].copy()
        f = f[f.mature & f.monitoring_date.ge(f.origin_date+pd.Timedelta(days=30))]
        require(f.assumed_available_at.le(f.monitoring_date).all() and f.actual.notna().all(), "EVIDENCE_MATURITY")
        keys = ["store_id", "item_id", "origin_date", "horizon", "target_date"]
        require(not f.duplicated(keys).any(), "EVIDENCE_DUPLICATE_KEYS")
        matched.append(set(map(tuple, f[keys].to_numpy())))
        add(23, "serving", checkpoint, [relative], f, metrics(f), LABELS["forecasts"],
            checkpoint=checkpoint, fit=fit, validation=val, comparison="native-serving-same-2100-cases",
            assumptions="Recorded sales target; repaired quantiles; different fit/validation windows. Descriptive comparison, no isolated method effect.")
    require(matched[0] == matched[1], "EVIDENCE_UNMATCHED_CHECKPOINT_CASES")
    from demandos.serving.publication import read_batch
    batch_paths = sorted((root/base/"publication/batches").glob("*/completed.json"))
    batch_rows = 0
    for path in batch_paths:
        frames, _ = read_batch(path.parent)
        require(not frames["forecasts"].duplicated(["store_id","item_id","origin_date","horizon"]).any(), "HISTORICAL_BATCH_KEYS")
        batch_rows += len(frames["forecasts"])
    require(len(batch_paths) == 7 and batch_rows == 4900, "HISTORICAL_PUBLICATION_COVERAGE")
    add(23,"serving","immutable-local-batches", [p.relative_to(root).as_posix() for p in batch_paths],
        ledger, dict(immutable_batches=len(batch_paths), forecast_rows=batch_rows), LABELS["faults"],
        grain="local immutable historical forecast batch", assumptions="Rechecked every batch seal and content hash; no live order dispatch or production SLO claim.")

    relative = "data/conformal/m5_h28/tables/intervals.parquet"
    intervals = table(relative)
    matched = []
    for method in ["neural_base_repaired", "neural_split_cqr", "neural_rolling_cqr"]:
        f = intervals[intervals.method.eq(method) & intervals.cohort.eq("sparse_stress")]
        require(len(f) > 0 and f.model_id.nunique() == 1, "CALIBRATION_CHECKPOINT")
        matched.append(set(map(tuple, f[["store_id", "item_id", "origin_date", "horizon", "model_id"]].to_numpy())))
        coverage = float((f.actual.ge(f.lower) & f.actual.le(f.upper)).mean())
        reported = table("data/conformal/m5_h28/tables/metrics_overall.parquet")
        value = reported[reported.method.eq(method) & reported.cohort.eq("sparse_stress")].coverage.item()
        require(abs(coverage-value) < 1e-12, "CALIBRATION_ARITHMETIC")
        add(17, "conformal", method, [relative], f, dict(n=len(f), coverage=coverage, width=float((f.upper-f.lower).mean())),
            LABELS["forecasts"], checkpoint=f.model_id.iloc[0], fit="2015-10-11", validation="2015-10-11",
            comparison="s17-same-neural-checkpoint-stress", assumptions="80% marginal interval target; temporal dependence, no unconditional finite-sample time-series guarantee.")
    require(matched[0] == matched[1] == matched[2], "CALIBRATION_UNMATCHED_CASES")

    relative = "data/optimization/m5_h28/tables/runs.parquet"
    runs = table(relative)
    for method in ["stochastic_milp", "proportional"]:
        f = runs[runs.study.eq("main") & runs.cost_scenario.eq("balanced") & runs.budget_factor.eq(.7)
                 & runs.method.eq(method) & runs.cohort.eq("continuity") & runs.lead_setting.eq("fixed_7")]
        require(len(f) == 39, "OPTIMIZATION_EVIDENCE_POPULATION")
        cost = f[["holding_cost", "lost_cost", "backlog_cost", "ordering_cost", "procurement_cost"]].sum(axis=1)-f.terminal_credit
        np.testing.assert_allclose(cost, f.operational_cost, atol=1e-7, rtol=0)
        add(20, "optimization", method, [relative], f,
            dict(series_runs=len(f), cost_mean=float(cost.mean()), fill=float(f.immediate_served_units.sum()/f.demand_units.sum())),
            LABELS["optimization"], grain="mean 28-day series-run cost; demand-weighted immediate fill",
            comparison="s20-main-balanced-b0.7-continuity", assumptions="Assumed balanced economics, fixed seven-day lead, same shared pool and hard constraints; fill/cost trade-off.")

    relative = "data/risk/m5_h28/tables/risk_frontier.parquet"
    losses = table(relative)
    scenario_path = "data/risk/m5_h28/tables/scenario_manifest.parquet"
    scenario_manifest = table(scenario_path)
    for method in ["mean", "cvar_2"]:
        f = losses[losses.method_id.eq(method) & losses.alpha.eq(.75)]
        scores = []
        for row in f.itertuples():
            scenarios = scenario_manifest[scenario_manifest.origin_date.eq(row.origin_date) & scenario_manifest.planning_family.eq("baseline")]
            demand = scenarios.pivot(index=["store_id","item_id"], columns="scenario_id", values="protection_demand").sort_index()
            probabilities = scenarios.groupby("scenario_id").probability.first().reindex(demand.columns).to_numpy()
            quantities = np.asarray(json.loads(row.quantities))
            loss = (40*np.abs(demand.to_numpy()-(13+quantities[:,None]))).sum(axis=0)
            score = tail(loss, probabilities, .75)
            require(abs(score["expected"]-row.expected) < 1e-7 and abs(score["cvar"]-row.cvar) < 1e-7, "PLANNING_RISK_ARITHMETIC")
            scores.append(score)
        require(len(scores) == 3, "RISK_EVIDENCE_ORIGINS")
        add(21, "risk", method, [relative,scenario_path], f,
            {k: float(np.mean([s[k] for s in scores])) for k in ["expected", "cvar"]}, LABELS["optimization"],
            grain="origin-mean pool day-zero planning mismatch cost", comparison="s21-static-baseline-planning",
            assumptions="Discrete scenario loss, alpha .75; planning mismatch differs from operating procurement/holding/lost cost.")
    relative = "data/risk/m5_h28/tables/pool_operating.parquet"
    operating = table(relative)
    for method in ["mean", "cvar_2"]:
        f = operating[operating.method_id.eq(method) & operating.scenario_family.eq("baseline") & operating.cohort.eq("ALL")
                      & operating.study.eq("replicated") & operating.world_kind.eq("prefix_bootstrap")]
        scores = [tail(g.operational_cost, np.full(len(g), 1/len(g)), .75) for _,g in f.groupby("origin_date")]
        add(21, "risk", method+"-operating", [relative], f,
            dict(expected=float(np.mean([x["expected"] for x in scores])), cvar=float(np.mean([x["cvar"] for x in scores]))),
            LABELS["inventory"], grain="origin-mean whole-pool simulated operating cost; eight worlds/origin",
            comparison="s21-baseline-eight-operating-worlds", assumptions="No direct comparison of magnitude to first-review planning loss; no guaranteed tail-risk reduction.")

    # Paired complete-policy MODEL_WORLD outcomes: reconstruct joins/differences.
    relative = "data/causal/m5_h28/tables/trajectory_runs.parquet"
    runs = table(relative)
    f = runs[runs.world_kind.eq("prefix_bootstrap") & runs.scenario_family.eq("baseline") & runs.cohort.eq("ALL")]
    keys = ["origin_date", "world_id", "resource_pool_id"]
    pair = f[f.policy_id.eq("mean")].merge(f[f.policy_id.eq("cvar_2")], on=keys, suffixes=("_mean", "_cvar"), validate="one_to_one")
    require(len(pair) == 24 and pair.demand_units_mean.eq(pair.demand_units_cvar).all(), "CAUSAL_PAIRED_WORLDS")
    add(22, "causal", "paired-baseline", [relative], f,
        dict(world_origin_pairs=len(pair), cost_difference=float((pair.operating_cost_cvar-pair.operating_cost_mean).mean()),
             fill_difference=float((pair.fill_cvar-pair.fill_mean).mean())), LABELS["causal"],
        grain="mean paired whole-pool CVaR-minus-mean intervention difference", comparison="s22-paired-baseline",
        assumptions="One shared resource pool; 24 conditional simulated world/origin pairs, no measured Walmart treatment effect.")

    # Independently reconstruct one known-propensity cross-fitted OPE estimate.
    logs = table("data/causal/m5_h28/tables/synthetic_logs.parquet")
    predictions = table("data/causal/m5_h28/tables/nuisance_predictions.parquet")
    nuisance = predictions[predictions.replication_id.eq(0) & predictions.model.eq("ridge")]
    f = logs[logs.replication_id.eq(0)].merge(nuisance, on=["replication_id", "context_id", "world_id", "origin_date"], validate="one_to_one")
    # Mean policy deterministically selects action zero; held-out origins only.
    w = f.logged_action.eq(0).to_numpy()/f.mu_0.to_numpy()
    y = f.reward.to_numpy(); dm = f.prediction_0.to_numpy()
    logged_prediction = np.where(f.logged_action.eq(0), f.prediction_0, f.prediction_1)
    values = dict(ips=float(np.mean(w*y)), snips=float(np.sum(w*y)/w.sum()), dm=float(dm.mean()), dr=float(np.mean(dm+w*(y-logged_prediction))))
    estimates = table("data/causal/m5_h28/tables/ope_estimates.parquet")
    for estimator, value in values.items():
        chosen = estimates[estimates.replication_id.eq(0) & estimates.model.eq("ridge") & estimates.propensity_source.eq("known")
                           & estimates.target.eq("mean") & estimates.population.eq("pooled") & estimates.clipping.eq("none") & estimates.estimator.eq(estimator)]
        require(len(chosen) == 1 and abs(chosen.estimate.item()-value) < 1e-7, "OPE_RECONCILIATION", estimator)
    add(22, "causal", "known-propensity-ridge-rep0-mean-policy", ["data/causal/m5_h28/tables/"+n+".parquet" for n in ["synthetic_logs", "nuisance_predictions", "ope_estimates"]],
        f, dict(n=len(f), **values), LABELS["logging"], grain="held-out context/world one-action negative simulated cost reward",
        comparison="s22-known-propensity-single-rep", horizon=14, assumptions="Synthetic randomized logging with known positivity; nuisance predictions cross-fitted chronologically; no actual M5 propensities.")
    return dict(schema_version=1, source_fingerprint=source_fingerprint, records=records,
                claim_labels=LABELS, comparison_rule="Match population, origin, horizon, outcome, grain, economics, availability and support. Isolated method attribution additionally requires same fitted checkpoint.")
