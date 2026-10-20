"""Read-only M5 extraction, frozen forecast provenance, and bounded scenarios."""
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import tomllib

import numpy as np
import pandas as pd

from demandos.warehouse.cli import settings, connection, build_artifacts
from demandos.features.pipeline import totals
from demandos.data.pipeline import fingerprint
from demandos.backtesting.pipeline import publish
from demandos.eda.pipeline import json_ready
from demandos.reconciliation.pipeline import peak_memory
from .core import Economics, Distribution, integer, normal_rop, cycle
from .experiment import KEY, CASE, METHODS, ORIGINS, STATUS, forecast_bundle, plan, evaluate, matched, summarize
from .scenarios import targets, period_distribution


def load_config(path):
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    if set(config) != {"inventory", "costs", "leads"}:
        raise ValueError("Expected inventory/costs/leads configuration")
    c = config["inventory"]
    expected = {"seed", "samples", "lookback_days", "block_days", "horizon", "currency", "sales_as_demand", "mode",
                "on_hand", "on_order", "backorders", "fixed_quantity", "order_multiple", "minimum_order", "tail_spread_floor"}
    if set(c) != expected or c["sales_as_demand"] is not True or c["mode"] != "lost_sales" or c["backorders"] != 0:
        raise ValueError("Explicit lost-sales / sales-as-demand scenario with no backlog required")
    for name, low, high in [("samples", 16, 2048), ("lookback_days", 28, 730), ("block_days", 1, 14), ("horizon", 28, 28)]:
        integer(c[name], name, low)
        if c[name] > high:
            raise ValueError(f"Bounded {name} exceeded")
    for name in ["seed", "on_hand", "on_order", "backorders", "fixed_quantity", "minimum_order"]:
        integer(c[name], name)
    integer(c["order_multiple"], minimum=1)
    if not isinstance(c["currency"], str) or not c["currency"].strip() or not np.isfinite(c["tail_spread_floor"]) or c["tail_spread_floor"] <= 0:
        raise ValueError("Explicit currency and positive tail spread floor required")
    for collection in (config["costs"], config["leads"]):
        if len(collection) < 3 or len({x["name"] for x in collection}) != len(collection):
            raise ValueError("At least three uniquely named cost/lead scenarios required")
    for cost in config["costs"]:
        if set(cost) != {"name", "price", "purchase", "salvage", "shortage_penalty", "overage_penalty"}:
            raise ValueError("Unknown/missing economic assumptions")
        Economics(**{k: v for k, v in cost.items() if k != "name"})
    for lead in config["leads"]:
        if set(lead) != {"name", "days", "probabilities", "review_days"} or len(set(lead["days"])) != len(lead["days"]):
            raise ValueError("Invalid lead scenario")
        integer(lead["review_days"], minimum=1)
        for day in lead["days"]:
            integer(day, minimum=1)
        Distribution(lead["days"], lead["probabilities"])
        if max(lead["days"])+lead["review_days"] > c["horizon"]:
            raise ValueError("Incomplete configured protection window; no silent truncation")
    return config


def read_inputs(root, sources):
    """Use actual saved forecasts and retain hashes of their complete provenance."""
    directories = {15: root/"data/reconciliation/m5_cohort_h28", 16: root/"data/quantiles/global_h28", 17: root/"data/conformal/m5_h28"}
    summaries = {}
    hashes = {}
    for season, directory in directories.items():
        summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "passed" or summary["source_fingerprints"] != sources:
            raise ValueError(f"Season {season} successful source-bound artifacts required")
        summaries[str(season)] = summary
        for path in [directory/"summary.json", *sorted((directory/"models").rglob("*"))]:
            if path.is_file():
                hashes[path.relative_to(root).as_posix()] = fingerprint(path)
    def read(season, name):
        path = directories[season]/"tables"/(name+".parquet")
        hashes[path.relative_to(root).as_posix()] = fingerprint(path)
        return pd.read_parquet(path)
    selected = read(16, "evaluation_cohort")
    if len(selected) != 25 or selected.duplicated(["store_id", "item_id"]).any() or selected.cohort.value_counts().to_dict() != {"continuity": 13, "sparse_stress": 12}:
        raise ValueError("Established disjoint 13/12 cohorts required")
    nodes = read(15, "cohort_node_map")
    point = read(15, "independent_bases")
    point = point[point.level.eq("ITEM_STORE")].merge(nodes[["node_id", "store_id", "item_id"]], on="node_id", validate="many_to_one")
    point = point.merge(selected, on=["store_id", "item_id"], validate="many_to_one")
    q = read(16, "forecasts")
    q = q[q.model.eq("lightgbm_quantiles")].copy()
    ledger = read(17, "forecast_score_ledger")
    ledger = ledger[ledger.family.eq("lightgbm") & ledger.origin_date.isin(ORIGINS)].copy()
    intervals = read(17, "intervals")
    intervals = intervals[intervals.method.eq("lightgbm_rolling_cqr")].copy()
    # Bind Season 17 reference files to exactly the saved Season 16 artifacts.
    for name, filename in [("summary", "summary.json"), ("forecasts", "tables/forecasts.parquet")]:
        if summaries["17"]["reference_provenance"][name] != fingerprint(directories[16]/filename):
            raise ValueError("Season 16/17 reference fingerprint mismatch")
    for record in summaries["15"]["checkpoint_provenance"]:
        directory = root/record["path"]
        for filename, expected in [("metadata.json", record["metadata_fingerprint"]), ("weights.pt", record["weights"])]:
            path = directory/filename
            if fingerprint(path) != expected:
                raise ValueError("Season 15 point checkpoint changed")
            hashes[path.relative_to(root).as_posix()] = expected
    for frame in (point, q, ledger, intervals):
        if len(frame) != 2100 or frame.duplicated(KEY+["horizon"]).any():
            raise ValueError("Incomplete or duplicated saved forecast coverage")
    bundles = {}
    for key, group in q.groupby(KEY):
        selection = lambda frame: frame.loc[frame.store_id.eq(key[0]) & frame.item_id.eq(key[1]) & frame.origin_date.eq(key[2])]
        bundles[key] = forecast_bundle(selection(point), group, selection(ledger), selection(intervals), key[2])
    provenance = dict(artifact_hashes=hashes, prior_model_provenance={
        "15": summaries["15"]["checkpoint_provenance"],
        "16": {k: summaries["16"][k] for k in ["tree_fits", "neural_fits"]},
        "17": summaries["17"]["models"]})
    return selected, bundles, q, provenance


def controlled():
    """Exact synthetic joint distributions: clearly separate from real M5."""
    from itertools import product
    independent = np.array(list(product([0, 10], repeat=4)))
    shared = np.array([[0]*4, [10]*4])
    weekly = np.array([[0, 0, 10, 0, 0, 10, 0]*2, [10, 0, 0, 10, 0, 0, 0]*2])
    burst = np.array([[0]*14, [0]*14, [0]*14, [0]*14, [20]*7+[0]*7])
    families = dict(independent_daily=independent, positive_shared_shocks=shared, repeated_weekly=weekly,
                    zero_inflated_burst_clusters=burst, smooth=np.full((2, 14), 5), all_zero=np.zeros((2, 14), int))
    rows = []
    for name, y in families.items():
        d = Distribution(y.sum(axis=1))
        for cu, co in [(10, 10), (90, 10), (10, 50)]:
            econ = Economics(cu, 0, 0, overage_penalty=co)
            q = d.quantile(econ.fractile)
            daily_sum = sum(Distribution(y[:, h]).quantile(.9) for h in range(y.shape[1]))
            rows.append(dict(family=name, status="EXACT_SYNTHETIC_TOY", days=y.shape[1], cu=cu, co=co,
                mean=d.mean, variance=d.variance, fractile=econ.fractile, optimal_q=q,
                cumulative_p90=d.quantile(.9), sum_daily_p90=daily_sum, **d.metrics(q, econ)))
    return pd.DataFrame(rows)


def distribution_tables(paths, key, config, actual):
    periods, curves, sensitivity = [], [], []
    for lead in config["leads"]:
        for name, y in paths.items():
            dl = period_distribution(y, lead["days"], lead["probabilities"])
            dp = period_distribution(y, lead["days"], lead["probabilities"], lead["review_days"])
            for kind, d in [("lead", dl), ("protection", dp)]:
                periods.append(pd.DataFrame(dict(zip(KEY, key)), index=np.arange(len(d.values))).assign(
                    lead_setting=lead["name"], distribution=name, period=kind, demand=d.values.astype(int), probability=d.p))
            for cost in config["costs"]:
                econ = Economics(**{k: v for k, v in cost.items() if k != "name"})
                quantities, costs = dp.curve(econ)
                optimum = dp.quantile(econ.fractile)
                if not np.isclose(dp.metrics(optimum, econ)["expected_cost"], costs.min(), atol=1e-7, rtol=1e-10):
                    raise AssertionError("Discrete cost-fractile target fails support enumeration")
                curves.append(pd.DataFrame(dict(zip(KEY, key)), index=np.arange(len(quantities))).assign(
                    lead_setting=lead["name"], distribution=name, cost_regime=cost["name"], quantity=quantities, expected_cost=costs))
                sensitivity.append(dict(zip(KEY, key), lead_setting=lead["name"], distribution=name, cost_regime=cost["name"],
                    lead_mean=dl.mean, lead_variance=dl.variance, protection_mean=dp.mean, protection_variance=dp.variance,
                    q=optimum, expected_cost=dp.metrics(optimum, econ)["expected_cost"],
                    realized_aggregate_target_cost=sum(p*float(econ.cost(optimum, np.sum(actual[:l+lead["review_days"]])))
                        for l, p in zip(lead["days"], lead["probabilities"])), status=STATUS))
    return pd.concat(periods, ignore_index=True), pd.concat(curves, ignore_index=True), pd.DataFrame(sensitivity)


def run(root, config):
    start = perf_counter()
    root, source, warehouse, env = settings(root)
    build_artifacts(root, source, warehouse)
    files = [source, root/"data/interim/sales_long.parquet", *sorted((root/"data/raw").glob("*.csv"))]
    before = {path.relative_to(root).as_posix(): fingerprint(path) for path in files}
    selected, bundles, labels, provenance = read_inputs(root, before)
    print("Verified saved point/quantile/conformal inputs; extracting only 25 series", flush=True)
    with connection(warehouse, env) as con:
        initial = totals(con)
        expected = json.loads((warehouse/"analytics_quality_report.json").read_text())["fact"]
        if any(str(value) != str(expected[name]) for name, value in initial.items()):
            raise ValueError("Verified warehouse totals changed")
        con.register("inventory_keys", selected[["store_id", "item_id"]])
        daily = con.execute("""select f.store_id,f.item_id,f.date,f.units_sold,f.sell_price
            from main.fact_daily_sales f join inventory_keys k using(store_id,item_id)
            where f.date > ? and f.date <= ? order by store_id,item_id,date""",
            [ORIGINS[0]-pd.Timedelta(days=config["inventory"]["lookback_days"]), ORIGINS[-1]+pd.Timedelta(days=28)]).df()
        extracted = len(daily)
        if totals(con) != initial:
            raise ValueError("Warehouse totals modified during extraction")
    decisions, evaluations, audits, gates, periods, curves, sensitivity, path_frames = [], [], [], [], [], [], [], []
    groups = {key: values.reset_index(drop=True) for key, values in daily.groupby(["store_id", "item_id"])}
    labels_by_key = dict(tuple(labels.groupby(KEY)))
    cohort_by_key = selected.set_index(["store_id", "item_id"]).cohort.to_dict()
    for key, bundle in bundles.items():
        history = groups[key[:2]]
        paths, current, audit = plan(history, key[2], bundle, config, key[:2])
        observed = history[history.date.gt(key[2]) & history.date.le(key[2]+pd.Timedelta(days=28))].rename(columns={"date": "target_date"})
        observed = targets(observed, key[2])
        reference = targets(labels_by_key[key], key[2])
        np.testing.assert_array_equal(observed.units_sold, reference.actual)
        current["cohort"] = cohort_by_key[key[:2]]
        outcome = evaluate(current, paths, observed.units_sold.to_numpy(int), config)
        outcome["cohort"] = cohort_by_key[key[:2]]
        decisions.append(current); evaluations.append(outcome)
        audits.append(dict(zip(KEY, key), cohort=cohort_by_key[key[:2]], **audit))
        for perturbation in ["mutate_future_sales", "remove_future_sales", "mutate_future_prices"]:
            altered = history.copy()
            future = altered.date.gt(key[2])
            if perturbation == "mutate_future_sales": altered.loc[future, "units_sold"] += 1000000
            elif perturbation == "remove_future_sales": altered = altered.loc[~future].copy()
            else: altered.loc[future, "sell_price"] = 999999.
            other_paths, other, other_audit = plan(altered, key[2], bundle, config, key[:2])
            pd.testing.assert_frame_equal(current.drop(columns="cohort"), other)
            if audit != other_audit or paths.keys() != other_paths.keys():
                raise AssertionError("Future data changed scenario provenance")
            for name in paths:
                np.testing.assert_array_equal(paths[name], other_paths[name])
            gates.append(dict(zip(KEY, key), perturbation=perturbation, passed=True, paths_distributions_decisions_unchanged=True))
        p, curve, s = distribution_tables(paths, key, config, observed.units_sold.to_numpy(int))
        periods.append(p); curves.append(curve); sensitivity.append(s)
        for name, y in paths.items():
            f = pd.DataFrame(y, columns=[f"demand_h{h}" for h in range(1, 29)])
            for column, value in dict(zip(KEY, key)).items(): f[column] = value
            f["distribution"], f["sample_id"], f["status"] = name, np.arange(len(y)), STATUS
            path_frames.append(f)
        if len(audits) % 25 == 0:
            print(f"Planned/evaluated {len(audits)}/75 series-origins; all future-data perturbations unchanged", flush=True)
    decisions, evaluations = pd.concat(decisions, ignore_index=True), pd.concat(evaluations, ignore_index=True)
    common_cases = len(bundles)*len(config["leads"])*len(config["costs"])
    if matched(evaluations) != common_cases or matched(decisions, method_columns=("method",)) != common_cases:
        raise ValueError("Incomplete common series-origin/lead/cost cases per policy")
    if set(evaluations.method) != set(METHODS) or len(decisions) != common_cases*len(METHODS) or len(evaluations) != common_cases*len(METHODS)*3:
        raise ValueError("Missing required policy or action")
    by_cohort = {name: int(count)*len(ORIGINS)*len(config["leads"])*len(config["costs"])
                 for name, count in selected.cohort.value_counts().items()}
    for cohort, expected_cases in by_cohort.items():
        if matched(evaluations[evaluations.cohort.eq(cohort)]) != expected_cases:
            raise ValueError("Continuity/stress decision coverage mismatch")
    if before != {path.relative_to(root).as_posix(): fingerprint(path) for path in files}:
        raise ValueError("Raw/interim/processed sources modified")
    if any(fingerprint(root/name) != digest for name, digest in provenance["artifact_hashes"].items()):
        raise ValueError("Prior forecasts/checkpoints modified")
    tables = dict(decisions=decisions, evaluations=evaluations, provenance=pd.DataFrame(audits),
        future_invariance=pd.DataFrame(gates), demand_paths=pd.concat(path_frames, ignore_index=True),
        period_distributions=pd.concat(periods, ignore_index=True), cost_curves=pd.concat(curves, ignore_index=True),
        dependence_sensitivity=pd.concat(sensitivity, ignore_index=True), controlled_scenarios=controlled(),
        sequential_demo=pd.DataFrame(cycle([[2, 2, 4, 1]], 2, 2, 3, 2, 5, trace=True)["daily"]),
        evaluation_cohort=selected)
    for name, extra in [("metrics", []), ("metrics_lead", ["lead_setting"]), ("metrics_regime", ["regime"]), ("metrics_origin", ["origin_date"])]:
        tables[name] = summarize(evaluations, ["cohort", "cost_regime", "method", "action", *extra])
    for frame in tables.values():
        if "status" not in frame:
            frame["status"] = STATUS
    summary = dict(status="passed", result_label=STATUS, config=config, source_fingerprints=before,
        inputs_unchanged=True, prior_artifacts_unchanged=True, warehouse_totals=initial,
        extracted_observations=extracted, evaluation_origins=ORIGINS.tolist(),
        daily_forecast_cases={"continuity": 1092, "sparse_stress": 1008}, series_origins=75,
        matched_cases_per_method_action=common_cases, matched_by_cohort=by_cohort,
        methods=METHODS, future_invariance_checks=len(gates), table_rows={k: len(v) for k, v in tables.items()},
        provenance=provenance, normal_hand_example=normal_rop(20, 5, 4, .95),
        metrics=tables["metrics"].to_dict("records"), runtime_seconds=perf_counter()-start, peak_process_mib=peak_memory(),
        versions={n: version(n) for n in ["numpy", "pandas", "pyarrow", "duckdb", "scipy"]},
        warnings=["SALES-AS-DEMAND simulation: no verified latent demand, stockouts, inventory, lead times or costs.",
            "Three marginal quantiles do not identify a distribution; piecewise CDF tails and historical rank coupling are assumptions.",
            "Conformal width inflation is a heuristic, not guaranteed service or an arbitrary upper quantile.",
            "Aggregate newsvendor optimum is not the optimum of a receipt-constrained sequential policy.",
            "Random lead probabilities are hypothetical and independent of demand; all policies use identical cases.",
            "Repeated origins/overlapping protection windows are comparisons, not independent real inventory cycles.",
            "All-zero-demand fill rates are undefined (JSON null). No model retraining or tuning on evaluated outcomes."],
        skipped_cases=[], training_warnings=0)
    target = root/"data/inventory/m5_h28"
    with TemporaryDirectory(prefix=".inventory-", dir=root/"data") as temporary:
        temporary = Path(temporary)
        (temporary/"tables").mkdir()
        for name, frame in tables.items(): frame.to_parquet(temporary/"tables"/(name+".parquet"), index=False)
        (temporary/"config.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
        from .report import report
        report(summary, tables, temporary)
        summary["runtime_seconds"] = perf_counter()-start
        summary["peak_process_mib"] = peak_memory()
        (temporary/"summary.json").write_text(json.dumps(json_ready(summary), default=str, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        publish(temporary, target)
    return summary, target
