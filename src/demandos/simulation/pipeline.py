"""Bounded real backtests, separate sensitivities, and source-bound publication."""
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import tomllib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from demandos.inventory.core import Economics, integer
from demandos.inventory.scenarios import period_distribution, bootstrap, seed_for, targets
from demandos.inventory.pipeline import load_config as inventory_config, read_inputs
from demandos.inventory.experiment import plan, ORIGINS, KEY
from demandos.warehouse.cli import settings, connection, build_artifacts
from demandos.features.pipeline import totals
from demandos.data.pipeline import fingerprint
from demandos.backtesting.pipeline import publish
from demandos.eda.pipeline import json_ready
from demandos.reconciliation.pipeline import peak_memory
from .engine import Costs, Supplier, simulate, RUN_KEYS, STATUS
from .policies import Policy, Order
from .metrics import run_metrics, aggregate, paired

SOURCES = {"historical_fractile", "point_mean_heuristic", "s16_quantile_paths", "s17_rolling_width_paths"}
STUDIES = {"backorders", "stock_0", "stock_100", "open_0", "review_3", "weekend_delay", "terminal_salvage", "bootstrap_block", "bootstrap_independent"}


def load_config(path):
    with Path(path).open("rb") as handle: c = tomllib.load(handle)
    if set(c) != {"simulation", "operations", "sensitivity", "warmup"}: raise ValueError("Unknown/missing simulation sections")
    s = c["simulation"]
    if set(s) != {"seed", "replications", "sources", "rules", "first_review_day", "fixed_quantity", "on_hand", "on_order", "mode", "review_days", "multiple", "minimum", "sales_as_demand"}:
        raise ValueError("Unknown/missing simulation parameters")
    if s["sales_as_demand"] is not True or s["mode"] != "lost_sales" or set(s["sources"]) != SOURCES or s["rules"] != ["rQ", "rS", "RS"]:
        raise ValueError("Explicit base lost-sales, four sources and three policies required")
    for name in ["seed", "fixed_quantity", "on_hand", "on_order", "minimum"]: integer(s[name])
    for name in ["replications", "first_review_day", "review_days", "multiple"]: integer(s[name], minimum=1)
    if s["replications"] > 4 or s["review_days"] > 14: raise ValueError("Simulation compute bound exceeded")
    if set(c["operations"]) != {"balanced", "shortage_expensive", "overage_expensive"}: raise ValueError("Three operating regimes required")
    for values in c["operations"].values(): Costs(**values, purchase=60.)
    if set(c["sensitivity"]) != {"studies"} or set(c["sensitivity"]["studies"]) != STUDIES or len(c["sensitivity"]["studies"]) != len(STUDIES):
        raise ValueError("Predeclared sensitivity studies required")
    w = c["warmup"]
    if set(w) != {"durations", "scored_days", "replications", "series_per_cohort"} or w["durations"] != [0, 28, 56] or w["scored_days"] != 56 or w["series_per_cohort"] != 1 or not 1 <= w["replications"] <= 4:
        raise ValueError("Bounded shared-tail warm-up design required")
    return c


class Sink:
    """Stream complete experiment groups; never accumulate a dense daily panel."""
    def __init__(self, directory):
        self.directory = Path(directory); self.directory.mkdir()
        self.writers, self.rows = {}, {}

    def write(self, name, frame):
        if frame.empty: return
        if "status" not in frame: frame = frame.assign(status=STATUS)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if name not in self.writers:
            self.writers[name] = pq.ParquetWriter(self.directory/(name+".parquet"), table.schema, compression="zstd")
            self.rows[name] = 0
        writer = self.writers[name]
        writer.write_table(table.cast(writer.schema)); self.rows[name] += len(frame)

    def close(self):
        for writer in self.writers.values(): writer.close()


def parameters(paths, bundle, lead, cost, review, sources):
    econ = Economics(**{k: v for k, v in cost.items() if k != "name"})
    values = {}
    for source in sources:
        if source == "point_mean_heuristic":
            daily = bundle["point"]
            r = int(np.ceil(sum(p*daily[:l].sum() for l, p in zip(lead["days"], lead["probabilities"])) ))
            s = int(np.ceil(sum(p*daily[:l+review].sum() for l, p in zip(lead["days"], lead["probabilities"])) ))
        else:
            y = paths["historical" if source == "historical_fractile" else source]
            r = period_distribution(y, lead["days"], lead["probabilities"]).quantile(econ.fractile)
            s = period_distribution(y, lead["days"], lead["probabilities"], review).quantile(econ.fractile)
        values[source] = (r, s)
    return values


def temporal_checks(demand, origin, policy, costs, supplier, initial):
    """Only PRE-k states/actions must remain fixed when later demand changes."""
    rows = []
    for mode in ["lost_sales", "backorders"]:
        baseline, events = simulate(demand, origin, policy, costs, supplier, initial_orders=initial, mode=mode)
        for k in [10, 20]:
            altered = np.asarray(demand).copy(); altered[k-1] += 10000
            actual, new_events = simulate(altered, origin, policy, costs, supplier, initial_orders=initial, mode=mode)
            pd.testing.assert_frame_equal(baseline.iloc[:k-1], actual.iloc[:k-1])
            old = events[events.event.eq("placed") & events.issue_day.lt(k)].reset_index(drop=True)
            new = new_events[new_events.event.eq("placed") & new_events.issue_day.lt(k)].reset_index(drop=True)
            pd.testing.assert_frame_equal(old, new)
            rows.append(dict(mode=mode, mutation_day=k, test="pre_k_states_actions_orders", passed=True))
    # Hidden lost demand cannot change observations after stock is exhausted.
    first, _ = simulate(np.full(28, 10000), origin, policy, costs, supplier, on_hand=0)
    second, _ = simulate(np.full(28, 100000), origin, policy, costs, supplier, on_hand=0)
    observed = ["ending_stock", "ending_on_order", "ordered", "position_before_order", "action_reason", "immediate_served"]
    pd.testing.assert_frame_equal(first[observed], second[observed])
    rows.append(dict(mode="lost_sales", mutation_day=1, test="hidden_lost_demand_not_observable", passed=True))
    return rows


def run(root, config):
    start = perf_counter()
    root, source, warehouse, env = settings(root); build_artifacts(root, source, warehouse)
    files = [source, root/"data/interim/sales_long.parquet", *sorted((root/"data/raw").glob("*.csv"))]
    before = {p.relative_to(root).as_posix(): fingerprint(p) for p in files}
    selected, bundles, labels, provenance = read_inputs(root, before)
    old_dir = root/"data/inventory/m5_h28"
    old_summary = json.loads((old_dir/"summary.json").read_text())
    if old_summary["status"] != "passed" or old_summary["source_fingerprints"] != before: raise ValueError("Verified Season 18 artifacts required")
    for p in [old_dir/"summary.json", old_dir/"tables/decisions.parquet"]:
        provenance["artifact_hashes"][p.relative_to(root).as_posix()] = fingerprint(p)
    old_plans = pd.read_parquet(old_dir/"tables/decisions.parquet")
    cfg18 = inventory_config(root/"configs/inventory.toml")
    if cfg18 != old_summary["config"]: raise ValueError("Season 18 planning configuration changed")
    provenance["artifact_hashes"]["configs/inventory.toml"] = fingerprint(root/"configs/inventory.toml")
    with connection(warehouse, env) as con:
        initial_totals = totals(con)
        if initial_totals["daily_rows"] != 58327370 or initial_totals["series"] != 30490: raise ValueError("M5 warehouse grain/count changed")
        expected = json.loads((warehouse/"analytics_quality_report.json").read_text())["fact"]
        if any(str(v) != str(expected[k]) for k, v in initial_totals.items()): raise ValueError("Warehouse integrity totals changed")
        con.register("simulation_keys", selected[["store_id", "item_id"]])
        daily = con.execute("""select f.store_id,f.item_id,f.date,f.units_sold,f.sell_price
            from main.fact_daily_sales f join simulation_keys k using(store_id,item_id)
            where date > ? and date <= ? order by store_id,item_id,date""",
            [ORIGINS[0]-pd.Timedelta(days=365), ORIGINS[-1]+pd.Timedelta(days=28)]).df()
    histories = dict(tuple(daily.groupby(["store_id", "item_id"])))
    label_groups = dict(tuple(labels.groupby(KEY)))
    cohorts = selected.set_index(["store_id", "item_id"]).cohort.to_dict()
    settings_base = config["simulation"]
    rows, gates, origin_gates, warm_inputs = [], [], [], []
    seen_shocks, seen_demands = set(), {}
    target = root/"data/simulation/m5_h28"
    with TemporaryDirectory(prefix=".simulation-", dir=root/"data") as temporary:
        temporary = Path(temporary); sink = Sink(temporary/"tables")
        def group(key, demand, params, lead, cost, *, experiment="main", replication=0, mode="lost_sales", on_hand=10,
                  on_order=3, review=7, terminal=0., demand_id="m5_recorded_sales_proxy", warmup=0, offset=0):
            series = json.dumps(list(key[:2]), separators=(",", ":"))
            stream = f"{series}|{key[2].date()}|rep{replication}"
            supplier = Supplier(tuple(lead["days"]), tuple(lead["probabilities"]), settings_base["seed"], stream, lead.get("weekend_delay", 0))
            initial_lead, initial_shock, _ = supplier.at(offset, key[2]+pd.Timedelta(days=offset))
            initial_orders = (Order("initial_0", 0, initial_lead, on_order, initial_lead, initial_shock),) if on_order else ()
            operating = Costs(**config["operations"][cost["name"]], purchase=cost["purchase"], terminal_value=terminal)
            case = dict(series_id=series, store_id=key[0], item_id=key[1], origin_date=key[2],
                cost_scenario_id=cost["name"], lead_scenario_id=lead["name"], demand_scenario_id=demand_id,
                replication_id=replication, experiment_id=experiment, cohort=cohorts[key[:2]], mode=mode,
                initial_on_hand=on_hand, initial_on_order=on_order, initial_arrival_day=initial_lead,
                review_days=review, terminal_value=terminal)
            shocks = []
            for d in range(offset, offset+len(demand)+1):
                shock_key = (series, key[2], replication, lead["name"], d)
                if shock_key not in seen_shocks:
                    date = key[2]+pd.Timedelta(days=d); l, shock, u = supplier.at(d, date)
                    shocks.append(dict(series_id=series, origin_date=key[2], replication_id=replication,
                        lead_scenario_id=lead["name"], calendar_day=d, supplier_shock_id=shock, uniform=u, sampled_lead=l))
                    seen_shocks.add(shock_key)
            sink.write("supplier_shocks", pd.DataFrame(shocks))
            demand_key = (series, key[2], replication if demand_id != "m5_recorded_sales_proxy" else 0, demand_id)
            exogenous = []
            for day, value in enumerate(demand, offset+1):
                date = key[2]+pd.Timedelta(days=day); identity = (*demand_key, date)
                if identity in seen_demands:
                    if seen_demands[identity] != value: raise AssertionError("Conflicting shared exogenous demand")
                    continue
                seen_demands[identity] = value
                exogenous.append(dict(zip(["series_id", "origin_date", "replication_id", "demand_scenario_id"], demand_key),
                    simulation_date=date, demand_proxy=int(value), observed_source_sales=int(value) if demand_id == "m5_recorded_sales_proxy" else None,
                    source_kind="recorded_M5_sales" if demand_id == "m5_recorded_sales_proxy" else "origin_safe_bootstrap"))
            sink.write("exogenous_demand", pd.DataFrame(exogenous))
            frames, event_frames = [], []
            for source_name, (r, s) in params.items():
                for rule in settings_base["rules"]:
                    policy = Policy(rule, r, s, settings_base["fixed_quantity"], review, settings_base["first_review_day"], settings_base["multiple"], settings_base["minimum"])
                    ledger, events = simulate(demand, key[2], policy, operating, supplier, on_hand=on_hand,
                        initial_orders=initial_orders, mode=mode, warmup=warmup, calendar_offset=offset)
                    meta = {**case, "policy_id": f"{experiment}:{source_name}:{rule}", "source": source_name, "rule": rule,
                            "regime": current_regime[key], "frozen_rop": r, "frozen_target": s}
                    rows.append({**meta, **run_metrics(ledger), "status": STATUS})
                    frames.append(ledger.assign(**meta))
                    if not events.empty:
                        events = events.assign(**meta)
                        events["event_date"] = key[2]+pd.to_timedelta(events.event_day+offset, unit="D")
                        events["arrival_date"] = key[2]+pd.to_timedelta(events.arrival_day+offset, unit="D")
                        event_frames.append(events)
            combined = pd.concat(frames, ignore_index=True)
            sink.write("daily_ledger", combined)
            if event_frames: sink.write("order_events", pd.concat(event_frames, ignore_index=True))
            action_columns = ["day_index", "simulation_date", "ordered", "action_reason", "action_threshold", "action_target", "position_before_order", "position_after_order"]
            sink.write("actions", combined[action_columns+list(meta)])
            reference = next(f for f in frames if f.source.iloc[0] == "historical_fractile" and f.rule.iloc[0] == "RS")
            daily_pairs = []
            for frame in frames:
                np.testing.assert_array_equal(frame.demand_proxy, reference.demand_proxy)
                np.testing.assert_array_equal(frame.supplier_shock_id, reference.supplier_shock_id)
                f = frame[RUN_KEYS+["experiment_id", "day_index", "simulation_date", "scored"]].copy()
                f["cost_difference"] = frame.operational_cost.to_numpy()-reference.operational_cost.to_numpy()
                f["reference_policy"] = "historical_fractile:RS"
                daily_pairs.append(f)
            sink.write("paired_daily_costs", pd.concat(daily_pairs, ignore_index=True))
        current_regime = {}
        try:
            for index, (key, bundle) in enumerate(bundles.items(), 1):
                history = histories[key[:2]]
                paths, plans, audit = plan(history, key[2], bundle, cfg18, key[:2])
                current_regime[key] = plans.regime.iloc[0]
                saved = old_plans[old_plans.store_id.eq(key[0]) & old_plans.item_id.eq(key[1]) & old_plans.origin_date.eq(key[2])]
                pd.testing.assert_frame_equal(plans.sort_values(["method", "lead_setting", "cost_regime"])[["rop", "target"]].reset_index(drop=True),
                    saved.sort_values(["method", "lead_setting", "cost_regime"])[["rop", "target"]].reset_index(drop=True))
                for change in ["mutate_future_sales", "remove_future_sales", "mutate_future_prices"]:
                    altered = history.copy(); mask = altered.date.gt(key[2])
                    if change == "mutate_future_sales": altered.loc[mask, "units_sold"] += 1000000
                    elif change == "remove_future_sales": altered = altered.loc[~mask]
                    else: altered.loc[mask, "sell_price"] = 999999.
                    pp, qq, _ = plan(altered, key[2], bundle, cfg18, key[:2])
                    pd.testing.assert_frame_equal(plans, qq)
                    for name in paths: np.testing.assert_array_equal(paths[name], pp[name])
                    origin_gates.append(dict(zip(KEY, key), test=change, passed=True))
                actual = targets(label_groups[key], key[2]).actual.to_numpy(int)
                future = history[history.date.gt(key[2]) & history.date.le(key[2]+pd.Timedelta(days=28))]
                np.testing.assert_array_equal(actual, future.units_sold)
                for cost in cfg18["costs"]:
                    for lead in cfg18["leads"]:
                        params = parameters(paths, bundle, lead, cost, settings_base["review_days"], settings_base["sources"])
                        for replication in range(settings_base["replications"]):
                            group(key, actual, params, lead, cost, replication=replication, on_hand=settings_base["on_hand"],
                                  on_order=settings_base["on_order"], review=settings_base["review_days"])
                cost = next(c for c in cfg18["costs"] if c["name"] == "balanced")
                lead = next(l for l in cfg18["leads"] if l["name"] == "fixed_7")
                for study in config["sensitivity"]["studies"]:
                    settings_study = dict(experiment=study)
                    review, scenario_lead, demand = 7, lead, actual
                    if study == "backorders": settings_study["mode"] = "backorders"
                    elif study == "stock_0": settings_study["on_hand"] = 0
                    elif study == "stock_100": settings_study["on_hand"] = 100
                    elif study == "open_0": settings_study["on_order"] = 0
                    elif study == "review_3": review = 3
                    elif study == "weekend_delay": scenario_lead = {**lead, "name": "fixed_7_weekend_plus_2", "weekend_delay": 2}
                    elif study == "terminal_salvage": settings_study["terminal"] = cost["salvage"]
                    else:
                        family = "historical" if study == "bootstrap_block" else "historical_independent"
                        demand = paths[family][0]
                        settings_study["demand_id"] = "origin_safe_"+family
                    params = parameters(paths, bundle, scenario_lead, cost, review, settings_base["sources"])
                    group(key, demand, params, scenario_lead, cost, review=review, **settings_study)
                random_lead = cfg18["leads"][-1]
                supplier = Supplier(tuple(random_lead["days"]), tuple(random_lead["probabilities"]), settings_base["seed"], f"temporal|{key}")
                init = (Order("initial_0", 0, 3, 3, 3, "fixed_initial"),)
                for rule in settings_base["rules"]:
                    r, s = parameters(paths, bundle, random_lead, cost, 7, ["s16_quantile_paths"])["s16_quantile_paths"]
                    checks = temporal_checks(actual, key[2], Policy(rule, r, s), Costs(**config["operations"]["balanced"], purchase=60), supplier, init)
                    gates.extend(dict(zip(KEY, key), rule=rule, **check) for check in checks)
                if key[2] == ORIGINS[0]: warm_inputs.append((key, history, paths, bundle, cost, lead))
                if index % 25 == 0: print(f"Simulated {index}/75 series-origins; conservation and temporal gates passed", flush=True)
            # Separate common-tail bootstrap study: exactly the same last 56 scored days.
            warm_selected = set(selected.groupby("cohort").head(1)[["store_id", "item_id"]].itertuples(index=False, name=None))
            for key, history, paths, bundle, cost, lead in warm_inputs:
                if key[:2] not in warm_selected: continue
                params = parameters(paths, bundle, lead, cost, 7, ["historical_fractile"])
                for block in [1, 7]:
                    for replication in range(config["warmup"]["replications"]):
                        path, _, _ = bootstrap(history, key[2], samples=1, horizon=112, block=block,
                            seed=seed_for(settings_base["seed"], "warmup", *key, replication))
                        for warmup in config["warmup"]["durations"]:
                            offset = 56-warmup
                            group(key, path[0, offset:], params, lead, cost, experiment=f"warmup_{warmup}_block{block}",
                                  replication=replication, demand_id=f"warmup_bootstrap_block{block}", warmup=warmup, offset=offset)
        finally:
            sink.close()
        runs = pd.DataFrame(rows)
        pairs, paired_summary = paired(runs)
        main = runs[runs.experiment_id.eq("main")]
        expected_runs = 75*3*3*settings_base["replications"]*len(settings_base["sources"])*3
        if len(main) != expected_runs or main.scored_days.ne(28).any() or main.warmup_days.ne(0).any(): raise AssertionError("Main coverage mismatch")
        if main.groupby(["cohort", "policy_id", "cost_scenario_id", "lead_scenario_id", "replication_id"]).scored_days.sum().unique().tolist() != [1092, 1008]:
            # Group sorting is deterministic, but compare sets for portability.
            if set(main.groupby(["cohort", "policy_id", "cost_scenario_id", "lead_scenario_id", "replication_id"]).scored_days.sum()) != {1092, 1008}:
                raise AssertionError("Continuity/stress daily coverage changed")
        generated = dict(runs=runs, paired_runs=pairs, paired_summary=paired_summary,
            temporal_checks=pd.DataFrame(gates), origin_checks=pd.DataFrame(origin_gates), evaluation_cohort=selected)
        for name, extra in [("metrics", []), ("metrics_origin", ["origin_date"]), ("metrics_regime", ["regime"])]:
            generated[name] = aggregate(runs, ["experiment_id", "cohort", "cost_scenario_id", "lead_scenario_id", "source", "rule", *extra])
        metrics = generated["metrics"]
        generated["sensitivity_metrics"] = metrics[~metrics.experiment_id.eq("main") & ~metrics.experiment_id.str.startswith("warmup_")].copy()
        generated["warmup_metrics"] = metrics[metrics.experiment_id.str.startswith("warmup_")].copy()
        generated["fulfillment_mode_metrics"] = metrics[metrics.experiment_id.isin(["main", "backorders"]) &
            metrics.cost_scenario_id.eq("balanced") & metrics.lead_scenario_id.eq("fixed_7")].copy()
        for name, frame in generated.items(): frame.to_parquet(temporary/"tables"/(name+".parquet"), index=False)
        from .verification import verify
        verification = verify(temporary)
        if before != {p.relative_to(root).as_posix(): fingerprint(p) for p in files}: raise ValueError("Source files modified")
        if any(fingerprint(root/p) != h for p, h in provenance["artifact_hashes"].items()): raise ValueError("Prior artifacts modified")
        with connection(warehouse, env) as con:
            if totals(con) != initial_totals: raise ValueError("Warehouse totals changed")
        summary = dict(status="passed", label=STATUS, config=config, inventory_config=cfg18, source_fingerprints=before,
            provenance=provenance, inputs_unchanged=True, warehouse_totals=initial_totals, extracted_rows=len(daily),
            main_runs=len(main), main_daily_rows=len(main)*28, unique_real_demand_paths=75, unique_real_daily_observations=2100,
            main_policy_count=12, daily_coverage_per_policy_scenario_replication={"continuity":1092,"sparse_stress":1008},
            total_runs=len(runs), table_rows={**sink.rows, **{k:len(v) for k,v in generated.items()}},
            conservation=dict(passed=True, daily_rows=sink.rows["daily_ledger"], maximum_gap=0, queue_id_reconciliation=True),
            independent_sql_gates=verification["independent_sql_gates"],
            versions={name: version(name) for name in ["numpy", "pandas", "pyarrow", "duckdb", "matplotlib"]},
            temporal_checks=len(gates), origin_checks=len(origin_gates), runtime_seconds=perf_counter()-start, peak_process_mib=peak_memory(),
            skipped_cases=[], warnings=["SCENARIO-BASED: recorded sales are a demand proxy, not verified uncensored demand.",
                "Origin targets are frozen stationary planning approximations, not daily reforecasting.",
                "Three marginal quantiles and conformal widths do not identify joint demand or inventory service guarantees.",
                "Procurement is paid at order, including terminal outstanding orders; no terminal backlog tail cost.",
                "Daily no-shortage rate is not replenishment CSL; no cycle segmentation is claimed.",
                "Paired uncertainty is descriptive series-origin clustering; repeated items/origins remain dependent.",
                "Weekend delays use the frozen baseline lead distribution for planning: explicit unmodeled-delay sensitivity."])
        from .report import report
        report(summary, generated, temporary)
        summary["runtime_seconds"], summary["peak_process_mib"] = perf_counter()-start, peak_memory()
        (temporary/"summary.json").write_text(json.dumps(json_ready(summary), default=str, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        (temporary/"config.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
        (temporary/"conservation.json").write_text(json.dumps(summary["conservation"], indent=2)+"\n", encoding="utf-8")
        publish(temporary, target)
    return summary, target
