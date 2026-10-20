"""Frozen forecast contracts, origin-safe planning, and matched single decisions."""
import hashlib

import numpy as np
import pandas as pd

from demandos.intermittent.diagnostics import statistics
from .core import Economics, Distribution, inventory_position, policy, cycle, service, counts
from .scenarios import bootstrap, seed_for, quantile_paths, period_distribution, targets

KEY = ["store_id", "item_id", "origin_date"]
CASE = KEY+["lead_setting", "cost_regime"]
METHODS = ["historical_mean", "point_mean_heuristic", "daily_median_sum_heuristic",
           "historical_fractile", "s16_quantile_paths", "s17_base_paths", "s17_rolling_width_paths"]
ORIGINS = pd.to_datetime(["2016-01-31", "2016-02-28", "2016-03-27"])
STATUS = "SIMULATED_SALES_AS_DEMAND"


def forecast_bundle(point, s16, s17, intervals, origin):
    """Drop every observed outcome before making decisions; enforce fit timing."""
    p, q, b, c = [targets(frame, origin) for frame in (point, s16, s17, intervals)]
    for frame, columns in [(p, ["fit_cutoff", "fit_target_max", "history_cutoff"]),
                           (q, ["fit_cutoff", "fit_target_max"]),
                           (b, ["model_fit_cutoff", "fit_target_max", "selection_cutoff"])]:
        if any(pd.to_datetime(frame[name]).isna().any() or pd.to_datetime(frame[name]).gt(origin).any() for name in columns):
            raise ValueError("Future-fitted/unknown forecast provenance")
    for frame in (q, b):
        values = frame[["q10", "q50", "q90"]].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (np.diff(values, axis=1) < 0).any():
            raise ValueError("Invalid repaired daily forecast quantiles")
    if not np.isfinite(p.prediction).all() or p.prediction.lt(0).any():
        raise ValueError("Invalid point forecast")
    if not np.isfinite(c[["lower", "upper"]]).all().all() or c.upper.lt(c.lower).any():
        raise ValueError("Invalid saved conformal interval")
    if not np.allclose(c.base_lower, b.q10) or not np.allclose(c.base_upper, b.q90) or not np.array_equal(c.model_id, b.model_id):
        raise ValueError("Conformal interval must use the same frozen base checkpoint")
    return dict(point=p.prediction.to_numpy(float), s16=q[["q10", "q50", "q90"]].to_numpy(float),
                s17=b[["q10", "q50", "q90"]].to_numpy(float),
                width=(c.upper-c.lower).to_numpy(float),
                model_id=str(b.model_id.iloc[0]), status=STATUS)


def plan(history, origin, bundle, config, identity):
    """No evaluation outcomes/prices in this API. All randomness is key-stable."""
    c = config["inventory"]
    seed = seed_for(c["seed"], *identity, str(pd.Timestamp(origin).date()))
    arguments = dict(samples=c["samples"], lookback=c["lookback_days"], horizon=c["horizon"], seed=seed)
    historical, ranks, provenance = bootstrap(history, origin, block=c["block_days"], **arguments)
    independent, _, _ = bootstrap(history, origin, block=1, **arguments)
    factor = np.maximum(1., bundle["width"]/np.maximum(bundle["s17"][:, 2]-bundle["s17"][:, 0], c["tail_spread_floor"]))
    paths = dict(historical=historical, historical_independent=independent,
                 s16_quantile_paths=quantile_paths(bundle["s16"], ranks, c["tail_spread_floor"]),
                 s17_base_paths=quantile_paths(bundle["s17"], ranks, c["tail_spread_floor"]),
                 s17_rolling_width_paths=quantile_paths(bundle["s17"], ranks, c["tail_spread_floor"], factor))
    diag = statistics(history, origin, window=c["lookback_days"])
    ip = inventory_position(c["on_hand"], c["on_order"], c["backorders"])
    rows = []
    for lead in config["leads"]:
        lead_dist = {name: period_distribution(y, lead["days"], lead["probabilities"]) for name, y in paths.items()}
        protection = {name: period_distribution(y, lead["days"], lead["probabilities"], lead["review_days"]) for name, y in paths.items()}
        for costs in config["costs"]:
            econ = Economics(**{k: v for k, v in costs.items() if k != "name"})
            for method in METHODS:
                distribution = method if method in paths else "historical"
                dl, dp = lead_dist[distribution], protection[distribution]
                rop, target = dl.quantile(econ.fractile), dp.quantile(econ.fractile)
                if method == "historical_mean":
                    rop, target = int(np.ceil(dl.mean)), int(np.ceil(dp.mean))
                elif method in ("point_mean_heuristic", "daily_median_sum_heuristic"):
                    daily = bundle["point"] if method == "point_mean_heuristic" else bundle["s16"][:, 1]
                    rop = int(np.ceil(sum(p*daily[:l].sum() for l, p in zip(lead["days"], lead["probabilities"]))))
                    target = int(np.ceil(sum(p*daily[:l+lead["review_days"]].sum() for l, p in zip(lead["days"], lead["probabilities"]))))
                decisions = policy(ip, rop, target, c["fixed_quantity"], c["order_multiple"], c["minimum_order"])
                metrics = dp.metrics(target, econ)
                optimum = dp.quantile(econ.fractile)
                rows.append(dict(store_id=identity[0], item_id=identity[1], origin_date=pd.Timestamp(origin),
                    lead_setting=lead["name"], cost_regime=costs["name"], method=method, distribution=distribution,
                    status=STATUS, cu=econ.cu, co=econ.co, critical_fractile=econ.fractile,
                    rop=rop, target=target, position=ip, signed_safety_offset=rop-dl.mean,
                    nonnegative_buffer=max(0., rop-dl.mean), lead_mean=dl.mean, lead_variance=dl.variance,
                    protection_mean=dp.mean, protection_variance=dp.variance,
                    p10=dp.quantile(.1), p50=dp.quantile(.5), p90=dp.quantile(.9),
                    scenario_optimal_target=optimum, own_expected_cost=metrics["expected_cost"],
                    own_expected_shortage=metrics["shortage"], own_expected_leftover=metrics["leftover"],
                    own_csl=metrics["csl"], own_fill_rate=metrics["fill_rate"],
                    own_distribution_regret=max(0., metrics["expected_cost"]-dp.metrics(optimum, econ)["expected_cost"]),
                    common_historical_expected_cost=protection["historical"].metrics(target, econ)["expected_cost"],
                    regime=diag["regime"], zero_sales_fraction=diag["zero_sales_fraction"], adi=diag["adi"], cv2=diag["cv2"],
                    **decisions))
    provenance.update(model_id=bundle["model_id"], max_width_factor=float(factor.max()),
                      history_digest=hashlib.sha256(historical.tobytes()).hexdigest())
    return paths, pd.DataFrame(rows), provenance


def evaluate(decisions, paths, actual, config):
    """Matched one-order outcomes, and separately labeled aggregate stock targets."""
    c = config["inventory"]
    actual = counts(actual)
    if actual.ndim != 1:
        raise ValueError("One observed target vector required")
    actual = actual.astype(int)[None, :]
    if actual.shape != (1, c["horizon"]):
        raise ValueError("Complete observed target vector required")
    rows = []
    for decision in decisions.to_dict("records"):
        lead = next(x for x in config["leads"] if x["name"] == decision["lead_setting"])
        costs = next(x for x in config["costs"] if x["name"] == decision["cost_regime"])
        econ = Economics(**{k: v for k, v in costs.items() if k != "name"})
        for action in ["periodic_up_to", "continuous_up_to", "continuous_fixed"]:
            quantity = decision[action]
            expected, observed = [], []
            all_sim_demand, all_sim_served, all_sim_weights = [], [], []
            for l, p in zip(lead["days"], lead["probabilities"]):
                sim = cycle(paths[decision["distribution"]], l, lead["review_days"], c["on_hand"], c["on_order"], quantity)
                outcome = cycle(actual, l, lead["review_days"], c["on_hand"], c["on_order"], quantity)
                expected.append(p*float(np.mean(econ.cu*sim["shortage"]+econ.co*sim["leftover"])))
                all_sim_demand.extend(sim["demand"]); all_sim_served.extend(sim["served"])
                all_sim_weights.extend(np.full(len(sim["demand"]), p/len(sim["demand"])))
                before, after = actual[0, :l].sum(), actual[0, l:l+lead["review_days"]].sum()
                # Clairvoyant oracle may choose a different order for each realized L.
                # Initial overhang cannot be discarded; pre-receipt lost sales are irrevocable.
                oracle_cost = econ.cu*max(0, before-c["on_hand"])+econ.co*max(0, max(0, c["on_hand"]-before)+c["on_order"]-after)
                realized_cost = float(econ.cu*outcome["shortage"][0]+econ.co*outcome["leftover"][0])
                if realized_cost < oracle_cost-1e-8:
                    raise AssertionError("Executed cost fell below clairvoyant lower bound")
                observed.append(dict(weight=p, demand=float(outcome["demand"][0]), served=float(outcome["served"][0]),
                    shortage=float(outcome["shortage"][0]), leftover=float(outcome["leftover"][0]),
                    stock_unit_days=float(outcome["stock_unit_days"][0]),
                    cost=realized_cost,
                    clairvoyant_cost=float(oracle_cost),
                    target_cost=float(econ.cost(decision["target"], outcome["demand"][0]))))
            obs = pd.DataFrame(observed)
            weighted = lambda name: float(obs.weight@obs[name])
            metrics = service(obs.demand.to_numpy(), obs.served.to_numpy(), obs.weight.to_numpy())
            sim_metrics = service(all_sim_demand, all_sim_served, all_sim_weights)
            rows.append({**{k: decision[k] for k in CASE+["method", "regime", "status"]},
                "action": action, "quantity": quantity, "trigger": decision["trigger"],
                "target": decision["target"], "rop": decision["rop"], "position": decision["position"],
                "expected_own_one_order_cost": sum(expected), "realized_one_order_cost": weighted("cost"),
                "realized_aggregate_target_cost": weighted("target_cost"),
                "clairvoyant_one_order_cost": weighted("clairvoyant_cost"),
                "clairvoyant_regret": max(0., weighted("cost")-weighted("clairvoyant_cost")),
                "demand": weighted("demand"), "served": weighted("served"),
                "shortage": weighted("shortage"), "leftover": weighted("leftover"),
                "stock_unit_days": weighted("stock_unit_days"), "csl": metrics["csl"], "fill_rate": metrics["fill_rate"],
                "own_scenario_csl": sim_metrics["csl"], "own_scenario_fill_rate": sim_metrics["fill_rate"]})
    return pd.DataFrame(rows)


def matched(frame, keys=CASE, method_columns=("method", "action")):
    if frame.empty or frame.duplicated(keys+list(method_columns)).any():
        raise ValueError("Empty or duplicate policy cases")
    reference = None
    for _, group in frame.groupby(list(method_columns)):
        cases = set(group[keys].itertuples(index=False, name=None))
        if reference is not None and cases != reference:
            raise ValueError("Unmatched valid inventory policy cases")
        reference = cases
    return len(reference)


def summarize(frame, groups):
    rows = []
    for key, values in frame.groupby(groups):
        demand = values.demand.sum()
        rows.append(dict(zip(groups, key if isinstance(key, tuple) else (key,)), cases=len(values),
            cost_mean=float(values.realized_one_order_cost.mean()),
            aggregate_target_cost_mean=float(values.realized_aggregate_target_cost.mean()),
            expected_own_cost_mean=float(values.expected_own_one_order_cost.mean()),
            shortage_mean=float(values.shortage.mean()), leftover_mean=float(values.leftover.mean()),
            stock_unit_days_mean=float(values.stock_unit_days.mean()), order_mean=float(values.quantity.mean()),
            csl=float(values.csl.mean()), fill_rate=float(values.served.sum()/demand) if demand else None,
            trigger_rate=float(values.trigger.mean()), clairvoyant_regret_mean=float(values.clairvoyant_regret.mean())))
    return pd.DataFrame(rows)
