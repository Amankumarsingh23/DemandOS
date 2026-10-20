"""SYNTHETIC fixtures: deterministic sequential state, cost and leakage contracts."""
from dataclasses import dataclass, fields
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from demandos.simulation.engine import Costs, Supplier, simulate, RUN_KEYS
from demandos.simulation.policies import Order, Observation, Action, Policy
from demandos.simulation.metrics import run_metrics, aggregate, paired
from demandos.simulation.pipeline import load_config, temporal_checks
from demandos.simulation.verification import verify
from demandos.data.pipeline import fingerprint

ORIGIN = pd.Timestamp("2016-02-28")
ZERO = Costs(0, 0, 0, 0, 0)


class Script:
    def __init__(self, orders): self.orders, self.seen = orders, []
    def decide(self, observation):
        self.seen.append(observation)
        return Action(self.orders.get(observation.day_index, 0), "SYNTHETIC_SCRIPT", 0, 0, observation.simulation_date)


class FixedTape:
    def __init__(self, days): self.days = days
    def at(self, day, date): return self.days.get(day, 1), f"SYNTHETIC_{day}", .5


class TransitionTests(unittest.TestCase):
    def test_lost_sales_10_plus_3_minus_15(self):
        daily, _ = simulate([15], ORIGIN, Script({}), ZERO, FixedTape({}),
                            initial_orders=[Order("initial", 0, 1, 3, 1, "fixture")])
        self.assertEqual(daily.immediate_served.tolist(), [13])
        self.assertEqual(daily.lost_units.tolist(), [2])
        self.assertEqual(daily.ending_stock.tolist(), [0])
        self.assertEqual(daily.ending_backlog.tolist(), [0])

    def test_exact_three_day_golden_integration(self):
        daily, events = simulate([6, 5, 4], ORIGIN, Script({1: 8}), ZERO, FixedTape({1: 2}),
                                 initial_orders=[Order("initial", 0, 3, 3, 3, "fixture")])
        for column, expected in dict(arrivals=[0, 0, 11], immediate_served=[6, 4, 4],
                                     lost_units=[0, 1, 0], ending_stock=[4, 0, 7], ending_on_order=[11, 11, 0]).items():
            self.assertEqual(daily[column].tolist(), expected)
        self.assertEqual(daily.demand_proxy.sum(), 15)
        self.assertEqual(daily.immediate_served.sum(), 14)
        self.assertEqual(10+daily.arrivals.sum(), daily.immediate_served.sum()+7)
        self.assertEqual(len(events[events.event.eq("received")]), 2)

    def test_backlog_carryover_delivery_and_no_double_count(self):
        daily, _ = simulate([5, 0, 1], ORIGIN, Script({1: 4}), ZERO, FixedTape({}), on_hand=2, mode="backorders")
        self.assertEqual(daily.immediate_served.tolist(), [2, 0, 1])
        self.assertEqual(daily.new_backlog.tolist(), [3, 0, 0])
        self.assertEqual(daily.old_backlog_fulfilled.tolist(), [0, 3, 0])
        self.assertEqual(daily.ending_backlog.tolist(), [3, 0, 0])
        metrics = run_metrics(daily)
        self.assertEqual(metrics["demand_units"], 6)
        self.assertEqual(metrics["immediate_fill_rate"], .5)
        self.assertEqual(metrics["backorders_fulfilled"], 3)
        self.assertEqual(metrics["fulfilled_backorder_mean_wait"], 1)
        self.assertEqual(metrics["backlog_unit_days"], 3)

    def test_fifo_wait_and_censored_pending_backlog(self):
        d, _ = simulate([3, 2, 0], ORIGIN, Script({2: 4}), ZERO, FixedTape({}), on_hand=0, mode="backorders")
        self.assertEqual(d.fulfilled_backorder_wait_unit_days.sum(), 7)
        self.assertEqual(d.ending_backlog.iloc[-1], 1)
        self.assertEqual(run_metrics(d)["fulfilled_backorder_mean_wait"], 7/4)

    def test_initial_backlog_is_served_before_new_demand(self):
        d, _ = simulate([4], ORIGIN, Script({}), ZERO, FixedTape({}), on_hand=2, initial_backlog=5, mode="backorders")
        self.assertEqual((d.old_backlog_fulfilled.iloc[0], d.immediate_served.iloc[0], d.ending_backlog.iloc[0]), (2, 0, 7))

    def test_one_day_lead_never_serves_issue_day(self):
        d, _ = simulate([5, 5], ORIGIN, Script({1: 5}), ZERO, FixedTape({}), on_hand=0)
        self.assertEqual(d.arrivals.tolist(), [0, 5])
        self.assertEqual(d.immediate_served.tolist(), [0, 5])

    def test_crossing_orders_and_multiple_open_orders(self):
        d, e = simulate([0]*9, ORIGIN, Script({1: 5, 2: 7}), ZERO, FixedTape({1: 7, 2: 2}), on_hand=0)
        self.assertEqual(d.arrivals.tolist(), [0, 0, 0, 7, 0, 0, 0, 5, 0])
        self.assertEqual(d.ending_on_order.tolist(), [5, 12, 12, 5, 5, 5, 5, 0, 0])
        self.assertEqual(e[e.event.eq("received")].order_id.tolist(), ["order_00002", "order_00001"])

    def test_simultaneous_arrivals_exactly_once(self):
        initial = [Order("a", 0, 2, 3, 2, "a"), Order("b", 0, 2, 4, 2, "b")]
        d, e = simulate([0]*4, ORIGIN, Script({}), ZERO, FixedTape({}), on_hand=0, initial_orders=initial)
        self.assertEqual(d.arrivals.tolist(), [0, 7, 0, 0])
        self.assertEqual(len(e[e.event.eq("received")]), 2)

    def test_queue_validation_and_noninteger_inputs(self):
        with self.assertRaises(ValueError): Order("bad", 1, 1, 3, 1, "x")
        with self.assertRaises(ValueError): simulate([.5], ORIGIN, Script({}), ZERO, FixedTape({}))
        with self.assertRaises(ValueError): simulate([1], ORIGIN, Script({}), ZERO, FixedTape({}), initial_backlog=1)
        q = Order("same", 0, 1, 1, 1, "x")
        with self.assertRaises(ValueError): simulate([1], ORIGIN, Script({}), ZERO, FixedTape({}), initial_orders=[q, q])

    def test_random_paths_all_daily_and_cumulative_invariants(self):
        demand = np.random.default_rng(17).integers(0, 15, 100)
        for mode in ["lost_sales", "backorders"]:
            d, _ = simulate(demand, ORIGIN, Policy("rS", 20, 40), ZERO, Supplier((1, 3, 7), (.2, .5, .3), 42, "fixture"), mode=mode)
            np.testing.assert_array_equal(d.opening_stock+d.arrivals, d.immediate_served+d.old_backlog_fulfilled+d.ending_stock)
            np.testing.assert_array_equal(d.demand_proxy, d.immediate_served+d.new_backlog+d.lost_units)
            self.assertEqual(d.ordered.sum(), d.arrivals.sum()+d.ending_on_order.iloc[-1])
            self.assertEqual(d.demand_proxy.sum(), d.immediate_served.sum()+d.old_backlog_fulfilled.sum()+d.lost_units.sum()+d.ending_backlog.iloc[-1])


class PolicyCostTests(unittest.TestCase):
    def observation(self, day=1, stock=20, pipeline=50, backlog=5):
        return Observation(day, day, (ORIGIN+pd.Timedelta(days=day)).date(), stock, backlog, pipeline, (), ())

    def test_position_and_fixed_vs_up_to_trigger(self):
        state = self.observation()
        self.assertEqual(state.position, 65)
        self.assertEqual(Policy("rQ", 80, 140, 50).decide(state).quantity, 50)
        self.assertEqual(Policy("rS", 80, 140, 50).decide(state).quantity, 75)
        self.assertEqual(Policy("rQ", 64, 140, 50).decide(state).quantity, 0)
        self.assertEqual(Policy("rS", 65, 60).decide(state).quantity, 0)

    def test_periodic_origin_calendar_and_integer_constraints(self):
        p = Policy("RS", 0, 140, review_days=7, multiple=10, minimum=80)
        self.assertEqual([p.decide(self.observation(day=d)).quantity for d in range(1, 10)], [80,0,0,0,0,0,0,80,0])

    def test_ip_before_after_order_counts_pipeline(self):
        d, _ = simulate([5, 5], ORIGIN, Policy("rS", 20, 30), ZERO, FixedTape({1: 7}))
        self.assertEqual(d.position_before_order.tolist(), [5, 25])
        self.assertEqual(d.ordered.tolist(), [25, 0])
        self.assertEqual(d.position_after_order.tolist(), [30, 25])

    def test_required_174_daily_cost_and_no_mismatch_overlay(self):
        c = Costs(2, 30, 0, 50, 10)
        booked = c.book(7, 1, 0, 8)
        self.assertEqual(booked["operational_cost"], 174)
        self.assertEqual(set(booked), {"holding_cost", "lost_cost", "backlog_cost", "ordering_cost", "procurement_cost", "terminal_credit", "operational_cost"})

    def test_purchase_booked_at_order_including_terminal_pipeline(self):
        d, _ = simulate([0,0], ORIGIN, Script({2: 8}), Costs(0,0,0,5,10), FixedTape({2:7}), on_hand=0)
        self.assertEqual(d.procurement_cost.tolist(), [0,80])
        self.assertEqual(d.operational_cost.sum(), 85)
        self.assertEqual(d.ending_on_order.iloc[-1], 8)

    def test_initial_pipeline_prepaid_terminal_credit_only_last_day(self):
        d, _ = simulate([0,0], ORIGIN, Script({}), Costs(0,0,0,0,10,5), FixedTape({}), on_hand=0,
                        initial_orders=[Order("prepaid",0,1,3,1,"x")])
        self.assertEqual(d.procurement_cost.sum(), 0)
        self.assertEqual(d.terminal_credit.tolist(), [0,15])
        with self.assertRaises(ValueError): Costs(0,0,0,0,1,2)

    def test_service_definitions_zero_denominator_and_no_csl_claim(self):
        d, _ = simulate([0,0], ORIGIN, Script({}), ZERO, FixedTape({}))
        m = run_metrics(d)
        self.assertIsNone(m["immediate_fill_rate"])
        self.assertIsNone(m["replenishment_csl"])
        self.assertEqual(m["daily_no_shortage_fraction"], 1)
        self.assertEqual(m["csl_status"], "not_defined_no_cycle_segmentation")

    def test_warmup_scoring_preserves_state(self):
        d, _ = simulate([2]*4, ORIGIN, Script({}), Costs(1,0,0,0,0), FixedTape({}), warmup=2)
        m = run_metrics(d)
        self.assertEqual((m["scored_days"],m["warmup_days"],m["demand_units"]), (2,2,4))
        self.assertEqual(m["holding_cost"], 4+2)
        with self.assertRaises(ValueError): simulate([1], ORIGIN, Script({}), ZERO, FixedTape({}), warmup=1)


class SafetyTests(unittest.TestCase):
    def saved_fixture(self, directory):
        directory = Path(directory); (directory/"tables").mkdir()
        ledger, events = simulate([6,5,4], ORIGIN, Script({1:8}), ZERO, FixedTape({1:2}),
            initial_orders=[Order("initial",0,3,3,3,"fixture")])
        meta = dict(series_id="SYNTHETIC",origin_date=ORIGIN,policy_id="fixture",cost_scenario_id="fixture",
            lead_scenario_id="fixture",demand_scenario_id="fixture",replication_id=0,experiment_id="fixture",initial_on_hand=10,initial_on_order=3)
        ledger=ledger.assign(**meta); events=events.assign(**meta)
        pairs=ledger[RUN_KEYS+["day_index","scored"]].assign(cost_difference=0.)
        run=pd.DataFrame([{**meta,**run_metrics(ledger),"operational_cost_reference":0.}])
        demand=ledger[["series_id","origin_date","replication_id","demand_scenario_id","simulation_date","demand_proxy"]]
        shocks=ledger[["series_id","origin_date","replication_id","lead_scenario_id","calendar_day"]]
        tables=dict(daily_ledger=ledger,order_events=events,exogenous_demand=demand,supplier_shocks=shocks,
                    paired_daily_costs=pairs,paired_runs=run,runs=run)
        for name,frame in tables.items(): frame.to_parquet(directory/"tables"/(name+".parquet"),index=False)
        return tables

    def test_independent_persisted_sql_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            tables=self.saved_fixture(directory)
            result=verify(directory)
            self.assertEqual(result["status"],"passed")
            self.assertEqual(result["daily_rows"],3)
            bad=tables["daily_ledger"].copy();bad.loc[0,"ending_stock"]+=1
            bad.to_parquet(Path(directory)/"tables/daily_ledger.parquet",index=False)
            with self.assertRaisesRegex(AssertionError,"daily_conservation"):verify(directory)

    def test_independent_sql_rejects_duplicate_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            tables=self.saved_fixture(directory)
            bad=pd.concat([tables["order_events"],tables["order_events"].iloc[-1:]])
            bad.to_parquet(Path(directory)/"tables/order_events.parquet",index=False)
            with self.assertRaisesRegex(AssertionError,"event_uniqueness"):verify(directory)

    def test_immutable_policy_observation_has_no_hidden_demand_or_price(self):
        names = {f.name for f in fields(Observation)}
        self.assertTrue({"demand", "lost", "lost_units", "future", "actual", "price"}.isdisjoint(names))
        p = Script({})
        simulate([5, 0, 1], ORIGIN, p, ZERO, FixedTape({}), on_hand=2)
        self.assertEqual(p.seen[0].fulfilled_sales_history, (2,))
        self.assertEqual(p.seen[0].backlog, 0)
        with self.assertRaises(AttributeError): p.seen[0].on_hand = 999

    def test_common_randomness_not_order_count_dependent(self):
        tape = Supplier((2,4,7), (.25,.5,.25),42,"common")
        a, _ = simulate([1]*28, ORIGIN, Script({1:3,3:4}), ZERO, tape)
        b, _ = simulate([1]*28, ORIGIN, Script({3:9}), ZERO, tape)
        self.assertEqual(a.sampled_lead.tolist(), b.sampled_lead.tolist())
        self.assertEqual(a.supplier_shock_id.tolist(), b.supplier_shock_id.tolist())
        self.assertEqual(tape.at(3,ORIGIN), tape.at(3,ORIGIN))

    def test_time_dependent_weekend_lead(self):
        tape = Supplier((1,), (1.,),42,"common",2)
        self.assertEqual(tape.at(1,pd.Timestamp("2016-02-29"))[0],1)
        self.assertEqual(tape.at(6,pd.Timestamp("2016-03-05"))[0],3)

    def test_exact_leap_dates_and_pre_k_nonanticipativity(self):
        y = np.arange(28)%5
        p = Policy("rS",5,20)
        tape = Supplier((1,7),(.5,.5),42,"fixture")
        checks = temporal_checks(y,ORIGIN,p,ZERO,tape,(Order("initial",0,3,3,3,"x"),))
        self.assertEqual(len(checks),5)
        self.assertTrue(all(c["passed"] for c in checks))
        d,_=simulate(y,ORIGIN,p,ZERO,tape)
        self.assertEqual(d.simulation_date.iloc[0],pd.Timestamp("2016-02-29"))
        self.assertEqual(d.simulation_date.iloc[-1],pd.Timestamp("2016-03-27"))

    def test_source_immutability_and_config(self):
        config=load_config(Path(__file__).resolve().parents[1]/"configs/simulation.toml")
        self.assertEqual(config["simulation"]["replications"],2)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"SYNTHETIC.csv"; source.write_text("demand\n1\n2\n")
            before=fingerprint(source)
            simulate(pd.read_csv(source).demand,ORIGIN,Script({}),ZERO,FixedTape({}))
            self.assertEqual(before,fingerprint(source))

    def test_paired_case_uniqueness_and_cluster_unit(self):
        rows=[]
        for series in ["SYNTHETIC_A","SYNTHETIC_B","SYNTHETIC_C"]:
            for replication in [0,1]:
                for rule in ["RS","rS"]:
                    d,_=simulate([1]*3,ORIGIN,Policy(rule,2,5),Costs(1,2,0,1,3),FixedTape({}),on_hand=1)
                    rows.append(dict(series_id=series,origin_date=ORIGIN,policy_id="test:"+rule,cost_scenario_id="fixture",
                        lead_scenario_id="fixture",demand_scenario_id="SYNTHETIC",replication_id=replication,
                        experiment_id="test",source="historical_fractile",rule=rule,cohort="fixture",**run_metrics(d)))
        runs=pd.DataFrame(rows); pairs,metrics=paired(runs)
        self.assertEqual(len(pairs),12)
        self.assertTrue(metrics.clusters.eq(3).all())
        with self.assertRaises(ValueError): paired(pd.concat([runs,runs.iloc[:1]]))
        with self.assertRaises(ValueError): paired(runs.iloc[1:])
        result=aggregate(runs,["cohort"])
        self.assertEqual(result.runs.iloc[0],12)


if __name__ == "__main__": unittest.main()
