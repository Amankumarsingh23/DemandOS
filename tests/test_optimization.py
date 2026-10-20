"""Exact SYNTHETIC optimization models and coordinated-timeline integration."""
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from demandos.optimization.model import Candidates,Resources,Fairness,basic_lp,solve,baseline,verify,status,build,fair_slacks
from demandos.optimization.examples import candidates,run_examples
from demandos.optimization.coordinator import coordinated,origin_snapshot
from demandos.optimization.pipeline import load_config
from demandos.inventory.core import Economics


class LinearIntegerTests(unittest.TestCase):
    def test_required_lp_vertices_duals_primal_value(self):
        r=basic_lp([8,5],[[4,2],[1,1]],[20,8])
        np.testing.assert_allclose(r["x"],[2,6]);np.testing.assert_allclose(r["shadow_prices"],[1.5,2])
        self.assertEqual(r["benefit"],46)
        self.assertEqual(np.array([20,8])@r["shadow_prices"],46)
        s=basic_lp([8,5],[[4,2],[1,1],[0,1]],[20,8,4])
        np.testing.assert_allclose(s["x"],[3,4]);self.assertEqual(s["benefit"],44)

    def test_knapsack_brute_force_and_relaxation_gap(self):
        r=run_examples()
        self.assertEqual(r["knapsack"]["value"],17)
        self.assertEqual(r["knapsack"]["quantity"],[1,0,1])
        self.assertAlmostEqual(r["integrality"]["gap"],7.3333333333)
        with self.assertRaises(AssertionError):verify(candidates([1,1],[6,5],[10,9]),Resources(10,100,{"supplier":100}),[1,1])

    def test_lp_is_not_an_implementable_integer_order(self):
        c=candidates([1,1],[6,5],[10,9]);r=Resources(10,100,{"supplier":100})
        relaxed=solve(c,r,relax=True)
        with self.assertRaises(AssertionError):verify(c,r,relaxed["quantity"])
        self.assertEqual(solve(c,r)["status"],"OPTIMAL")

    def test_integer_moq_case_pack_and_tight_bounds(self):
        c=candidates([17,5],[1,1],[4,5],pack=[6,6],moq=[12,12])
        r=Resources(30,30,{"supplier":30});s=solve(c,r)
        np.testing.assert_array_equal(s["quantity"],[12,0])
        self.assertEqual(s["matrix"]["case_upper"],[2,0])
        with self.assertRaises(AssertionError):verify(c,r,[6,0])

    def test_shared_supplier_receiving_and_item_caps(self):
        c=replace(candidates([10,10],[1,1],[10,5],stores=["a","b"]),item_cap=np.array([3,10]))
        r=Resources(100,100,{"supplier":7},{"b":2})
        q=solve(c,r)["quantity"];np.testing.assert_array_equal(q,[3,2]);verify(c,r,q)

    def test_infeasible_forced_minimum_not_fabricated(self):
        c=candidates([10,10],[1,1],[1,1],minimum=[5,5]);r=Resources(100,8,{"supplier":100})
        s=solve(c,r);self.assertEqual(s["status"],"INFEASIBLE");self.assertIsNone(s["quantity"])
        self.assertEqual(s["diagnostic"]["forced_minimum_budget"],10)
        self.assertIn("warehouse_procurement_flow",s["diagnostic"]["violating_families"])

    def test_time_limit_incumbent_validated_before_usage(self):
        c=candidates([2],[1],[5]);r=Resources(10,10,{"supplier":10})
        mock=SimpleNamespace(status=1,x=np.array([.5,.5]),mip_gap=.5,mip_dual_bound=-10,message="synthetic time limit")
        with patch("demandos.optimization.model.milp",return_value=mock):
            result=solve(c,r);self.assertEqual(result["status"],"NUMERICAL_ERROR");self.assertIsNone(result["quantity"])
        mock.x=np.array([1.,1.])
        with patch("demandos.optimization.model.milp",return_value=mock):
            result=solve(c,r);self.assertEqual(result["status"],"FEASIBLE_NOT_PROVEN_OPTIMAL");verify(c,r,result["quantity"])

    def test_local_shadow_prices_match_small_rhs_perturbations(self):
        base=basic_lp([8,5],[[4,2],[1,1]],[20,8])
        for index in [0,1]:
            rhs=np.array([20.,8.]);rhs[index]+=.001
            changed=basic_lp([8,5],[[4,2],[1,1]],rhs)
            self.assertAlmostEqual((changed["benefit"]-base["benefit"])/.001,base["shadow_prices"][index])

    def test_invalid_fractional_request_and_zero_packs(self):
        with self.assertRaises(ValueError):candidates([1.2],[1],[1])
        with self.assertRaises(ValueError):candidates([1],[1],[1],pack=[0])
        with self.assertRaises(ValueError):Resources(-1,1,{})

    def test_fair_minimum_hard_and_soft_slack(self):
        c=candidates([10,10],[1,1],[9,1],stores=["a","b"]);r=Resources(10,100,{"supplier":100})
        q=solve(c,r,Fairness(minimum_share=.3))["quantity"];self.assertGreaterEqual(q[1],3)
        self.assertEqual(solve(c,Resources(1,100,{"supplier":100}),Fairness(minimum_share=.5))["status"],"INFEASIBLE")
        soft=Fairness(minimum_share=.5,soft=True,penalty=100)
        result=solve(c,Resources(1,100,{"supplier":100}),soft)
        self.assertEqual(result["status"],"OPTIMAL");verify(c,Resources(1,100,{"supplier":100}),result["quantity"],soft)
        self.assertGreater(sum(fair_slacks(c,result["quantity"],soft)["minimum_shortfall"].values()),0)

    def test_fair_disparity_linearized_and_zero_requests(self):
        c=candidates([10,10,0],[1,1,1],[9,1,0],stores=["a","b","zero"]);r=Resources(10,100,{"supplier":100})
        f=Fairness(maximum_disparity=0)
        q=solve(c,r,f)["quantity"];np.testing.assert_array_equal(q,[5,5,0])
        self.assertNotIn("zero",fair_slacks(c,q,f)["shares"])

    def test_baselines_feasible_with_fees_packs_and_moq(self):
        c=replace(candidates([17,15],[2,3],[5,4],pack=[3,3],moq=[9,6]),fee=np.array([5.,5.]))
        r=Resources(40,15,{"supplier":14})
        for method in ["greedy","proportional"]:verify(c,r,baseline(c,r,method))

    def test_statuses_and_nonoptimal_incumbents(self):
        for code,x,gap,expected in [(0,[1],.1,"FEASIBLE_NOT_PROVEN_OPTIMAL"),(1,[1],.1,"FEASIBLE_NOT_PROVEN_OPTIMAL"),(1,None,None,"TIME_LIMIT_NO_INCUMBENT"),(2,None,None,"INFEASIBLE"),(3,None,None,"UNBOUNDED"),(4,None,None,"NUMERICAL_ERROR")]:
            self.assertEqual(status(SimpleNamespace(status=code,x=x,mip_gap=gap)),expected)

    def test_stable_input_order_and_repeatable_ties(self):
        c=candidates([3,3],[1,1],[1,1]);r=Resources(3,10,{"supplier":10})
        np.testing.assert_array_equal(solve(c,r)["quantity"],solve(c,r)["quantity"])
        with self.assertRaises(ValueError):replace(c,keys=("B","A"))


class StochasticTests(unittest.TestCase):
    def test_probability_weighting_matches_direct_enumeration(self):
        c=candidates([5],[1],[1]);r=Resources(5,10,{"supplier":10})
        d=np.array([[0,5]]);p=np.array([.8,.2])
        solved=solve(c,r,demand=d,probabilities=p,stock=[0],cu=1,co=1)
        costs=[float(p@(np.maximum(d[0]-q,0)+np.maximum(q-d[0],0))) for q in range(6)]
        self.assertEqual(solved["quantity"][0],0);self.assertAlmostEqual(solved["objective"],min(costs))

    def test_shared_first_stage_and_evpi(self):
        c=candidates([10,10],[1,1],[1,1]);r=Resources(10,100,{"supplier":100});d=np.array([[0,10],[10,0]])
        a=solve(c,r,demand=d,probabilities=[.5,.5],stock=[0,0],cu=9,co=1)
        # Merely choosing which outcome is later realized is not an input to solve.
        b=solve(c,r,demand=d.copy(),probabilities=[.5,.5],stock=[0,0],cu=9,co=1)
        np.testing.assert_array_equal(a["quantity"],b["quantity"])
        self.assertEqual(len(a["quantity"]),2)
        self.assertAlmostEqual(run_examples()["evpi"]["evpi"],50)

    def test_scenario_validation_and_no_procurement_overlay(self):
        c=candidates([5],[60],[1]);r=Resources(1000,100,{"supplier":100})
        s=solve(c,r,demand=[[3]],probabilities=[1],stock=[0],cu=40,co=20)
        self.assertEqual(s["objective"],0);self.assertEqual(s["quantity"][0],3)
        with self.assertRaises(ValueError):solve(c,r,demand=[[3,4]],probabilities=[.1,.2],stock=[0])
        with self.assertRaises(ValueError):solve(c,r,demand=[[-1]],probabilities=[1],stock=[0])


def fixture():
    metadata=pd.DataFrame(dict(series_id=["A","B"],store_id=["s1","s2"],item_id=["i1","i2"],dept_id=["d","d"],cohort=["continuity","sparse_stress"],volume=[1.,2.]))
    settings=dict(seed=42,purchase=1.,order_fee=1.,on_hand=2,on_order=1,receiving_factor=1.,review_days=7,solver_seconds=2.,holding=.2,lost_penalty=10.,backlog_daily=1.)
    return metadata,settings,dict(days=[1,3],probabilities=[.5,.5]),np.array([[5,10],[10,5]])


class CoordinationTests(unittest.TestCase):
    def execute(self,method="stochastic_milp",actual=None,**kwargs):
        metadata,settings,lead,scenario=fixture()
        return coordinated(np.full((2,28),2) if actual is None else actual,pd.Timestamp("2016-02-28"),metadata,[10,10],scenario,settings,
            method=method,costs=Economics(10,1,0),lead=lead,**kwargs)

    def test_two_phase_shared_budget_and_supplier_no_reset(self):
        r=self.execute();b=r["resources"]
        self.assertTrue(b.budget_after.ge(-1e-7).all())
        np.testing.assert_allclose(b.budget_before.iloc[1:],b.budget_after.iloc[:-1])
        self.assertAlmostEqual(b.committed.sum()+b.budget_after.iloc[-1],b.budget_initial.iloc[0])
        a=r["allocations"]
        self.assertTrue(a.approved.le(a.candidate_quantity).all())
        self.assertEqual(len(r["daily"]),56)

    def test_all_baselines_integrate_and_pack_safe(self):
        for method in ["proportional","greedy","deterministic_milp"]:
            r=self.execute(method=method,pack=3,moq=3)
            self.assertTrue((r["allocations"].approved%3).eq(0).all())

    def test_pre_k_all_series_states_and_approvals_identical(self):
        base=self.execute();future=np.full((2,28),2);future[1,9]=1000;other=self.execute(actual=future)
        cols=["series_id","day_index","approved","position"]
        a=base["allocations"];b=other["allocations"]
        pd.testing.assert_frame_equal(a.loc[a.day_index.lt(10),cols],b.loc[b.day_index.lt(10),cols])

    def test_hidden_lost_demand_and_backorder_conservation(self):
        a=self.execute(actual=np.full((2,28),1000));b=self.execute(actual=np.full((2,28),10000))
        pd.testing.assert_frame_equal(a["allocations"][["approved","position"]],b["allocations"][["approved","position"]])
        d=self.execute(mode="backorders")["daily"]
        self.assertEqual(d.lost_units.sum(),0)
        np.testing.assert_array_equal(d.opening_backlog+d.new_backlog-d.old_backlog_fulfilled,d.ending_backlog)

    def test_config_and_leap_day(self):
        c=load_config(Path(__file__).resolve().parents[1]/"configs/optimization.toml")
        self.assertEqual(c["optimization"]["scenario_samples"],16)
        r=self.execute();self.assertEqual(r["daily"].simulation_date.min(),pd.Timestamp("2016-02-29"))
        self.assertEqual(r["daily"].simulation_date.max(),pd.Timestamp("2016-03-27"))

    def test_origin_snapshot_has_no_realized_demand_argument(self):
        metadata,settings,lead,scenario=fixture()
        a=origin_snapshot(metadata,[10,10],scenario,settings,Economics(10,1,0))
        b=origin_snapshot(metadata,[10,10],scenario.copy(),settings,Economics(10,1,0))
        np.testing.assert_array_equal(a["quantity"],b["quantity"])

    def test_forecast_sources_share_frozen_resource_reference(self):
        metadata,settings,lead,scenario=fixture();settings["resource_targets"]=[10,10]
        runs=[coordinated(np.full((2,28),2),pd.Timestamp("2016-02-28"),metadata,target,scenario,settings,
            method="proportional",costs=Economics(10,1,0),lead=lead) for target in [[10,10],[20,30]]]
        self.assertEqual(runs[0]["resources"].budget_initial.iloc[0],runs[1]["resources"].budget_initial.iloc[0])
        self.assertEqual(runs[0]["resources"].flow_limit.iloc[0],runs[1]["resources"].flow_limit.iloc[0])


if __name__=="__main__":unittest.main()
