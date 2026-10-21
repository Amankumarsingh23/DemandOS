"""SYNTHETIC mathematical goldens and risk/queue hard gates, not M5 evidence."""
from dataclasses import replace
from itertools import product
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from demandos.risk.metrics import tail, epigraph_cvar, worst_probability, frontier
from demandos.risk.model import Scenarios, solve, losses
from demandos.risk.scenarios import Closure, planning, shifted
from demandos.optimization.model import Resources, Fairness, solve as nominal, verify, fair_slacks
from demandos.optimization.examples import candidates
from demandos.simulation.engine import simulate, Costs, Supplier
from demandos.simulation.policies import Order, Policy
from demandos.optimization.coordinator import coordinated
from demandos.inventory.core import Economics, Distribution
from demandos.risk.pipeline import load_config, summarize_operating
from scipy.optimize import milp
from pathlib import Path


class TailTests(unittest.TestCase):
    def test_ten_loss_goldens(self):
        x=[10,12,14,16,18,20,22,24,30,100];p=np.full(10,.1)
        for a,v,c in [(.9,30,100),(.8,24,65)]:
            self.assertEqual(tail(x,p,a)["var"],v);self.assertAlmostEqual(tail(x,p,a)["cvar"],c)
            self.assertAlmostEqual(epigraph_cvar(x,p,a)["cvar"],c)

    def test_four_loss_epigraph(self):
        r=tail([10,20,30,100],[.25]*4,.75)
        self.assertEqual(r["var"],30);self.assertEqual(r["cvar"],100)
        self.assertEqual(30+np.array([0,0,0,70]).sum()/(4*.25),100)
        self.assertAlmostEqual(epigraph_cvar([10,20,30,100],[.25]*4,.75)["cvar"],100)

    def test_unequal_fractional_ties_zero_and_signed_losses(self):
        r=tail([0,10,10,100,1000],[.6,.1,.2,.1,0],.75)
        self.assertAlmostEqual(r["cvar"],46);self.assertEqual(r["var"],10)
        self.assertAlmostEqual(epigraph_cvar([-10,-5], [.5,.5],.5)["cvar"],-5)
        self.assertEqual(tail([5,5,5],[.2,.3,.5],.8)["var"],5)

    def test_invalid_probabilities_levels_nonfinite(self):
        for p in [[-1,2],[0,0],[.3,.6],[float("nan"),1]]:
            with self.assertRaises(ValueError):tail([1,2],p)
        for a in [0,1,-1,float("nan")]:
            with self.assertRaises(ValueError):tail([1],[1],a)
        with self.assertRaises(ValueError):tail([float("inf")],[1])

    def test_rare_severe_loss(self):
        self.assertGreater(tail([10,10000],[.99,.01],.9)["cvar"],tail([10,100],[.99,.01],.9)["cvar"])

    def test_multiple_optimal_thresholds(self):
        # Entire eta interval [30,100] minimizes this discrete-tail objective.
        for eta in [30,50,100]:
            self.assertEqual(eta+np.maximum(np.array([10,20,30,100])-eta,0).sum(),100)

    def test_probability_endpoints_and_frontier(self):
        self.assertEqual(worst_probability(100,1000)["cost"],280)
        self.assertEqual(worst_probability(1000,100)["shock_probability"],.05)
        f=pd.DataFrame({"expected":[10,11,9,10],"cvar":[20,21,25,20]})
        self.assertEqual(frontier(f),[True,False,True,True])


class RiskModels(unittest.TestCase):
    def setUp(self):
        self.c=candidates([20],[1],[1]);self.r=Resources(100,100,{"supplier":100})
        self.s=Scenarios([[5,10,15]],[.8,.1,.1],("low","mid","high"))

    def test_newsvendor_bruteforce_golden(self):
        costs=np.array([losses([q],[0],self.s,4,1) for q in range(21)])
        self.assertEqual(int(costs.max(axis=1).argmin()),13)
        s=solve(self.c,self.r,self.s,[0],method="minimax",cu=4,co=1)
        self.assertEqual(s["quantity"].tolist(),[13]);self.assertEqual(s["objective"],8)
        mean=costs@self.s.probabilities
        np.testing.assert_array_equal(np.flatnonzero(np.isclose(mean,mean.min())),np.arange(5,11))
        self.assertAlmostEqual(mean.min(),6)
        self.assertEqual(Distribution([5,10,15],[.8,.1,.1]).quantile(.8),5)

    def test_lambda_zero_existing_milp_and_brute_force(self):
        old=nominal(self.c,self.r,demand=self.s.demand,probabilities=self.s.probabilities,stock=[0],cu=4,co=1)
        new=solve(self.c,self.r,self.s,[0],method="mean_cvar",penalty=0,cu=4,co=1)
        self.assertAlmostEqual(new["objective"],old["objective"])
        self.assertIn(new["quantity"][0],range(5,11));self.assertIn(old["quantity"][0],range(5,11))

    def test_minimax_regret_matrix(self):
        costs=np.array([[0,100],[20,40],[50,20]])
        oracle=costs.min(axis=0);np.testing.assert_array_equal(oracle,[0,20])
        regret=(costs-oracle).max(axis=1)
        self.assertEqual(int(regret.argmin()),1);self.assertEqual(regret.min(),20)
        # Allocation regret model also matches enumerated oracle/feasible set.
        all_costs=np.array([losses([q],[0],self.s,4,1) for q in range(21)])
        r=solve(self.c,self.r,self.s,[0],method="regret",cu=4,co=1)
        self.assertAlmostEqual(r["objective"],(all_costs-all_costs.min(axis=0)).max(axis=1).min())

    def test_cvar_objective_ceiling_and_infeasible(self):
        for weight in [.5,2]:
            r=solve(self.c,self.r,self.s,[0],method="mean_cvar",penalty=weight,alpha=.8,cu=4,co=1)
            brute=[tail(losses([q],[0],self.s,4,1),self.s.probabilities,.8) for q in range(21)]
            self.assertAlmostEqual(r["objective"],min(s["expected"]+weight*s["cvar"] for s in brute))
        self.assertEqual(solve(self.c,self.r,self.s,[0],method="ceiling",risk_limit=0,cu=4,co=1)["status"],"INFEASIBLE")
        feasible=solve(self.c,self.r,self.s,[0],method="ceiling",risk_limit=100,cu=4,co=1)
        self.assertLessEqual(feasible["scores"]["cvar"],100)

    def test_dro_bruteforce(self):
        s=Scenarios([[5,7,15,18]],[.45,.45,.05,.05],("n1","n2","s1","s2"))
        r=solve(self.c,self.r,s,[0],method="dro",cu=4,co=1)
        brute=[]
        for q in range(21):
            l=losses([q],[0],s,4,1);brute.append(worst_probability(l[:2].mean(),l[2:].mean())["cost"])
        self.assertAlmostEqual(r["objective"],min(brute))

    def test_integrality_moq_resources_and_hard_soft_fairness(self):
        c=replace(candidates([12,12],[1,1],[1,1],stores=["a","b"],pack=[3,3],moq=[6,6]),fee=np.array([2.,2.]))
        r=Resources(10,10,{"supplier":10},{"a":6,"b":6});s=Scenarios([[5,12],[5,12]],[.5,.5],("a","b"))
        soft=Fairness(minimum_share=.5,soft=True,penalty=100)
        for m in ["mean","mean_cvar","minimax","regret","dro"]:
            result=solve(c,r,s,[0,0],method=m,penalty=.5,fairness=soft)
            verify(c,r,result["quantity"],soft);self.assertGreater(result["preference_cost"],0)
        self.assertEqual(solve(c,r,s,[0,0],fairness=Fairness(minimum_share=.5))["status"],"INFEASIBLE")

    def test_invalid_and_duplicate_scenarios(self):
        for d,p,ids in [([[1,2]],[.5,.5],("a","a")),([[1.5]],[1],("a",)),([[-1]],[1],("a",)),([[1]],[0],("a",))]:
            with self.assertRaises(ValueError):Scenarios(d,p,ids)
        with self.assertRaises(ValueError):solve(self.c,self.r,self.s,[0],penalty=-1)

    def test_status_and_fractional_incumbent(self):
        mock=SimpleNamespace(status=1,x=None,mip_gap=None,mip_dual_bound=0,message="synthetic timeout")
        with patch("demandos.risk.model.milp",return_value=mock):
            self.assertEqual(solve(self.c,self.r,self.s,[0])["status"],"TIME_LIMIT_NO_INCUMBENT")
        with patch("demandos.risk.model.nominal_solve",return_value={"status":"FEASIBLE_NOT_PROVEN_OPTIMAL","objective":1}):
            self.assertEqual(solve(self.c,self.r,self.s,[0],method="regret")["status"],"ORACLE_NOT_PROVEN_OPTIMAL")

    def test_zero_probability_support_and_zero_resource(self):
        s=Scenarios([[0,100]],[1,0],("n","s"))
        r=solve(self.c,Resources(0,0,{"supplier":0}),s,[0],method="minimax")
        self.assertEqual(r["quantity"].tolist(),[0]);self.assertEqual(r["objective"],4000)
        self.assertEqual(r["scores"]["expected"],0)

    def test_fractional_and_invalid_incumbents_rejected(self):
        # mean model: n,z and 3 shortage + 3 excess variables.
        mock=SimpleNamespace(status=1,x=np.array([.5,.5,5,10,15,0,0,0.]),mip_gap=.5,mip_dual_bound=0,message="synthetic")
        with patch("demandos.risk.model.milp",return_value=mock):
            self.assertEqual(solve(self.c,self.r,self.s,[0])["status"],"NUMERICAL_ERROR")

    def test_stable_worst_and_repeatability(self):
        a=solve(self.c,self.r,self.s,[0],method="minimax",cu=4,co=1)
        b=solve(self.c,self.r,self.s,[0],method="minimax",cu=4,co=1)
        self.assertEqual(a["quantity"].tolist(),b["quantity"].tolist())
        self.assertEqual(a["matrix"]["sha256"],b["matrix"]["sha256"])

    def test_roundoff_auxiliary_reconstruction_preserves_order(self):
        mock=SimpleNamespace(status=0,x=np.array([10.,1.,-3e-7,0.,5.,5.,0.,0.]),mip_gap=0.,
                             mip_dual_bound=1.5,message="synthetic sub-micro cost auxiliary")
        with patch("demandos.risk.model.milp",return_value=mock):
            r=solve(self.c,self.r,self.s,[0],cu=4,co=1)
        self.assertEqual(r["quantity"].tolist(),[10]);self.assertAlmostEqual(r["objective"],6)
        self.assertGreater(r["raw_residuals"]["lower"],1e-7);self.assertEqual(r["residuals"]["lower"],0)

    def test_presolve_recovery_identical_model_and_bounded_time(self):
        first=SimpleNamespace(status=4,x=None,message="synthetic presolve transformation failure")
        calls=[]
        def backend(*args,**kwargs):
            calls.append((args,kwargs))
            if len(calls)==1:return first
            return milp(*args,**kwargs)
        with patch("demandos.risk.model.milp",side_effect=backend):
            r=solve(self.c,self.r,self.s,[0],cu=4,co=1)
        self.assertEqual(r["status"],"OPTIMAL");self.assertEqual(len(r["solver_attempts"]),2)
        np.testing.assert_array_equal(calls[0][0][0],calls[1][0][0])
        np.testing.assert_array_equal(calls[0][1]["constraints"].A.toarray(),calls[1][1]["constraints"].A.toarray())
        self.assertFalse(calls[1][1]["options"]["presolve"])
        self.assertLess(calls[1][1]["options"]["time_limit"],3.)


class CoordinatedRiskTests(unittest.TestCase):
    def setup_problem(self):
        metadata=pd.DataFrame(dict(series_id=["a","b"],store_id=["s1","s2"],item_id=["i1","i2"],
                                   dept_id=["d","d"],cohort=["continuity","sparse_stress"],volume=[1.,2.]))
        s=dict(on_hand=10,on_order=3,purchase=60.,order_fee=5.,receiving_factor=1.,seed=42,review_days=7,
               solver_seconds=3.,holding=.2,lost_penalty=40.,backlog_daily=2.,resource_targets=np.array([30,30]))
        return metadata,s

    def run_risk(self,actual,method="mean",closure=None):
        metadata,s=self.setup_problem();sc=Scenarios([[20,25,30,40],[20,25,30,40]],[.45,.45,.05,.05],("a","b","c","d"))
        def allocate(c,r,f,stock,unused,econ,day):
            return solve(c,r,sc,stock,method=method,penalty=.5,fairness=f,cu=econ.cu,co=econ.co)
        return coordinated(actual,pd.Timestamp("2016-01-31"),metadata,[30,30],sc.demand,s,method="stochastic_milp",
                           costs=Economics(100,60,20),lead=dict(days=[7],probabilities=[1.]),allocator=allocate,delivery_schedule=closure)

    def test_all_risk_methods_shared_stateful_resources(self):
        for method in ["mean","mean_cvar","minimax","regret","dro"]:
            r=self.run_risk(np.full((2,28),3),method,Closure())
            self.assertEqual(len(r["daily"]),56)
            self.assertTrue(r["resources"].budget_after.ge(0).all())
            self.assertTrue(r["solver"].status.eq("OPTIMAL").all())

    def test_all_series_earlier_prefix_demand_and_supply(self):
        y=np.full((2,28),3);a=self.run_risk(y)
        changed=y.copy();changed[0,9]=100000
        for b in [self.run_risk(changed),self.run_risk(y,closure=Closure(10,16))]:
            for name in ["daily","allocations","resources"]:
                pd.testing.assert_frame_equal(a[name][a[name].day_index.lt(10)].reset_index(drop=True),
                                              b[name][b[name].day_index.lt(10)].reset_index(drop=True))

    def test_zero_demand_service_denominators(self):
        from demandos.simulation.metrics import run_metrics
        r=self.run_risk(np.zeros((2,28),int))
        for _,g in r["daily"].groupby("series_id"):
            self.assertIsNone(run_metrics(g)["immediate_fill_rate"])

    def test_config_compute_boundary(self):
        c=load_config(Path(__file__).resolve().parents[1]/"configs/risk.toml")
        self.assertEqual(c["risk"]["planning_scenarios"],8)


class DisruptionTests(unittest.TestCase):
    def run_path(self,closure=None,mode="lost_sales"):
        return simulate(np.full(28,3),pd.Timestamp("2016-01-31"),Policy("RS",0,20),Costs(1,40,2,5,60),
                        Supplier((7,),(1.,),42,"same"),initial_orders=(Order("initial",0,10,5,10,"known"),),
                        mode=mode,delivery_schedule=closure)

    def test_outstanding_retiming_conservation(self):
        for mode in ["lost_sales","backorders"]:
            d,e=self.run_path(Closure(),mode)
            retimed=e[e.event.eq("retimed")]
            self.assertGreater(len(retimed),0);self.assertTrue(retimed.arrival_day.eq(15).all())
            for oid,g in e.groupby("order_id"):
                first=g[g.event.isin(["placed","initial_outstanding"])].iloc[0]
                terminal=g[g.event.isin(["received","terminal_outstanding"])]
                self.assertEqual(len(terminal),1);self.assertEqual(first.quantity,terminal.quantity.iloc[0])
            self.assertEqual(5+d.ordered.sum(),d.arrivals.sum()+d.ending_on_order.iloc[-1])
            self.assertTrue(d[["stock_conservation_gap","demand_conservation_gap","queue_conservation_gap"]].eq(0).all().all())

    def test_shock_prefix_and_no_future_information(self):
        a,_=self.run_path();b,_=self.run_path(Closure())
        pd.testing.assert_frame_equal(a[a.day_index.lt(8)],b[b.day_index.lt(8)])
        self.assertNotEqual(a.arrivals.tolist(),b.arrivals.tolist())

    def test_finite_stress_and_prefix_scenarios(self):
        paths=np.ones((2,8,28),int)
        s=planning(paths);self.assertEqual(s.demand.shape,(2,8));self.assertAlmostEqual(s.probabilities.sum(),1)
        self.assertTrue((shifted(paths,"surge")>=paths).all())
        self.assertGreater(shifted(paths,"burst")[...,7:14].sum(),paths[...,7:14].sum())


if __name__ == "__main__": unittest.main()
