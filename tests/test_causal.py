"""SYNTHETIC causal/OPE goldens, isolation, assignment and stateful hard gates."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from demandos.causal.estimators import evaluate,probabilities,cluster_intervals,trajectory_weights
from demandos.causal.logging import FEATURES,logging_probabilities,assign,target_probabilities,estimator_view
from demandos.causal.nuisance import fit_predict
from demandos.causal.demonstrations import controlled,sequential
from demandos.causal.simulation import catalog,one_allocation,intervention,prefix_equal
from demandos.risk.model import Scenarios


def logs(origin="2016-01-31",prefix="train",n=60):
    rng=np.random.default_rng(42);a=rng.binomial(1,.5,n)
    data={name:rng.normal(size=n) for name in FEATURES}
    data.update(context_id=[prefix+str(i) for i in range(n)],origin_date=pd.Timestamp(origin),resource_pool_id="SYNTHETIC_POOL",
                world_id=np.arange(n)%8,scenario_id="SYNTHETIC",logged_action=a,true_propensity=.5,mu_0=.5,mu_1=.5,
                reward=2*a+data["risk_score"],outcome_available_date=pd.Timestamp(origin)+pd.Timedelta(days=14),
                model_provenance="synthetic unit test",scenario_status="SIMULATED_SYNTHETIC_LOGGING_MODEL_WORLD",
                eligible_actions="[0,1]",replication_id=0)
    return pd.DataFrame(data)


class EstimatorGoldens(unittest.TestCase):
    def test_required_ips_nine(self):
        r=evaluate([1,0,1,0],[10,4,8,6],[[.5,.5]]*4,[[0,1]]*4)
        self.assertEqual(r["ips"],9);self.assertEqual(r["snips"],9)
        np.testing.assert_array_equal(r["weights"],[2,0,2,0]);self.assertEqual(r["ess"],2)

    def test_snips_five_ten(self):
        r=evaluate([1,0,0,0],[10,4,8,6],[[.5,.5]]*4,[[0,1]]*4)
        self.assertEqual(r["ips"],5);self.assertEqual(r["snips"],10)

    def test_dr_eighteen_and_action_mismatch(self):
        self.assertEqual(evaluate([1],[12],[[.75,.25]],[[0,1]],[[4,10]])["dr"],18)
        r=evaluate([0],[999],[[.75,.25]],[[0,1]],[[4,10]])
        self.assertEqual(r["dr"],10);self.assertEqual(r["weights"][0],0)

    def test_stochastic_dm_correct_action_average(self):
        r=evaluate([1],[12],[[.5,.5]],[[.25,.75]],[[4,10]])
        self.assertEqual(r["dm"],8.5);self.assertEqual(r["dr"],11.5)

    def test_zero_matching_sample_not_structural_overlap_failure(self):
        r=evaluate([0,0],[3,4],[[.5,.5]]*2,[[0,1]]*2,[[2,5]]*2)
        self.assertEqual(r["ips"],0);self.assertIsNone(r["snips"]);self.assertEqual(r["ess"],0)
        self.assertEqual(r["dr"],5);self.assertEqual(r["snips_status"],"UNDEFINED_ZERO_WEIGHT")

    def test_unobserved_action_missing_support_not_fixed_by_dm_or_clip(self):
        for clip in [None,2]:
            r=evaluate([0],[3],[[1,0]],[[0,1]],[[2,100]],clip=clip)
            self.assertEqual(r["status"],"NONIDENTIFIABLE_MISSING_SUPPORT")
            for estimator in ["ips","snips","dm","dr"]:self.assertIsNone(r[estimator])

    def test_clipping_separate_biased_sensitivity(self):
        r=evaluate([1],[12],[[.9,.1]],[[0,1]],[[0,10]],clip=2)
        self.assertEqual(r["status"],"BIASED_CLIPPED_SENSITIVITY");self.assertEqual(r["ips"],24)

    def test_nonfinite_noninteger_impossible_actions_and_invalid_weights(self):
        for mu in [[[.2,.2]],[[-.1,1.1]],[[float("nan"),1]]]:
            with self.assertRaises(ValueError):evaluate([1],[1],mu,[[0,1]])
        for a,r,mu in [([.5],[1],[[.5,.5]]),([2],[1],[[.5,.5]]),([1],[float("inf")],[[.5,.5]]),([1],[1],[[1,0]])]:
            with self.assertRaises(ValueError):evaluate(a,r,mu,[[0,1]])

    def test_no_all_action_rewards_argument(self):
        with self.assertRaises(ValueError):evaluate([1],[[1,2]],[[.5,.5]],[[0,1]])

    def test_cluster_bootstrap_does_not_count_daily_rows(self):
        r=cluster_intervals([0,1,0,1],[1,2,1,2],[[.5,.5]]*4,[[.5,.5]]*4,[[1,2]]*4,[0,0,1,1],samples=30)
        self.assertEqual(r["clusters"],2);self.assertIn("FEW_CLUSTERS",r["status"])


class LoggingAndNuisance(unittest.TestCase):
    def test_randomization_reproducible_and_normalized(self):
        p=logging_probabilities([-100,0,100]);np.testing.assert_allclose(p.sum(axis=1),1)
        self.assertTrue((p>=.2-1e-12).all())
        first=[assign("context",w,0,p[1]) for w in range(20)]
        self.assertEqual(first,[assign("context",w,0,p[1]) for w in range(20)])
        self.assertEqual(set(first),{0,1})

    def test_contract_rejects_oracle_and_future_fields(self):
        frame=logs();estimator_view(frame)
        for name in ["oracle_0","future_actual_sales","unchosen_reward"]:
            with self.assertRaises(ValueError):estimator_view(frame.assign(**{name:1}))

    def test_mismatched_selected_propensity_and_duplicate_unit_rejected(self):
        with self.assertRaises(ValueError):estimator_view(logs().assign(true_propensity=.7))
        with self.assertRaises(ValueError):estimator_view(pd.concat([logs(),logs()]))

    def test_chronological_holdout_and_prediction_invariance_to_eval_rewards(self):
        train=logs();test=logs("2016-02-28","test")
        for model in ["ridge","forest","misspecified"]:
            p,mu,a=fit_predict(train,test,model=model)
            other,mu2,b=fit_predict(train,test.assign(reward=999999.),model=model)
            np.testing.assert_array_equal(p,other);np.testing.assert_array_equal(mu,mu2)
            self.assertEqual(a["training_digest"],b["training_digest"])
            self.assertLess(pd.Timestamp(a["outcome_maturity_max"]),pd.Timestamp(a["evaluation_origin_min"]))

    def test_immature_training_and_shared_clusters_rejected(self):
        test=logs("2016-02-28","test")
        with self.assertRaises(ValueError):fit_predict(logs().assign(outcome_available_date=pd.Timestamp("2016-03-01")),test)
        with self.assertRaises(ValueError):fit_predict(logs(),logs("2016-02-28"))

    def test_known_estimated_propensity_separate_and_not_reward_fitted(self):
        train=logs();test=logs("2016-02-28","test")
        _,p,_=fit_predict(train,test)
        _,other,_=fit_predict(train.assign(reward=-12345.),test)
        np.testing.assert_array_equal(p,other)
        np.testing.assert_allclose(p.sum(axis=1),1)

    def test_predeclared_stochastic_and_deterministic_targets(self):
        frame=logs()
        for name in ["mean","cvar","mixture","risk_rule"]:
            p=target_probabilities(frame,name);np.testing.assert_allclose(p.sum(axis=1),1)


class ControlledAndSequential(unittest.TestCase):
    def test_double_robustness_and_omitted_confounder(self):
        _,summary,confounding=controlled(replications=30,n=3000)
        dr=summary[summary.estimator.eq("dr")]
        good=dr[dr.correct_propensity|dr.correct_outcome]
        self.assertTrue((good.bias.abs()<.2).all())
        self.assertGreater(dr[~dr.correct_propensity&~dr.correct_outcome].bias.iloc[0],3)
        self.assertGreater(confounding.naive_difference.mean(),8)
        self.assertLess(abs(confounding.randomized_benchmark.mean()-2),.3)

    def test_sequential_two_to_28(self):
        r=trajectory_weights(np.ones((1,28),int),np.full((1,28,2),.5),np.tile([0.,1.],(1,28,1)))
        self.assertAlmostEqual(r["weights"][0]/268435456,1)

    def test_sequential_zero_match_vs_support_failure(self):
        r=trajectory_weights(np.zeros((3,4),int),np.full((3,4,2),.5),np.tile([0.,1.],(3,4,1)))
        self.assertEqual(r["status"],"NO_MATCHING_TRAJECTORIES");self.assertEqual(r["ess"],0)
        r=trajectory_weights(np.zeros((3,4),int),np.tile([1.,0.],(3,4,1)),np.tile([0.,1.],(3,4,1)))
        self.assertEqual(r["status"],"NONIDENTIFIABLE_MISSING_SUPPORT")

    def test_sequential_stateful_diagnostic_bounds(self):
        result,ledger=sequential(n=64)
        self.assertEqual(len(result),8);self.assertEqual(len(ledger),512)
        self.assertTrue(result.ess.le(64+1e-7).all())


class StatefulTracks(unittest.TestCase):
    def setup_pool(self):
        metadata=pd.DataFrame(dict(series_id=["a","b"],store_id=["s1","s2"],item_id=["i1","i2"],dept_id=["d","d"],
                                   cohort=["continuity","sparse_stress"],volume=[1.,2.]))
        settings=dict(on_hand=10,on_order=3,purchase=60.,order_fee=5.,receiving_factor=1.,seed=42,review_days=7,
                      solver_seconds=3.,holding=.2,lost_penalty=40.,backlog_daily=2.,resource_targets=[30,30])
        scenarios=Scenarios([[20,25,30,40],[20,25,30,40]],[.45,.45,.05,.05],("a","b","c","d"))
        return metadata,settings,scenarios

    def test_catalogue_is_whole_feasible_vector(self):
        from demandos.optimization.model import verify
        m,s,sc=self.setup_pool();q,c,r,_=catalog(m,[30,30],sc,s,budget=.1)
        self.assertEqual(q.shape,(2,2))
        for a in q:verify(c,r,a)

    def test_one_allocation_cost_consistency_and_no_later_orders(self):
        m,_,_=self.setup_pool();r=one_allocation(np.ones((2,14),int),pd.Timestamp("2016-01-31"),m,[6,0])
        self.assertEqual(r["daily"].ordered.sum(),0)
        self.assertAlmostEqual(r["runs"].total_cost.sum(),r["daily"].operational_cost.sum()+365)
        self.assertAlmostEqual(r["runs"].reward.sum(),-r["runs"].total_cost.sum())
        with self.assertRaises(ValueError):one_allocation(np.ones((2,14),int),pd.Timestamp("2016-01-31"),m,[.5,0])

    def test_complete_policy_reset_and_future_prefix(self):
        m,s,sc=self.setup_pool();y=np.full((2,28),3)
        for action in [0,1]:
            first=intervention(y,pd.Timestamp("2016-01-31"),m,[30,30],sc,s,action=action)
            reset=intervention(y,pd.Timestamp("2016-01-31"),m,[30,30],sc,s,action=action)
            pd.testing.assert_frame_equal(first["daily"],reset["daily"])
            changed=y.copy();changed[0,9]=100000
            other=intervention(changed,pd.Timestamp("2016-01-31"),m,[30,30],sc,s,action=action)
            prefix_equal(first,other,10)
            self.assertTrue(first["resources"].budget_after.ge(0).all())


if __name__=="__main__":unittest.main()
