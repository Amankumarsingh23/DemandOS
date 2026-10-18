"""Deterministic synthetic Season 11 contracts; no real M5 rows are fabricated."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd


class DiagnosticTests(unittest.TestCase):
    def history(self,values):
        return pd.DataFrame({"date":pd.date_range("2020-01-01",periods=len(values)),"units_sold":values})

    def test_adi_population_cv2_and_all_quadrants_equality(self):
        from demandos.intermittent.diagnostics import statistics,quadrant
        history=self.history([0,0,3,0,0,0,2,0,0,4,0,0])
        result=statistics(history,"2020-01-12")
        self.assertEqual(result["adi"],4)
        self.assertAlmostEqual(result["positive_sales_population_variance"],2/3)
        self.assertAlmostEqual(result["cv2"],2/27)
        self.assertEqual(result["regime"],"intermittent")
        self.assertEqual(result["days_since_positive"],2)
        for adi,cv2,expected in ((1.,0.,"smooth"),(1.,.49,"erratic"),(1.32,0.,"intermittent"),(1.32,.49,"lumpy")):
            self.assertEqual(quadrant(adi,cv2),expected)

    def test_all_zero_one_positive_no_history_and_missing_days(self):
        from demandos.intermittent.diagnostics import statistics,prefix
        for values,status in (([],"no_observations"),([0]*10,"all_zero"),([0,1,0],"insufficient_positive_history")):
            result=statistics(self.history(values),"2020-01-10" if len(values)!=3 else "2020-01-03")
            self.assertEqual(result["diagnostic_status"],status)
            self.assertIsNone(result["cv2"])
            if not any(values): self.assertIsNone(result["adi"])
            if sum(v>0 for v in values)==1: self.assertEqual(result["positive_sales_population_variance"],0.)
        with self.assertRaises(ValueError):
            prefix(self.history([0,1,2]).drop(index=1),"2020-01-03")
        with self.assertRaises(ValueError):
            prefix(self.history([-1,1]),"2020-01-02")

    def test_future_regime_invariance_and_visible_history_detection(self):
        from demandos.intermittent.diagnostics import describe,prefix,cold_status
        history=self.history([0,1]*20)
        changed=history.copy();changed.loc[changed.date.gt(pd.Timestamp("2020-01-28")),"units_sold"]=np.nan
        self.assertEqual(describe(history,"2020-01-28"),describe(changed,"2020-01-28"))
        for n,status in ((0,"no_observed_history"),(3,"short_under_7"),(7,"short_under_28"),(28,"established")):
            observed=prefix(history,"2020-01-28",n)
            self.assertEqual(len(observed),n)
            self.assertEqual(cold_status(observed),status)
        empty=prefix(history,"2020-01-28",0)
        self.assertEqual(cold_status(empty,item_elsewhere=True),"new_store_existing_item")
        self.assertEqual(cold_status(empty,known_category=False),"unknown_category")
        self.assertEqual(cold_status(self.history([0]*28)),"observed_all_zero")


class PriorTests(unittest.TestCase):
    def setUp(self):
        from demandos.intermittent.config import Config
        self.config=replace(Config(),minimum_prior_rows=1,kappa=2.)
        self.daily=pd.DataFrame([{"store_id":s,"item_id":i,"cat_id":"C","date":d,"units_sold":v}
            for s,i,v in (("S","held_out",999),("S","donor",4),("T","held_out",999),("T","other",2))
            for d in pd.date_range("2020-01-01",periods=4)])
        self.empty=pd.DataFrame({"date":pd.Series(dtype="datetime64[ns]"),"units_sold":pd.Series(dtype=float)})

    def test_leave_item_out_and_shrinkage_arithmetic(self):
        from demandos.intermittent.priors import PriorState
        state=PriorState.from_daily(self.daily,"2020-01-03",self.config)
        self.assertEqual(state.estimate("S","held_out","C",self.empty,"2020-01-03")["prior_prediction"],4.)
        observed=pd.DataFrame({"date":pd.date_range("2020-01-02",periods=2),"units_sold":[0,0]})
        row=state.estimate("S","held_out","C",observed,"2020-01-03")
        self.assertEqual(row["prior_prediction"],2.)
        self.assertEqual(row["prior_rows"],3)
        self.assertEqual(row["prior_items"],1)

    def test_prior_future_and_own_item_exclusion_invariance(self):
        from demandos.intermittent.priors import PriorState
        changed=self.daily.copy()
        changed.loc[changed.date.gt(pd.Timestamp("2020-01-03"))|changed.item_id.eq("held_out"),"units_sold"]+=100000
        first=PriorState.from_daily(self.daily,"2020-01-03",self.config)
        second=PriorState.from_daily(changed,"2020-01-03",self.config)
        self.assertEqual(first.estimate("S","held_out","C",self.empty,"2020-01-03"),second.estimate("S","held_out","C",self.empty,"2020-01-03"))
        with self.assertRaises(ValueError):
            first.estimate("S","held_out","C",self.empty,"2020-01-04")

    def test_prior_unavailable_and_existing_item_new_store(self):
        from demandos.intermittent.priors import PriorState
        state=PriorState.from_daily(self.daily,"2020-01-03",self.config)
        self.assertEqual(state.estimate("new_store","held_out","C",self.empty,"2020-01-03")["prior_mean"],3.)
        absent=state.estimate("S","held_out",None,self.empty,"2020-01-03")
        self.assertIsNone(absent["prior_prediction"])
        self.assertEqual(absent["prior_status"],"unavailable_no_history_or_prior")
        observed=pd.DataFrame({"date":[pd.Timestamp("2020-01-03")],"units_sold":[2]})
        self.assertEqual(state.estimate("S","held_out",None,observed,"2020-01-03")["prior_prediction"],2.)


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ("duckdb","lightgbm")),"Install analytics/boosting extras")
class HurdleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_boosting import BoostingTests
        BoostingTests.setUpClass()
        cls.training=BoostingTests.training.copy()
        cls.validation=BoostingTests.validation.copy()
        cls.inference=BoostingTests.inference.copy()
        cls.features=BoostingTests.features
        cls.fc=BoostingTests.feature_config
        cls.config=replace(BoostingTests.config,iterations=6,patience=2,threads=1)
        cls.training["target_units_sold"]=np.where(np.arange(len(cls.training))%3==0,0,np.arange(len(cls.training))%7+1)
        cls.validation["target_units_sold"]=np.where(np.arange(len(cls.validation))%3==0,0,3)

    def model(self):
        from demandos.intermittent.hurdle import HurdleModel
        return HurdleModel(self.features,self.config,self.fc)

    def test_occurrence_positive_targets_product_and_serialization_unknowns(self):
        from demandos.intermittent.hurdle import HurdleModel
        model=self.model().fit(self.training,allowed_cutoff="2016-03-27",validation=self.validation,validation_cutoff="2016-04-24")
        self.assertEqual(model.state["positive_rows"],int(self.training.target_units_sold.gt(0).sum()))
        self.assertEqual(model.occurrence.native_parameters["objective"],"binary")
        self.assertEqual(model.size.metadata["training_rows"],model.state["positive_rows"])
        predicted=model.predict(self.inference)
        np.testing.assert_allclose(predicted.prediction,predicted.p_positive*predicted.conditional_positive_mean)
        self.assertTrue(predicted.p_positive.between(0,1).all())
        self.assertTrue(predicted.conditional_positive_mean.ge(1).all())
        with TemporaryDirectory() as directory:
            model.save(directory);restored=HurdleModel.load(directory)
            np.testing.assert_allclose(predicted.prediction,restored.predict(self.inference).prediction,rtol=1e-10,atol=1e-10)
            unknown=self.inference.copy();unknown["item_id"]="unseen";unknown["relative_level_change_7"]=np.nan
            self.assertTrue(np.isfinite(restored.predict(unknown).prediction).all())
        self.assertFalse(predicted.duplicated(["store_id","item_id","origin_date","horizon"]).any())
        self.assertEqual(len(predicted),84)

    def test_degenerate_partitions_and_refit_reset(self):
        model=self.model().fit(self.training,allowed_cutoff="2016-03-27")
        model.fit(self.training.assign(target_units_sold=0),allowed_cutoff="2016-03-27")
        self.assertIsNone(model.occurrence);self.assertIsNone(model.size)
        self.assertTrue(model.predict(self.inference).prediction.eq(0).all())
        model.fit(self.training.assign(target_units_sold=2),allowed_cutoff="2016-03-27")
        self.assertTrue(model.predict(self.inference).p_positive.eq(1).all())
        sparse=self.training.assign(target_units_sold=0)
        sparse.loc[sparse.index[:3],"target_units_sold"]=5
        model.fit(sparse,allowed_cutoff="2016-03-27")
        self.assertEqual(model.state["size_constant"],5.)
        self.assertIsNone(model.size)

    def test_no_future_training_or_early_stopping_and_no_target_predictor(self):
        from demandos.intermittent.hurdle import HurdleModel
        with self.assertRaises(ValueError):
            self.model().fit(self.training,allowed_cutoff="2016-03-01")
        with self.assertRaises(ValueError):
            self.model().fit(self.training,allowed_cutoff="2016-03-27",validation=self.training,validation_cutoff="2016-04-24")
        with self.assertRaises(ValueError):
            HurdleModel(self.features+["target_units_sold"],self.config,self.fc)
        model=self.model().fit(self.training,allowed_cutoff="2016-03-27")
        with self.assertRaises(ValueError):
            model.predict(self.inference.assign(actual=1))


class PolicyTests(unittest.TestCase):
    def validation(self):
        return pd.DataFrame([{"store_id":f"S{i}","item_id":"I","origin_date":pd.Timestamp("2015-12-06"),
            "horizon":h,"target_date":pd.Timestamp("2015-12-06")+pd.Timedelta(days=h),"regime":"intermittent",
            "model":model,"prediction":value,"actual":2.,"status":"ok","visible_history_days":365}
            for i in range(6) for h in range(1,29) for model,value in (("pooled",4.),("hurdle",0.),("tsb",2.),("seasonal_naive",3.))])

    def test_frozen_routes_convex_weights_and_serialization(self):
        from demandos.intermittent.policy import SegmentPolicy
        policy=SegmentPolicy().fit(self.validation(),validation_end="2016-01-03",first_evaluation="2016-01-31")
        rule=policy.state["rules"][0]
        self.assertEqual(rule["route"],"tsb")
        self.assertAlmostEqual(sum(rule["weights"].values()),1.)
        self.assertTrue(all(w>=0 for w in rule["weights"].values()))
        with TemporaryDirectory() as directory:
            path=Path(directory)/"policy.json";policy.save(path)
            self.assertEqual(policy.state,SegmentPolicy.load(path).state)
        with self.assertRaises(ValueError):
            SegmentPolicy().fit(self.validation(),validation_end="2016-02-28",first_evaluation="2016-01-31")

    def test_low_evidence_default_missing_candidate_and_cold_override(self):
        from demandos.intermittent.policy import SegmentPolicy
        policy=SegmentPolicy().fit(self.validation().query("store_id == 'S0'"),validation_end="2016-01-03",first_evaluation="2016-01-31")
        self.assertEqual(policy.state["rules"][0]["route"],"pooled")
        inputs=self.validation().query("store_id == 'S0'").copy()
        inputs["origin_date"]=pd.Timestamp("2016-02-28")
        inputs["target_date"]=inputs.origin_date+pd.to_timedelta(inputs.horizon,unit="D")
        inputs=inputs.drop(columns="actual")
        out,decisions=policy.predict(inputs,pd.Timestamp("2016-02-28"))
        self.assertTrue(out.prediction.eq(4.).all())
        prior=inputs[inputs.model.eq("pooled")].assign(model="cold_prior",prediction=1.,visible_history_days=0)
        cold=inputs.assign(visible_history_days=0)
        out,decisions=policy.predict(pd.concat([cold,prior]),pd.Timestamp("2016-02-28"))
        self.assertTrue(out.prediction.eq(1.).all())
        self.assertTrue(decisions.decision_reason.eq("cold_override").all())

    def test_policy_duplicate_alignment_and_unavailable_cases(self):
        from demandos.intermittent.policy import SegmentPolicy
        with self.assertRaises(ValueError):
            SegmentPolicy().fit(pd.concat([self.validation(),self.validation().iloc[:1]]),validation_end="2016-01-03",first_evaluation="2016-01-31")
        policy=SegmentPolicy().fit(self.validation(),validation_end="2016-01-03",first_evaluation="2016-01-31")
        rows=self.validation().copy()
        rows["origin_date"]=pd.Timestamp("2016-02-28");rows["target_date"]=rows.origin_date+pd.to_timedelta(rows.horizon,unit="D")
        rows["status"]="skipped";rows["prediction"]=np.nan
        predicted,_=policy.predict(rows,"2016-02-28")
        self.assertTrue(predicted.prediction.isna().all());self.assertTrue(predicted.status.eq("skipped").all())


class EvaluationTests(unittest.TestCase):
    def test_full_grid_leap_date_no_hidden_history_scale_and_fanout(self):
        from demandos.intermittent.evaluation import MODELS,attach_actuals
        origin=pd.Timestamp("2016-02-28")
        cohort=pd.DataFrame([{"store_id":"S","item_id":"I","series_id":"store_item/S/I","cohort":"cold_simulation",
                              "scenario":"new_product_0","visible_days":0}])
        dates=pd.date_range("2016-01-01","2016-03-27")
        frames={("S","I"):pd.DataFrame({"date":dates,"units_sold":np.arange(len(dates))%3})}
        forecasts=pd.DataFrame([{"store_id":"S","item_id":"I","origin_date":origin,"horizon":h,
            "target_date":origin+pd.Timedelta(days=h),"model":m,"prediction":0.,"raw_prediction":0.,"status":"ok",
            "reason":None,"regime":"no_observations","cold_status":"no_observed_history","visible_history_days":0}
            for h in range(1,29) for m in MODELS])
        scored=attach_actuals(forecasts,cohort,frames,[origin])
        self.assertEqual(len(scored),28*len(MODELS))
        self.assertTrue(scored.scale.isna().all())
        self.assertTrue(scored.scale_status.eq("no_visible_history").all())
        self.assertTrue(scored[scored.horizon.eq(1)].target_date.eq(pd.Timestamp("2016-02-29")).all())
        for bad in (pd.concat([forecasts,forecasts.iloc[:1]]),forecasts.iloc[1:]):
            with self.assertRaises(ValueError):
                attach_actuals(bad,cohort,frames,[origin])


if __name__=="__main__":
    unittest.main()
