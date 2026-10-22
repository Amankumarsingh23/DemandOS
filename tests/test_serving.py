"""Deterministic SYNTHETIC fixtures; real M5 replay is a separate integration audit."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from demandos.serving.core import Blocked,canonical,digest,frame_hash,writer_lock
from demandos.serving.features import extract,eligible_history,parity,validate_frame
from demandos.serving.registry import Registry
from demandos.serving.publication import publish,read_batch,request_id
from demandos.serving.monitoring import psi_probabilities,ks,drift,reference_bins,frequencies,mature_labels,performance,matched_performance,incident
from demandos.serving.routing import canary,fallback


def fixture():
    from test_features import fixture as existing
    con,config,sales,calendar,keys,_,mapping=existing();con.close()
    for name in ["event_name_1","event_name_2","event_type_1","event_type_2"]:calendar[name]=calendar[name].astype("string")
    snapshot=sales[["store_id","item_id","date","units_sold"]].assign(available_at=sales.date)
    return config,snapshot,calendar,keys,mapping


def registry_fixture(root):
    root=Path(root);model=root/"trusted/model.txt";model.parent.mkdir();model.write_text("synthetic model identity, not actual fitted weights")
    registry=Registry(root,root/"registry",[model.parent])
    metadata=dict(model_id="synthetic",kind="historical_empirical",training_cutoff="2015-01-01",validation_window=["2014-12-01","2014-12-31"],
                  feature_schema_version="test",feature_contract_digest="f",horizons=list(range(1,29)),output_schema=["q10","q50","q90"],dependencies={},historical_provenance="SYNTHETIC UNIT FIXTURE")
    registry.register("v1",metadata,[model],asof="2016-01-31")
    return registry,model,metadata


class FeatureServingTests(unittest.TestCase):
    def test_shared_transforms_exact_lags_and_golden_mean(self):
        c,s,calendar,keys,mapping=fixture();f,_=extract(s,calendar,keys,"2016-02-28",c,mapping)
        self.assertEqual(len(f),84)
        row=f.iloc[0];history=s[s.store_id.eq(row.store_id)&s.date.le(row.origin_date)].units_sold
        self.assertEqual(row.origin_lag_1,history.iloc[-2]);self.assertEqual(row.rolling_mean_7_at_origin,history.tail(7).mean())
        self.assertEqual(np.mean([10,12,8,14,11,9,13]),11);self.assertAlmostEqual(np.mean([12,8,14,11,9,13,40]),107/7)

    def test_future_mutations_do_not_change_features_or_snapshot_identity(self):
        c,s,cal,k,m=fixture();f,h=extract(s,cal,k,"2016-02-28",c,m)
        changed=s.copy();changed.loc[changed.date.gt("2016-02-28"),"units_sold"]+=100000
        other,d=extract(changed,cal,k,"2016-02-28",c,m);parity(f,other);self.assertEqual(h,d)

    def test_late_arrival_excluded_and_critical_gap_blocks(self):
        c,s,cal,k,m=fixture();origin=pd.Timestamp("2016-02-28")
        changed=s.copy();changed.loc[changed.date.eq(origin),"available_at"]=origin+pd.Timedelta(days=1)
        self.assertFalse(eligible_history(changed,origin).date.eq(origin).any())
        with self.assertRaisesRegex(Blocked,"STALE_SOURCE"):extract(changed,cal,k,origin,c,m)

    def test_stale_incomplete_duplicate_schema_and_invalid_sales(self):
        c,s,cal,k,m=fixture()
        cases=[s[~s.date.eq("2016-02-28")],s[~s.date.eq("2016-02-20")],pd.concat([s,s.iloc[[0]]]),s.drop(columns="units_sold"),s.assign(units_sold=-1)]
        for bad in cases:
            with self.subTest(),self.assertRaises(Blocked):extract(bad,cal,k,"2016-02-28",c,m)

    def test_parity_catches_lag_null_and_encoding_skew(self):
        c,s,cal,k,m=fixture();f,_=extract(s,cal,k,"2016-02-28",c,m)
        for name,value in [("origin_lag_1",-1),("last_known_price",0.),("store_id_code",999)]:
            changed=f.copy();changed.loc[0,name]=value
            with self.assertRaisesRegex(Blocked,"TRAINING_SERVING_SKEW"):parity(f,changed)
        changed=f.copy();changed.loc[0,"store_id_code"]=999
        with self.assertRaisesRegex(Blocked,"ENCODING_MISMATCH"):validate_frame(changed,c,m)

    def test_future_feature_and_wrong_type_block(self):
        c,s,cal,k,m=fixture();f,_=extract(s,cal,k,"2016-02-28",c,m)
        bad=f.copy();bad["max_sales_source_date"]+=pd.Timedelta(days=1)
        with self.assertRaises(Blocked):validate_frame(bad,c,m)
        bad=f.copy();bad["horizon"]=bad.horizon.astype(float)
        with self.assertRaisesRegex(Blocked,"FEATURE_TYPES"):validate_frame(bad,c,m)
        bad=f.copy();bad["rolling_mean_7_at_origin"]=bad.rolling_mean_7_at_origin.astype(int)
        with self.assertRaisesRegex(Blocked,"FEATURE_TYPES"):validate_frame(bad,c,m)


class RegistryTests(unittest.TestCase):
    def test_immutable_registration_digest_and_untrusted_paths(self):
        with TemporaryDirectory() as tmp:
            reg,model,metadata=registry_fixture(tmp)
            with self.assertRaisesRegex(Blocked,"MODEL_NOT_APPROVED"):reg.verify("v1",approved=True)
            reg.transition("v1","approved",reason="unit test",asof="2016-01-31");reg.verify("v1",approved=True)
            with self.assertRaisesRegex(Blocked,"IMMUTABLE_REGISTRY_CONFLICT"):reg.register("v1",{**metadata,"model_id":"changed"},[model],asof="x")
            model.write_text("tampered")
            with self.assertRaisesRegex(Blocked,"ARTIFACT_DIGEST_MISMATCH"):reg.verify("v1")
            with self.assertRaises(Blocked):reg._artifact("../outside.pkl")

    def test_valid_atomic_rollback_and_blocked_missing_target(self):
        with TemporaryDirectory() as tmp:
            reg,p,meta=registry_fixture(tmp)
            for name in ["v1","v2"]:
                if name=="v2":reg.register(name,{**meta,"model_id":"v2"},[p],asof="x")
                reg.transition(name,"approved",reason="unit",asof="x");reg.activate("forecast",name,contract_digest="f",reason="unit",asof="x")
            reg.activate("forecast","v1",contract_digest="f",reason="rollback",asof="x",rollback=True)
            self.assertEqual(reg.resolve("forecast")["model_version"],"v1")
            p.unlink();before=reg.state()
            with self.assertRaises(Blocked):reg.activate("forecast","v2",contract_digest="f",reason="missing",asof="x",rollback=True)
            self.assertEqual(before,reg.state())

    def test_compatibility_future_fit_retirement_and_state_corruption(self):
        with TemporaryDirectory() as tmp:
            reg,p,meta=registry_fixture(tmp)
            with self.assertRaisesRegex(Blocked,"INCOMPATIBLE_FEATURE_CONTRACT"):reg.verify("v1",contract_digest="other")
            with self.assertRaisesRegex(Blocked,"MODEL_FIT_AFTER_ORIGIN"):reg.verify("v1",origin="2014-01-01")
            reg.transition("v1","approved",reason="unit",asof="x");reg.transition("v1","retired",reason="unit",asof="x")
            with self.assertRaises(Blocked):reg.verify("v1")
            state=reg.state();state["aliases"]["forecast"]="v1";(reg.directory/"state.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(Blocked,"REGISTRY_STATE_CORRUPTION"):reg.state()


class PublicationTests(unittest.TestCase):
    def test_identity_and_canonical_serialization(self):
        self.assertEqual(digest({"b":2,"a":1}),digest({"a":1,"b":2}))
        entry=dict(artifact_digest="a",model_version="v",horizons=[1,2])
        a,_=request_id(entry,"f","s","2016-01-31",["cohort"],{},channel="approved")
        b,_=request_id(entry,"f","changed","2016-01-31",["cohort"],{},channel="approved")
        self.assertNotEqual(a,b)

    def test_atomic_retry_interruption_conflict_and_corruption(self):
        with TemporaryDirectory() as tmp:
            forecast=pd.DataFrame({"prediction":[1.,2.]});decisions=pd.DataFrame({"approved_quantity":[1,0]})
            validator=lambda f,d: self.assertEqual((len(f),len(d)),(2,2))
            with self.assertRaisesRegex(Blocked,"INJECTED_INTERRUPTED_STAGE"):publish(tmp,{"id":1},forecast,decisions,validator,interrupt=True)
            self.assertFalse((Path(tmp)/"batches").exists())
            batch,first,status=publish(tmp,{"id":1},forecast,decisions,validator)
            self.assertEqual(status,"PUBLISHED_LOCAL_SIMULATION")
            _,second,status=publish(tmp,{"id":1},forecast,decisions,validator)
            self.assertEqual(first,second);self.assertEqual(status,"REUSED_NO_NEW_ACTIONS")
            with self.assertRaisesRegex(Blocked,"IDEMPOTENCY_CONFLICT"):publish(tmp,{"id":1},forecast*2,decisions,validator)
            path=Path(tmp)/"batches"/batch/"forecasts.parquet";path.write_bytes(path.read_bytes()+b"corrupted")
            with self.assertRaisesRegex(Blocked,"PUBLICATION_CORRUPTION"):read_batch(path.parent)

    def test_incomplete_manifest_and_lock_do_not_publish(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Blocked,"PUBLICATION_INCOMPLETE"):read_batch(tmp)
            with writer_lock(tmp):
                with self.assertRaisesRegex(Blocked,"LOCAL_WRITER_BUSY"):
                    with writer_lock(tmp):self.fail("concurrent writer acquired lock")


class MonitorTests(unittest.TestCase):
    policy=dict(minimum_samples=20,psi_warning=.25,ks_warning=.30,missingness_warning=.1,version="test-v1",window="test")

    def test_psi_goldens_zero_bins_equal_and_renormalization(self):
        self.assertAlmostEqual(psi_probabilities([.5,.5],[.3,.7]),.169459572,places=8)
        self.assertEqual(psi_probabilities([.5,.5],[.5,.5]),0)
        self.assertTrue(np.isfinite(psi_probabilities([1,0],[0,1])))
        self.assertAlmostEqual(psi_probabilities([5,5],[3,7]),.169459572,places=8)

    def test_ks_golden_ties_constant_and_empty(self):
        self.assertEqual(ks([1,2,3,4],[3,4,5,6]),.5)
        self.assertEqual(ks([1,1,1],[1,1]),0);self.assertEqual(ks([1,1],[2,2]),1)
        self.assertIsNone(ks(np.array([],float),[1,2]))

    def test_fixed_bins_severe_shift_missingness_and_typing(self):
        a=np.arange(25,dtype=float);b=a+100
        bins=reference_bins(a);self.assertEqual(len(frequencies(b,bins)),len(bins))
        self.assertEqual(drift(a,b,"x",self.policy,origin="x")["severity"],"warning")
        self.assertEqual(drift(a,np.full(25,np.nan),"x",self.policy,origin="x")["severity"],"warning")
        self.assertEqual(drift(a,a,"x",self.policy,origin="x")["psi"],0)
        with self.assertRaises(Blocked):ks(["string"],["type"])
        with self.assertRaises(Blocked):drift([],a,"x",self.policy,origin="x")

    def test_label_maturity_eight_horizons_and_no_zero_fill(self):
        origin=pd.Timestamp("2016-01-31");dates=pd.date_range(origin+pd.Timedelta(days=1),periods=28)
        f=pd.DataFrame(dict(store_id="s",item_id="i",origin_date=origin,horizon=np.arange(1,29),target_date=dates,prediction=2.,q10=0.,q50=2.,q90=4.))
        labels=pd.DataFrame(dict(store_id="s",item_id="i",date=dates,units_sold=2))
        ledger=mature_labels(f,labels,"2016-02-10",lag_days=2);self.assertEqual(ledger.loc[ledger.mature,"horizon"].tolist(),list(range(1,9)))
        self.assertTrue(ledger.loc[~ledger.mature,"actual"].isna().all());m=performance(ledger)
        self.assertEqual(m["mature"],8);self.assertEqual(m["unavailable"],20);self.assertEqual(m["mae"],0);self.assertEqual(m["coverage"],1)
        missing=mature_labels(f,labels.iloc[1:],"2016-03-05");self.assertEqual(performance(missing)["missing_labels"],1)

    def test_zero_denominator_missing_and_matched_cohorts(self):
        f=pd.DataFrame(dict(store_id=["s"],item_id=["i"],origin_date=pd.to_datetime(["2016-01-31"]),horizon=[1],target_date=pd.to_datetime(["2016-02-01"]),prediction=[1.],q10=[0.],q50=[1.],q90=[2.]))
        labels=pd.DataFrame(dict(store_id=["s"],item_id=["i"],date=pd.to_datetime(["2016-02-01"]),units_sold=[0]))
        l=mature_labels(f,labels,"2016-02-10");self.assertIsNone(performance(l)["wape"])
        matched_performance(l,l)
        with self.assertRaises(Blocked):matched_performance(l,l.iloc[:0])

    def test_incident_severity_and_deterministic_pool_routing(self):
        a=canary("shared",.5);self.assertEqual(a,canary("shared",.5))
        self.assertEqual(canary("shared",0)["route"],"approved");self.assertEqual(canary("shared",1)["route"],"candidate")
        i=incident("signal","blocking","2016-01-31",disposition="publication blocked")
        self.assertEqual(i["event_id"],incident("signal","blocking","2016-01-31",disposition="publication blocked")["event_id"])
        with TemporaryDirectory() as tmp:
            reg,p,meta=registry_fixture(tmp)
            self.assertEqual(fallback(reg,"f","2016-01-31","STALE_SOURCE")["mode"],"BLOCKED_HUMAN_REVIEW")


class BatchBoundaryTests(unittest.TestCase):
    """Small real callable pipeline with a declared synthetic baseline recipe."""
    def setUp(self):
        from demandos.serving.core import read_json
        from demandos.serving.decisions import frozen_inputs
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.config,self.snapshot,self.calendar,self.keys,self.mapping=fixture()
        self.keys=self.keys.assign(cohort="synthetic",volume=1.)
        self.keys["series_id"]=[json.dumps([s,i],separators=(",",":")) for s,i in self.keys[["store_id","item_id"]].itertuples(index=False,name=None)]
        self.keys=self.keys.sort_values("series_id").reset_index(drop=True)
        self.origin=pd.Timestamp("2016-02-28")
        recipe=self.root/"configs/serving_baseline.json";recipe.parent.mkdir()
        recipe.write_text(json.dumps(dict(days=56,trained_weights=False,algorithm="historical_empirical",levels=[.1,.5,.9],method="inverted_cdf_including_zeros")))
        self.registry=Registry(self.root,self.root/"registry",[recipe.parent])
        metadata=dict(model_id="synthetic-baseline",kind="historical_empirical",training_cutoff=None,validation_window=None,feature_schema_version="test",feature_contract_digest="f",
                      horizons=list(range(1,29)),output_schema=["q10","q50","q90"],dependencies={},historical_provenance="SYNTHETIC FIXTURE / NO FITTED WEIGHTS")
        self.registry.register("baseline",metadata,[recipe],asof=str(self.origin))
        self.settings=dict(on_hand=10,on_order=3,purchase=60.,order_fee=5.,receiving_factor=1.,solver_seconds=3.,seed=42,review_days=7,
                           holding=.2,lost_penalty=40.,backlog_daily=2.)
        _,self.scenario,self.resources,_=frozen_inputs(self.snapshot,self.keys,self.origin,self.settings)

    def serve(self,channel="approved",**kwargs):
        from demandos.serving.batch import serve
        if channel!="shadow" and self.registry.state()["status"]["baseline"]=="candidate":
            self.registry.transition("baseline","approved",reason="synthetic tests",asof=str(self.origin))
        return serve(self.registry,self.root/"publication","baseline",self.snapshot,self.calendar,self.keys,self.origin,self.config,self.mapping,"f",{},
                     self.settings,self.scenario,self.resources,channel=channel,**kwargs)

    def test_callable_baseline_batch_and_effectively_once_retry(self):
        first=self.serve();second=self.serve()
        self.assertEqual((len(first["forecasts"]),len(first["decisions"])),(84,3))
        self.assertEqual(first["publication"],second["publication"])
        self.assertEqual(second["publication_status"],"REUSED_NO_NEW_ACTIONS")

    def test_serving_role_alias_resolves_verified_version(self):
        from demandos.serving.batch import serve
        self.registry.transition("baseline","approved",reason="synthetic alias",asof=str(self.origin))
        self.registry.activate("forecast","baseline",contract_digest="f",reason="unit alias",asof=str(self.origin))
        batch=serve(self.registry,self.root/"publication","forecast",self.snapshot,self.calendar,self.keys,self.origin,self.config,self.mapping,"f",{},
                    self.settings,self.scenario,self.resources)
        self.assertTrue(batch["forecasts"].model_version.eq("baseline").all())

    def test_shadow_has_no_active_actions_and_unapproved_canary_blocks(self):
        from demandos.serving.routing import publish_canary
        batch=self.serve("shadow");self.assertTrue(batch["decisions"].empty)
        self.assertEqual(len(batch["comparison_decisions"]),3)
        with self.assertRaisesRegex(Blocked,"MODEL_NOT_APPROVED"):
            publish_canary(self.root/"canary",canary("pool",1),batch,self.registry,"f")

    def test_whole_pool_canary_immutable_reference_retry(self):
        from demandos.serving.routing import publish_canary
        batch=self.serve();route=canary("one_shared_pool",1)
        a,x=publish_canary(self.root/"canary",route,batch,self.registry,"f")
        b,y=publish_canary(self.root/"canary",route,batch,self.registry,"f")
        self.assertEqual(x,y);self.assertEqual(len(list((self.root/"canary").glob("*.json"))),1)
        self.assertEqual(a.model_version.nunique(),1);pd.testing.assert_frame_equal(a,b)

    def test_expected_snapshot_digest_blocks_before_publication(self):
        with self.assertRaisesRegex(Blocked,"SOURCE_SNAPSHOT_DIGEST"):self.serve(expected_source_digest="wrong")
        self.assertFalse((self.root/"publication/batches").exists())

    def test_publication_checks_every_forecast_and_decision_lineage(self):
        batch=self.serve()
        f=batch["forecasts"].copy();f.loc[0,"source_snapshot_digest"]="wrong"
        with self.assertRaisesRegex(Blocked,"PUBLICATION_LINEAGE"):batch["validator"](f,batch["decisions"])
        d=batch["decisions"].copy();d.loc[0,"target_units"]+=1
        with self.assertRaisesRegex(Blocked,"DECISION_FORECAST_LINEAGE"):batch["validator"](batch["forecasts"],d)

    def test_nonfinite_negative_crossed_duplicate_and_wrong_dates_reject(self):
        batch=self.serve();f=batch["forecasts"];d=batch["decisions"]
        for column,value in [("q50",float("nan")),("q10",-1.),("q90",-1.),("model_digest","wrong"),("target_date",pd.Timestamp("2017-01-01"))]:
            bad=f.copy();bad.loc[0,column]=value
            with self.subTest(column=column),self.assertRaises(Blocked):batch["validator"](bad,d)
        with self.assertRaises(Blocked):batch["validator"](pd.concat([f,f.iloc[[0]]]),d)

    def test_optimizer_infeasible_is_not_an_automatic_zero_order(self):
        from demandos.serving.decisions import plan
        batch=self.serve()
        with self.assertRaisesRegex(Blocked,"OPTIMIZER_INFEASIBLE"):
            plan(batch["forecasts"],self.keys,self.settings,self.scenario,self.resources,force_infeasible=True)

    def test_pack_moq_and_resource_overrun_block(self):
        from demandos.serving.decisions import validate_decisions
        batch=self.serve();original=batch["decisions"]
        for qty,pack,moq in [(5,6,0),(1,1,12),(100000000,1,0)]:
            d=original.copy();d.loc[0,"approved_quantity"]=qty;d.loc[0,"pack"]=pack;d.loc[0,"moq"]=moq
            with self.subTest(),self.assertRaises(Blocked):validate_decisions(d)

    def test_fallback_revalidates_contract_and_hard_input_failures(self):
        self.serve();self.registry.activate("fallback","baseline",contract_digest="f",reason="unit",asof=str(self.origin))
        self.assertEqual(fallback(self.registry,"f",self.origin,"ARTIFACT_DIGEST_MISMATCH")["mode"],"TRUSTED_ELIGIBLE_BASELINE")
        self.assertEqual(fallback(self.registry,"bad",self.origin,"ARTIFACT_DIGEST_MISMATCH")["mode"],"BLOCKED_HUMAN_REVIEW")
        self.assertEqual(fallback(self.registry,"f",self.origin,"INCOMPLETE_HISTORY")["mode"],"BLOCKED_HUMAN_REVIEW")


if __name__=="__main__":unittest.main()
