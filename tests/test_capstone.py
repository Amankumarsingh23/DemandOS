"""Deterministic SYNTHETIC integration fixtures; never presented as actual M5."""
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tomllib

import numpy as np
import pandas as pd

from demandos.capstone.config import CONFIG, LABELS, prerequisites, run_identity, table_label
from demandos.capstone.pipeline import select, capacities
from demandos.capstone.audit import audit, forecast_checks, trajectory_checks, storage_checks, prefix
from demandos.capstone.documentation import introduction_checks, validate
from demandos.capstone.evidence import metrics
from demandos.capstone.publication import publish, seal, check_seal
from demandos.capstone.faults import exercise
from demandos.capstone.__main__ import main
from demandos.serving.core import Blocked, atomic_json, read_json
from demandos.serving.features import extract
from demandos.serving.monitoring import mature_labels
from demandos.serving.decisions import plan, downstream, target_vector
from demandos.risk.pipeline import origin_problem
from demandos.risk.model import Scenarios
from demandos.causal.simulation import intervention
from demandos.risk.scenarios import Closure

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    metadata = pd.DataFrame(dict(store_id=["S1","S2"], item_id=["I1","I2"],
        series_id=['["S1","I1"]','["S2","I2"]'], dept_id=["D","D"],
        cat_id=["FOODS","FOODS"], volume=[1.,1.], cohort=["continuity","sparse_stress"]))
    settings = tomllib.loads((ROOT/"configs/optimization.toml").read_text())["optimization"]
    settings.update(holding=.2,lost_penalty=40.,backlog_daily=2.,order_fee=5.)
    entry = dict(artifact_digest="SYNTHETIC-NOT-FITTED",model_version="synthetic-unit-fixture",
                 training_cutoff="2016-01-31",validation_window=["2015-12-01","2016-01-03"])
    rows = []
    for m in metadata.itertuples():
        for h in range(1,29):
            rows.append(dict(store_id=m.store_id,item_id=m.item_id,origin_date=pd.Timestamp(CONFIG["origin"]),horizon=h,
                target_date=pd.Timestamp(CONFIG["origin"])+pd.Timedelta(days=h),q10=1.,q50=2.,q90=6.,prediction=2.,
                model_version=entry["model_version"],model_digest=entry["artifact_digest"],feature_contract_digest="synthetic",prediction_id=f"{m.series_id}:{h}"))
    forecasts = pd.DataFrame(rows)
    basis = np.array([35,35])
    scenarios = Scenarios(np.array([[10,20,25,30,35,40,45,50]]*2),np.full(8,.125),tuple(str(i) for i in range(8)))
    _, resources, _ = origin_problem(metadata,basis,settings)
    decisions,_ = plan(forecasts,metadata,settings,scenarios,resources)
    demand = np.full((2,28),5,dtype=int)
    result,_ = downstream(demand,pd.Timestamp(CONFIG["origin"]),metadata,forecasts,scenarios,settings,basis)
    altered = demand.copy();altered[0,13] += 1000
    perturbed,_ = downstream(altered,pd.Timestamp(CONFIG["origin"]),metadata,forecasts,scenarios,settings,basis)
    closure = intervention(demand,pd.Timestamp(CONFIG["origin"]),metadata,target_vector(forecasts,metadata),scenarios,
        {**settings,"resource_targets":basis},closure=Closure(8,14))
    snapshot = forecasts[["store_id","item_id","target_date"]].rename(columns={"target_date":"date"}).assign(units_sold=5)
    maturity = mature_labels(forecasts,snapshot,CONFIG["monitoring_date"],lag_days=2)
    return metadata,forecasts,entry,decisions,result,perturbed,closure,snapshot,maturity,capacities(metadata,resources)


def write_fixture(directory):
    m,f,e,d,r,p,c,s,l,cap = fixture()
    frames = dict(metadata=m,forecasts=f,decisions=d,snapshot=s,maturity=l)
    for pre,res in [("",r),("perturbed_",p),("closure_",c)]:
        frames.update({pre+n:res[n] for n in ["daily","events","allocations","resources","solver"]})
        frames[pre+"solver"] = res["solver"][["day_index","status"]]
    (directory/"tables").mkdir()
    for n,frame in frames.items():frame.to_parquet(directory/"tables"/(n+".parquet"),index=False)
    manifest = dict(config=CONFIG,labels=LABELS,model=e,feature_contract_digest="synthetic",storage_capacities=cap,
                    table_labels={name:table_label(name) for name in frames},
                    sources={"synthetic":"source-digest"},models={"synthetic":"model-digest"})
    atomic_json(directory/"manifest.json",manifest)
    atomic_json(directory/"summary.json",dict(forecast_metrics=metrics(l),simulation_metrics=trajectory_checks(r,2)))
    atomic_json(directory/"evidence.json",dict(claim_labels=LABELS,records=[]))
    atomic_json(directory/"faults.json",[dict(accepted=True,label=LABELS["faults"])]*5)


class CapstoneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parts = fixture()

    def test_identity_canonical_and_sensitive_to_every_frozen_input(self):
        base=dict(code="a",source="b",model="c",seed=42,origin=CONFIG["origin"],cohort=["x"],horizon=28)
        self.assertEqual(run_identity(base),run_identity(dict(reversed(list(base.items())))))
        for name in base:
            changed={**base,name:"changed"}
            self.assertNotEqual(run_identity(base),run_identity(changed))

    def test_selection_independent_of_future_outcomes_and_stable_to_input_order(self):
        m=pd.DataFrame(dict(series_id=[str(i) for i in range(10)],cohort=["continuity"]*5+["sparse_stress"]*5))
        pd.testing.assert_frame_equal(select(m),select(m.sample(frac=1,random_state=42)))
        self.assertEqual(select(m).series_id.tolist(),["0","1","2","5","6","7"])

    def test_golden_forecast_coverage_and_dates(self):
        m,f,e,*_=self.parts
        forecast_checks(f,m,CONFIG["origin"],e,"synthetic")
        for bad in [f.iloc[:-1],pd.concat([f,f.iloc[[0]]]),f.assign(target_date=f.target_date+pd.Timedelta(days=1))]:
            with self.subTest(),self.assertRaises(Blocked):forecast_checks(bad,m,CONFIG["origin"],e,"synthetic")

    def test_quantiles_and_native_provenance(self):
        m,f,e,*_=self.parts
        for bad in [f.assign(q10=100.),f.assign(q90=np.nan),f.assign(model_digest="other"),f.assign(feature_contract_digest="other")]:
            with self.subTest(),self.assertRaises(Blocked):forecast_checks(bad,m,CONFIG["origin"],e,"synthetic")
        with self.assertRaisesRegex(Blocked,"MODEL_CUTOFF"):
            forecast_checks(f,m,CONFIG["origin"],{**e,"training_cutoff":"2016-02-01"},"synthetic")

    def test_future_sales_and_price_are_excluded_from_features(self):
        from test_serving import fixture as feature_fixture
        c,s,cal,k,m=feature_fixture()
        f,h=extract(s,cal,k,"2016-02-28",c,m)
        changed=s.copy();changed.loc[changed.date.gt("2016-02-28"),"units_sold"]+=10000
        changed["sell_price"]=99999.
        other,g=extract(changed,cal,k,"2016-02-28",c,m)
        pd.testing.assert_frame_equal(f,other);self.assertEqual(h,g)

    def test_inventory_cost_resource_conservation(self):
        result=self.parts[4]
        measured=trajectory_checks(result,2)
        self.assertGreater(measured["cost"],0);self.assertLessEqual(measured["budget_used"],measured["budget_initial"])
        for name,column in [("daily","ending_stock"),("daily","operational_cost"),("resources","budget_after"),("allocations","activation")]:
            bad={k:v.copy() if isinstance(v,pd.DataFrame) else v for k,v in result.items()}
            bad[name].loc[0,column]+=1
            with self.subTest(column=column),self.assertRaises((Blocked,AssertionError)):trajectory_checks(bad,2)

    def test_duplicate_order_and_receipt_blocked(self):
        result=self.parts[4]
        for table in ["allocations","events"]:
            bad={**result,table:pd.concat([result[table],result[table].iloc[[0]]],ignore_index=True)}
            with self.subTest(),self.assertRaises((Blocked,AssertionError,pd.errors.MergeError)):trajectory_checks(bad,2)

    def test_physical_storage_distinct_from_flow_quota(self):
        m,_,_,d,r,*_=self.parts
        storage_checks(r["daily"],d,m,self.parts[-1])
        with self.assertRaisesRegex(Blocked,"STORAGE"):
            storage_checks(r["daily"],d,m,{"S1":0.,"S2":0.})

    def test_all_past_states_and_actions_invariant_to_day14_demand(self):
        prefix(self.parts[4],self.parts[5],14)
        bad={k:v.copy() if isinstance(v,pd.DataFrame) else v for k,v in self.parts[5].items()}
        bad["allocations"].loc[0,"candidate_quantity"]+=1
        with self.assertRaises(AssertionError):prefix(self.parts[4],bad,14)

    def test_whole_order_delays_preserve_identity_and_quantity(self):
        closure=self.parts[6]
        self.assertTrue(closure["events"].event.eq("retimed").any())
        trajectory_checks(closure,2)

    def test_atomic_complete_publication_reuses_without_duplicate_orders(self):
        with TemporaryDirectory() as tmp:
            path,reused=publish(Path(tmp),write_fixture,audit);self.assertFalse(reused)
            first=check_seal(path)
            again,reused=publish(Path(tmp),write_fixture,audit)
            self.assertTrue(reused);self.assertEqual(path,again);self.assertEqual(first,check_seal(path))
            self.assertEqual(len(list((Path(tmp)/"runs").iterdir())),1)

    def test_same_identity_different_content_rejected(self):
        with TemporaryDirectory() as tmp:
            publish(Path(tmp),write_fixture,audit)
            def changed(stage):
                write_fixture(stage)
                f=pd.read_parquet(stage/"tables/forecasts.parquet");f.loc[0,"q10"]+=.1
                f.to_parquet(stage/"tables/forecasts.parquet",index=False)
                l=pd.read_parquet(stage/"tables/maturity.parquet");l.loc[0,"q10"]+=.1
                l.to_parquet(stage/"tables/maturity.parquet",index=False)
                summary=read_json(stage/"summary.json");summary["forecast_metrics"]=metrics(l)
                atomic_json(stage/"summary.json",summary)
            with self.assertRaisesRegex(ValueError,"RUN_CONTENT_CONFLICT"):publish(Path(tmp),changed,audit)

    def test_independent_audit_detects_corruption_even_after_resealing(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp);write_fixture(p);seal(p);audit(p)
            f=pd.read_parquet(p/"tables/daily.parquet");f.loc[0,"ending_stock"]+=1
            f.to_parquet(p/"tables/daily.parquet",index=False);seal(p)
            with self.assertRaises((Blocked,AssertionError)):audit(p)

    def test_summary_arithmetic_is_not_trusted(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp);write_fixture(p)
            s=read_json(p/"summary.json");s["simulation_metrics"]["cost"]=0;atomic_json(p/"summary.json",s)
            with self.assertRaisesRegex(Blocked,"SUMMARY_ARITHMETIC"):audit(p,sealed=False)

    def test_dataset_and_model_snapshot_mismatch_block(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp);write_fixture(p)
            for sources,models in [({"changed":"source"},{"synthetic":"model-digest"}),({"synthetic":"source-digest"},{"changed":"model"})]:
                data=SimpleNamespace(source_hashes=sources,model_hashes=models)
                with self.subTest(),patch("demandos.serving.inputs.verify_sources"),self.assertRaisesRegex(Blocked,"INPUT_DIGEST_MISMATCH"):
                    audit(p,data=data,sealed=False)

    def test_scientific_labels_must_not_relabel_simulation_as_observed(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp);write_fixture(p)
            m=read_json(p/"manifest.json");m["labels"]["inventory"]=LABELS["source"];atomic_json(p/"manifest.json",m)
            with self.assertRaisesRegex(Blocked,"SCIENTIFIC_LABEL"):audit(p,sealed=False)

    def test_missing_prerequisites_and_dry_run_do_not_write_or_load_models(self):
        with TemporaryDirectory() as tmp:
            before=list(Path(tmp).rglob("*"))
            with patch("demandos.capstone.pipeline.load",side_effect=AssertionError("must not load")),redirect_stdout(StringIO()):
                self.assertEqual(main(["--root",tmp,"--dry-run"]),1)
                self.assertEqual(main(["--root",tmp,"--check-prereqs"]),1)
            self.assertEqual(before,list(Path(tmp).rglob("*")))
            missing=prerequisites(tmp)["missing"]
            self.assertIn("data/raw/calendar.csv",missing);self.assertTrue(any("model.txt" in p for p in missing))

    def test_all_five_actual_fault_exercises(self):
        from test_serving import fixture as feature_fixture,registry_fixture
        c,s,cal,k,m=feature_fixture();features,_=extract(s,cal,k,"2016-02-28",c,m)
        with TemporaryDirectory() as tmp:
            reg,model,meta=registry_fixture(tmp)
            # Same production verifier on synthetic registry files; no native weights invented.
            reg.register("historical-56d-v1",meta,[model],asof="synthetic")
            reg.register("s16-lgb-20160131",meta,[model],asof="synthetic")
            data=SimpleNamespace(feature_config=c,mapping=m)
            results=exercise(data,features,self.parts[3],reg,"f")
            self.assertEqual(len(results),5);self.assertTrue(all(x["accepted"] for x in results))
            self.assertEqual(results[-1]["disposition"],"REUSED_NO_NEW_ACTIONS")

    def test_documentation_rejects_stale_introductions(self):
        with self.assertRaisesRegex(Blocked,"STALE_README"):
            introduction_checks("Current milestone: Season 14","Implemented scope: Seasons 1–24")
        with self.assertRaisesRegex(Blocked,"STALE_ARCHITECTURE"):
            introduction_checks("Season 24","Implemented scope: Seasons 1–11")

    def test_internal_documentation_links_and_current_scope(self):
        result=validate(ROOT);self.assertGreater(result["internal_links"],100)

    def test_broken_and_machine_specific_links_block(self):
        for link in ["missing.md","C:/private/path.md"]:
            with TemporaryDirectory() as tmp:
                r=Path(tmp);(r/"docs").mkdir()
                (r/"README.md").write_text("Season 24\n[bad]("+link+")",encoding="utf-8")
                (r/"docs/architecture.md").write_text("Implemented scope: Seasons 1–24",encoding="utf-8")
                with self.subTest(),self.assertRaises(Blocked):validate(r)

    def test_metric_arithmetic_and_zero_denominator(self):
        f=pd.DataFrame(dict(actual=[0,2],q10=[0.,0.],q50=[0.,1.],q90=[1.,3.]))
        got=metrics(f);self.assertEqual(got["mae"],.5);self.assertEqual(got["wape"],.5);self.assertEqual(got["coverage"],1.)
        self.assertIsNone(metrics(f.assign(actual=0))["wape"])


if __name__ == "__main__":unittest.main()
