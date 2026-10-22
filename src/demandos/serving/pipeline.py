"""End-to-end LOCAL historical serving, injected incidents and verified run pointer."""
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import platform
import tomllib
import numpy as np
import pandas as pd

from demandos.reconciliation.pipeline import peak_memory
from .inputs import load,verify_sources,ORIGINS
from .core import atomic_json,read_json,digest,file_hash,require,STATUS
from .features import contract,parity,CASE
from .bootstrap import initialize,APPROVED,CANDIDATE,BASELINE
from .batch import serve
from .decisions import frozen_inputs,validate_decisions
from .faults import exercise
from .experiments import temporal,monitors,simulate_controllers
from .routing import canary,publish_canary,fallback
from .monitoring import performance,incident


def load_config(path):
    config=tomllib.loads(Path(path).read_text(encoding="utf-8"))
    require(set(config)=={"serving","alerts"},"SERVING_CONFIG")
    s=config["serving"];a=config["alerts"]
    require(s==dict(seed=42,origins=[str(o.date()) for o in ORIGINS],horizon=28,series=25,label_lag_days=2,canary_fraction=.5,
                    solver_seconds=3.,determinism_atol=1e-10,simulation_worlds=4),"PREDECLARED_REPLAY_CONFIG")
    require(a==dict(version="historical-demo-v1",minimum_samples=20,psi_warning=.25,ks_warning=.30,missingness_warning=.10,
                    mae_relative_warning=.20,coverage_drop_warning=.10,fallback_rate_warning=.10,simulated_cost_increase_warning=.10,
                    simulated_fill_drop_warning=.05,window="one origin; reference frozen at 2016-01-31"),"PREDECLARED_ALERT_POLICY")
    return config


def run(root,config):
    started=perf_counter();data=load(Path(root));root=data.root
    feature_contract=contract(root,data.feature_config,data.mapping);contract_digest=digest(feature_contract)
    code={p.name:file_hash(p) for p in sorted((root/"src/demandos/serving").glob("*.py"))}
    environment={p:version(p) for p in ["numpy","pandas","pyarrow","duckdb","scipy","lightgbm"]}
    run_config=dict(config=config,code_digest=digest(code),environment=environment,python=platform.python_version(),
                    assumed_decision_settings=data.settings,
                    decision_implementations={p:file_hash(root/"src/demandos"/p) for p in ["inventory/core.py","optimization/model.py","optimization/coordinator.py",
                        "risk/model.py","risk/pipeline.py","risk/scenarios.py","simulation/engine.py","simulation/policies.py"]},
                    historical_data_scope="25 series; assumed event-date availability; same fixed retrospective origins")
    run_id=digest(dict(contract=contract_digest,run_config=run_config))[:20]
    base=root/"data/serving/m5_h28";output=base/"runs"/run_id;output.mkdir(parents=True,exist_ok=True)
    tables=output/"tables";tables.mkdir(exist_ok=True)
    atomic_json(output/"feature_contract.json",feature_contract);atomic_json(output/"config.json",run_config)
    registry,entries=initialize(data,output/"registry",contract_digest)
    bundles={};retry=[];parities=[];shadow=[];incidents=[];features=[]
    for origin in ORIGINS:
        _,scenario,resource,_=frozen_inputs(data.snapshot,data.metadata,origin,data.settings)
        for name,channel in [(APPROVED,"approved"),(CANDIDATE,"shadow")]:
            kwargs=dict(channel=channel)
            batch=serve(registry,output/"publication",name,data.snapshot,data.calendar,data.metadata,origin,data.feature_config,data.mapping,
                        contract_digest,run_config,data.settings,scenario,resource,**kwargs)
            batch["directory"]=output/"publication";bundles[(name,origin)]=batch
            again=serve(registry,output/"publication",name,data.snapshot,data.calendar,data.metadata,origin,data.feature_config,data.mapping,
                        contract_digest,run_config,data.settings,scenario,resource,**kwargs)
            require(again["publication_status"]=="REUSED_NO_NEW_ACTIONS" and again["publication"]==batch["publication"],"REPLAY_IDEMPOTENCY")
            np.testing.assert_allclose(batch["forecasts"][["q10","q50","q90"]],again["forecasts"][["q10","q50","q90"]],atol=1e-10,rtol=0)
            retry.append(dict(batch_id=batch["batch_id"],model_version=name,origin=str(origin),passed=True,retry_status=again["publication_status"]))
            p=parity(data.reference_features[data.reference_features.origin_date.eq(origin)],batch["features"])
            parities.append(dict(model_version=name,origin_date=origin,**p))
            shadow.append(dict(model_version=name,origin_date=origin,channel=channel,forecast_rows=len(batch["forecasts"]),published_actions=len(batch["decisions"]),
                               runtime=batch["predictor"]["seconds"],valid=True,constraint_violations=0,plan_status=batch["decision_diagnostic"]["status"],
                               model_training_cutoff=registry.entry(name)["training_cutoff"],fit_provenance="different frozen fits; raw quantiles separately identified; no conformal correction"))
            if name==APPROVED:features.append(batch["features"])
        print(f"Real historical batch/retry/parity PASS {origin.date()}: 700 approved / 700 shadow predictions",flush=True)
    first=bundles[(APPROVED,ORIGINS[0])]
    _,scenario,resource,_=frozen_inputs(data.snapshot,data.metadata,ORIGINS[0],data.settings)
    faults,events,injections,recovery=exercise(data,registry,first,contract_digest,config,scenario,resource,output/"faults")
    incidents.extend(events)
    # A real supported fallback forecast is produced after the isolated injected model failure.
    fallback_event=next(e for e in events if e["signal"]=="BASELINE_FALLBACK")
    safe=serve(registry,output/"publication",BASELINE,data.snapshot,data.calendar,data.metadata,ORIGINS[0],data.feature_config,data.mapping,
               contract_digest,run_config,data.settings,scenario,resource,channel="fallback",fallback_mode=recovery["mode"],incident_ids=[fallback_event["event_id"]])
    safe["directory"]=output/"publication";bundles[(BASELINE,ORIGINS[0])]=safe
    replay=serve(registry,output/"publication",BASELINE,data.snapshot,data.calendar,data.metadata,ORIGINS[0],data.feature_config,data.mapping,
                 contract_digest,run_config,data.settings,scenario,resource,channel="fallback",fallback_mode=recovery["mode"],incident_ids=[fallback_event["event_id"]])
    retry.append(dict(batch_id=safe["batch_id"],passed=replay["publication"]==safe["publication"],retry_status=replay["publication_status"]))
    future=temporal(data,registry,bundles,contract_digest,config)
    drifts,metrics,label_table,matched,events=monitors(data,bundles,config);incidents.extend(events)
    atomic_json(output/"drift.json",drifts.to_dict("records"))
    if registry.state()["status"][CANDIDATE]=="candidate":
        registry.transition(CANDIDATE,"approved",reason="SIMULATED canary only: verified digest, fit/feature/output/resource gates; no future-label selection",asof="2016-01-31 assumed local approval")
    if registry.state()["aliases"].get("canary")!=CANDIDATE:
        registry.activate("canary",CANDIDATE,contract_digest=contract_digest,reason="separate local canary role; does not change champion",asof="2016-01-31")
    route=canary("SELECTED_COHORT_DC",config["serving"]["canary_fraction"],config["serving"]["seed"])
    canaries=[];routing=[]
    for origin in ORIGINS:
        name=CANDIDATE if route["route"]=="candidate" else APPROVED
        chosen=bundles[(name,origin)]
        frame,identity=publish_canary(output/"canary",route,chosen,registry,contract_digest)
        again,second=publish_canary(output/"canary",route,chosen,registry,contract_digest)
        require(identity==second,"CANARY_RETRY");pd.testing.assert_frame_equal(frame,again)
        canaries.append(frame);routing.append(dict(origin_date=origin,canary_id=identity,model_version=name,forecast_batch_id=chosen["batch_id"],**route))
        incidents.append(incident("SIMULATED_CANARY_ROUTED","informational",origin,model_version=name,batch_id=chosen["batch_id"],feature_digest=contract_digest,
                                  evidence=dict(canary_id=identity,route=route),disposition="independent local simulation namespace; no external orders"))
    # Alias rollback is separate from forecasts/actions; old batches remain byte-identical.
    published_before={p.relative_to(output).as_posix():file_hash(p) for p in (output/"publication/batches").rglob("*") if p.is_file()}
    registry.activate("forecast",CANDIDATE,contract_digest=contract_digest,reason="synthetic incident response exercise; no dispatch",asof="2016-03-27 local exercise")
    registry.activate("forecast",APPROVED,contract_digest=contract_digest,reason="restore prior approved version after synthetic incident",asof="2016-03-27 local exercise",rollback=True)
    require(published_before=={p.relative_to(output).as_posix():file_hash(p) for p in (output/"publication/batches").rglob("*") if p.is_file()},"ROLLBACK_MUTATED_PUBLICATION")
    incidents.append(incident("LOCAL_ALIAS_ROLLBACK","informational","2016-03-27 local exercise",model_version=APPROVED,feature_digest=contract_digest,
                              evidence=dict(previous=CANDIDATE,restored=APPROVED,published_batches_unchanged=True),disposition="prior approved alias restored; no duplicate actions"))
    runs,solvers,temporal_checks,risk,sim_counts=simulate_controllers(data,bundles,config,output/"simulation")
    paired=runs[runs.model_version.eq(APPROVED)].merge(runs[runs.model_version.eq(CANDIDATE)],on=["origin_date","world_kind","world_id"],suffixes=("_approved","_candidate"),validate="one_to_one")
    for row in paired.itertuples():
        if row.cost_candidate>row.cost_approved*(1+config["alerts"]["simulated_cost_increase_warning"]) or row.fill_candidate<row.fill_approved-config["alerts"]["simulated_fill_drop_warning"]:
            incidents.append(incident("DELAYED_SIMULATED_DECISION_GUARDRAIL","warning",row.origin_date+pd.Timedelta(days=30),model_version=CANDIDATE,
                threshold=config["alerts"],evidence=dict(world_id=row.world_id,world_kind=row.world_kind,cost_difference=row.cost_candidate-row.cost_approved,fill_difference=row.fill_candidate-row.fill_approved),
                disposition="scenario warning after outcomes; never changes origin routing; no real deployment"))
    fallback_rate=1/len(bundles)
    if fallback_rate>config["alerts"]["fallback_rate_warning"]:
        incidents.append(incident("ELEVATED_INJECTED_FALLBACK_RATE","warning","2016-03-27 replay summary",threshold=config["alerts"]["fallback_rate_warning"],
                                   evidence=dict(fallback_batches=1,total_forecast_batches=len(bundles),rate=fallback_rate),disposition="synthetic fault drill, not measured live reliability"))
    frames=dict(features=pd.concat(features,ignore_index=True),forecast_ledger=pd.concat([b["forecasts"] for b in bundles.values()],ignore_index=True),
        decision_ledger=pd.concat([b["decisions"] for b in bundles.values() if not b["decisions"].empty],ignore_index=True),
        feature_parity=pd.DataFrame(parities),shadow=pd.DataFrame(shadow),future_checks=future,faults=faults,forecast_metrics=metrics,maturity_ledger=label_table,
        matched_forecast_metrics=matched,canary_decisions=pd.concat(canaries,ignore_index=True),canary_routing=pd.DataFrame(routing),
        simulation_runs=runs,simulation_solver=solvers,simulation_temporal_checks=temporal_checks,simulation_risk=risk,simulation_paired=paired)
    for name,frame in frames.items():frame.to_parquet(tables/(name+".parquet"),index=False)
    atomic_json(output/"incidents.json",incidents);atomic_json(output/"retry_proof.json",retry)
    from .verification import verify
    audit=verify(output,data,registry,contract_digest)
    verify_sources(data)
    full=label_table[label_table.monitor_window.eq("complete")]
    overall={model:performance(group) for model,group in full.groupby("model_version")}
    summary=dict(status="passed",scope=STATUS,run_id=run_id,config=config,feature_contract_digest=contract_digest,environment=environment,
        python=platform.python_version(),code_fingerprints=code,source_fingerprints=data.source_hashes,historical_artifact_hashes=data.model_hashes,
        warehouse_totals={k:str(v) if k=="known_revenue" else v for k,v in data.warehouse_totals.items()},inputs_models_unchanged=True,
        cohort=dict(series=25,continuity=13,sparse_stress=12,origins=[str(x.date()) for x in ORIGINS],real_paths=75,target_rows=2100),
        registry_entries=[dict(model_version=e["model_version"],artifact_digest=e["artifact_digest"],training_cutoff=e["training_cutoff"],kind=e["kind"]) for e in entries],
        registry_state=registry.state(),forecast_rows=len(frames["forecast_ledger"]),primary_forecast_rows=4200,origin_decisions=len(frames["decision_ledger"]),
        canary_actions=len(frames["canary_decisions"]),canary=route,forecast_metrics=overall,fallback_rate=fallback_rate,
        forecast_retries=len(retry),feature_parity_checks=len(parities),future_checks=len(future),simulation=sim_counts,simulation_temporal_checks=len(temporal_checks),
        fault_cases=len(faults),fault_dispositions=faults.status.value_counts().to_dict(),incident_counts=pd.Series([e["severity"] for e in incidents]).value_counts().to_dict(),
        drift_reports=len(drifts)+len(injections),observed_drift_warnings=int(drifts.severity.eq("warning").sum()),injected_shift_warnings=sum(r["severity"]=="warning" for r in injections),
        solver_statuses=solvers.status.value_counts().to_dict(),max_mip_gap=float(solvers.gap.max()) if solvers.gap.notna().any() else None,
        verification=audit,runtime_seconds=perf_counter()-started,peak_memory_mib=peak_memory(),
        table_rows={k:len(v) for k,v in frames.items()},ineligible=["No live service, actual inventory timestamps/costs/order dispatch or measured Walmart policy effects.",
            "No neural/checkpoint families outside this predeclared two-native-model comparison; no retraining.",
            "Candidate is the older Season17 PRE-calibration frozen tree ensemble, not conformal intervals or an improved successor.",
            "One real resource pool; canary exposures are dependent retrospective episodes, not randomized independent business evidence.",
            "Four conditional bootstrap worlds give descriptive simulated CVaR, not a real tail guarantee."])
    atomic_json(output/"summary.json",summary)
    from .report import write_report
    write_report(summary,frames,drifts.to_dict("records")+injections,output/"serving_report.md")
    atomic_json(base/"latest_verified.json",dict(run_id=run_id,path=output.relative_to(root).as_posix(),summary_sha256=file_hash(output/"summary.json"),verification_sha256=file_hash(output/"verification.json")))
    return summary,output
