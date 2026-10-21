"""Source-bound one-step OPE and paired complete-policy experiment runner."""
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter
import tomllib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from demandos.risk.pipeline import dump
from demandos.backtesting.pipeline import publish
from demandos.reconciliation.pipeline import peak_memory
from .inputs import load,verify_sources
from .bandit import contexts,potential_outcomes,make_logs,estimate,audit_oracle
from .trajectories import run as run_trajectories,compare
from .demonstrations import controlled,sequential
from .estimators import evaluate
from .simulation import STATUS


def load_config(path):
    config=tomllib.loads(Path(path).read_text(encoding="utf-8"))
    expected=dict(seed=42,worlds=8,logging_replications=32,one_step_days=14,initial_stock=[0,10,30],budget_factors=[.1,.7],flow_factors=[.5,.8],
        actions=["mean","cvar_2"],targets=["mean","cvar","mixture","risk_rule"],outcome_models=["ridge","forest","misspecified"],
        propensity_sources=["known","estimated"],bootstrap_samples=200,controlled_replications=200,controlled_n=2000,
        sequential_trajectories=2048,solver_seconds=3.,sales_as_demand=True)
    if set(config)!={"causal"} or config["causal"]!=expected:raise ValueError("Predeclared bounded causal experiment required; no outcome-driven tuning")
    return config["causal"]


def support_demonstrations():
    result=evaluate([0],[3.],[[1.,0.]],[[0.,1.]],[[2.,100.]],clip=2.)
    if result["status"]!="NONIDENTIFIABLE_MISSING_SUPPORT" or any(result[n] is not None for n in ["ips","snips","dm","dr"]):
        raise AssertionError("Missing support incorrectly identified")
    zero=evaluate([0,0],[3.,4.],[[.5,.5]]*2,[[0.,1.]]*2,[[2.,5.]]*2)
    return dict(missing_support=result,finite_sample_zero_matches={k:v for k,v in zero.items() if not isinstance(v,np.ndarray)})


def run(root,config):
    started=perf_counter();data=load(Path(root));root=data.root
    output=root/"data/causal/m5_h28"
    with TemporaryDirectory(prefix=".causal-",dir=root/"data") as scratch:
        tmp=Path(scratch)
        context,actions,future_checks,catalog_solves=contexts(data,config)
        oracle,one_counts=potential_outcomes(data,config,context,actions,tmp/"oracle")
        logs=make_logs(context,oracle,config["logging_replications"])
        estimates,nuisance_provenance=estimate(logs,config)
        # The estimator has already returned: only now does audit use unchosen Y.
        audit_tables=audit_oracle(logs,oracle,estimates["ope_estimates"],estimates["nuisance_predictions"])
        trajectories,solver,temporal,trajectory_counts=run_trajectories(data,config,tmp/"tables")
        contrasts=compare(trajectories,config["bootstrap_samples"])
        repetitions,controlled_summary,confounding=controlled(config["controlled_replications"],config["controlled_n"])
        sequence,sequence_weights=sequential(config["sequential_trajectories"])
        tables=dict(contexts=context,action_catalog=actions,synthetic_logs=logs,context_future_checks=future_checks,
            source_future_checks=data.prefix_checks,trajectory_runs=trajectories,trajectory_solver=solver,trajectory_temporal_checks=temporal,
            controlled_repetitions=repetitions,controlled_summary=controlled_summary,confounding=confounding,
            sequential_diagnostics=sequence,sequential_weights=sequence_weights,cohort=data.metadata,**estimates,**audit_tables,**contrasts)
        for name,frame in tables.items():frame.to_parquet(tmp/"tables"/(name+".parquet"),index=False)
        dump(tmp/"config.json",config);dump(tmp/"catalog_solver.json",catalog_solves)
        dump(tmp/"nuisance_provenance.json",nuisance_provenance);dump(tmp/"support_demonstrations.json",support_demonstrations())
        context_record=dict(source_fingerprints=data.source_hashes,provenance=data.provenance,warehouse_totals=data.warehouse_totals,
                            one_step_counts=one_counts,trajectory_counts=trajectory_counts,config=config,runtime_before_audit=perf_counter()-started)
        dump(tmp/"execution_context.json",context_record)
        checkpoint=(root/"data/causal/unverified_computation").resolve()
        if not checkpoint.is_relative_to((root/"data/causal").resolve()):raise ValueError("Unsafe diagnostic copy")
        shutil.copytree(tmp,checkpoint,dirs_exist_ok=True)
        from .verification import verify_outputs
        audit=verify_outputs(tmp)
        verify_sources(data)
        summary=dict(status="passed",scenario_status=STATUS,config=config,source_fingerprints=data.source_hashes,
            provenance=data.provenance,warehouse_totals=data.warehouse_totals,inputs_unchanged=True,
            cohort_counts=data.metadata.cohort.value_counts().to_dict(),original_observed_paths=75,observed_target_rows=2100,
            contexts=len(context),one_step_context_worlds=len(oracle)//2,oracle_pool_interventions=len(oracle),logging_rows=len(logs),
            logging_replications=config["logging_replications"],development_rows_per_replication=96,evaluation_rows_per_replication=192,
            one_step_daily_rows=one_counts["daily"],trajectory_counts=trajectory_counts,temporal_checks=len(temporal),
            future_source_checks=len(data.prefix_checks),future_context_checks=len(future_checks),
            catalog_solver_statuses=pd.Series([s["status"] for s in catalog_solves]).value_counts().to_dict(),
            trajectory_solver_statuses=solver.status.value_counts().to_dict(),
            max_catalog_mip_gap=max(float(s.get("gap") or 0) for s in catalog_solves),
            max_trajectory_mip_gap=float(solver.gap.max()),
            max_mip_gap=max(float(solver.gap.max()),max(float(s.get("gap") or 0) for s in catalog_solves)),
            runtime_seconds=perf_counter()-started,peak_memory_mib=peak_memory(),
            versions={p:version(p) for p in ["numpy","pandas","scipy","scikit-learn","duckdb","pyarrow"]},
            verification=audit,table_rows={p.relative_to(tmp).as_posix():pq.ParquetFile(p).metadata.num_rows for p in tmp.rglob("*.parquet")},
            assumptions=dict(track_a="one origin-issued feasible vector, 14-day reward, no later orders; initial procurement charged once; pretrained nuisance fits Jan31 selected rewards only",
                track_b="complete 28-day mean vs mean+2CVaR policies; identical reset/world; reward is negative full simulated operating cost",
                uncertainty="eight synthetic world-panel clusters, fixed fitted nuisances; one real historical pool and three dependent retrospective origins; no reliable real-world confidence coverage",
                identification="known synthetic assignment; no actual Walmart propensities, inventory logs, uncensored demand or causal savings"))
        dump(tmp/"summary.json",summary)
        from .report import write_report
        write_report(summary,tables,tmp/"causal_report.md")
        publish(tmp,output)
    return summary,output
