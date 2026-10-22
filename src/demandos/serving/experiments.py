"""Bounded replay invariance, descriptive monitors and matched simulated controllers."""
import numpy as np
import pandas as pd

from demandos.inventory.scenarios import bootstrap,seed_for
from demandos.causal.simulation import prefix_equal,intervention
from demandos.risk.scenarios import Closure
from demandos.risk.metrics import tail
from .core import require,digest
from .features import extract,parity,CASE
from .prediction import predict
from .decisions import frozen_inputs,plan,target_vector,downstream
from .bootstrap import APPROVED,CANDIDATE
from .monitoring import drift,mature_labels,performance,matched_performance,incident


def temporal(data,registry,bundles,contract_digest,config):
    checks=[]
    for origin in sorted({k[1] for k in bundles if k[0]==APPROVED}):
        for mode in ["mutate_future_sales","remove_future_sales","mutate_future_prices"]:
            changed=data.snapshot.copy();future=changed.date.gt(origin)
            if mode=="mutate_future_sales":changed.loc[future,"units_sold"]+=1000000
            elif mode=="remove_future_sales":changed=changed.loc[~future]
            else:changed.loc[future,"sell_price"]=999999.
            f,source=extract(changed,data.calendar,data.metadata,origin,data.feature_config,data.mapping)
            basis,scenario,resource,_=frozen_inputs(changed,data.metadata,origin,data.settings)
            for model in [APPROVED,CANDIDATE]:
                base=bundles[(model,origin)];parity(base["features"],f,atol=config["serving"]["determinism_atol"])
                require(source==base["source_digest"],"FUTURE_SNAPSHOT_LEAKAGE")
                other,_=predict(registry,model,f,changed,contract_digest)
                columns=["raw_q10","raw_q50","raw_q90","q10","q50","q90","prediction"]
                np.testing.assert_allclose(other[columns],base["forecasts"][columns],atol=1e-10,rtol=0)
                q,_=plan(other,data.metadata,data.settings,scenario,resource)
                np.testing.assert_array_equal(q.approved_quantity,base["comparison_decisions"].approved_quantity)
                checks.append(dict(origin_date=origin,model_version=model,check=mode,features_equal=True,snapshot_equal=True,
                                   predictions_equal=True,all_pool_actions_equal=True,passed=True))
    return pd.DataFrame(checks)


def monitors(data,bundles,config):
    records=[];metrics=[];labels=[];events=[];paired=[]
    model_names=sorted({k[0] for k in bundles});origins=sorted({k[1] for k in bundles})
    reference=bundles[(APPROVED,origins[0])]
    ref=reference["features"].drop_duplicates(["store_id","item_id"])
    for origin in origins:
        batch=bundles[(APPROVED,origin)];current=batch["features"].drop_duplicates(["store_id","item_id"])
        signals={"rolling_mean_7_at_origin":(ref.rolling_mean_7_at_origin.to_numpy(float),current.rolling_mean_7_at_origin.to_numpy(float)),
                 "rolling_zero_fraction_28_at_origin":(ref.rolling_zero_fraction_28_at_origin.to_numpy(float),current.rolling_zero_fraction_28_at_origin.to_numpy(float)),
                 "sales_at_origin":(ref.sales_at_origin.to_numpy(float),current.sales_at_origin.to_numpy(float)),
                 "prediction_q50":(reference["forecasts"].q50.to_numpy(float),batch["forecasts"].q50.to_numpy(float)),
                 "forecast_width":((reference["forecasts"].q90-reference["forecasts"].q10).to_numpy(float),(batch["forecasts"].q90-batch["forecasts"].q10).to_numpy(float))}
        for name,(a,b) in signals.items():
            record=drift(a,b,name,config["alerts"],origin=origin);record["injection"]="none";records.append(record)
            if record["severity"]=="warning":events.append(incident("HISTORICAL_DESCRIPTIVE_DRIFT","warning",origin,batch_id=batch["batch_id"],model_version=APPROVED,
                                                                      threshold=config["alerts"],evidence=record,disposition="descriptive warning; no automatic failure claim"))
    observations=data.snapshot[["store_id","item_id","date","units_sold"]]
    for (model,origin),batch in bundles.items():
        for when,offset in [("early",10),("complete",30)]:
            ledger=mature_labels(batch["forecasts"],observations,origin+pd.Timedelta(days=offset),lag_days=config["serving"]["label_lag_days"])
            ledger=ledger.merge(data.metadata[["store_id","item_id","cohort"]],on=["store_id","item_id"],validate="many_to_one")
            ledger["monitor_window"]=when;labels.append(ledger)
            for cohort in ["ALL","continuity","sparse_stress"]:
                part=ledger if cohort=="ALL" else ledger[ledger.cohort.eq(cohort)]
                metrics.append(dict(model_version=model,origin_date=origin,window=when,cohort=cohort,**performance(part)))
            require(int(ledger.mature.sum())==(200 if when=="early" else 700),"MATURITY_HORIZON_GOLDEN")
    label_table=pd.concat(labels,ignore_index=True)
    for origin in origins:
        for when in ["early","complete"]:
            a=label_table[label_table.model_version.eq(APPROVED)&label_table.origin_date.eq(origin)&label_table.monitor_window.eq(when)]
            b=label_table[label_table.model_version.eq(CANDIDATE)&label_table.origin_date.eq(origin)&label_table.monitor_window.eq(when)]
            first,second=matched_performance(a,b)
            paired.append(dict(origin_date=origin,window=when,mature=first["mature"],approved_mae=first["mae"],candidate_mae=second["mae"],
                               approved_coverage=first["coverage"],candidate_coverage=second["coverage"],candidate_minus_approved_mae=second["mae"]-first["mae"]))
            if second["mae"]>first["mae"]*(1+config["alerts"]["mae_relative_warning"]) or second["coverage"]<first["coverage"]-config["alerts"]["coverage_drop_warning"]:
                events.append(incident("MATURE_MATCHED_FORECAST_DETERIORATION","warning",origin+pd.Timedelta(days=10 if when=="early" else 30),model_version=CANDIDATE,
                                       threshold=config["alerts"],evidence=paired[-1],disposition="delayed retrospective warning; never fed into origin forecasts"))
    return pd.DataFrame(records),pd.DataFrame(metrics),label_table,pd.DataFrame(paired),events


def simulate_controllers(data,bundles,config,directory):
    from pathlib import Path
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    runs=[];solver=[];checks=[];counts=dict(pool_runs=0,daily_rows=0,replays=0)
    cohorts=data.metadata
    for origin in sorted({k[1] for k in bundles}):
        basis,scenario,resource,_=frozen_inputs(data.snapshot,cohorts,origin,data.settings)
        actual=np.stack([data.snapshot.loc[data.snapshot.store_id.eq(m.store_id)&data.snapshot.item_id.eq(m.item_id)&data.snapshot.date.gt(origin)&data.snapshot.date.le(origin+pd.Timedelta(days=28))].sort_values("date").units_sold.to_numpy(int) for m in cohorts.itertuples()])
        worlds=[]
        for m in cohorts.itertuples():
            h=data.snapshot[data.snapshot.store_id.eq(m.store_id)&data.snapshot.item_id.eq(m.item_id)]
            worlds.append(bootstrap(h,origin,samples=config["serving"]["simulation_worlds"],lookback=56,block=7,horizon=28,
                                    seed=seed_for(42,"serving-evaluation",origin,m.series_id))[0])
        worlds=np.stack(worlds)
        models=[model for model in [APPROVED,CANDIDATE] if (model,origin) in bundles]
        for model in models:
            batch=bundles[(model,origin)]
            for kind,world,y in [("observed_sales_proxy",-1,actual)]+[("eligible_history_bootstrap",w,worlds[:,w,:]) for w in range(config["serving"]["simulation_worlds"])]:
                result,metric=downstream(y,origin,cohorts,batch["forecasts"],scenario,data.settings,basis)
                meta=dict(model_version=model,origin_date=origin,world_kind=kind,world_id=world,scenario_status="SIMULATED_STATE_AND_COSTS; M5_RECORDED_SALES_PROXY")
                stem=f"{origin:%Y%m%d}-{model}-{kind}-{world}"
                for name in ["daily","events","allocations","resources","solver"]:result[name].assign(**meta).to_parquet(directory/(stem+"-"+name+".parquet"),index=False)
                solver.append(result["solver"].assign(**meta))
                runs.append({**meta,**metric});counts["pool_runs"]+=1;counts["daily_rows"]+=len(result["daily"])
                if kind=="observed_sales_proxy":
                    reset,_=downstream(y,origin,cohorts,batch["forecasts"],scenario,data.settings,basis);pd.testing.assert_frame_equal(result["daily"],reset["daily"])
                    checks.append({**meta,"check":"full_reset","passed":True});counts["replays"]+=1
                    changed=y.copy();changed[0,9]+=100000
                    other,_=downstream(changed,origin,cohorts,batch["forecasts"],scenario,data.settings,basis);prefix_equal(result,other,10)
                    checks.append({**meta,"check":"future_demand_day10","passed":True});counts["replays"]+=1
                    hidden=changed.copy();hidden[0,9]+=100000
                    hidden_result,_=downstream(hidden,origin,cohorts,batch["forecasts"],scenario,data.settings,basis)
                    pd.testing.assert_frame_equal(other["allocations"],hidden_result["allocations"])
                    checks.append({**meta,"check":"hidden_unmet_not_observable","passed":True});counts["replays"]+=1
                    supply=intervention(y,origin,cohorts,target_vector(batch["forecasts"],cohorts),scenario,{**data.settings,"resource_targets":basis},action=0,closure=Closure(10,16))
                    prefix_equal(result,supply,10);checks.append({**meta,"check":"future_supply_day10","passed":True});counts["replays"]+=1
        print(f"Complete simulated serving controllers/reset/prefix checks {origin.date()}",flush=True)
    run_table=pd.DataFrame(runs);risk=[]
    for (model,origin),g in run_table[run_table.world_kind.eq("eligible_history_bootstrap")].groupby(["model_version","origin_date"]):
        risk.append(dict(model_version=model,origin_date=origin,worlds=len(g),**tail(g.cost,np.full(len(g),1/len(g)),.75),
                         interpretation="four conditional simulated pool worlds; CVaR(.75) is worst of four, no real risk guarantee"))
    return run_table,pd.concat(solver,ignore_index=True),pd.DataFrame(checks),pd.DataFrame(risk),counts
