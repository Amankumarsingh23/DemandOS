"""Independent persisted-output arithmetic and integrity gates before verified pointer."""
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

from .core import require,read_json,digest,atomic_json,file_hash
from .features import CASE,parity
from .prediction import validate_output
from .decisions import validate_decisions,target_vector
from .publication import read_batch
from .bootstrap import APPROVED,CANDIDATE,BASELINE


def verify(directory,data,registry,contract_digest):
    directory=Path(directory);tables=directory/"tables";checks={}
    def gate(name,condition=True):require(bool(condition),"INDEPENDENT_AUDIT_FAILED",name);checks[name]=True
    features=pd.read_parquet(tables/"features.parquet")
    parity(data.reference_features,features);gate("saved_training_reference_feature_parity")
    for (store,item,origin),g in features.groupby(["store_id","item_id","origin_date"]):
        history=data.snapshot[data.snapshot.store_id.eq(store)&data.snapshot.item_id.eq(item)&data.snapshot.date.le(origin)].sort_values("date")
        y=history.units_sold.to_numpy(float);row=g.iloc[0]
        for lag in data.feature_config.lags:np.testing.assert_allclose(g[f"origin_lag_{lag}"],y[-lag-1],atol=0)
        for w in data.feature_config.windows:
            np.testing.assert_allclose(g[f"rolling_mean_{w}_at_origin"],sum(y[-w:])/w,atol=1e-10)
            np.testing.assert_allclose(g[f"rolling_zero_fraction_{w}_at_origin"],sum(y[-w:]==0)/w,atol=1e-10)
        gate("independent_origin_lags_means_zero_rates")
    gate("no_unverified_historical_prices",features.known_price_weeks.eq(0).all() and features.last_known_price.isna().all())
    forecasts=[];decisions=[];batch_rows=[]
    for path in (directory/"publication/batches").iterdir():
        frames,manifest=read_batch(path);f,d=frames["forecasts"],frames["decisions"]
        entry=registry.verify(manifest["request"]["model_version"],contract_digest=contract_digest)
        selected=features[features.origin_date.eq(f.origin_date.iloc[0])]
        from .batch import LINEAGE
        validate_output(f.drop(columns=LINEAGE),selected,entry,contract_digest);validate_decisions(d)
        require(f.batch_id.eq(manifest["batch_id"]).all() and f.source_snapshot_digest.eq(manifest["request"]["source_snapshot_digest"]).all(),"FORECAST_PUBLICATION_LINEAGE")
        if manifest["request"]["channel"]=="shadow":gate("shadow_never_publishes_actions",d.empty)
        else:
            require(len(d)==25 and d.feature_contract_digest.eq(contract_digest).all(),"DECISION_LINEAGE")
            np.testing.assert_array_equal(d.target_units,target_vector(f,data.metadata))
            for r in d.itertuples():
                ids=f.loc[f.store_id.eq(r.store_id)&f.item_id.eq(r.item_id),"prediction_id"].tolist()
                require(json.loads(r.prediction_ids)==ids,"DECISION_PREDICTION_LINEAGE")
            decisions.append(d)
        forecasts.append(f);batch_rows.append(dict(batch_id=manifest["batch_id"],rows=len(f),actions=len(d),channel=manifest["request"]["channel"]))
    f=pd.concat(forecasts,ignore_index=True)
    gate("forecast_unique_complete_model_keys",len(f)==4900 and not f.duplicated(["model_version",*CASE]).any())
    gate("each_model_origin_horizon_complete",f.groupby(["model_version","origin_date","store_id","item_id"]).horizon.nunique().eq(28).all())
    for origin in sorted(f.origin_date.unique()):
        actual=f[f.model_version.eq(APPROVED)&f.origin_date.eq(origin)].sort_values(CASE)
        reference=data.reference_forecasts[data.reference_forecasts.origin_date.eq(origin)].sort_values(CASE)
        np.testing.assert_allclose(actual[["q10","q50","q90"]],reference[["q10","q50","q90"]],atol=1e-10,rtol=0)
    gate("saved_season16_forecast_goldens")
    gate("all_persisted_batches_and_resource_vectors_verified",len(batch_rows)==7 and sum(r["actions"] for r in batch_rows)==100)
    future=pd.read_parquet(tables/"future_checks.parquet");temporal=pd.read_parquet(tables/"simulation_temporal_checks.parquet")
    gate("future_source_features_predictions_all_pool_actions",len(future)==18 and future.passed.all())
    gate("reset_prefix_supplier_hidden_demand",len(temporal)==24 and temporal.passed.all())
    drifts=read_json(directory/"drift.json")+read_json(directory/"faults/drift_injections.json")
    for row in drifts:
        p=np.maximum(np.array(row["reference_probabilities"]),row["epsilon"]);q=np.maximum(np.array(row["current_probabilities"]),row["epsilon"])
        p/=p.sum();q/=q.sum();np.testing.assert_allclose(row["psi"],sum((q-p)*np.log(q/p)),atol=1e-12)
        a=np.array(row["reference_values"],float);b=np.array(row["current_values"],float);a=np.sort(a[np.isfinite(a)]);b=np.sort(b[np.isfinite(b)])
        if len(a) and len(b):
            points=np.unique(np.r_[a,b]);distance=max(abs(np.searchsorted(a,points,side="right")/len(a)-np.searchsorted(b,points,side="right")/len(b)))
            np.testing.assert_allclose(row["ks"],distance,atol=1e-12)
    gate("independent_all_psi_and_ecdf_ks")
    labels=pd.read_parquet(tables/"maturity_ledger.parquet")
    observed=labels.loc[labels.mature,["store_id","item_id","target_date","actual"]].merge(
        data.snapshot[["store_id","item_id","date","units_sold"]].rename(columns={"date":"target_date"}),
        on=["store_id","item_id","target_date"],validate="many_to_one")
    np.testing.assert_array_equal(observed.actual,observed.units_sold)
    gate("mature_labels_match_actual_warehouse_sales")
    gate("available_labels_only",labels.loc[labels.mature,"assumed_available_at"].le(labels.loc[labels.mature,"monitoring_date"]).all() and labels.loc[~labels.mature,"actual"].isna().all())
    gate("maturity_eight_horizon_golden",labels.loc[labels.monitor_window.eq("early"),"mature"].to_numpy().tolist()==labels.loc[labels.monitor_window.eq("early"),"horizon"].le(8).tolist())
    metrics=pd.read_parquet(tables/"forecast_metrics.parquet")
    for row in metrics.itertuples():
        part=labels[labels.model_version.eq(row.model_version)&labels.origin_date.eq(row.origin_date)&labels.monitor_window.eq(row.window)]
        if row.cohort!="ALL":part=part[part.cohort.eq(row.cohort)]
        part=part[part.mature];y=part.actual.to_numpy(float);error=np.abs(y-part.prediction.to_numpy(float))
        require(len(part)==row.mature,"METRIC_DENOMINATOR")
        np.testing.assert_allclose(row.mae,sum(error)/len(part),atol=1e-12)
        if y.sum():np.testing.assert_allclose(row.wape,sum(error)/sum(y),atol=1e-12)
        else:require(pd.isna(row.wape),"WAPE_ZERO_DENOMINATOR")
        e=y[:,None]-part[["q10","q50","q90"]].to_numpy(float);levels=np.array([.1,.5,.9])
        np.testing.assert_allclose(row.pinball,np.where(e>=0,levels*e,(levels-1)*e).mean(),atol=1e-12)
        np.testing.assert_allclose(row.coverage,((y>=part.q10)&(y<=part.q90)).mean(),atol=1e-12)
        np.testing.assert_allclose(row.width,(part.q90-part.q10).mean(),atol=1e-12)
        np.testing.assert_allclose(row.observed_units,y.sum(),atol=1e-12)
    gate("independent_mature_mae_wape_pinball_coverage")
    sim=directory/"simulation"
    with duckdb.connect() as con:
        for name in ["daily","events","allocations","resources"]:con.read_parquet(str(sim/("*-"+name+".parquet")),union_by_name=True).create_view(name)
        physical=con.execute("""select count(*) from daily where opening_stock+arrivals<>immediate_served+old_backlog_fulfilled+ending_stock
            or demand_proxy<>immediate_served+lost_units+new_backlog or opening_on_order-arrivals+ordered<>ending_on_order
            or opening_backlog+new_backlog<>ending_backlog+old_backlog_fulfilled
            or least(ending_stock,ending_backlog,ending_on_order)<0""").fetchone()[0]
        gate("independent_simulated_state_queue_conservation",physical==0)
        cost=con.execute("""select count(*) from daily where abs(operational_cost-(.2*ending_stock+40*lost_units+2*ending_backlog+60*ordered+5*(ordered>0)::int-terminal_credit))>1e-7""").fetchone()[0]
        gate("independent_simulated_cost_components",cost==0)
        complete=con.execute("""select count(*) from (select model_version,origin_date,world_kind,world_id,series_id from daily
            group by all having count(*)<>28 or count(distinct day_index)<>28)""").fetchone()[0]
        gate("independent_complete_simulation_paths",complete==0 and con.execute("select count(*) from daily").fetchone()[0]==21000)
        balances=con.execute("""select count(*) from resources where budget_after < -1e-7 or flow_used>flow_limit+1e-7 or abs(budget_before-committed-budget_after)>1e-7""").fetchone()[0]
        gate("independent_hard_cash_and_flow",balances==0)
        invalid=con.execute("""select count(*) from allocations where approved<0 or approved>candidate_quantity or approved<>round(approved)
            or approved%pack<>0 or (approved>0 and approved<moq)""").fetchone()[0]
        gate("independent_simulated_pack_moq_requests",invalid==0)
        keys="model_version,origin_date,world_kind,world_id,series_id"
        lifecycle=con.execute(f"""select count(*) from (select {keys},order_id from events group by all having
            count(*) filter(where event in ('placed','initial_outstanding'))<>1 or count(*) filter(where event in ('received','terminal_outstanding'))<>1
            or count(distinct quantity)<>1)""").fetchone()[0]
        receipts=con.execute(f"""select count(*) from daily d full join (select {keys},event_day as day_index,sum(quantity) as received_units
            from events where event='received' group by all) e using({keys},day_index) where coalesce(d.arrivals,0)<>coalesce(e.received_units,0)""").fetchone()[0]
        gate("independent_order_identity_receipts",lifecycle==0 and receipts==0)
        common=con.execute("""select count(*) from (select origin_date,world_kind,world_id,series_id,day_index from daily group by all
            having count(*)<>2 or count(distinct demand_proxy)<>1 or count(distinct supplier_shock_id)<>1
            or (min(day_index)=1 and (count(distinct opening_stock)<>1 or count(distinct opening_on_order)<>1)))""").fetchone()[0]
        gate("independent_matched_exogenous_worlds_and_resets",common==0)
        allocation=con.execute("select * from allocations").df();resources=con.execute("select * from resources").df()
        group=["model_version","origin_date","world_kind","world_id"]
        perday=dict(tuple(allocation.groupby(group+["day_index"])))
        for _,r in resources.groupby(group):
            last=None;cash=None
            for row in r.sort_values("day_index").itertuples():
                a=perday[tuple(getattr(row,k) for k in group+["day_index"])];new=json.loads(row.supplier_remaining)
                old=json.loads(row.supplier_initial) if last is None else last;use=a.groupby("supplier").approved.sum()
                require(all(v>=-1e-7 and abs(old[k]-use.get(k,0)-v)<1e-7 for k,v in new.items()),"INDEPENDENT_SUPPLIER_LEDGER")
                require(cash is None or abs(cash-row.budget_before)<1e-7,"INDEPENDENT_CASH_RESET")
                np.testing.assert_allclose(a.purchase@a.approved+a.fee@a.activation,row.committed,atol=1e-7)
                for store,limit in json.loads(row.receiving_limit).items():require(a.loc[a.store_id.eq(store),"approved"].sum()<=limit+1e-7,"INDEPENDENT_RECEIVING_CAP")
                last=new;cash=row.budget_after
        gate("independent_persistent_supplier_cash_procurement_store_limits")
        aggregated=con.execute("""select model_version,origin_date,world_kind,world_id,sum(operational_cost) as total_cost,sum(demand_proxy) as demand,
            sum(immediate_served) as served,sum(lost_units) as lost from daily group by all""").df()
    runs=pd.read_parquet(tables/"simulation_runs.parquet")
    merged=runs.merge(aggregated,on=["model_version","origin_date","world_kind","world_id"],validate="one_to_one")
    np.testing.assert_allclose(merged.cost,merged.total_cost,atol=1e-6)
    for column in ["demand","served","lost"]:np.testing.assert_array_equal(merged[column+"_x"],merged[column+"_y"])
    gate("independent_simulated_pool_costs",len(merged)==30)
    risk=pd.read_parquet(tables/"simulation_risk.parquet")
    for row in risk.itertuples():
        values=np.sort(runs.loc[runs.model_version.eq(row.model_version)&runs.origin_date.eq(row.origin_date)&runs.world_kind.eq("eligible_history_bootstrap"),"cost"].to_numpy(float))
        require(len(values)==4,"RISK_WORLD_COUNT");np.testing.assert_allclose([row.expected,row.var,row.cvar,row.worst],[values.mean(),values[2],values[-1],values[-1]],atol=1e-6)
    gate("independent_four_world_risk")
    canary=pd.read_parquet(tables/"canary_decisions.parquet")
    gate("whole_pool_canary_unique_actions",len(canary)==75 and canary.groupby("origin_date").model_version.nunique().eq(1).all() and not canary.action_id.duplicated().any())
    for path in (directory/"canary").glob("*.json"):
        m=read_json(path);require(m["publication_digest"]==digest({k:v for k,v in m.items() if k!="publication_digest"}),"CANARY_PUBLICATION_INTEGRITY")
    gate("canary_approval_and_publication_integrity")
    state=registry.state();gate("rollback_restored_prior_verified_approved",state["aliases"]["forecast"]==APPROVED)
    faults=pd.read_parquet(tables/"faults.parquet");gate("all_predeclared_faults_block_warn_or_recover",faults.passed.all() and len(faults)>=25)
    retry=read_json(directory/"retry_proof.json");gate("idempotent_no_duplicate_completed_batches",all(x["passed"] for x in retry))
    result=dict(status="passed",independent_gates=len(checks),checks=checks,batches=len(batch_rows),forecast_rows=len(f),
                published_origin_actions=100,simulated_canary_actions=75,simulation_daily_rows=21000,forecast_metrics_verified=len(metrics),drift_reports_verified=len(drifts))
    atomic_json(directory/"verification.json",result)
    return result
