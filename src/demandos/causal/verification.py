"""Independent persisted-artifact checks; oracle is used here, after fitting."""
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from .logging import estimator_view,assign


def verify_outputs(directory):
    root=Path(directory);checks={}
    with duckdb.connect() as con:
        con.execute("set threads=2");con.execute("set memory_limit='1GB'")
        for p in (root/"tables").glob("*.parquet"):con.read_parquet(str(p)).create_view(p.stem)
        con.read_parquet(str(root/"oracle/potential_outcomes.parquet")).create_view("oracle_values")
        for name in ["daily","events","runs"]:con.read_parquet(str(root/"oracle/tables"/(name+".parquet"))).create_view("one_"+name)
        def gate(name,sql):
            n=con.execute(sql).fetchone()[0]
            if n:raise AssertionError(f"Causal gate {name}: {n} failures")
            checks[name]=True
        one="context_id,world_id,action,series_id"
        trajectory="origin_date,world_kind,world_id,scenario_family,policy_id,series_id"
        pool="origin_date,world_kind,world_id,scenario_family,policy_id"
        gate("one_step_unique_complete",f"select count(*) from (select {one} from one_daily group by all having count(*)<>14 or count(distinct day_index)<>14 or min(day_index)<>1 or max(day_index)<>14)")
        gate("trajectory_unique_complete",f"select count(*) from (select {trajectory} from trajectory_daily group by all having count(*)<>28 or count(distinct day_index)<>28 or min(day_index)<>1 or max(day_index)<>28)")
        for name in ["one_daily","trajectory_daily"]:
            gate(name+"_physical",f"""select count(*) from {name} where opening_stock+arrivals<>immediate_served+old_backlog_fulfilled+ending_stock
                or demand_proxy<>immediate_served+lost_units+new_backlog or opening_backlog+new_backlog<>ending_backlog+old_backlog_fulfilled
                or opening_on_order-arrivals+ordered<>ending_on_order or least(ending_stock,ending_backlog,ending_on_order,ordered)<0""")
            gate(name+"_cost",f"""select count(*) from {name} where abs(holding_cost-.2*ending_stock)>1e-7 or abs(lost_cost-40*lost_units)>1e-7
                or abs(procurement_cost-60*ordered)>1e-7 or abs(ordering_cost-5*(ordered>0)::int)>1e-7 or abs(backlog_cost-2*ending_backlog)>1e-7
                or abs(operational_cost-holding_cost-lost_cost-backlog_cost-ordering_cost-procurement_cost+terminal_credit)>1e-7""")
        gate("no_later_orders_track_a","select count(*) from one_daily where ordered<>0")
        gate("track_a_initial_cash_and_reward","""select count(*) from one_runs where abs(decision_cash-60*initial_action_quantity-5*(initial_action_quantity>0)::int)>1e-7
            or abs(total_cost-operational_cost-decision_cash)>1e-7 or abs(reward+total_cost)>1e-7""")
        gate("track_a_aggregate_rewards","""select count(*) from (select context_id,world_id,action,sum(reward) as calculated_reward,sum(total_cost) as calculated_cost from one_runs group by all) r
            join oracle_values o using(context_id,world_id,action) where abs(r.calculated_reward-o.reward)>1e-6 or abs(r.calculated_cost-o.total_cost)>1e-6""")
        gate("chosen_reward_consistency","""select count(*) from synthetic_logs l join oracle_values o
            on l.context_id=o.context_id and l.world_id=o.world_id and l.logged_action=o.action where abs(l.reward-o.reward)>1e-7""")
        gate("oracle_pair_keys","select count(*) from (select context_id,world_id from oracle_values group by all having count(*)<>2 or count(distinct action)<>2)")
        gate("one_step_common_world", """select count(*) from (select context_id,world_id,series_id,day_index from one_daily group by all
            having count(*)<>2 or count(distinct demand_proxy)<>1 or (min(day_index)=1 and count(distinct opening_stock)<>1))""")
        gate("trajectory_common_world","""select count(*) from (select origin_date,world_kind,world_id,scenario_family,series_id,day_index from trajectory_daily group by all
            having count(*)<>2 or count(distinct demand_proxy)<>1 or count(distinct supplier_shock_id)<>1)""")
        gate("track_b_approved_equals_orders",f"""select count(*) from trajectory_daily d join trajectory_allocations a using({trajectory},day_index)
            where d.ordered<>a.approved or d.position_before_order<>a.position or a.approved>a.candidate_quantity or a.approved<0 or a.approved<>round(a.approved)""")
        gate("persistent_budget","""select count(*) from (select *,lag(budget_after) over(partition by origin_date,world_kind,world_id,scenario_family,policy_id order by day_index) as previous_budget
            from trajectory_resources) where abs(budget_before-committed-budget_after)>1e-7 or budget_after < -1e-7 or flow_used>flow_limit+1e-7
            or (day_index>1 and abs(previous_budget-budget_before)>1e-7) or (day_index=1 and budget_before<>budget_initial)""")
        gate("purchases_equal_persistent_commitments",f"""select count(*) from (select {pool},day_index,sum(procurement_cost+ordering_cost) as cash from trajectory_daily group by all) d
            join trajectory_resources r using({pool},day_index) where abs(d.cash-r.committed)>1e-7""")
        for prefix,key in [("one",one),("trajectory",trajectory)]:
            gate(prefix+"_order_lifecycle",f"""select count(*) from (select {key},order_id from {prefix}_events group by all having
                count(*) filter(where event in ('placed','initial_outstanding'))<>1 or count(*) filter(where event in ('received','terminal_outstanding'))<>1
                or count(distinct quantity)<>1 or min(lead_time)<1 or max(abs(arrival_day-issue_day-lead_time))<>0
                or max(case when event='received' then abs(event_day-arrival_day) else 0 end)<>0)""")
            gate(prefix+"_receipt_matches_stock",f"""select count(*) from {prefix}_daily d full join
                (select {key},event_day as day_index,sum(quantity) as received_units from {prefix}_events where event='received' group by all) e
                using({key},day_index) where coalesce(d.arrivals,0)<>coalesce(e.received_units,0)""")
        gate("forward_retiming","select count(*) from trajectory_events where event='retimed' and (arrival_day<=previous_arrival_day or event_day<>previous_arrival_day)")
        gate("finite_delayed_logging_contract","select count(*) from synthetic_logs where mu_0<=0 or mu_1<=0 or abs(mu_0+mu_1-1)>1e-12 or outcome_available_date<=origin_date")
        gate("future_context_isolation","select count(*) from context_future_checks where not passed or not all_25_actions_equal or not propensities_equal")
        gate("source_future_isolation","select count(*) from source_future_checks where not passed")
        gate("trajectory_temporal_isolation","select count(*) from trajectory_temporal_checks where not passed")
        gate("solver_statuses","select count(*) from trajectory_solver where status not in ('OPTIMAL','FEASIBLE_NOT_PROVEN_OPTIMAL') or (status='OPTIMAL' and gap>1e-7)")
        gate("trajectory_reward_orientation","select count(*) from trajectory_runs where abs(reward+operating_cost)>1e-7")
        gate("trajectory_run_totals",f"""select count(*) from (
            select {pool},cohort,sum(operational_cost) as total_cost,sum(demand_proxy) as demand,sum(immediate_served) as served,sum(lost_units) as lost from trajectory_daily group by all
            union all select {pool},'ALL' as cohort,sum(operational_cost) as total_cost,sum(demand_proxy) as demand,sum(immediate_served) as served,sum(lost_units) as lost from trajectory_daily group by all) d
            join trajectory_runs r using({pool},cohort) where abs(d.total_cost-r.operating_cost)>1e-6 or d.demand<>r.demand_units or d.served<>r.served_units or d.lost<>r.lost_units
            or (d.demand>0 and abs(d.served*1./d.demand-r.fill)>1e-10) or (d.demand=0 and r.fill is not null)""")
    logs=estimator_view(pd.read_parquet(root/"tables/synthetic_logs.parquet"))
    expected_mu=.2+.6/(1+np.exp(-logs.risk_score.to_numpy()))
    np.testing.assert_allclose(expected_mu,logs.mu_1,atol=1e-12)
    for r in logs.itertuples():
        if assign(r.context_id,r.world_id,r.replication_id,[r.mu_0,r.mu_1])!=r.logged_action:raise AssertionError("Assignment not reproducible")
    checks["predecision_known_randomization_reproduced"]=True
    provenance=json.loads((root/"nuisance_provenance.json").read_text())
    for row in provenance:
        if pd.Timestamp(row["outcome_maturity_max"])>pd.Timestamp(row["evaluation_origin_min"]):raise AssertionError("Immature nuisance labels")
        if pd.Timestamp(row["fit_origin_max"])>=pd.Timestamp(row["evaluation_origin_min"]):raise AssertionError("Chronological nuisance leakage")
    checks["nuisance_chronological_maturity"]=True
    verify_catalogue(root);verify_resource_ledgers(root)
    checks["independent_whole_vector_feasibility"]=True
    checks["independent_persistent_supplier_store_cash_flow"]=True
    count=verify_ope(root,logs);checks["independent_all_ope_formulas_and_alignment"]=True
    verify_oracle_values(root,logs);checks["independent_oracle_target_values_and_estimator_errors"]=True
    verify_trajectory_contrasts(root);checks["independent_paired_cost_service_delay_fairness"]=True
    verify_sequential(root);checks["independent_log_domain_trajectory_arithmetic"]=True
    risk=pd.read_parquet(root/"tables/trajectory_operating_risk.parquet")
    runs=pd.read_parquet(root/"tables/trajectory_runs.parquet")
    for row in risk.itertuples():
        g=runs[runs.origin_date.eq(row.origin_date)&runs.scenario_family.eq(row.scenario_family)&runs.policy_id.eq(row.policy_id)&runs.cohort.eq(row.cohort)&runs.world_kind.eq("prefix_bootstrap")]
        x=np.sort(g.operating_cost.to_numpy(float));assert len(x)==8
        np.testing.assert_allclose([row.expected,row.var,row.cvar,row.worst],[x.mean(),x[5],x[-2:].mean(),x[-1]],atol=1e-6)
    checks["independent_equal_mass_operating_tail"]=True
    controlled=pd.read_parquet(root/"tables/controlled_summary.parquet")
    good=controlled[controlled.estimator.eq("dr")&(controlled.correct_propensity|controlled.correct_outcome)]
    bad=controlled[controlled.estimator.eq("dr")&~controlled.correct_propensity&~controlled.correct_outcome]
    if (good.bias.abs()>.25).any() or bad.bias.iloc[0]<3:raise AssertionError("Controlled structural DR demonstration failed")
    checks["controlled_double_robustness_and_confounding"]=True
    result=dict(status="passed",checks=checks,independent_gates=len(checks),ope_estimates_verified=count,
                logger_rows=len(logs),real_independent_resource_pools=1,conditional_synthetic_world_panels=8)
    (root/"verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def verify_catalogue(root):
    frame=pd.read_parquet(root/"tables/action_catalog.parquet")
    for _,g in frame.groupby(["context_id","action"]):
        if len(g)!=25 or g.series_id.nunique()!=25 or (g.quantity<0).any() or (g.quantity>g.request).any() or (g.quantity!=np.floor(g.quantity)).any():raise AssertionError("Invalid whole feasible catalogue")
        cash=float(g.purchase@g.quantity+g.fee@(g.quantity>0));flow=float(g.volume@g.quantity)
        if cash>g.budget_limit.iloc[0]+1e-7 or flow>g.flow_limit.iloc[0]+1e-7:raise AssertionError("Catalogue hard resource violation")
        for col,limits in [("supplier",g.supplier_limits.iloc[0]),("store_id",g.store_limits.iloc[0])]:
            actual=g.groupby(col).quantity.sum()
            for name,limit in json.loads(limits).items():
                if actual.get(name,0)>limit+1e-7:raise AssertionError("Catalogue shared cap violation")
    oracle=pd.read_parquet(root/"oracle/potential_outcomes.parquet")
    for cid,g in frame.groupby("context_id"):
        q=g.pivot(index="series_id",columns="action",values="quantity")
        if q[0].equals(q[1]):
            y=oracle[oracle.context_id.eq(cid)].pivot(index="world_id",columns="action",values="reward")
            np.testing.assert_allclose(y[0],y[1],atol=1e-7)


def verify_resource_ledgers(root):
    allocations=pd.read_parquet(root/"tables/trajectory_allocations.parquet")
    resources=pd.read_parquet(root/"tables/trajectory_resources.parquet")
    keys=["origin_date","world_kind","world_id","scenario_family","policy_id","day_index"]
    groups=dict(tuple(allocations.groupby(keys)))
    for _,g in resources.groupby(keys[:-1]):
        previous=None
        for row in g.sort_values("day_index").itertuples():
            key=tuple(getattr(row,k) for k in keys);a=groups[key]
            np.testing.assert_allclose(a.purchase@a.approved+a.fee@a.activation,row.committed,atol=1e-7)
            np.testing.assert_allclose(a.volume@a.approved,row.flow_used,atol=1e-7)
            initial=json.loads(row.supplier_initial);remaining=json.loads(row.supplier_remaining)
            last=initial if previous is None else previous;use=a.groupby("supplier").approved.sum()
            for name in initial:
                if abs(last[name]-use.get(name,0)-remaining[name])>1e-7 or remaining[name]<-1e-7:raise AssertionError("Persistent supplier accounting")
            previous=remaining
            use=a.groupby("store_id").approved.sum()
            for name,limit in json.loads(row.receiving_limit).items():
                if use.get(name,0)>limit+1e-7:raise AssertionError("Store order quota")


def verify_ope(root,logs):
    predictions=pd.read_parquet(root/"tables/nuisance_predictions.parquet")
    estimates=pd.read_parquet(root/"tables/ope_estimates.parquet")
    cache={};verified=0
    diagnostics=pd.read_parquet(root/"tables/overlap_diagnostics.parquet")
    diagnostic_keys=["replication_id","model","propensity_source","target","population"]
    diagnostic_map=diagnostics.set_index(diagnostic_keys)
    for key,g in predictions.groupby(["replication_id","model"]):
        rep,model=key
        logged=logs[logs.replication_id.eq(rep)&logs.origin_date.gt(pd.Timestamp("2016-01-31"))]
        cache[key]=logged.merge(g,on=["replication_id","context_id","world_id","origin_date"],validate="one_to_one")
    for key,g in estimates.groupby(["replication_id","model","propensity_source","target","population","clipping"]):
        rep,model,source,target,population,clipping=key
        x=cache[(rep,model)]
        if population!="pooled":x=x[x.origin_date.eq(pd.Timestamp(population))]
        a=x.logged_action.to_numpy(int);r=x.reward.to_numpy(float)
        mu=x[["mu_0","mu_1"] if source=="known" else ["estimated_mu_0","estimated_mu_1"]].to_numpy(float)
        pred=x[["prediction_0","prediction_1"]].to_numpy(float)
        p=np.zeros(len(x)) if target=="mean" else np.ones(len(x)) if target=="cvar" else np.full(len(x),.5) if target=="mixture" else np.where(x.risk_score>=0,.8,.2)
        pi=np.c_[1-p,p];w=np.array([pi[i,v]/mu[i,v] for i,v in enumerate(a)])
        if clipping!="none":w=np.minimum(w,2.)
        direct=np.array([sum(pi[i,j]*pred[i,j] for j in range(2)) for i in range(len(x))])
        scores=dict(ips=float(np.sum(w*r)/len(x)),snips=float(np.sum(w*r)/w.sum()) if w.sum() else None,
                    dm=float(np.mean(direct)),dr=float(np.mean([direct[i]+w[i]*(r[i]-pred[i,a[i]]) for i in range(len(x))])))
        for row in g.itertuples():
            np.testing.assert_allclose(row.estimate,scores[row.estimator],atol=1e-6,rtol=1e-10);verified+=1
        if clipping=="none":
            diag=diagnostic_map.loc[(rep,model,source,target,population)]
            np.testing.assert_allclose(diag.ess,(w.sum()**2)/(w@w) if w@w else 0.,atol=1e-8)
            np.testing.assert_allclose([diag.max_weight,diag.weight_mean,diag.weight_variance],[w.max(),w.mean(),w.var()],atol=1e-10)
            if diag.matching_actions!=(w>0).sum() or diag.positivity_violations or diag.target_support_coverage!=1:
                raise AssertionError("Incorrect overlap/ESS diagnostics")
    return verified


def verify_oracle_values(root,logs):
    oracle=pd.read_parquet(root/"oracle/potential_outcomes.parquet")
    values=pd.read_parquet(root/"tables/oracle_policy_values.parquet")
    base=logs[logs.replication_id.eq(0)&logs.origin_date.gt(pd.Timestamp("2016-01-31"))]
    joined=base.merge(oracle.pivot(index=["context_id","world_id"],columns="action",values="reward").rename(columns={0:"y0",1:"y1"}).reset_index(),
                      on=["context_id","world_id"],validate="one_to_one")
    for row in values.itertuples():
        x=joined if row.population=="pooled" else joined[joined.origin_date.eq(pd.Timestamp(row.population))]
        p=np.zeros(len(x)) if row.target=="mean" else np.ones(len(x)) if row.target=="cvar" else np.full(len(x),.5) if row.target=="mixture" else np.where(x.risk_score>=0,.8,.2)
        np.testing.assert_allclose(row.oracle_value,((1-p)*x.y0+p*x.y1).mean(),atol=1e-7)
    comparisons=pd.read_parquet(root/"tables/oracle_comparisons.parquet")
    np.testing.assert_allclose(comparisons.error,comparisons.estimate-comparisons.oracle_value,atol=1e-7)
    summaries=pd.read_parquet(root/"tables/ope_repeated_summary.parquet")
    keys=["model","propensity_source","target","population","estimator"]
    groups=dict(tuple(comparisons[comparisons.clipping.eq("none")].groupby(keys)))
    for row in summaries.itertuples():
        g=groups[tuple(getattr(row,k) for k in keys)]
        np.testing.assert_allclose([row.estimate_mean,row.bias,row.rmse,row.std],
                                  [g.estimate.mean(),g.error.mean(),np.sqrt(np.mean(g.error**2)),g.estimate.std()],atol=1e-7)


def verify_trajectory_contrasts(root):
    runs=pd.read_parquet(root/"tables/trajectory_runs.parquet")
    paired=pd.read_parquet(root/"tables/trajectory_paired.parquet")
    for metric in ["operating_cost","fill","lost_units","delay_unit_days","allocation_disparity"]:
        np.testing.assert_allclose(paired[metric+"_difference"],paired[metric+"_cvar"]-paired[metric+"_mean"],atol=1e-8)
    np.testing.assert_allclose(paired.reward_ate,-paired.operating_cost_difference,atol=1e-8)
    allocation=pd.read_parquet(root/"tables/trajectory_allocations.parquet")
    cohort=pd.read_parquet(root/"tables/cohort.parquet")[["series_id","cohort"]]
    expanded=pd.concat([allocation.merge(cohort,on="series_id",validate="many_to_one"),allocation.assign(cohort="ALL")],ignore_index=True)
    keys=["origin_date","world_kind","world_id","scenario_family","policy_id","cohort"]
    st=expanded.groupby(keys+["day_index","store_id"])[["candidate_quantity","approved"]].sum().reset_index()
    st=st[st.candidate_quantity.gt(0)].assign(share=lambda x:x.approved/x.candidate_quantity)
    daily=st.groupby(keys+["day_index"]).share.agg(["min","max"]).reset_index()
    daily["disparity"]=daily["max"]-daily["min"]
    disparity=daily.groupby(keys).disparity.mean().rename("recalculated_disparity").reset_index()
    check=runs.merge(disparity,on=keys,how="left",validate="one_to_one")
    np.testing.assert_allclose(check.allocation_disparity,check.recalculated_disparity.fillna(0),atol=1e-9)
    events=pd.read_parquet(root/"tables/trajectory_events.parquet")
    delay=events[events.event.eq("retimed")].copy()
    delay["delay_unit_days"]=(delay.arrival_day-delay.previous_arrival_day)*delay.quantity
    expanded=pd.concat([delay,delay.assign(cohort="ALL")],ignore_index=True)
    aggregate=expanded.groupby(keys).agg(recalculated_delay=("delay_unit_days","sum"),recalculated_events=("order_id","size")).reset_index()
    check=runs.merge(aggregate,on=keys,how="left",validate="one_to_one")
    np.testing.assert_allclose(check.delay_unit_days,check.recalculated_delay.fillna(0),atol=1e-8)
    np.testing.assert_array_equal(check.delay_order_events,check.recalculated_events.fillna(0))


def verify_sequential(root):
    ledger=pd.read_parquet(root/"tables/sequential_weights.parquet")
    result=pd.read_parquet(root/"tables/sequential_diagnostics.parquet")
    for row in result.itertuples():
        g=ledger[ledger.horizon.eq(row.horizon)&ledger.target_probability.eq(row.target_probability)]
        w=g.weight.to_numpy(float);r=g.return_value.to_numpy(float)
        np.testing.assert_allclose(w,np.exp(g.log_weight),atol=1e-12)
        ess=w.sum()**2/(w@w) if w@w else 0.
        np.testing.assert_allclose([row.ess,row.max_weight,row.weight_variance,row.trajectory_ips],
                                  [ess,w.max(),w.var(),np.mean(w*r)],atol=1e-7)
        if row.matching!=(w>0).sum():raise AssertionError("Trajectory matching count")
        if w.sum():np.testing.assert_allclose(row.trajectory_snips,w@r/w.sum(),atol=1e-9)
