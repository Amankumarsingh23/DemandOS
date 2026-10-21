"""Track A: one origin-issued whole-pool allocation; no subsequent decisions."""
from itertools import product
import json
import numpy as np
import pandas as pd
from demandos.inventory.experiment import ORIGINS
from demandos.inventory.scenarios import bootstrap,seed_for
from demandos.simulation.pipeline import Sink
from .logging import FEATURES,METADATA,logging_probabilities,assign,target_probabilities,estimator_view
from .simulation import catalog,one_allocation,STATUS
from .nuisance import fit_predict
from .estimators import evaluate,cluster_intervals


def build_worlds(data,track,worlds=8):
    result={}
    for origin in ORIGINS:
        rows=[]
        for m in data.metadata.itertuples():
            paths,_,_=bootstrap(data.histories[(m.store_id,m.item_id)],origin,samples=worlds,
                                 seed=seed_for(42,"causal-"+track,m.series_id,origin))
            rows.append(paths)
        result[origin]=np.stack(rows)
        from demandos.risk.pipeline import hashlib_array
        if len({hashlib_array(result[origin][:,j,:]) for j in range(worlds)})!=worlds:
            raise AssertionError("Need distinct complete joint evaluation worlds")
    return result


def contexts(data,config):
    rows=[];catalog_rows=[];checks=[];diagnostics=[]
    for origin in ORIGINS:
        for stock,budget,flow in product(config["initial_stock"],config["budget_factors"],config["flow_factors"]):
            cid=f"{origin.date()}|stock={stock}|budget={budget}|flow={flow}"
            q,c,r,solves=catalog(data.metadata,data.targets[origin],data.scenarios[origin],data.settings,
                                stock=stock,budget=budget,flow=flow,seconds=config["solver_seconds"])
            features={**data.history_features[origin],"on_hand":stock,"budget_factor":budget,"flow_factor":flow,
                "target_units":int(data.targets[origin].sum()),"mean_units":int(q[0].sum()),"cvar_units":int(q[1].sum()),
                "mean_cash":solves[0]["resource_use"]["budget"],"cvar_cash":solves[1]["resource_use"]["budget"],
                "mean_disparity":solves[0]["fairness"]["disparity"],"cvar_disparity":solves[1]["fairness"]["disparity"],
                "risk_score":1.5*(1-stock/30)+(.7-budget)/.6+.5*(.8-flow)/.3-1.}
            p=logging_probabilities(features["risk_score"])
            row=dict(context_id=cid,origin_date=origin,resource_pool_id="SELECTED_COHORT_DC",**features,mu_0=p[0],mu_1=p[1],
                     eligible_actions="[0,1]",model_provenance="frozen historical fractile; S15-21 provenance retained",scenario_status=STATUS)
            rows.append(row)
            for action in [0,1]:
                diagnostics.append(dict(context_id=cid,origin_date=origin,action=action,**solves[action]))
                for i,m in enumerate(data.metadata.itertuples()):
                    catalog_rows.append(dict(context_id=cid,origin_date=origin,action=action,series_id=m.series_id,store_id=m.store_id,item_id=m.item_id,
                        supplier=m.dept_id,quantity=int(q[action,i]),request=int(c.request[i]),purchase=float(c.purchase[i]),fee=float(c.fee[i]),volume=float(c.volume[i]),
                        initial_stock=stock,initial_prepaid_pipeline=3,target=int(data.targets[origin][i]),
                        budget_limit=r.budget,flow_limit=r.flow_volume,supplier_limits=json.dumps(r.suppliers),store_limits=json.dumps(r.receiving),scenario_status=STATUS))
            for perturbation,other in data.future_scenarios[origin].items():
                changed,_,_,extra=catalog(data.metadata,data.targets[origin],other,data.settings,stock=stock,budget=budget,flow=flow,seconds=config["solver_seconds"])
                np.testing.assert_array_equal(q,changed)
                np.testing.assert_array_equal(p,logging_probabilities(features["risk_score"]))
                checks.append(dict(context_id=cid,origin_date=origin,check=perturbation,all_25_actions_equal=True,propensities_equal=True,passed=True))
                diagnostics.extend(dict(context_id=cid,origin_date=origin,action=a,purpose=perturbation,**s) for a,s in enumerate(extra))
    return pd.DataFrame(rows),pd.DataFrame(catalog_rows),pd.DataFrame(checks),diagnostics


def potential_outcomes(data,config,context,catalogue,directory):
    worlds=build_worlds(data,"bandit",config["worlds"]);rows=[]
    directory.mkdir();sink=Sink(directory/"tables")
    try:
        for index,c in context.iterrows():
            action_rows=catalogue[catalogue.context_id.eq(c.context_id)]
            for world in range(config["worlds"]):
                y=worlds[c.origin_date][:,world,:14]
                for action in [0,1]:
                    q=action_rows[action_rows.action.eq(action)].set_index("series_id").loc[data.metadata.series_id,"quantity"].to_numpy(int)
                    result=one_allocation(y,c.origin_date,data.metadata,q,stock=int(c.on_hand),world_id=world)
                    meta=dict(context_id=c.context_id,action=action,world_id=world,scenario_status=STATUS)
                    for name in ["daily","events","runs"]:sink.write(name,result[name].assign(**meta))
                    g=result["runs"];demand=g.demand_units.sum()
                    rows.append(dict(context_id=c.context_id,origin_date=c.origin_date,resource_pool_id=c.resource_pool_id,world_id=world,action=action,
                        reward=float(g.reward.sum()),total_cost=float(g.total_cost.sum()),operating_cost=float(g.operational_cost.sum()),decision_cash=float(g.decision_cash.sum()),
                        fill=float(g.immediate_served_units.sum()/demand) if demand else None,lost_units=int(g.lost_units.sum()),
                        outcome_available_date=c.origin_date+pd.Timedelta(days=14),scenario_status=STATUS))
            if index%6==0:print(f"One-step oracle context {index+1}/{len(context)}; feasible whole-pool actions",flush=True)
    finally:sink.close()
    oracle=pd.DataFrame(rows)
    oracle.to_parquet(directory/"potential_outcomes.parquet",index=False)
    return oracle,sink.rows


def make_logs(context,oracle,replications=32):
    """Evaluator-side logging generator; only chosen reward leaves this function."""
    truth=oracle.pivot(index=["context_id","world_id"],columns="action",values="reward")
    rows=[]
    for replication in range(replications):
        for c in context.itertuples():
            p=np.array([c.mu_0,c.mu_1])
            for world in range(8):
                action=assign(c.context_id,world,replication,p)
                rows.append({**{name:getattr(c,name) for name in FEATURES},"context_id":c.context_id,"origin_date":c.origin_date,
                    "resource_pool_id":c.resource_pool_id,"world_id":world,"scenario_id":"prefix_only_historical_joint_world",
                    "logged_action":action,"true_propensity":p[action],"mu_0":p[0],"mu_1":p[1],"reward":float(truth.loc[(c.context_id,world),action]),
                    "outcome_available_date":c.origin_date+pd.Timedelta(days=14),"model_provenance":c.model_provenance,
                    "scenario_status":STATUS,"eligible_actions":"[0,1]","replication_id":replication})
    return estimator_view(pd.DataFrame(rows))


def estimate(logs,config):
    """Fit/evaluate without any potential-outcome table/path argument."""
    results=[];predictions=[];provenance=[];diagnostics=[];contributions=[]
    for replication,g in logs.groupby("replication_id",sort=True):
        train=g[g.origin_date.eq(ORIGINS[0])].reset_index(drop=True)
        evaluation=g[g.origin_date.gt(ORIGINS[0])].reset_index(drop=True)
        known=evaluation[["mu_0","mu_1"]].to_numpy(float)
        for model in config["outcome_models"]:
            pred,estimated,record=fit_predict(train,evaluation,model=model,seed=42)
            provenance.append(dict(replication_id=replication,**record))
            for i,row in evaluation.iterrows():
                predictions.append(dict(replication_id=replication,context_id=row.context_id,world_id=row.world_id,origin_date=row.origin_date,
                    model=model,prediction_0=pred[i,0],prediction_1=pred[i,1],estimated_mu_0=estimated[i,0],estimated_mu_1=estimated[i,1]))
            for prop in config["propensity_sources"]:
                assumed=known if prop=="known" else estimated
                for target in config["targets"]:
                    pi=target_probabilities(evaluation,target)
                    if ((pi>0)&(known<=0)).any():raise AssertionError("Estimated propensities/clipping cannot restore missing TRUE logger support")
                    for population,mask in [("pooled",np.ones(len(evaluation),bool)),(str(ORIGINS[1].date()),evaluation.origin_date.eq(ORIGINS[1]).to_numpy()),
                                            (str(ORIGINS[2].date()),evaluation.origin_date.eq(ORIGINS[2]).to_numpy())]:
                        x=evaluation.loc[mask];a=x.logged_action.to_numpy(int);r=x.reward.to_numpy(float)
                        result=evaluate(a,r,assumed[mask],pi[mask],pred[mask])
                        if result["positivity_violations"]:raise AssertionError("Primary OPE overlap failure")
                        meta=dict(replication_id=replication,model=model,propensity_source=prop,target=target,population=population,n=len(x),
                                  scenario_status=STATUS,clipping="none")
                        ci=cluster_intervals(a,r,assumed[mask],pi[mask],pred[mask],x.world_id.to_numpy(),samples=config["bootstrap_samples"]) if replication==0 else None
                        for name in ["ips","snips","dm","dr"]:
                            interval=None if ci is None else ci["intervals"][name]
                            results.append({**meta,"estimator":name,"estimate":result[name],"status":result["status"],
                                "lower":None if interval is None else interval[0],"upper":None if interval is None else interval[1],
                                "interval_status":"not_computed_replication" if ci is None else ci["status"],"world_clusters":8})
                        diagnostics.append({**meta,**{name:result[name] for name in ["ess","max_weight","weight_mean","weight_variance","matching_actions","min_logging_probability","max_logging_probability","positivity_violations","target_support_coverage"]},
                                            "assignment_counts":str(result["assignment_counts"]),"propensity_rmse_vs_true":float(np.sqrt(np.mean((assumed[mask]-known[mask])**2)))})
                        if replication==0 and population=="pooled":
                            clipped=evaluate(a,r,assumed[mask],pi[mask],pred[mask],clip=2.)
                            for name in ["ips","snips","dm","dr"]:
                                results.append({**meta,"estimator":name,"estimate":clipped[name],"status":clipped["status"],"clipping":"cap_2_BIASED_SENSITIVITY",
                                                "lower":None,"upper":None,"interval_status":"not_computed","world_clusters":8})
                        if replication==0 and population=="pooled":
                            for i,(index,row) in enumerate(x.iterrows()):
                                contributions.append({**meta,"context_id":row.context_id,"world_id":row.world_id,"reward":r[i],"action":int(a[i]),
                                    "weight":result["weights"][i],"ips":result["ips_contributions"][i],"dm":result["dm_contributions"][i],"dr":result["dr_contributions"][i]})
        if replication%8==0:print(f"Observed-only chronological nuisance fits/OPE replication {replication+1}/{config['logging_replications']}",flush=True)
    return dict(ope_estimates=pd.DataFrame(results),nuisance_predictions=pd.DataFrame(predictions),overlap_diagnostics=pd.DataFrame(diagnostics),
                ope_contributions=pd.DataFrame(contributions)),provenance


def audit_oracle(logs,oracle,estimates,predictions):
    """Oracle first joins predictions AFTER fit_predict/estimate have returned."""
    lookup=oracle.pivot(index=["context_id","world_id"],columns="action",values="reward")
    values=[];ate=[]
    base=logs[logs.replication_id.eq(0)&logs.origin_date.gt(ORIGINS[0])].copy()
    y=lookup.loc[pd.MultiIndex.from_frame(base[["context_id","world_id"]])].to_numpy(float)
    for population,mask in [("pooled",np.ones(len(base),bool)),(str(ORIGINS[1].date()),base.origin_date.eq(ORIGINS[1]).to_numpy()),
                            (str(ORIGINS[2].date()),base.origin_date.eq(ORIGINS[2]).to_numpy())]:
        for target in ["mean","cvar","mixture","risk_rule"]:
            pi=target_probabilities(base.loc[mask],target)
            values.append(dict(population=population,target=target,oracle_value=float(np.mean(np.sum(pi*y[mask],axis=1))),n=int(mask.sum()),scenario_status=STATUS))
        x=base.loc[mask];a=x.logged_action.to_numpy(int);r=x.reward.to_numpy(float);p=x.mu_1.to_numpy(float)
        controls=(a==0)*p/(1-p);treated=a==1
        ate.append(dict(population=population,oracle_ate=float((y[mask,1]-y[mask,0]).mean()),
            oracle_att_actual_logged_treated=float((y[mask,1]-y[mask,0])[treated].mean()) if treated.any() else None,
            naive_difference=float(r[treated].mean()-r[~treated].mean()) if treated.any() and (~treated).any() else None,
            normalized_ipw_att=float(r[treated].mean()-controls@r/controls.sum()) if treated.any() and controls.sum() else None,
            estimand="reward advantage of CVaR one-shot vector over mean; finite empirical pool-context/world population",scenario_status=STATUS))
    truth=pd.DataFrame(values)
    comparison=estimates.merge(truth,on=["population","target"],validate="many_to_one",suffixes=("","_oracle"))
    comparison["error"]=comparison.estimate-comparison.oracle_value
    summaries=[]
    for key,g in comparison[comparison.clipping.eq("none")].groupby(["model","propensity_source","target","population","estimator"]):
        summaries.append(dict(zip(["model","propensity_source","target","population","estimator"],key),
            replications=len(g),oracle_value=float(g.oracle_value.iloc[0]),estimate_mean=float(g.estimate.mean()),bias=float(g.error.mean()),
            rmse=float(np.sqrt(np.mean(g.error**2))),std=float(g.estimate.std()),scenario_status=STATUS))
    audit=predictions.merge(oracle[["context_id","world_id","action","reward"]].pivot(index=["context_id","world_id"],columns="action",values="reward").rename(columns={0:"oracle_0",1:"oracle_1"}).reset_index(),
                            on=["context_id","world_id"],validate="many_to_one")
    errors=[]
    for model,g in audit.groupby("model"):
        errors.append(dict(model=model,all_action_rmse=float(np.sqrt(np.mean(np.r_[g.prediction_0-g.oracle_0,g.prediction_1-g.oracle_1]**2))),
                           use="post-fit audit only; realized counterfactual error includes demand-world noise"))
    return dict(oracle_policy_values=truth,oracle_comparisons=comparison,ope_repeated_summary=pd.DataFrame(summaries),
                one_step_ate_att=pd.DataFrame(ate),nuisance_oracle_errors=pd.DataFrame(errors))
