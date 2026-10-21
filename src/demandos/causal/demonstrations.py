"""Controlled structural goldens, confounding and sequential instability."""
import numpy as np
import pandas as pd
from scipy.special import expit
from .estimators import evaluate,trajectory_weights


def controlled(replications=200,n=2000,seed=42):
    """Independent synthetic units; known conditional means, NOT M5 rewards.

    X confounds A and Y. Y(a)=10X+2a+epsilon; logger .1/.9 by X.
    Population V(always1)=7, ATE=ATT=2. Wrong nuisances omit X.
    """
    rows=[];confounding=[]
    for replication in range(replications):
        rng=np.random.default_rng(seed+replication);x=rng.binomial(1,.5,n);noise=rng.normal(0,1,n)
        p=.1+.8*x;a=(rng.random(n)<p).astype(int);r=10*x+2*a+noise
        mu=np.c_[1-p,p];pi=np.tile([0.,1.],(n,1));correct=np.c_[10*x,10*x+2]
        for prop_good,outcome_good in [(True,True),(True,False),(False,True),(False,False)]:
            assumed=mu if prop_good else np.full((n,2),.5)
            m=correct if outcome_good else np.full((n,2),5.)
            result=evaluate(a,r,assumed,pi,m)
            for estimator in ["ips","snips","dm","dr"]:
                value=result[estimator]
                if estimator=="snips":influence=result["weights"]*(r-value)/result["weights"].mean()
                else:influence=result[estimator+"_contributions"]
                se=float(np.std(influence,ddof=1)/np.sqrt(n))
                rows.append(dict(replication_id=replication,correct_propensity=prop_good,correct_outcome=outcome_good,estimator=estimator,
                    estimate=value,truth=7.,error=value-7.,se=se,lower=value-1.96*se,upper=value+1.96*se,
                    covered=abs(value-7.)<=1.96*se,interval_status="IID_SYNTHETIC_WALD_ASYMPTOTIC; misspecification invalidates coverage claims"))
        pi0=1-pi
        ipw=evaluate(a,r,mu,pi)["ips"]-evaluate(a,r,mu,pi0)["ips"]
        dr=evaluate(a,r,mu,pi,correct)["dr"]-evaluate(a,r,mu,pi0,correct)["dr"]
        random_a=(rng.random(n)<.5).astype(int);random_r=10*x+2*random_a+noise
        confounding.append(dict(replication_id=replication,truth_ate=2.,truth_att=2.,
            naive_difference=float(r[a==1].mean()-r[a==0].mean()),known_ipw=ipw,correct_dm=2.,correct_dr=dr,
            omitted_confounder_dr=evaluate(a,r,np.full((n,2),.5),pi,np.full((n,2),5.))["dr"]-
                                  evaluate(a,r,np.full((n,2),.5),pi0,np.full((n,2),5.))["dr"],
            randomized_benchmark=float(random_r[random_a==1].mean()-random_r[random_a==0].mean())))
    frame=pd.DataFrame(rows)
    summary=[]
    for key,g in frame.groupby(["correct_propensity","correct_outcome","estimator"]):
        summary.append(dict(zip(["correct_propensity","correct_outcome","estimator"],key),replications=len(g),
                            bias=float(g.error.mean()),std=float(g.estimate.std()),rmse=float(np.sqrt(np.mean(g.error**2))),coverage=float(g.covered.mean())))
    return frame,pd.DataFrame(summary),pd.DataFrame(confounding)


def sequential(n=2048,seed=42):
    rng=np.random.default_rng(seed);a=rng.binomial(1,.5,(n,28));noise=rng.normal(0,.1,(n,28))
    def returns(actions):
        state=np.zeros(n);reward=[]
        for t in range(28):
            reward.append(2*actions[:,t]-state**2)
            state=.8*state+.3*actions[:,t]+noise[:,t]
        return np.cumsum(np.column_stack(reward),axis=1)
    logged=returns(a);rows=[];ledgers=[]
    for p in [1.,.75]:
        on_policy=rng.binomial(1,p,(n,28));oracle=returns(on_policy)
        for horizon in [1,7,14,28]:
            mu=np.full((n,horizon,2),.5);pi=np.tile([1-p,p],(n,horizon,1))
            result=trajectory_weights(a[:,:horizon],mu,pi)
            w=result["weights"];total=logged[:,horizon-1]
            rows.append(dict(horizon=horizon,target_probability=p,n=n,ess=result["ess"],matching=result["matching_trajectories"],
                max_weight=result["max_weight"],weight_variance=result["weight_variance"],status=result["status"],
                trajectory_ips=float(np.mean(w*total)),trajectory_snips=float(w@total/w.sum()) if w.sum() else None,
                on_policy_monte_carlo=float(oracle[:,horizon-1].mean()),
                interpretation="illustrative stateful synthetic process; not inventory-controller OPE or empirical reliability"))
            for i in range(n):ledgers.append(dict(trajectory_id=i,horizon=horizon,target_probability=p,log_weight=float(result["log_weights"][i]),weight=float(w[i]),return_value=float(total[i])))
    assert np.isclose(np.exp(28*np.log(2)),268435456)
    return pd.DataFrame(rows),pd.DataFrame(ledgers)
