"""Observed-action-only contextual OPE and explicit overlap failures."""
import numpy as np
from scipy.special import logsumexp


def probabilities(values, name="policy"):
    p=np.asarray(values,float)
    if p.ndim!=2 or not len(p) or not p.shape[1] or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1,atol=1e-12,rtol=0):
        raise ValueError(f"Finite normalized {name} probability vectors required")
    return p


def evaluate(actions, rewards, logging, target, predictions=None, *, clip=None):
    """No oracle argument: rewards must contain ONLY the selected-action outcome.

    Full logging vectors are required to check support even for unchosen actions.
    A support failure makes all identified policy-value estimates unavailable,
    including DM. Clipping is a separately labeled biased sensitivity.
    """
    mu=probabilities(logging,"logging");pi=probabilities(target,"target")
    a=np.asarray(actions);r=np.asarray(rewards,float);n,k=mu.shape
    if pi.shape!=mu.shape or a.shape!=(n,) or r.shape!=(n,) or not np.isfinite(r).all() or not np.isfinite(a).all() or (a!=np.floor(a)).any() or (a<0).any() or (a>=k).any():
        raise ValueError("Aligned integer actions and finite selected rewards required")
    a=a.astype(int);chosen=mu[np.arange(n),a]
    if (chosen<=0).any():raise ValueError("Impossible logged action: zero true/assumed logging probability")
    m=None if predictions is None else np.asarray(predictions,float)
    if m is not None and (m.shape!=mu.shape or not np.isfinite(m).all()):raise ValueError("Finite aligned all-action PREDICTIONS required")
    support=(pi>0)&(mu<=0)
    base=dict(n=n,actions=k,positivity_violations=int(support.sum()),target_support_coverage=float(1-support.sum()/max(1,(pi>0).sum())),
              assignment_counts=np.bincount(a,minlength=k).tolist(),min_logging_probability=float(mu.min()),max_logging_probability=float(mu.max()),
              reward_orientation="larger reward is better; inventory reward = negative simulated cost")
    if support.any():return {**base,"status":"NONIDENTIFIABLE_MISSING_SUPPORT","ips":None,"snips":None,"dm":None,"dr":None,"ess":None}
    w=pi[np.arange(n),a]/chosen
    if clip is not None:
        if not np.isfinite(clip) or clip<=0:raise ValueError("Finite positive clipping cap required")
        w=np.minimum(w,clip)
    direct=None if m is None else (pi*m).sum(axis=1)
    dr=None if m is None else direct+w*(r-m[np.arange(n),a])
    denominator=float(w.sum());square=float(w@w)
    return {**base,"status":"IDENTIFIED_BY_ASSUMED_LOGGING_DESIGN" if clip is None else "BIASED_CLIPPED_SENSITIVITY",
            "ips":float(np.mean(w*r)),"snips":float(w@r/denominator) if denominator else None,
            "snips_status":"AVAILABLE" if denominator else "UNDEFINED_ZERO_WEIGHT",
            "dm":None if direct is None else float(direct.mean()),"dr":None if dr is None else float(dr.mean()),
            "ess":denominator**2/square if square else 0.,"max_weight":float(w.max()),"weight_mean":float(w.mean()),
            "weight_variance":float(w.var()),"matching_actions":int((w>0).sum()),"weight_sum":denominator,
            "weights":w,"ips_contributions":w*r,"dm_contributions":direct,"dr_contributions":dr,"clip":clip}


def cluster_intervals(actions,rewards,logging,target,predictions,clusters,*,samples=200,seed=42):
    """Percentile world-block bootstrap, conditional on the fitted nuisances.

    Does not refit nuisance models or imply reliable real-world coverage with
    few/shared historical pools. Blocks include all origins/context strata.
    """
    labels=np.asarray(clusters);unique=np.unique(labels)
    if labels.shape!=(len(actions),) or len(unique)<2 or samples<20:raise ValueError("At least two complete trajectory/world blocks required")
    groups=[np.flatnonzero(labels==c) for c in unique];rng=np.random.default_rng(seed)
    draws={name:[] for name in ["ips","snips","dm","dr"]}
    a=np.asarray(actions);r=np.asarray(rewards);mu=np.asarray(logging);pi=np.asarray(target);m=np.asarray(predictions)
    for _ in range(samples):
        idx=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
        result=evaluate(a[idx],r[idx],mu[idx],pi[idx],m[idx])
        for name in draws:
            if result[name] is not None:draws[name].append(result[name])
    return dict(clusters=len(unique),bootstrap_samples=samples,
                status="HEURISTIC_MODEL_WORLD_BLOCK_INTERVAL_FIXED_NUISANCE_FEW_CLUSTERS",
                intervals={name:np.quantile(v,[.025,.975]).tolist() if v else None for name,v in draws.items()})


def trajectory_weights(actions, logging, target):
    """Complete-trajectory weights in log space, not daily-row OPE."""
    a=np.asarray(actions);mu=np.asarray(logging,float);pi=np.asarray(target,float)
    if a.ndim!=2 or mu.ndim!=3 or pi.shape!=mu.shape or mu.shape[:2]!=a.shape:
        raise ValueError("Aligned trajectory x time x action probabilities required")
    probabilities(mu.reshape(-1,mu.shape[-1]),"sequential logger")
    probabilities(pi.reshape(-1,pi.shape[-1]),"sequential target")
    if not np.isfinite(a).all() or (a!=np.floor(a)).any() or (a<0).any() or (a>=mu.shape[-1]).any():raise ValueError("Invalid sequential actions")
    a=a.astype(int);chosen_mu=np.take_along_axis(mu,a[...,None],axis=2)[...,0]
    chosen_pi=np.take_along_axis(pi,a[...,None],axis=2)[...,0]
    if (chosen_mu<=0).any():raise ValueError("Impossible sequential logged action")
    if ((pi>0)&(mu<=0)).any():return dict(status="NONIDENTIFIABLE_MISSING_SUPPORT",weights=None,ess=None)
    with np.errstate(divide="ignore"):
        logw=(np.log(chosen_pi)-np.log(chosen_mu)).sum(axis=1)
    if np.max(logw)>np.log(np.finfo(float).max):raise ValueError("Trajectory weights exceed floating-point range; retain log weights")
    w=np.exp(logw);matching=int(np.isfinite(logw).sum())
    ess=float(np.exp(2*logsumexp(logw)-logsumexp(2*logw))) if matching else 0.
    return dict(status="AVAILABLE" if matching else "NO_MATCHING_TRAJECTORIES",log_weights=logw,weights=w,
                ess=ess,matching_trajectories=matching,max_weight=float(w.max()),weight_variance=float(w.var()))
