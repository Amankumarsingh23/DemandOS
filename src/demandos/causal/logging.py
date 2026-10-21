"""Assignment uses only a fixed predecision context and independently keyed U."""
import numpy as np
from scipy.special import expit
from demandos.inventory.scenarios import seed_for

FEATURES=("history_mean","history_zero_rate","history_cv","on_hand","budget_factor","flow_factor",
          "target_units","mean_units","cvar_units","mean_cash","cvar_cash","mean_disparity","cvar_disparity","risk_score")
METADATA=("context_id","origin_date","resource_pool_id","world_id","scenario_id","logged_action","true_propensity",
          "mu_0","mu_1","reward","outcome_available_date","model_provenance","scenario_status","eligible_actions","replication_id")


def logging_probabilities(risk_score):
    z=np.asarray(risk_score,float)
    if not np.isfinite(z).all():raise ValueError("Finite PREDECISION risk score required")
    p=.2+.6*expit(z)
    return np.stack([1-p,p],axis=-1)


def assign(context_id,world_id,replication,probabilities,seed=42):
    p=np.asarray(probabilities,float)
    if p.shape!=(2,) or not np.isfinite(p).all() or (p<=0).any() or not np.isclose(p.sum(),1,atol=1e-12,rtol=0):raise ValueError("Strict positive binary logging vector required")
    uniform=np.random.default_rng(seed_for(seed,"causal-assignment",context_id,world_id,replication)).random()
    return int(uniform>=p[0])


def target_probabilities(frame,name):
    n=len(frame)
    if name=="mean":p=np.zeros(n)
    elif name=="cvar":p=np.ones(n)
    elif name=="mixture":p=np.full(n,.5)
    elif name=="risk_rule":p=np.where(frame.risk_score.to_numpy()>=0,.8,.2)
    else:raise ValueError("Unknown predeclared target policy")
    return np.c_[1-p,p]


def estimator_view(frame):
    """Reject any extra columns rather than accidentally selecting an oracle target."""
    if set(frame.columns)!=set(FEATURES)|set(METADATA):
        raise ValueError("Estimator input must be the selected-outcome logging contract; no oracle/future columns")
    import pandas as pd
    if frame.duplicated(["context_id","world_id","replication_id"]).any():raise ValueError("Duplicate logging unit")
    if not np.isfinite(frame[list(FEATURES)].to_numpy(float)).all() or not np.isfinite(frame.reward).all():raise ValueError("Finite eligible features and selected rewards required")
    from .estimators import probabilities
    mu=probabilities(frame[["mu_0","mu_1"]].to_numpy(float),"true logger")
    a=frame.logged_action.to_numpy(float)
    if (a!=np.floor(a)).any() or (a<0).any() or (a>1).any() or not np.isfinite(a).all():raise ValueError("Binary integer logged action required")
    if not np.allclose(mu[np.arange(len(a)),a.astype(int)],frame.true_propensity,atol=1e-12,rtol=0):raise ValueError("Selected propensity does not match full known logger vector")
    dates=pd.to_datetime(frame.origin_date);available=pd.to_datetime(frame.outcome_available_date)
    if dates.isna().any() or available.isna().any() or (available<=dates).any():raise ValueError("Explicit delayed outcome maturity required")
    return frame.copy()
