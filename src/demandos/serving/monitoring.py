"""Frozen-bin PSI, descriptive KS and strictly mature-label forecast scoring."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from .core import require, digest
from .features import KEY


def psi_probabilities(reference,current,*,epsilon=1e-6):
    p,q=np.asarray(reference,float),np.asarray(current,float)
    require(p.ndim==1 and p.shape==q.shape and len(p)>0 and np.isfinite(p).all() and np.isfinite(q).all() and (p>=0).all() and (q>=0).all() and p.sum()>0 and q.sum()>0, "PSI_PROBABILITIES")
    require(np.isfinite(epsilon) and epsilon>0,"PSI_SMOOTHING")
    p=p/p.sum();q=q/q.sum();p=np.maximum(p,epsilon);q=np.maximum(q,epsilon);p/=p.sum();q/=q.sum()
    return float(np.sum((q-p)*np.log(q/p)))


def numeric(values):
    x=np.asarray(values)
    require(np.issubdtype(x.dtype,np.number),"DRIFT_FEATURE_TYPE")
    x=x.astype(float);require(not np.isinf(x).any(),"DRIFT_NONFINITE")
    return x


def reference_bins(values):
    x=numeric(values);known=x[np.isfinite(x)]
    inside=np.unique(np.quantile(known,np.linspace(0,1,11))) if len(known) else np.array([])
    return np.r_[-np.inf,inside,np.inf]


def frequencies(values,bins):
    x=numeric(values);edges=np.asarray(bins,float)
    require(len(x)>0 and edges.ndim==1 and len(edges)>=2 and np.all(np.diff(edges)>0) and edges[0]==-np.inf and edges[-1]==np.inf,"DRIFT_BINS_OR_EMPTY")
    counts=np.r_[np.histogram(x[np.isfinite(x)],bins=edges)[0],np.isnan(x).sum()]
    return counts/len(x)


def ks(reference,current):
    a,b=numeric(reference),numeric(current);a=a[np.isfinite(a)];b=b[np.isfinite(b)]
    return float(ks_2samp(a,b).statistic) if len(a) and len(b) else None


def drift(reference,current,feature,policy,*,origin):
    a,b=numeric(reference),numeric(current);bins=reference_bins(a)
    require(len(a)>0 and len(b)>0,"DRIFT_EMPTY_SAMPLE")
    p,q=frequencies(a,bins),frequencies(b,bins);psi=psi_probabilities(p,q);distance=ks(a,b)
    enough=min(np.isfinite(a).sum(),np.isfinite(b).sum())>=policy["minimum_samples"]
    missing=float(np.isnan(b).mean());missing_delta=missing-float(np.isnan(a).mean())
    warn=bool((enough and (psi>policy["psi_warning"] or distance is not None and distance>policy["ks_warning"])) or missing_delta>policy["missingness_warning"])
    return dict(feature=feature,origin_date=str(origin),reference_n=len(a),current_n=len(b),reference_finite=int(np.isfinite(a).sum()),current_finite=int(np.isfinite(b).sum()),
                reference_values=[None if np.isnan(x) else float(x) for x in a],current_values=[None if np.isnan(x) else float(x) for x in b],
                reference_bins=["-inf" if x==-np.inf else "+inf" if x==np.inf else float(x) for x in bins],missing_bin="last; NaN separate; never filled as zero",
                reference_probabilities=p.tolist(),current_probabilities=q.tolist(),epsilon=1e-6,psi=psi,ks=distance,
                missing_fraction=missing,missing_delta=missing_delta,minimum_samples=policy["minimum_samples"],threshold_version=policy["version"],
                window=policy["window"],severity="warning" if warn else "informational",sample_status="SUFFICIENT_DESCRIPTIVE_ONLY" if enough else "INSUFFICIENT_FINITE_SAMPLE",
                interpretation="fixed reference bins; no IID KS p-value or causal failure claim")


def mature_labels(forecasts,labels,monitoring_date,*,lag_days=2):
    require(type(lag_days) is int and lag_days>=0,"LABEL_LAG")
    require({*KEY,"date","units_sold"}.issubset(labels) and not labels.duplicated([*KEY,"date"]).any(),"LABEL_KEYS")
    x=labels[[*KEY,"date","units_sold"]].rename(columns={"date":"target_date","units_sold":"actual"}).copy()
    x["label_observation_date"]=pd.to_datetime(x.target_date)
    x["assumed_available_at"]=x.label_observation_date+pd.Timedelta(days=lag_days)
    frame=forecasts.merge(x,on=[*KEY,"target_date"],how="left",validate="many_to_one")
    observed=frame.actual.notna()
    values=frame.loc[observed,"actual"].to_numpy(float)
    require(np.isfinite(values).all() and (values>=0).all() and (values==np.floor(values)).all(),"INVALID_LABELS")
    frame["mature"]=observed & frame.assumed_available_at.le(pd.Timestamp(monitoring_date))
    frame["maturity_status"]=np.where(~observed,"MISSING_LABEL",np.where(frame.mature,"MATURE","NOT_YET_AVAILABLE"))
    frame["monitoring_date"]=pd.Timestamp(monitoring_date);frame["label_lag_days"]=lag_days
    # Unavailable labels are masked in the monitor ledger too, not merely ignored by metrics.
    frame.loc[~frame.mature,"actual"]=np.nan
    return frame


def performance(ledger):
    mature=ledger[ledger.mature];n=len(mature);missing=int(ledger.maturity_status.eq("MISSING_LABEL").sum())
    if not n:return dict(requested=len(ledger),mature=0,unavailable=len(ledger),missing_labels=missing,mae=None,wape=None,pinball=None,coverage=None,width=None,observed_units=0.,wape_status="UNAVAILABLE_NO_MATURE_LABELS")
    y=mature.actual.to_numpy(float);pred=mature.prediction.to_numpy(float);denominator=float(np.abs(y).sum())
    errors=y[:,None]-mature[["q10","q50","q90"]].to_numpy(float);q=np.array([.1,.5,.9])
    return dict(requested=len(ledger),mature=n,unavailable=len(ledger)-n,missing_labels=missing,mae=float(np.abs(y-pred).mean()),
                wape=float(np.abs(y-pred).sum()/denominator) if denominator else None,observed_units=denominator,
                wape_status="AVAILABLE" if denominator else "UNDEFINED_ZERO_OBSERVED_UNITS",pinball=float(np.maximum(q*errors,(q-1)*errors).mean()),
                coverage=float(((y>=mature.q10)&(y<=mature.q90)).mean()),width=float((mature.q90-mature.q10).mean()))


def matched_performance(approved,candidate):
    keys=[*KEY,"origin_date","horizon","target_date"]
    a=approved[approved.mature];b=candidate[candidate.mature]
    require(set(map(tuple,a[keys].to_numpy()))==set(map(tuple,b[keys].to_numpy())),"UNMATCHED_MONITOR_COHORT")
    require(len(a)>0,"NO_MATURE_COMPARISON")
    return performance(a),performance(b)


def incident(signal,severity,asof,*,model_version=None,batch_id=None,feature_digest=None,threshold=None,evidence=None,disposition="recorded",policy_version="historical-demo-v1"):
    require(severity in {"blocking","warning","informational"},"INCIDENT_SEVERITY")
    event=dict(signal=signal,severity=severity,asof=str(asof),model_version=model_version,batch_id=batch_id,
               feature_contract_digest=feature_digest,threshold=threshold,evidence=evidence,disposition=disposition,
               publication_id=batch_id,policy_version=policy_version,scenario_status="SYNTHETIC_FAULT_OR_HISTORICAL_LOCAL_MONITOR")
    return dict(event_id=digest(event),**event)
