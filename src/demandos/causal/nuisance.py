"""Chronological observed-reward-only nuisance fits; oracle-free APIs."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .logging import FEATURES,estimator_view


def design(frame,action):
    x=frame[list(FEATURES)].to_numpy(float)
    a=np.broadcast_to(np.asarray(action,float),(len(x),))
    return np.c_[x,a,x*a[:,None]]


def fit_predict(training,evaluation,*,model="ridge",seed=42):
    train=estimator_view(training);test=estimator_view(evaluation)
    if set(train.context_id)&set(test.context_id):raise ValueError("Training/evaluation context clusters overlap")
    maturity=pd.to_datetime(train.outcome_available_date).max();decision=pd.to_datetime(test.origin_date).min()
    if maturity>decision or pd.to_datetime(train.origin_date).max()>=decision:raise ValueError("Training outcomes not mature at the evaluation origin")
    if set(train.logged_action)!={0,1}:raise ValueError("Both logged actions required in development sample")
    if model=="ridge":regressor=make_pipeline(StandardScaler(),Ridge(alpha=10.))
    elif model=="forest":regressor=RandomForestRegressor(n_estimators=64,max_depth=5,min_samples_leaf=4,random_state=seed,n_jobs=1)
    elif model=="misspecified":regressor=None
    else:raise ValueError("Unknown bounded nuisance model")
    if regressor is None:pred=np.full((len(test),2),float(train.reward.mean()))
    else:
        regressor.fit(design(train,train.logged_action),train.reward.to_numpy(float))
        pred=np.column_stack([regressor.predict(design(test,a)) for a in [0,1]])
    propensity=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=seed))
    propensity.fit(train[list(FEATURES)],train.logged_action)
    if propensity.classes_.tolist()!=[0,1]:raise AssertionError("Unexpected propensity action order")
    mu_hat=propensity.predict_proba(test[list(FEATURES)])
    digest=hashlib.sha256(pd.util.hash_pandas_object(train[["context_id","world_id","logged_action","reward"]],index=False).to_numpy().tobytes()).hexdigest()
    provenance=dict(model=model,training_rows=len(train),evaluation_rows=len(test),fit_origin_max=str(pd.to_datetime(train.origin_date).max()),
                    outcome_maturity_max=str(maturity),evaluation_origin_min=str(decision),training_digest=digest,
                    features=list(FEATURES),split="Jan31 development only; Feb28/Mar27 held out; no oracle API argument",seed=seed)
    return pred,mu_hat,provenance
