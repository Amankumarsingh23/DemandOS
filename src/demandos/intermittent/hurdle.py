"""Binary occurrence times conditional-positive squared-error mean regression."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

from demandos.boosting.models import GlobalModel, TARGET, validate_rows
from demandos.boosting.config import BoostConfig
from demandos.features.config import FeatureConfig
from demandos.features.contracts import KEYS


class OccurrenceModel(GlobalModel):
    def _fit_native(self,X,y,Xv,yv):
        c = self.config
        params = {"objective":"binary","metric":"binary_logloss","learning_rate":c.learning_rate,
                  "num_leaves":c.num_leaves,"max_depth":c.max_depth,"min_child_samples":c.min_child_samples,
                  "lambda_l1":c.reg_alpha,"lambda_l2":c.reg_lambda,"feature_fraction":c.colsample,
                  "bagging_fraction":c.subsample,"bagging_freq":1,"num_threads":c.threads,"seed":c.seed,
                  "deterministic":True,"force_col_wise":True,"verbosity":1}
        train = lgb.Dataset(X,label=y,categorical_feature=self.categories)
        valid = lgb.Dataset(Xv,label=yv,reference=train,categorical_feature=self.categories) if Xv is not None else None
        self.booster = lgb.train(params,train,num_boost_round=c.iterations,valid_sets=[valid] if valid is not None else None,
            callbacks=[lgb.early_stopping(c.patience,verbose=False)] if valid is not None else [])
        self.rounds = self.booster.best_iteration or self.booster.current_iteration()
        self.native_parameters = params


class HurdleModel:
    def __init__(self,features,config=BoostConfig(),feature_config=FeatureConfig()):
        GlobalModel("lightgbm",features,config,feature_config)  # Reuse the predictor allowlist.
        self.features,self.config,self.feature_config = list(features),config,feature_config
        self.occurrence,self.size,self.state = None,None,{}

    def fit(self,training,*,allowed_cutoff,validation=None,validation_cutoff=None,rounds=None):
        validate_rows(training,labelled=True,cutoff=allowed_cutoff)
        if len(training)>self.config.max_rows or training.origin_date.nunique()>self.config.max_origins or len(training[["store_id","item_id"]].drop_duplicates())>self.config.training_series:
            raise ValueError("Hurdle training resource cap exceeded")
        self.occurrence,self.size = None,None
        if validation is not None:
            validate_rows(validation,labelled=True,cutoff=validation_cutoff)
            if not validation.target_date.gt(pd.Timestamp(allowed_cutoff)).all() or not validation.origin_date.ge(pd.Timestamp(allowed_cutoff)).all():
                raise ValueError("Hurdle validation must chronologically follow fitting labels")
        positive = training[TARGET].gt(0)
        self.state = {"allowed_cutoff":str(pd.Timestamp(allowed_cutoff).date()),"training_rows":len(training),
                      "positive_rows":int(positive.sum()),"occurrence_status":"learned","size_status":"learned",
                      "p_constant":None,"size_constant":None,"size_objective":"untransformed positive-target squared error; score floor 1",
                      "validation_cutoff":str(pd.Timestamp(validation_cutoff).date()) if validation is not None else None}
        if positive.nunique() == 1:
            self.state.update(p_constant=float(positive.iloc[0]),occurrence_status="constant_single_occurrence_class")
        else:
            c = replace(self.config,iterations=rounds["occurrence"]) if rounds and rounds["occurrence"] else self.config
            binary = training.assign(**{TARGET:positive.astype(int)})
            valid = validation.assign(**{TARGET:validation[TARGET].gt(0).astype(int)}) if validation is not None else None
            self.occurrence = OccurrenceModel("lightgbm",self.features,c,self.feature_config).fit(binary,
                allowed_cutoff=allowed_cutoff,validation=valid,validation_cutoff=validation_cutoff)
        positive_rows = training.loc[positive].copy()
        if len(positive_rows) < 20:
            self.state.update(size_constant=float(positive_rows[TARGET].mean()) if len(positive_rows) else 0.,
                              size_status="observed_positive_mean_small_partition" if len(positive_rows) else "not_learned_no_positive_labels")
        else:
            c = replace(self.config,iterations=rounds["size"]) if rounds and rounds["size"] else self.config
            valid = validation.loc[validation[TARGET].gt(0)].copy() if validation is not None else None
            if valid is not None and valid.empty:
                valid = None
                self.state["size_validation_status"] = "no_positive_validation_fixed_budget"
            self.size = GlobalModel("lightgbm",self.features,c,self.feature_config).fit(positive_rows,
                allowed_cutoff=allowed_cutoff,validation=valid,validation_cutoff=validation_cutoff if valid is not None else None)
        self.state["rounds"] = {"occurrence":self.occurrence.rounds if self.occurrence else 0,"size":self.size.rounds if self.size else 0}
        return self

    def predict(self,features):
        validate_rows(features)
        if not self.state or not features.origin_date.ge(pd.Timestamp(self.state["allowed_cutoff"])).all() or not set(self.features).issubset(features):
            raise ValueError("Fitted origin-valid hurdle state and feature schema required")
        p = self.occurrence.predict(features).raw_prediction.to_numpy(float) if self.occurrence else np.repeat(self.state["p_constant"],len(features))
        raw_size = self.size.predict(features).raw_prediction.to_numpy(float) if self.size else np.repeat(self.state["size_constant"],len(features))
        size = np.maximum(raw_size,1.) if self.state["positive_rows"] else np.zeros(len(features))
        if not np.isfinite(p).all() or not np.isfinite(size).all() or (p<0).any() or (p>1).any():
            raise ValueError("Invalid hurdle probabilities or conditional mean")
        result = features[KEYS+["target_date"]].reset_index(drop=True).copy()
        result["model"] = "hurdle"
        result["p_positive"],result["raw_positive_size"],result["conditional_positive_mean"] = p,raw_size,size
        result["raw_prediction"],result["prediction"] = p*size,p*size
        result["hurdle_status"] = self.state["occurrence_status"]+";"+self.state["size_status"]
        return result

    def save(self,directory):
        directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
        for name,model in (("occurrence",self.occurrence),("positive_size",self.size)):
            if model is not None:
                model.save(directory/name)
        payload = {"features":self.features,"config":asdict(self.config),"feature_config":asdict(self.feature_config),"state":self.state,
                   "occurrence_learned":self.occurrence is not None,"size_learned":self.size is not None}
        (directory/"hurdle.json").write_text(json.dumps(payload,indent=2,default=str)+"\n",encoding="utf-8")

    @classmethod
    def load(cls,directory):
        directory = Path(directory); raw = json.loads((directory/"hurdle.json").read_text())
        model = cls(raw["features"],BoostConfig(**raw["config"]),FeatureConfig(**raw["feature_config"]))
        model.state = raw["state"]
        model.occurrence = GlobalModel.load(directory/"occurrence") if raw["occurrence_learned"] else None
        model.size = GlobalModel.load(directory/"positive_size") if raw["size_learned"] else None
        return model
