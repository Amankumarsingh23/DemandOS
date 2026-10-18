"""A few frozen regime rules and convex blends learned only on earlier losses."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Config


CANDIDATES = ("pooled","hurdle","tsb","seasonal_naive")
KEYS = ["store_id","item_id","origin_date","horizon","target_date"]
BLENDS = ({"pooled":1.},{"pooled":.5,"hurdle":.5},{"pooled":.5,"tsb":.5},{"pooled":.5,"seasonal_naive":.5})


class SegmentPolicy:
    def __init__(self,config=Config()):
        self.config,self.state = config,{}

    def fit(self,validation,*,validation_end,first_evaluation):
        end = pd.Timestamp(validation_end)
        if validation.empty or end >= pd.Timestamp(first_evaluation) or pd.to_datetime(validation.target_date).gt(end).any():
            raise ValueError("Routing/weight validation must end before evaluation")
        source = validation[validation.model.isin(CANDIDATES)].copy()
        if source.duplicated(KEYS+["model"]).any() or source.groupby(KEYS).actual.nunique(dropna=False).gt(1).any():
            raise ValueError("Unique aligned validation forecasts/actuals required")
        if source.groupby(KEYS).regime.nunique(dropna=False).gt(1).any():
            raise ValueError("Origin regime must agree across candidate rows")
        available = source[source.status.eq("ok")]
        if not np.isfinite(available[["actual","prediction"]].to_numpy(float)).all():
            raise ValueError("Finite permitted validation outcomes required")
        counts = available.groupby(KEYS).model.nunique()
        common = counts[counts.eq(len(CANDIDATES))].reset_index()[KEYS]
        aligned = available.merge(common,on=KEYS,validate="many_to_one")
        evidence = []
        for regime, group in source.groupby("regime",sort=True):
            good = aligned[aligned.regime.eq(regime)]
            base = good[good.model.eq("pooled")].sort_values(KEYS)
            series = len(base[["store_id","item_id"]].drop_duplicates())
            enough = series >= self.config.minimum_validation_series and len(base) >= self.config.minimum_validation_rows
            losses,blends = {},[]
            if enough:
                matrix = good.pivot(index=KEYS,columns="model",values="prediction").sort_index()
                y = base.set_index(KEYS).actual.reindex(matrix.index).to_numpy(float)
                losses = {name:float(np.mean(np.abs(y-matrix[name].to_numpy(float)))) for name in CANDIDATES}
                for weights in BLENDS:
                    pred = sum(weight*matrix[name].to_numpy(float) for name,weight in weights.items())
                    blends.append(float(np.mean(np.abs(y-pred))))
                route = min(CANDIDATES,key=lambda name:(losses[name],CANDIDATES.index(name)))
                weights = dict(BLENDS[int(np.argmin(blends))])
            else:
                route,weights = "pooled",{"pooled":1.}
            evidence.append({"regime":regime,"validation_series":series,"validation_rows":len(base),
                             "rule_status":"selected" if enough else "insufficient_evidence_default_pooled",
                             "route":route,"weights":weights,"candidate_mae":losses,"blend_mae":blends})
        canonical = source[KEYS+["model","prediction","actual","regime","status"]].sort_values(KEYS+["model"]).to_csv(index=False,lineterminator="\n")
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        self.state = {"version":"season11_regime_policy_v1","validation_end":str(end.date()),
                      "first_evaluation":str(pd.Timestamp(first_evaluation).date()),"evidence_sha256":digest,
                      "default_route":"pooled","rules":evidence,"candidates":list(CANDIDATES),
                      "blend_candidates":list(BLENDS),"cold_override":"cold_prior when visible_history_days < short_history"}
        return self

    def predict(self,forecasts,origin):
        if not self.state or pd.Timestamp(origin) < pd.Timestamp(self.state["first_evaluation"]):
            raise ValueError("Frozen policy is valid only after its validation period")
        if forecasts.duplicated(KEYS+["model"]).any() or not pd.to_datetime(forecasts.origin_date).eq(pd.Timestamp(origin)).all():
            raise ValueError("Unique origin-aligned candidate predictions required")
        rules = {row["regime"]:row for row in self.state["rules"]}
        outputs,decisions = [],[]
        for key,group in forecasts.groupby(KEYS,sort=True):
            row = group.iloc[0]
            known = {r.model:r for r in group.itertuples() if r.status == "ok" and pd.notna(r.prediction)}
            rule = rules.get(row.regime,{"route":"pooled","weights":{"pooled":1.},"rule_status":"unseen_regime_default_pooled"})
            cold = row.visible_history_days < self.config.short_history
            for method in ("router","ensemble"):
                desired = {"cold_prior":1.} if cold else {rule["route"]:1.} if method == "router" else rule["weights"]
                if any(v<0 or not np.isfinite(v) for v in desired.values()) or not np.isclose(sum(desired.values()),1.):
                    raise ValueError("Policy weights must be finite nonnegative and normalized")
                effective,reason = desired,"cold_override" if cold else rule["rule_status"]
                if not set(desired).issubset(known):
                    effective = {"pooled":1.} if "pooled" in known else {"cold_prior":1.} if "cold_prior" in known else {}
                    reason = "candidate_unavailable_explicit_default" if effective else "no_candidate_or_prior_available"
                value = sum(weight*known[name].prediction for name,weight in effective.items()) if effective else np.nan
                base = dict(zip(KEYS,key))
                outputs.append({**base,"model":method,"raw_prediction":value,"prediction":value,
                                "status":"ok" if effective else "skipped","reason":reason})
                decisions.append({**base,"method":method,"regime":row.regime,"requested_weights":json.dumps(desired,sort_keys=True),
                    "effective_weights":json.dumps(effective,sort_keys=True),"decision_reason":reason,
                    "policy_version":self.state["version"],"evidence_sha256":self.state["evidence_sha256"],
                    "validation_end":self.state["validation_end"]})
        return pd.DataFrame(outputs),pd.DataFrame(decisions)

    def save(self,path):
        Path(path).write_text(json.dumps({"config":asdict(self.config),"state":self.state},indent=2)+"\n",encoding="utf-8")

    @classmethod
    def load(cls,path):
        raw = json.loads(Path(path).read_text()); policy = cls(Config(**raw["config"])); policy.state=raw["state"]
        return policy
