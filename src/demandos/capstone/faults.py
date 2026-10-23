"""Isolated reversible exercises. No original data or native model is modified."""
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd

from demandos.serving.core import Blocked, digest, require
from demandos.serving.features import validate_frame
from demandos.serving.decisions import validate_decisions
from .config import LABELS


def exercise(data, features, decisions, registry, contract_digest):
    rows = []

    def blocked(name, call):
        try:
            call()
        except (Blocked, ValueError, AssertionError) as exc:
            rows.append(dict(fault=name, disposition="BLOCKED", signal=str(exc).split(":")[0], accepted=True, label=LABELS["faults"]))
        else:
            raise AssertionError("Fault was not blocked: "+name)

    bad = features.copy()
    bad["max_sales_source_date"] = bad.origin_date+pd.Timedelta(days=1)
    blocked("future-ineligible feature", lambda: validate_frame(bad, data.feature_config, data.mapping))
    blocked("incompatible fallback contract", lambda: registry.verify("historical-56d-v1", contract_digest="incompatible"))
    bad = decisions.copy(); bad["approved_quantity"] = bad.request+1000000
    blocked("invalid hard resource allocation", lambda: validate_decisions(bad))
    # Use an actual Registry verifier on temporary, explicitly SYNTHETIC identity bytes.
    from demandos.serving.registry import Registry
    with TemporaryDirectory(prefix="demandos-capstone-fault-") as tmp:
        root = Path(tmp); model = root/"identity.txt"; model.write_text("SYNTHETIC DIGEST FIXTURE", encoding="utf-8")
        local = Registry(root, root/"registry", [root])
        metadata = {k:v for k,v in registry.entry("historical-56d-v1").items() if k in ["model_id","kind","training_cutoff","validation_window","feature_schema_version","feature_contract_digest","horizons","output_schema","dependencies","historical_provenance"]}
        local.register("fixture", metadata, [model], asof="synthetic")
        model.write_text("CORRUPTED SYNTHETIC COPY", encoding="utf-8")
        blocked("corrupt artifact digest", lambda: local.verify("fixture"))
    # Exercise the existing atomic publisher twice with identical full vectors.
    from demandos.serving.publication import publish, request_id
    with TemporaryDirectory(prefix="demandos-capstone-duplicate-") as tmp:
        # Its validator is the actual decision hard-resource validator.
        entry = registry.entry("s16-lgb-20160131")
        _, request = request_id(entry, contract_digest, digest("synthetic-request-namespace"), features.origin_date.iloc[0],
                                decisions[["store_id","item_id"]].to_dict("records"), {"capstone_fault":True}, channel="isolated-fault")
        # publication's forecast-key fixture need only match actual frames; no dispatch occurs.
        forecast = features[["store_id","item_id","origin_date","horizon","target_date"]].copy()
        validator = lambda f,d: validate_decisions(d)
        publish(Path(tmp), request, forecast, decisions, validator)
        outcome = publish(Path(tmp), request, forecast, decisions, validator)
        require(outcome[2] == "REUSED_NO_NEW_ACTIONS", "DUPLICATE_REQUEST_NOT_REUSED")
        rows.append(dict(fault="duplicate publication request", disposition="REUSED_NO_NEW_ACTIONS", signal="idempotent reuse", accepted=True, label=LABELS["faults"]))
    return rows
