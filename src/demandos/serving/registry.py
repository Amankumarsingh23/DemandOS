"""Immutable model manifests and one atomically replaced local lifecycle state."""
from importlib.metadata import version
from pathlib import Path
import re

from .core import atomic_json, read_json, digest, file_hash, require, writer_lock


class Registry:
    def __init__(self, root, directory, trusted_roots):
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        self.trusted = [Path(p).resolve() for p in trusted_roots]
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, name):
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", name), "REGISTRY_ID", name)
        return self.directory/"entries"/(name+".json")

    def _artifact(self, relative):
        path = (self.root/relative).resolve()
        require(path.is_relative_to(self.root) and any(path.is_relative_to(p) for p in self.trusted), "UNTRUSTED_ARTIFACT", relative)
        require(path.suffix in {".txt", ".json"}, "UNSAFE_ARTIFACT_FORMAT", relative)
        require(path.is_file(), "ARTIFACT_UNAVAILABLE", relative)
        return path

    def state(self):
        path = self.directory/"state.json"
        state = read_json(path) if path.exists() else dict(status={}, aliases={}, events=[])
        prior = None
        for event in state["events"]:
            require(event["previous"] == prior and event["event_id"] == digest({k:v for k,v in event.items() if k!="event_id"}), "REGISTRY_AUDIT_CORRUPTION")
            prior = event["event_id"]
        # Mutable routing/status must agree with the immutable history content.
        statuses, aliases = {}, {}
        for event in state["events"]:
            if event["action"] == "status": statuses[event["model_version"]] = event["status"]
            else: aliases[event["alias"]] = event["model_version"]
        require(statuses == state["status"] and aliases == state["aliases"], "REGISTRY_STATE_CORRUPTION")
        return state

    def _event(self, state, **data):
        event = dict(previous=state["events"][-1]["event_id"] if state["events"] else None, **data)
        state["events"].append(dict(**event, event_id=digest(event)))
        atomic_json(self.directory/"state.json", state)

    def register(self, name, metadata, artifact_paths, *, asof):
        self._path(name)
        files = {str(Path(p).resolve().relative_to(self.root).as_posix()): file_hash(self._artifact(str(Path(p).resolve().relative_to(self.root)))) for p in artifact_paths}
        require(bool(files), "ARTIFACT_UNAVAILABLE")
        required = {"model_id", "kind", "training_cutoff", "validation_window", "feature_schema_version", "feature_contract_digest",
                    "horizons", "output_schema", "dependencies", "historical_provenance"}
        require(required.issubset(metadata), "REGISTRY_METADATA", sorted(required-set(metadata)))
        require(metadata["kind"] in {"lightgbm_quantiles", "historical_empirical"}, "UNSUPPORTED_MODEL")
        require(metadata["horizons"]==list(range(1,29)) and metadata["output_schema"][:3]==["q10","q50","q90"],"REGISTRY_OUTPUT_CONTRACT")
        if metadata["training_cutoff"] is not None:
            import pandas as pd
            cutoff=pd.Timestamp(metadata["training_cutoff"]);window=metadata["validation_window"]
            require(not pd.isna(cutoff) and cutoff==cutoff.normalize() and len(window)==2 and pd.Timestamp(window[0])<=pd.Timestamp(window[1])<=cutoff,"REGISTRY_FIT_PROVENANCE")
        body = dict(**metadata, model_version=name, artifact_files=files, artifact_digest=digest(files),
                    initial_status="candidate", approval_lineage="state.json immutable hash-linked events")
        entry = dict(**body, entry_digest=digest(body))
        with writer_lock(self.directory):
            path = self._path(name)
            if path.exists(): require(read_json(path)==entry, "IMMUTABLE_REGISTRY_CONFLICT", name)
            else: atomic_json(path, entry)
            state = self.state()
            if name not in state["status"]:
                state["status"][name] = "candidate"
                self._event(state, action="status", model_version=name, status="candidate", asof=asof, reason="registered trusted existing artifact/explicit baseline recipe")
        return entry

    def entry(self, name):
        path = self._path(name)
        require(path.exists(), "REGISTRY_ENTRY_UNAVAILABLE", name)
        entry = read_json(path)
        require(entry["entry_digest"] == digest({k:v for k,v in entry.items() if k!="entry_digest"}), "REGISTRY_ENTRY_CORRUPTION")
        require(entry["artifact_digest"] == digest(entry["artifact_files"]), "REGISTRY_ENTRY_CORRUPTION")
        return entry

    def verify(self, name, *, contract_digest=None, origin=None, approved=False):
        entry = self.entry(name)
        require(self.state()["status"].get(name) in ({"approved"} if approved else {"candidate", "approved"}), "MODEL_NOT_APPROVED", name)
        if contract_digest is not None: require(entry["feature_contract_digest"]==contract_digest, "INCOMPATIBLE_FEATURE_CONTRACT")
        if origin is not None and entry["training_cutoff"] is not None:
            import pandas as pd
            require(pd.Timestamp(entry["training_cutoff"])<=pd.Timestamp(origin), "MODEL_FIT_AFTER_ORIGIN")
        for relative, expected in entry["artifact_files"].items():
            require(file_hash(self._artifact(relative))==expected, "ARTIFACT_DIGEST_MISMATCH", relative)
        for package, expected in entry["dependencies"].items():
            require(version(package)==expected, "DEPENDENCY_MISMATCH", package)
        return entry

    def transition(self, name, status, *, reason, asof):
        self.verify(name)
        require(status in {"approved", "retired"} and bool(reason), "REGISTRY_TRANSITION")
        with writer_lock(self.directory):
            state = self.state(); old = state["status"][name]
            require(old == "candidate" and status=="approved" or old=="approved" and status=="retired", "REGISTRY_TRANSITION", f"{old}->{status}")
            require(status!="retired" or name not in state["aliases"].values(), "ACTIVE_MODEL_RETIREMENT")
            state["status"][name] = status
            self._event(state, action="status", model_version=name, status=status, reason=reason, asof=asof)

    def activate(self, alias, name, *, contract_digest, reason, asof, rollback=False):
        require(alias in {"forecast", "fallback", "canary"} and bool(reason), "REGISTRY_ALIAS")
        self.verify(name, contract_digest=contract_digest, approved=True)
        with writer_lock(self.directory):
            state = self.state(); previous = state["aliases"].get(alias)
            if rollback:
                require(any(e.get("alias")==alias and e["model_version"]==name for e in state["events"]), "ROLLBACK_NOT_PRIOR_APPROVED")
                require(previous != name, "ROLLBACK_ALREADY_ACTIVE")
            state["aliases"][alias] = name
            self._event(state, action="rollback" if rollback else "alias", alias=alias, model_version=name,
                        previous_model=previous, reason=reason, asof=asof)
        return previous

    def resolve(self, alias, **kwargs):
        name = self.state()["aliases"].get(alias)
        require(name is not None, "ALIAS_UNAVAILABLE", alias)
        return self.verify(name, approved=True, **kwargs)
