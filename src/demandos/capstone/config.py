"""Frozen selection and cheap, read-only prerequisite discovery."""
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import subprocess

from demandos.serving.core import digest, file_hash

CONFIG = dict(origin="2016-01-31", horizon=28, per_cohort=3, seed=42,
              selection="first three store/item keys per established cohort, sorted by series_id",
              label_lag_days=2, monitoring_date="2016-03-01", perturbation_day=14,
              physical_storage="per store: initial inventory plus entire supplier entitlement, volume-weighted upper bound",
              tolerance=1e-8, native_checkpoint="s16-lgb-20160131")
LABELS = {
    "source": "REAL_OBSERVED_SOURCE",
    "forecasts": "HISTORICAL_FORECAST_EVALUATION",
    "inventory": "SIMULATED_INVENTORY_OUTCOME",
    "optimization": "SIMULATED_OPTIMIZATION_OUTCOME",
    "logging": "SYNTHETIC_POLICY_LOG",
    "causal": "MODEL_WORLD_CAUSAL_DEMO",
    "faults": "LOCAL_SERVING_FAULT_EXERCISE",
}


def table_label(name):
    if name in {"snapshot", "metadata", "features"}:
        return LABELS["source"]
    if name in {"forecasts", "maturity"}:
        return LABELS["forecasts"]
    if name == "decisions" or name.endswith(("allocations", "resources", "solver")):
        return LABELS["optimization"]
    return LABELS["inventory"]


def prerequisites(root):
    root = Path(root)
    required = ["data/raw/" + n for n in ("sales_train_validation.csv", "calendar.csv", "sell_prices.csv")]
    required += ["data/interim/sales_long.parquet", "data/processed/daily_sales.parquet",
                 "data/warehouse/demandos.duckdb", "data/warehouse/dbt_build_record.json",
                 "data/warehouse/analytics_quality_report.json", "configs/serving_baseline.json",
                 "configs/optimization.toml", "configs/simulation.toml"]
    for base in ["data/quantiles/global_h28", "data/conformal/m5_h28"]:
        required.append(base + "/summary.json")
        required += [f"{base}/models/lightgbm/{q}/frozen_evaluation/{n}"
                     for q in ("p10", "p50", "p90") for n in ("metadata.json", "model.txt")]
    required += ["data/quantiles/global_h28/" + n for n in (
        "feature_mapping.json", "tables/evaluation_cohort.parquet", "tables/inference_audit.parquet", "tables/forecasts.parquet")]
    # The final scientific audit requires saved evidence, not just historical prose.
    required += [f"data/{s}/m5_h28/tables/{n}.parquet" for s, ns in {
        "conformal": ["intervals", "metrics_overall"],
        "optimization": ["runs", "metrics"],
        "risk": ["risk_frontier", "scenario_manifest", "scenario_losses", "operating_risk", "pool_operating"],
        "causal": ["synthetic_logs", "nuisance_predictions", "ope_estimates", "trajectory_runs"],
    }.items() for n in ns]
    required.append("data/serving/m5_h28/latest_verified.json")
    pointer = root/"data/serving/m5_h28/latest_verified.json"
    if pointer.is_file():
        import json
        try:
            relative = json.loads(pointer.read_text(encoding="utf-8"))["path"]
            run = (root/relative).resolve()
            if not run.is_relative_to(root.resolve()/"data/serving/m5_h28"):
                raise ValueError("serving pointer leaves its artifact directory")
            required += [relative+"/tables/maturity_ledger.parquet", relative+"/registry/state.json"]
            required += [p.relative_to(root.resolve()).as_posix() for p in (run/"publication/batches").glob("*/*.parquet")]
            if len(list((run/"publication/batches").glob("*/completed.json"))) != 7:
                required.append(relative+"/publication/batches/<seven complete historical batches>")
        except (ValueError, KeyError):
            required.append("valid data/serving/m5_h28/latest_verified.json pointer")
    missing = [p for p in required if not (root / p).is_file()]
    packages = {}
    for name in ["pandas", "pyarrow", "duckdb", "lightgbm", "scipy", "dbt-duckdb"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            missing.append("installed package: " + name)
    return dict(ready=not missing, missing=missing, required_files=required, packages=packages,
                mode="presence only; no hashes, model loads, writes, downloads or training")


def code_identity(root):
    root = Path(root)
    paths = [root / "pyproject.toml"]
    for folder, suffixes in [("src", {".py"}), ("configs", {".json", ".toml"}), ("dbt", {".sql", ".yml"})]:
        paths += [p for p in (root / folder).rglob("*") if p.is_file() and p.suffix in suffixes
                  and not any(x in p.parts for x in ("target", "logs", "dbt_packages"))]
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths)}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return dict(git_commit=commit, code_digest=digest(files), files=files)


def run_identity(manifest):
    return digest(manifest)
