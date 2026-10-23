# Reproduce the DemandOS capstone

This is a local, CPU-first historical replay. It does not train models, download
data, run a web service or transmit purchase orders. A fresh clone contains code
and documentation; the large input data and frozen native weights are ignored.

## Runtime and installation

Python 3.11+ is the package contract. The verified local environment uses Python
3.13, Pandas 3.0.6, PyArrow 23.0.1, DuckDB 1.5.6, LightGBM 4.7.0,
SciPy 1.18.1 and dbt-duckdb 1.11.0. Native checkpoint metadata requires its saved
LightGBM version. Optional neural tests require the repository's neural extra;
installing its PyTorch wheel depends on platform availability.

From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[analytics,eda,boosting,neural,reconciliation]"
.\.venv\Scripts\python -m demandos.capstone --root . --check-prereqs
.\.venv\Scripts\python -m demandos.capstone --root . --dry-run
```

Both inspection modes only check presence and installed versions. Missing inputs
produce an explicit list and exit code 1, without manufacturing data or weights.
On macOS/Linux the Python modules use portable paths; replace the executable with
`.venv/bin/python`. That platform has not been verified by this capstone run.

## Obtain inputs separately

Obtain the legitimate M5 Forecasting Accuracy validation dataset under its
applicable terms. Extract/copy the original CSVs into these exact locations:

```text
data/raw/sales_train_validation.csv
data/raw/calendar.csv
data/raw/sell_prices.csv
```

Keep the originals unchanged. Rebuild ingestion/warehouse only if their validated
outputs are missing; rebuilding is outside the inexpensive default demo:

```powershell
.\.venv\Scripts\python -m demandos.data.cli --data-dir data --chunk-size 128
.\.venv\Scripts\python -m demandos.warehouse --root . build
```

The [warehouse guide](warehouse.md) describes its build/verify commands.
The full input is 30,490 series × 1,913 dates = 58,327,370 recorded observations.

## Restore eligible artifacts before the demo

Use your own verified outputs from the earlier seasons; a clone cannot recreate
the trained checkpoints simply by running the capstone. Preserve the original
paths, metadata, native `model.txt` files, source fingerprints and versions.

The adapter reuses the Season 23 loader and registry bootstrap, so it requires
both three-quantile ensembles under:

```text
data/quantiles/global_h28/models/lightgbm/{p10,p50,p90}/frozen_evaluation/
data/conformal/m5_h28/models/lightgbm/{p10,p50,p90}/frozen_evaluation/
```

Each directory needs `metadata.json` and `model.txt`. It also requires the
quantile feature mapping, fixed cohort, inference audit, forecast table and both
summary JSONs, together with the saved S17/S20/S21/S22 scientific evidence and
the S23 verified-run pointer/tables. `--check-prereqs` lists the exact files.
The native S16 ensemble is used for the demo; S17 is inspected for evidence and
registered separately. No neural model is loaded by the demo.

`configs/serving_baseline.json` is an existing 56-day historical empirical recipe,
not fitted weights. It is registered to exercise fallback compatibility; the
default capstone fails if native prerequisites are unavailable and does not
silently substitute a baseline. Earlier-season reproduction commands remain in
the [README history](../README.md#historical-season-implementation-details).

## Execute, independently verify and repeat

```powershell
.\.venv\Scripts\python -m demandos.capstone --root .
.\.venv\Scripts\python -m demandos.capstone --root . --verify
.\.venv\Scripts\python -m demandos.capstone --root .
.\.venv\Scripts\python -m unittest discover -s tests -v
```

The real-data dbt acceptance command uses the repository's portable environment:

```powershell
.\.venv\Scripts\python -c "from demandos.warehouse.cli import settings,invoke_dbt; r,s,w,e=settings('.'); invoke_dbt(r,e,'test')"
```

The default selects the first three sorted keys in each established continuity
and sparse-stress cohort, at origin 2016-01-31. It generates 168 forecast rows,
six noncommitted origin previews, and three 168-row simulation trajectories:
recorded-sales proxy, day-14 perturbation, and a whole-order receiving closure.
The actual sequential controller issues its own post-demand orders; preview
orders are not booked again as operating cash.

Outputs are under ignored `data/capstone/m5_h28/`. `latest_verified.json` names
the immutable `runs/<identity-prefix>/` directory. It contains `manifest.json`,
`feature_contract.json`, `tables/*.parquet`, `summary.json`, `evidence.json`,
`faults.json`, `verification.json`, `runtime.json`, `completed.json` and
`capstone_report.md`. `last_invocation.json` reports repeat/reuse and total runtime.
All final files are staged, audited, sealed by SHA256 and atomically renamed.

Identical inputs, code, Git base commit and local environment must produce equal
keys/actions and numerically equal results within absolute tolerance 1e-8.
Timing diagnostics are excluded from identity and equivalence. A changed code
digest or Git commit intentionally produces a different run identity. No
distributed transaction guarantee is asserted.

## Interpret the result

Read [scientific evidence](capstone_results.md) before comparing models or costs.
Raw sales may be censored. Inventory, costs, supplier quotas, storage ceilings,
availability stamps and receiving closures are assumptions. The physical storage
ceiling is a conservative initial-stock-plus-entitlement upper bound; it is
checked independently from procurement flow quotas and is not a binding-capacity
optimization study. Partial deliveries and real supplier/order integration are
not implemented. Use [system design](system_design_defense.md) for deployment gaps.
