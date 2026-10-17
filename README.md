# DemandOS

DemandOS is a learning-driven retail demand intelligence project. Its goal is to
turn credible sales forecasts and uncertainty estimates into cost-aware inventory
decisions.

## Current milestone: Season 2 — ingestion and validation

This repository contains the project foundation and a local M5 CSV ingestion
pipeline. Forecasting models, inventory optimization, APIs, and dashboards will
be added in later sessions. There are no model results or datasets in Git.

The verified source dataset is M5: daily product/store sales, calendar
information, and selling prices. Sales are observations, not necessarily true
demand: stockouts can hide purchases customers wanted to make. Any future demand
estimation or inventory simulation must state its assumptions explicitly.

## Local setup

Requires Python 3.11 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -c "import demandos; print(demandos.__version__)"
```

On macOS/Linux, use `.venv/bin/python` in place of
`.\.venv\Scripts\python`.

Run the package smoke check and synthetic ingestion tests after installation:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `configs/` | Configuration files as components are introduced |
| `data/raw/` | Original input data; keep unchanged |
| `data/interim/` | Intermediate transformations |
| `data/processed/` | Prepared analysis and modeling data |
| `docs/architecture.md` | Scope, assumptions, and planned system flow |
| `docs/decisions/` | Architecture decision records |
| `notebooks/` | Exploratory analysis and learning experiments |
| `src/demandos/` | Installable Python package |
| `scripts/` | Repeatable project commands when needed |
| `tests/` | Automated checks |

Data contents, local environments, secrets, and generated build files are ignored
by Git. Obtain datasets separately under their applicable terms; no download is
automated at this milestone.

## M5 ingestion

Place these extracted CSVs in `data/raw/` (ZIP files are not pipeline inputs):

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

```powershell
.\.venv\Scripts\python -m demandos.data.cli --data-dir data --chunk-size 128
```

Missing files are listed explicitly and the command exits with status 1 without
creating outputs. The pipeline does not download, extract, or fabricate data.

Implemented checks cover required source columns, non-null identifiers, finite
nonnegative integer sales, contiguous day identifiers, unique series/calendar/
price keys, calendar dates and week mappings, event null pairs, SNAP flags,
consistent item/store hierarchy, and finite positive source prices. Prices over
1000 produce a review warning rather than an arbitrary rejection. Blank source
prices are rejected; absent records in the price join remain null.

Sales are read in configurable wide-row chunks, melted, joined to calendar by
`d`, and left-joined to prices by `(store_id, item_id, wm_yr_wk)`. Both joins
enforce many-to-one cardinality and preserve sales rows. All calendar columns
are retained without adding engineered features. Global series uniqueness and
unique dates establish the daily key across chunks.

Outputs, replaced after a successful validation run:

- `data/interim/sales_long.parquet`: identifiers, `d`, and `units_sold`.
- `data/processed/daily_sales.parquet`: long sales with calendar and prices.
- `data/processed/quality_report.json`: actual counts, coverage, missing prices,
  zero-sales rate, duplicate checks, warnings, and source SHA-256 fingerprints.

Raw inputs remain untouched. Output is written to temporary files first;
validation failures preserve the previous successful outputs. Publication of
the three files is sequential, not a transactional multi-file operation.
Row order and Parquet row groups can differ with chunk size; logical records
remain the same. The calendar and weekly price lookup fit in memory; the much
larger daily sales expansion is streamed, not accumulated in memory. Lower
`--chunk-size` if daily chunks are too large for your machine.

Tests use deterministic miniature **synthetic fixtures, not actual M5 rows**.
They verify transformation values, join cardinality, missing-price preservation,
invalid inputs, cross-chunk duplicates, repeatability, and raw-file preservation.

## Verified Season 2 real-data results

The full local `sales_train_validation.csv` run completed with `--chunk-size 128`.
These are measured results for that source version, not synthetic test results:

| Metric | Verified result |
| --- | --- |
| Store-product series | 30,490 |
| Observed days per series | 1,913 |
| Daily rows in each Parquet output | 58,327,370 |
| Distinct products / stores | 3,049 / 10 |
| Sales date range | 2011-01-29 through 2016-04-24 |
| Calendar source rows | 1,969 |
| Weekly price source rows | 6,841,121 |
| Daily rows with missing prices | 12,299,413 (21.09%) |
| Positive-sales rows with missing prices | 0 |
| Zero-sales daily rows | 39,777,094 (68.20%) |

All source-key and join-cardinality checks passed. Both saved Parquet row counts
and streamed missing-price/zero-sales totals were independently checked against
the quality report. Raw source SHA-256 hashes remained unchanged. Missing prices
remain null and do not prove a stockout or lack of product availability.

The complete reshape is `30,490 * 1,913 = 58,327,370` rows. Reproduce it using
the installation and ingestion commands above; obtain the raw CSVs separately.
The JSON report records source fingerprints so runs using different inputs can
be distinguished. All 9 automated tests passed; those tests use synthetic data
and are separate from this full-data verification.

See [architecture](docs/architecture.md) and the
[foundation decision](docs/decisions/0001-project-foundation.md) for scope and
design rationale.
