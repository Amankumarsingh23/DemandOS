# DemandOS

DemandOS is a learning-driven retail demand intelligence project. Its goal is to
turn credible sales forecasts and uncertainty estimates into cost-aware inventory
decisions.

## Current milestone: Season 9 — global gradient-boosted forecasts

This repository contains validated M5 ingestion and a local DuckDB/dbt retail
analytics warehouse, with reproducible retail EDA, figures, and ABC/XYZ and
store-item intermittency classifications, plus time-series diagnostics and
origin-aware temporal utilities. Twelve point-forecast baselines now run on a
small training-selected cohort with explicit coverage, rolling-origin MAE/WAPE
and train-only RMSSE evaluation, plus expanding/sliding window comparisons.
Direct-horizon features now combine origin-safe sales history, target calendars,
verified-as-of historical-price interfaces and training-only categorical mappings.
LightGBM, XGBoost and CatBoost now fit pooled, horizon-aware observed-sales models
with chronological validation and origin-specific refits. Inventory optimization,
APIs, and dashboards remain future work. Generated
datasets, forecasts, and reports stay
outside Git; verified results are documented below.

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
| `dbt/` | Portable profile, staging/core/mart SQL, macros, and data tests |
| `sql/examples/` | Runnable teaching queries, separate from transformations |
| `src/demandos/warehouse/` | dbt runner and SQL quality/performance reporting |
| `data/warehouse/` | Ignored local DuckDB database, spill files, and reports |
| `src/demandos/eda/` | Configured descriptive analysis, segmentation, charts, and reports |
| `sql/eda/` | Bounded store-item statistics and complete-week product reductions |
| `data/eda/` | Ignored generated EDA tables, PNG figures, Markdown, and quality reports |
| `src/demandos/timeseries/` | Guarded diagnostics, training-only cohort selection, and temporal contracts |
| `data/diagnostics/` | Ignored generated diagnostic tables, figures, and reports |

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
be distinguished. All 9 Season 2 automated tests passed; those tests use
synthetic data and are separate from this full-data verification.

## DuckDB and dbt warehouse

After Season 2 ingestion, run from the repository root:

```powershell
.\.venv\Scripts\python -m pip install -c configs/analytics-constraints.txt -e ".[analytics]"
.\.venv\Scripts\python -m demandos.warehouse build
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m demandos.warehouse examples
```

`build` executes dbt on the actual local Parquet, then verifies source/fact/mart
totals, runs all six teaching SQL files, and writes the analytics-quality report.
It does not rerun CSV ingestion. Staging and fact models are views; dimensions
and four aggregate marts are persisted tables in `data/warehouse/demandos.duckdb`.

Implemented dimensions: product, store, observed calendar, and known weekly
prices. Marts summarize store/day, category/day, product/day, and store-wide
historical performance. Counts of positive-sales items describe selling
activity, not stock availability. Known revenue and complete revenue are separate
metrics: positive sales with missing price imply unknown revenue; zero sales
imply zero recorded revenue even when the price is unknown.

`python -m demandos.warehouse check` rechecks summaries of a built warehouse.
`python -m demandos.warehouse docs` generates local dbt documentation and lineage.
See [warehouse operations, grains, and metric contracts](docs/warehouse.md) for
direct dbt commands, portable paths, resource settings, and limitations.

Local generated artifacts, all ignored by Git:

- `data/warehouse/analytics_quality_report.json`: real metrics, reconciliations,
  source fingerprint, dbt outcomes, measured node/query timings, and SQL results.
- `data/warehouse/representative_explain.txt`: actual filtered-query plan.
- `dbt/target/`: compiled SQL, lineage manifest, test/build results, generated docs.

The optional analytics extra keeps ingestion dependencies separate. Warehouse
integration tests use synthetic Parquet in a temporary directory, including
positive sales with unknown prices and deliberately corrupted dimensions/marts.
Install the extra to run those tests; otherwise they are explicitly skipped.

Verified on the full M5 validation history: **10 models, 36/36 dbt tests, and
14/14 Python tests passed**. Facts preserve 58,327,370 rows and 65,695,409 units.
Historical selling-price-based revenue totals **187,676,570.02**; no positive-sales
rows have unknown prices. Price completeness is 78.91% by observation count.
Marts contain 19,130 store/day, 5,739 category/day, 5,832,737 product/day, and
10 store-performance rows. All four reconcile to the source totals.

The measured successful dbt build took 84.44 seconds; a filtered store-aggregation
query took 0.20 seconds on this Windows machine (2 DuckDB threads, 2GB managed
buffer budget). Timings depend on hardware and caches. EXPLAIN showed a Parquet
scan with projected store/units columns and a date filter at the scan. See the
quality report and [warehouse documentation](docs/warehouse.md) for details.

## Season 4: reproduce EDA and segmentation

Use the existing Season 3 warehouse; no CSV re-ingestion or fact duplication:

```powershell
.\.venv\Scripts\python -m pip install -c configs/analytics-constraints.txt -e ".[analytics,eda]"
.\.venv\Scripts\python -m demandos.eda
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Configuration is in `configs/eda.toml`. To compute labels strictly through a
training cutoff, for example:

```powershell
.\.venv\Scripts\python -m demandos.eda --end 2015-12-31
```

Every run writes under `data/eda/<start_YYYYMMDD>_<end_YYYYMMDD>/`:

- `tables/`: reduced Parquet summaries, product segments, store-item regimes,
  daily/weekly/monthly patterns, selected outlier flags, and additive STL examples.
- `figures/`: nine actual-data PNG charts for a full-history run.
- `business_report.md`: measured findings, interpretations, follow-up questions,
  definitions, caveats, and figure links.
- `quality_report.json`: source fingerprint, config, reconciliation, classifications,
  package versions, runtime, and artifact sizes in rows.

Default window: **2011-01-29 through 2016-04-24**. ABC and XYZ both classify products
across stores. ABC uses complete known-price revenue and deterministic crossing-item
80%/95% boundaries. XYZ uses **273 complete Saturday-Friday weeks**, excluding one
partial week, and population weekly CV cutoffs 0.5/1.0. Store-item regimes use
ADI 1.32 and positive-sales-size CV² 0.49, with unsupported histories kept separate.

Verified ABC counts: **A 1,264; B 923; C 862**. Store-item counts: **smooth 980;
intermittent 23,102; erratic 497; lumpy 5,911**. The top 610 products (approximately
20%) contribute **59.48%** of eligible known-price revenue. CA_3 leads recorded
volume at **11,188,180 units**; Saturday's pooled weekday index is **1.210**.
All five summary grains reconcile to **58,327,370 observations**, **65,695,409 units**,
and **187,676,570.02** historical selling-price-based revenue. These are descriptive
sales patterns, not stockout, profitability, demand, or causal conclusions.

The first full-data EDA run took **27.54s** locally; the final verified regeneration
took **96.71s** under different local load. Tests include synthetic
population-statistic edge cases, zero/missing histories, ABC ties and crossings,
XYZ boundaries, one-to-one joins, reconciliation, prior-only averages, and a
complete SQL analysis with a training cutoff. All **28 Python tests** and a fresh
run of **36 dbt tests** passed. Generated figures/reports/data remain ignored.
See [EDA definitions, outputs, and limitations](docs/eda.md) for details.

## Season 5: diagnostics and origin-time safety

Use the same optional dependencies and existing warehouse:

```powershell
.\.venv\Scripts\python -m pip install -c configs/analytics-constraints.txt -e ".[analytics,eda]"
.\.venv\Scripts\python -m demandos.timeseries
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Choices are in `configs/timeseries.toml`. Default chronological boundaries:
training through **2016-02-28**, validation **2016-02-29–2016-03-27**, test
**2016-03-28–2016-04-24**. Season 4 cohort labels are recomputed using training
data only; future target observations do not influence scaling or segmentation.

Implemented: grouped historical lags, origin-relative features for horizons
1/7/14/28, prior/trailing rolling statistics, first/second/weekly differences,
guarded ACF/PACF and ADF/KPSS, descriptive additive STL, chronological boundaries,
and expanding/rolling origin examples. Forecasting is implemented separately in
Season 6 below.

The verified illustrative cohort contains **13 store-item series, CA_3, and three
categories**. Of 68 original/transformed combinations, **64** have valid classical
diagnostics and **four** are explicitly skipped for extreme sparsity. Original
ADF/KPSS results include **13 conflicting pairs**, two compatible with
nonstationarity, one compatible with stationarity, and one skip. These are
exploratory selected-series results, not population stationarity verdicts.
CA_3's training-series lag-7 ACF is **0.760**; HOUSEHOLD's is **0.881**.

Outputs: `data/diagnostics/train_20160228/summary.json`, `diagnostic_report.md`,
reusable Parquet tables, and 12 figures. The first full run took **34.27s** locally;
the final verified regeneration took **33.35s**. Each report records its actual
runtime. **41 Python tests** and **36 dbt tests**
passed, including deliberate leakage violations and future-perturbation tests.
See [time-series definitions and contracts](docs/timeseries.md) for estimator
choices, guardrails, feature availability assumptions, and exact output grains.

## Season 6: statistical forecasting and one fixed holdout

Reuse the installed analytics/EDA extras and verified warehouse:

```powershell
.\.venv\Scripts\python -m pip install -c configs/analytics-constraints.txt -e ".[analytics,eda]"
.\.venv\Scripts\python -m demandos.forecasting --root . --config configs/forecasting.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m demandos.warehouse check
```

Implemented: naive, seasonal naive, trailing moving average, SES, Holt linear,
Holt damped, additive weekly Holt-Winters, ARIMA, SARIMA, original Croston, SBA,
and TSB. All forecast observed sales, not uncensored demand. Raw predictions are
retained; scoring uses `max(0, raw)` without altering actuals. A trailing moving
average averages observations; ARIMA's MA(q) models past forecast-error shocks.

The real experiment uses **13 store-item series, CA_3, and three categories**,
selected with training-only regimes/ABC/XYZ. Origin: **2016-02-28**. All models
use the same last **730 training days**, forecast the full **2016-02-29–2016-03-27**
28-day vector, and expose h=1/7/14/28 errors. The later 28-day test block remains
unused. SARIMA runs on four training-selected series; no holdout tuning or future
prices are used. Defaults are explicit in `configs/forecasting.toml`.

Verified outcomes: **184 successful, 16 skipped, four failed fits** from 204
model/series combinations, producing **5,152 forecast rows**. Thirteen skips
are the SARIMA cap; three have fewer than two positive events. Four additive
Holt-Winters fits fail convergence and retain their four warnings. **52 negative
raw horizon predictions** are retained and clipped for scoring. No silent fallback.

On the **eight common store-item series** with successful fits for all 11
non-SARIMA models, seasonal naive has **MAE 11.362, WAPE 37.991%**;
Holt-Winters has **11.671, 39.027%**; naive has **15.290, 51.127%**.
On all three common category aggregates, Holt-Winters has **MAE 1,024.056,
WAPE 7.314%**, versus seasonal naive **1,283.679, 9.169%**. These comparison
panels have different grains and are never pooled. A single selected holdout
does not establish a robust model winner. Season 7 below adds RMSSE and
rolling-origin evaluation separately, preserving these Season 6 results.

Outputs under ignored `data/forecasts/origin_20160228/`: `summary.json`,
`forecast_report.md`, two forecast-vs-actual figures, and Parquet tables for raw
and clipped forecasts, training/cohort labels, fit metadata/warnings, coverage,
available/common-sample metrics, segments, horizons, and exact comparison members.
The final verified regeneration took **8.73s** locally. **54 Python tests** and a fresh
run of **36 real-data dbt tests** passed, including temporal invariance and
raw/processed input immutability checks.
See [forecasting contracts and measured results](docs/forecasting.md) for all model
metrics, initialization, bounds, edge cases, and validation evidence.

## Season 7: rolling-origin backtesting and metric comparisons

Use the same installed dependencies, unchanged models, and verified warehouse:

```powershell
.\.venv\Scripts\python -m demandos.backtesting --root . --config configs/backtesting.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,output,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

Default origins: **2016-01-31, 2016-02-28, 2016-03-27**, each with **28 target
days**, covering February 1 through April 24 without overlap. All 12 existing
methods run with expanding history and a **730-day sliding window**; ARIMA is
capped at eight series and SARIMA at four. The cohort comprises **13 store-item
series, CA_3 and three categories**, frozen using only training through January
31. This reuses the Season 6 selection rules but changes one sparse example to
avoid using February-fitted labels for January forecasts.

Verified output: **1,224 model/series/origin/window cases**, **34,272 planned
target rows**, **1,060 successful fits** (29,680 predictions), **142 skips** and
**22 convergence failures/warnings**. Failed/skipped predictions remain null in
the planned output; no fallback or origin removal occurs. Of 142 skips, 132 are
explicit ARIMA/SARIMA caps and ten are all-zero histories unsupported by numerical
fits. All inputs remain unchanged.

Common simple-method results on **13 store-item series × three expanding-window
origins**, 1,092 targets per method:

| Model | MAE | Pooled WAPE | Macro RMSSE |
| --- | ---: | ---: | ---: |
| Naive | 18.780 | 65.080% | 0.931 |
| Seasonal naive | 13.132 | 45.506% | 0.789 |
| Moving average | 13.076 | 45.313% | 0.742 |

Macro RMSSE averages **38 valid series-origin cases**; one all-zero training
case is explicitly undefined. Seasonal naive has lower MAE at the first origin;
moving average at the next two. Rankings also vary by horizon and segment.
SARIMA's expanding-window WAPE **25.815%** versus seasonal naive **28.491%**
uses a separate **two-series / six-case subset**, not the broader table.
These are illustrative cohort results, **not official M5 WRMSSE**.
The final verified benchmark took **31.75s** locally. **70 Python tests** and
**36 fresh real-data dbt tests** passed; source fingerprints and forecast
row/date/key/metric reconciliation checks passed.

Ignored outputs: `data/backtests/20160131_20160327_h28_s28/`, containing a
Markdown report, JSON summary, exact generated-file manifest, three figures,
planned forecasts, coverage, scales, common-case rankings, origin/horizon/band
metrics, paired losses, variability, ABC/XYZ/regime summaries and window
sensitivity in Parquet. See [backtesting contracts and measured findings](docs/backtesting.md)
for metric denominators, reproducibility, runtime and validation evidence.

## Season 8: direct-horizon feature engineering

Use the existing analytics/EDA installation and verified warehouse:

```powershell
.\.venv\Scripts\python -m demandos.features --root . --config configs/features.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
```

The canonical grain is **(store_id, item_id, origin_date, horizon)**, with
`target_date = origin_date + horizon` for days **1–28**. Origin means end-of-day
after that day's sales become available. `sales_at_origin` uses that day;
`origin_lag_1` uses the preceding day. Rolling windows include the origin and
require complete observed windows. Labels live in a separate Parquet file and
are never predictive columns.

Implemented: lags 1/7/14/28/56; 7/14/28-day mean, sum, population std, max,
zero fraction and positive count; recent-level changes; target weekday/month,
weekend, events, state-specific SNAP and cyclical encodings; raw identifiers and
stable training-only category codes. Historical price/relative-price features
require explicit observation/availability evidence. **Retrospective M5 prices
remain unavailable by default**, with null values and true missingness flags.
Target calendars are assumed scheduled and known at origin; announcement
archives are not available to verify that assumption.

Verified full-panel pass: **30,490 origin rows** at **2016-01-31**, reducing
**1,737,930 bounded historical observations**, without full-panel horizon
expansion. A diverse **13-series** cohort selected through **2015-10-10** produces
**1,456 training feature rows and separate labels** from four earlier origins,
plus **1,092 inference rows** at the three Season 7 origins. All 28 horizons are
present. Mapping/selection is fitted before every generated origin; training
labels end January 31, excluding the February–April evaluation targets.

Whole feature tables match after future actuals are changed or removed at each
evaluation origin. Default future-price exclusion is checked on real data;
non-null historical price behavior and future-price perturbation are tested
using explicit synthetic as-of evidence. Full-panel result DataFrame memory is
about **15.5 MB**; the existing two-thread/2GB DuckDB buffer configuration is reused.
No forecasting or ML model is trained by this pipeline.
The final verified run took **76.47s** locally (**3.97s** for the full-panel
origin reduction). **82 Python tests** and **36 fresh real-data dbt tests** passed.

Outputs under ignored `data/features/direct_h28/`: training/inference feature
Parquet, separate training labels, compact origin tables, per-origin partitions,
saved category mappings, feature dictionaries, actual sample rows and a JSON/
Markdown validation report. Add `--inference-only` to omit supervised tables,
`--cohort <path>` for explicit store-item keys, or `--price-evidence <path>` for
verified historical-price knowledge stamps. See [feature contracts and commands](docs/features.md).

## Season 9: global gradient-boosted forecasting

Use the verified warehouse and a CPU Python environment. The exact tested
boosting constraints require **Python 3.12+** (tested on **3.13.15**); the base
ingestion package still supports Python 3.11+. Broader dependency ranges in the
boosting extra are not a claim that every version combination was tested.

```powershell
.\.venv\Scripts\python -m pip install -c configs/analytics-constraints.txt -c configs/boosting-constraints.txt -e ".[analytics,eda,boosting]"
.\.venv\Scripts\python -m pip check
.\.venv\Scripts\python -m demandos.boosting --root . --config configs/boosting.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,output,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

One global model per library learns across **96 early-prefix-selected series**,
**58 historical origins**, and **154,560 eligible labeled examples**. It uses
**57 predictors**, retaining native categories and horizon, excluding duplicate
category codes, targets and audit fields. Numerical nulls remain missing;
categories use fitting-fold vocabularies with an explicit unknown token.
Retrospective M5 price predictors remain unavailable under Season 8 rules.

Development fits use **133,056 labels observed by 2015-12-06**, with **2,688**
validation targets on December 7–January 3. Early stopping selects **141/131/133**
rounds for LightGBM/XGBoost/CatBoost. Those counts are frozen before evaluation.
Refits at **2016-01-31, 2016-02-28 and 2016-03-27** use respectively **143,808,
149,184 and 154,560** labels, each with `target_date <= origin`.

Measured on identical **13 series × three origins × 28 days**, **1,092 targets
per method**:

| Model | MAE | Pooled WAPE | Macro RMSSE |
| --- | ---: | ---: | ---: |
| LightGBM | 12.898 | 44.695% | 0.725 |
| XGBoost | 13.561 | 46.994% | 0.823 |
| CatBoost | 14.516 | 50.301% | 0.748 |
| Seasonal naive | 13.132 | 45.506% | 0.789 |
| Moving average | 13.076 | 45.313% | 0.742 |
| TSB | 13.015 | 45.102% | 0.729 |
| Naive | 18.780 | 65.080% | 0.931 |

Macro RMSSE uses **38 valid series-origin cases**, with one undefined zero-scale
case. All **273 forecast cases** succeeded, producing **7,644 forecast rows**;
all **169 global/local fitting attempts** succeeded. Five recorded LightGBM
warning entries report no further positive-gain splits; no errors or fallbacks
were hidden. Save/reload checks pass for all 13 saved global models, including
development and the ablation. Real-data future perturbations pass for all nine
library/origin combinations. Raw/interim/processed fingerprints and warehouse
totals remain unchanged. **93 Python tests, 36 fresh real-data dbt tests and
dependency consistency checks passed**. The final verified experiment took **126.80s**
with two fitting threads; the historical pool DataFrame occupied **103.1 MB**,
which is not a bound on total process memory.

LightGBM improves pooled WAPE by **0.812 percentage points** against seasonal
naive, but loses at the January origin, on the 15–28-day band and in some
segments. Recent sales levels and item identity rank highly in tree importance;
validation permutation checks also highlight recent sales history. Removing
event features at the same 141 rounds improves internal-validation MAE from
**3.901 to 3.866**; it does not change the primary evaluation specification.
Importance is predictive, not causal.

This is a small **known Season 7 benchmark**, not a pristine test, a full-panel
accuracy estimate, official M5 WRMSSE, or evidence of inventory savings. Only
**7 of the 13 evaluation series** occur in the independent training panel;
unseen identities use the unknown-category policy. Global fitting uses a growing
sample of labels from 2014 onward; statistical baselines use their full available
history. Earlier outcomes may enter later refits once observed, without selecting
parameters from their scores.

Ignored outputs: `data/boosting/global_h28/`, containing separate feature/label
Parquet files, native model files and metadata, forecasts, aligned metrics,
coverage, eligibility audits, importance, validation ablation, two figures,
`summary.json`, `experiment_report.md` and the generated-file manifest.
See [global model contracts and reproducibility](docs/boosting.md) for API use,
exact cutoffs, resource caps, artifact details and limitations. Horizon-specific
models, alternative pooling and model-family ensembles remain Season 10 work.

See [architecture](docs/architecture.md) and the
[foundation decision](docs/decisions/0001-project-foundation.md) for scope and
design rationale.
