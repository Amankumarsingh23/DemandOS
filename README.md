# DemandOS

DemandOS is a learning-driven retail demand intelligence project. Its goal is to
turn credible sales forecasts and uncertainty estimates into cost-aware inventory
decisions.

## Current milestone: Season 14 — Temporal Fusion Transformer

This repository contains validated M5 ingestion and a local DuckDB/dbt retail
analytics warehouse, with reproducible retail EDA, figures, and ABC/XYZ and
store-item intermittency classifications, plus time-series diagnostics and
origin-aware temporal utilities. Twelve point-forecast baselines now run on a
small training-selected cohort with explicit coverage, rolling-origin MAE/WAPE
and train-only RMSSE evaluation, plus expanding/sliding window comparisons.
Direct-horizon features now combine origin-safe sales history, target calendars,
verified-as-of historical-price interfaces and training-only categorical mappings.
LightGBM, XGBoost and CatBoost now fit pooled, horizon-aware observed-sales models
with chronological validation and origin-specific refits. A separate experiment
now compares pooled, recursive, direct horizon-specific and independent vector
forecasting, plus an earlier-validation-selected ensemble.
Origin-safe intermittency diagnostics, a global hurdle mean model, cold-history
simulations, supported category priors and frozen segment policies are now
implemented separately. Global PyTorch LSTM and GRU encoders now add masked
56-day sequences and direct 28-day observed-sales forecasts with chronological
validation and train-only preprocessing. Target-only generic and interpretable
N-BEATS now add backcast residuals, additive forecasts and attributed polynomial/
weekly-Fourier components. A compact global point TFT now adds static/past/future
variable selection, gated residual networks, recurrent local processing and
masked shared-value temporal attention. Inventory optimization,
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
exact cutoffs, resource caps, artifact details and limitations. Season 10 extends
this reference without changing the Season 9 experiment or its results.

## Season 10: compare multi-horizon architectures

Use the same tested installation and verified warehouse as Season 9:

```powershell
.\.venv\Scripts\python -m demandos.multihorizon --root . --config configs/multihorizon.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,output,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

The pooled reference shares trees across horizons. Recursive LightGBM fits h=1,
then feeds its own fractional nonnegative predictions into freshly recomputed
lags, rolling statistics, zero/positive rates and level changes. Direct models
fit one global LightGBM per horizon. The **independent multi-output wrapper**
maps one origin row to a 28-value vector through 28 separate LightGBM children;
it requires complete observed target vectors and has **no joint output loss**.
One ensemble experiment selects pooled/seasonal-naive weights from 0/0.5/1 using
the strictly earlier December validation block, then freezes them.

All methods reuse the 96-series historical training panel and 13-series,
three-origin benchmark. Each historical feature respects its own origin, each
fitting label must already be observed by the fitting cutoff, and vector samples
require all 28 labels. Models refit at each evaluation origin with frozen
December-selected iterations. Native category/unknown handling, numerical
missingness and default historical-price exclusion follow Season 9.

Ignored `data/strategies/global_h28/` contains native model states, vector and
ensemble contracts, canonical forecasts, exact common-case metrics, fit/coverage
and computational-cost tables, recursive traces, leakage checks, five figures
and a measured architecture report.

Verified comparison on **1,092 identical targets per method**:

| Strategy | MAE | Pooled WAPE | Macro RMSSE |
| --- | ---: | ---: | ---: |
| Pooled LightGBM | 12.898 | 44.695% | 0.725 |
| Recursive LightGBM | 18.360 | 63.624% | 0.904 |
| Direct, 28 models | 14.167 | 49.095% | 0.837 |
| Independent multi-output, 28 children | 14.187 | 49.164% | 0.838 |
| Validation-selected ensemble | 12.898 | 44.695% | 0.725 |
| Seasonal naive | 13.132 | 45.506% | 0.789 |
| Moving average | 13.076 | 45.313% | 0.742 |

The ensemble selected **100% pooled / 0% seasonal naive**; it adds no accuracy
gain. Pooled forecasts reproduce Season 9 exactly. All **232 native fits** and
**156 local baseline fits** succeeded; **351 forecast cases / 9,828 rows** cover
all nine methods including naive and TSB. RMSSE has 38 valid cases and one
undefined scale. The **214** recorded native warning entries report no further
positive-gain splits; there are no hidden fallbacks or nonfinite forecasts.

Recursive MAE grows from **12.301** on days 1–7 to **21.404** on days 15–28.
Direct models slightly beat pooled on days 8–14 (**12.768 vs 12.815**) but lose
overall and on days 15–28 (**16.950 vs 14.549**). They improve the selected
erratic example but lose on the selected smooth/intermittent cases. Vector and
direct predictions match on days 15–28 because their eligible samples and
selected iterations coincide there. This does not imply jointly learned outputs.

Native model counts per origin are **1 pooled / 1 recursive / 28 direct /
28 vector**. Across three evaluation refits, fitting took **22.78 / 1.18 /
37.19 / 27.51s**, and strategy prediction took **0.32 / 33.08 / 9.57 / 14.26s**
respectively. Total measured execution, including development, model validation,
serialization and invariance checks, took **554.06s**. These are machine-specific
observations, not intrinsic speed ratios. The known 13-series benchmark is not
a pristine test, a full-panel estimate, or official M5 WRMSSE.

All **105 Python tests** (including 12 new synthetic architecture tests) and
**36 fresh real-data dbt tests** passed, with no skips. All 15 architecture/
ensemble-origin future-perturbation checks pass; source fingerprints and
warehouse totals remain unchanged. Serialized native models retain fit-only
categories, cutoff/digest evidence and reload-equivalent predictions.

See [multi-horizon contracts and commands](docs/multihorizon.md)
for precise algorithmic distinctions, label eligibility, serialization and
retrospective-benchmark limitations.

## Season 11: sparse observed sales and cold starts

Use the same tested installation and warehouse:

```powershell
.\.venv\Scripts\python -m demandos.intermittent --root . --config configs/intermittent.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,output,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

Implemented diagnostics calculate origin-safe ADI and positive-size population
CV², occurrence/zero summaries, days since a positive sale, and explicit
all-zero/insufficient-history statuses. Global hurdle forecasts multiply a
binary positive-sales probability by an untransformed positive-label regression
mean. Priors use pre-origin store-category/category observations, exclude the
target item in every store, and expose sample support and unavailable cases.

A small regime router and conservative blend select rules on December
validation, with minimum evidence thresholds and logged defaults. The real
experiment separates **13 continuity series**, **24 sparse stress series** and
**18 history/metadata-restricted cold simulations**. All cold item IDs are absent
from global training; withheld history is never fabricated as observed zeros.
No-history ML cases explicitly skip the established observed-history feature
contract. Other baselines preserve their minimum-history requirements.

M5 has **no verified stockout, inventory, introduction-date or lost-sales ground
truth**. Zero sales, missing prices and intermittency do not establish stockouts.
This implementation predicts recorded sales and does not recover exact true
demand or claim inventory savings. The known January–March comparison is
retrospective; cold histories are simulated restrictions on real observations.

Outputs remain ignored under `data/intermittent/global_h28/`, with models,
probability/size forecasts, origin diagnostics, prior support, policy evidence,
coverage/metrics, timing/warnings, four figures and a measured report. See
[Season 11 contracts and commands](docs/intermittent.md).

On the same **1,092 continuity targets per method**, pooled/hurdle/router/ensemble
WAPE is **44.695% / 46.372% / 46.159% / 44.910%**. Ensemble macro RMSSE is
**0.715** versus pooled **0.725**, with 38 valid cases and one undefined scale;
it does not improve pooled MAE/WAPE. On **1,960 shared sparse-stress targets**,
TSB/hurdle/pooled WAPE is **56.745% / 60.033% / 64.814%**. Cold rankings use a
separate **504-target supported intersection**; no-history priors and unavailable
cases are reported separately rather than mixed into an unfair comparison.

The hurdle occurrence Brier score is **0.163264** on continuity targets,
**0.124317** on sparse targets and **0.215825** on supported cold targets;
calibration remains imperfect. Across all ten methods, **46,200 planned rows**
contain **38,976 successes / 7,224 explicit skips**. All 12 native fits and
serialization checks passed. Source/warehouse integrity and real future
perturbation checks passed; all **118 Python tests** passed without skips.
All **36 fresh real-data dbt tests** passed.
The final measured experiment took **246.12 seconds**. Detailed calibration,
conditional-positive errors, support, timing and limitations are in the report.

## Season 12: global LSTM and GRU forecasts

Install the CPU PyTorch wheel, then the optional neural extra in the existing
analytics/EDA/boosting environment:

```powershell
.\.venv\Scripts\python -m pip install torch -c configs/neural-constraints.txt --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -e ".[analytics,eda,boosting,neural]" -c configs/analytics-constraints.txt -c configs/boosting-constraints.txt -c configs/neural-constraints.txt
.\.venv\Scripts\python -m demandos.neural --root . --config configs/neural.toml
```

Both genuine recurrent encoders use 56-day historical inputs, observed-history
masks, training-only static embeddings and scheduled calendar covariates for
direct 28-day point forecasts. Prices remain unavailable without verified as-of
evidence. Complete-vector fitting cutoffs and December validation prevent future
labels from entering training or epoch selection. Checkpoints persist weights,
normalization and vocabularies. Padding never becomes an observed zero.

The verified real experiment trains on **96 series**, with **4,704 development
windows + 96 validation windows**, then **5,088 / 5,280 / 5,472** complete
vectors at January 31 / February 28 / March 27, 2016. December selects 12 LSTM
and 3 GRU refit epochs. On identical **1,092 observed-sales targets per model**:

| Model | MAE | Pooled WAPE | Macro RMSSE |
|---|---:|---:|---:|
| GRU | 12.892712 | 44.6777% | 0.723991 |
| Pooled LightGBM | 12.897550 | 44.6945% | 0.724525 |
| TSB | 13.015273 | 45.1024% | 0.729239 |
| LSTM | 13.115248 | 45.4489% | 0.732582 |
| Seasonal naive | 13.131868 | 45.5065% | 0.788598 |

GRU's tiny pooled advantage is inconsistent across origins; this previously
examined 13-series benchmark does not establish a general neural-model ranking.
All forecasts succeed; RMSSE has 38 defined cases and one undefined training
scale. These are selected-cohort metrics, not official M5 WRMSSE.

LSTM/GRU have **8,001 / 6,593 parameters**. Local evaluation fits take
**17.06–18.65 / 6.96–7.86 seconds**. Model reloads and future-sales removal,
future-sales/price/label perturbations pass at all three origins. Source hashes
and warehouse totals remain unchanged. Short/cold-history mask checks establish
input support, not validated cold-start accuracy. The initial full experiment
took **373.4 seconds** on two CPU threads; no GPU is required.
All **134 Python tests** passed without skips, and all **36 fresh real-data dbt
tests** passed without warnings or failures.

Ignored outputs under `data/neural/global_h28/` include model/preprocessing
state, loss curves, predictions, matched metrics, coverage, paired losses,
timings, tensor/leakage checks, figures, JSON summary and a Markdown report.
See [neural contracts, measured results and verification commands](docs/neural.md).

## Season 13: target-only N-BEATS

Using the existing PyTorch environment and verified Season 12 reference artifacts:

```powershell
.\.venv\Scripts\python -m demandos.nbeats --root . --config configs/nbeats.toml
```

Both variants consume **56 observed-sales days plus a valid-history mask** and
produce **28 forecasts**, without calendars, prices, SNAP or ID embeddings.
Generic N-BEATS uses three independent dense residual blocks with learned bases;
interpretable N-BEATS uses two shared trend applications followed by two shared
weekly-seasonality applications. Backcasts subtract from masked residuals and
block forecasts add from zero. Raw components are retained before nonnegative
scoring; the attributed components are not causal explanations.

Real M5 training reuses **96 series**, **4,704 development + 96 validation
windows**, and **5,088 / 5,280 / 5,472** eligible vectors at the three evaluation
origins. December selects **two epochs for each variant** within a fixed 15-epoch
budget. Generic/interpretable models have **41,040 / 25,236 parameters**.

| Model | MAE | Pooled WAPE | Macro RMSSE |
|---|---:|---:|---:|
| Interpretable N-BEATS | 12.357747 | 42.8239% | 0.712729 |
| Generic N-BEATS | 12.527406 | 43.4118% | 0.712214 |
| GRU | 12.892712 | 44.6777% | 0.723991 |
| Pooled LightGBM | 12.897550 | 44.6945% | 0.724525 |
| TSB | 13.015273 | 45.1024% | 0.729239 |
| LSTM | 13.115248 | 45.4489% | 0.732582 |
| Seasonal naive | 13.131868 | 45.5065% | 0.788598 |

All seven methods share exactly **1,092 targets** each, with 38 defined RMSSE
cases and one undefined scale. Interpretable has the lowest pooled MAE/WAPE;
generic has the lowest macro RMSSE. LightGBM still performs better in the first
week and on the eight lumpy series. This previously examined 13-series benchmark,
with different feature sets, does not establish universal architecture superiority.

All six real N-BEATS origin checks pass future-sales mutation/removal, price/label
exclusion, checkpoint reload and component reconciliation. Original source
fingerprints and warehouse totals remain unchanged. Initial experiment runtime
was **93.75 seconds** on two CPU threads, reusing checked Season 12 reference
predictions. Generated checkpoints, block/component tables, metrics, coverage,
training curves, five figures and reports remain ignored under
`data/nbeats/global_h28/`. See [N-BEATS contracts and results](docs/nbeats.md).
All **150 Python tests** passed without skips, including 16 new N-BEATS tests;
all **36 fresh real-data dbt tests** passed without warnings or failures.

## Season 14: compact global point TFT

Use the existing PyTorch environment, verified warehouse and Season 12/13
reference artifacts:

```powershell
.\.venv\Scripts\python -m demandos.tft --root . --config configs/tft.toml
```

The TFT combines static embeddings and three Variable Selection Networks,
contextual Gated Residual Networks with GLU/LayerNorm, a static-initialized LSTM
encoder-decoder, static enrichment, interpretable shared-value/averaged-head
attention and gated skip pathways. A linear point head replaces the original
quantile heads; this season implements normalized MSE point forecasting only.

Inputs preserve **past `[B,56,10]`, future-known `[B,28,7]`, static `[B,5]`**;
outputs are **`[B,28]`**. Prices stay unavailable without verified as-of evidence.
Future realized sales/prices never enter the decoder. Training-only vocabularies
and normalization, observed-history masks, packed recurrence and causal/padding
attention masks preserve the temporal contract. Scheduled calendars remain an
explicit known-ahead assumption.

The fixed architecture uses hidden size 32, four attention heads and dropout 0.1.
December validation selects five epochs within an eight-epoch cap. Source data,
checkpoints, predictions, VSN/attention diagnostics, figures and reports remain
ignored under `data/tft/global_h28/`. Diagnostics describe internal model behavior,
not causal effects, stockouts or price elasticity. See
[TFT architecture, commands and measured results](docs/tft.md).

The verified model has **184,335 parameters**, selected at epoch 5 with December
validation MSE **1.550650**. It trains on the same **96 series**, with **4,704
development + 96 validation vectors** and **5,088 / 5,280 / 5,472** eligible
vectors for January 31 / February 28 / March 27, 2016 refits.

| Model | MAE | Pooled WAPE | Macro RMSSE |
|---|---:|---:|---:|
| Interpretable N-BEATS | 12.357747 | 42.8239% | 0.712729 |
| Generic N-BEATS | 12.527406 | 43.4118% | 0.712214 |
| GRU | 12.892712 | 44.6777% | 0.723991 |
| Pooled LightGBM | 12.897550 | 44.6945% | 0.724525 |
| TSB | 13.015273 | 45.1024% | 0.729239 |
| LSTM | 13.115248 | 45.4489% | 0.732582 |
| TFT | 13.297411 | 46.0801% | 0.742327 |
| Seasonal naive | 13.131868 | 45.5065% | 0.788598 |

All eight systems share exactly **1,092 successful targets** each; RMSSE has 38
defined cases and one undefined scale. TFT's pooled MAE/WAPE is worse than the
existing baselines in this limited retrospective experiment. Its small first-week
gain over interpretable N-BEATS does not hold across longer horizons or the
smooth/lumpy groups. No benchmark-based retuning was performed.

Mean VSN weights emphasize category, historical sales and future horizon/weekday;
attention emphasizes historical context. These are internal diagnostics, not
causal explanations. All three real origin checks pass future-sales mutation/
removal, price/label exclusion, checkpoint/preprocessing reload and probability/
mask assertions, with **zero forecast difference**. Source fingerprints and
warehouse totals are unchanged. The initial bounded experiment took **814.48 s**
on two CPU threads, reusing verified prior predictions. See the guide for measured
fit/inference times, case variation, resource limits and interpretation caveats.
All **167 Python tests** passed without skips, including 17 TFT tests. All
**36 fresh real-data dbt tests** passed without warnings or failures. Independent
saved-model verification reproduced all three origins and confirmed all 50
generated artifacts are present.

## Season 15: sparse grouped forecast reconciliation

```powershell
.\.venv\Scripts\python -m demandos.reconciliation --root . --config configs/reconciliation.toml
```

Requires the verified warehouse, Season 13 interpretable N-BEATS artifacts and
the existing neural environment plus the `reconciliation` SciPy extra. The full
12-level M5 membership is verified with a sparse **42,840 × 30,490** summing
matrix and **365,880** nonzeros. The bounded forecasting experiment uses a
clearly named **398-node / 102-bottom cohort**, preserving the exact original
13-series continuity view.

Valid saved N-BEATS checkpoints forecast all cohort bottoms; direct observed-
aggregate seasonal naive supplies independent upper-level forecasts. Bottom-up,
top-down, store-department middle-out, OLS, structural WLS, historical-variance
WLS and nonnegative WLS each cover **33,432 identical cases**. Shrinkage MinT is
implemented, but the actual covariance panel has rank **195/398** (392 paired
rows, 14 historical origins), so its gate explicitly falls back to diagonal
WLS. No real MinT score is claimed.

Bottom-up achieves all-level cohort MAE **8.201057**, pooled WAPE **20.1693%** and
macro RMSSE **0.727991**. The separate 13-bottom view reproduces N-BEATS WAPE
**42.8239%** exactly; these two cohort/grain scores are not directly comparable.
The initial maximum incoherence of **539.320287 units** becomes **zero** for
every reconciled method. The constrained method has zero negative predictions;
signed OLS/WLS outputs remain explicit.

All six future-sales/price perturbation checks pass. Source fingerprints,
reference models and the **58,327,370-row** warehouse remain unchanged. The final
real run took **87.05 s**, with **984.2 MiB** peak process working set. All
**193 Python tests** and **36 fresh real-data dbt tests** pass. Artifacts remain
ignored under `data/reconciliation/m5_cohort_h28/`. See
[reconciliation contracts, level results and reproduction](docs/reconciliation.md)
for historical fitting cutoffs, covariance fallbacks and limitations. Results
are cohort metrics, not official M5 WRMSSE.

## Season 16: observed-sales probabilistic forecasts

```powershell
.\.venv\Scripts\python -m demandos.quantiles --root . --config configs/quantiles.toml
```

Three genuine LightGBM quantile objectives (P10/P50/P90), a **42,384-parameter
N-BEATS-based joint monotone quantile network**, and a trailing empirical
baseline share exact 28-day target keys. December validation selects **68/47/160
tree rounds** and **eight neural epochs**; learned fitting labels then freeze
at **2016-01-31** for all three origins. No evaluation targets tune models or
calibration corrections. Later observed sales update inference histories only.

On the separate **1,092-case original continuity** view:

| Model | Mean pinball | 80% interval coverage | Mean width | Interval score |
|---|---:|---:|---:|---:|
| Quantile LightGBM | 5.238256 | 72.99% | 36.1432 | 71.0812 |
| Joint neural quantiles | 4.344992 | 60.99% | 40.1436 | 68.0127 |
| Historical empirical | 5.031685 | 78.94% | 36.1795 | 72.2143 |

The separately reported **12-series prefix-sparse stress** view has 1,008 cases
per method. Its empirical baseline has the lowest mean pinball. The neural
softplus lower bound excludes exact observed zeros, causing only **37.60%**
stress interval coverage despite ordered finite outputs; this limitation is
explicitly tested and retained. Better pinball does not establish calibrated
intervals. Quantiles concern observed sales, not uncensored true demand.

Raw LightGBM forecasts have 28 crossings; deterministic nonnegative sorting
repairs them without using actual targets. Raw/repaired metrics, inclusive
coverage/tie counts, interval widths/scores and per-origin/horizon/regime results
remain visible. Full coverage totals **6,300 forecasts**. All **216 Python tests**,
**36 fresh dbt tests** and **nine real future-sales/price invariance gates** pass;
source fingerprints and the **58,327,370-row** warehouse remain unchanged.

The real run trained eight development/final models in **177.80 s**, with
**1,275.0 MiB** peak process working set and zero training warnings/skips.
Artifacts remain ignored under `data/quantiles/global_h28/`. The existing
environment/dependencies suffice. See [quantile methods, actual results and
reproduction](docs/quantiles.md), including why marginal quantiles cannot be
summed through Season 15's hierarchy. No conformal or inventory work is included.

See [architecture](docs/architecture.md) and the
[foundation decision](docs/decisions/0001-project-foundation.md) for scope and
design rationale.
