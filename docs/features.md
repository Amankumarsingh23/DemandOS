# Season 8: feature contracts and reproducible generation

## Run and source prerequisites

From the repository root, reuse the existing extras and verified warehouse:

```powershell
.\.venv\Scripts\python -m pip install -c configs/analytics-constraints.txt -e ".[analytics,eda]"
.\.venv\Scripts\python -m demandos.features --root . --config configs/features.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,output,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

Use `.venv/bin/python` on Unix. Required inputs are the unchanged processed
Parquet and the existing verified DuckDB/dbt warehouse. Source/build fingerprints
must match; the CLI never downloads data, rewrites inputs, rebuilds the warehouse,
or fits forecasting models. Raw/interim/processed SHA-256s and warehouse
row/unit/revenue totals are checked before and after generation.

`configs/features.toml` declares lags, windows, training origins, mapping cutoff
and cohort limits. Evaluation origins/horizon are read from the Season 7
`configs/backtesting.toml`; the demonstration requires its 28-day horizon.
Lags/windows are bounded to 365 days; the default selected cohort is capped at
14 series and explicit horizon expansion at 256 (configurable up to 1,024).
There is no full-panel 28-horizon expansion.

Optional commands:

```powershell
.\.venv\Scripts\python -m demandos.features --inference-only
.\.venv\Scripts\python -m demandos.features --cohort data/raw/feature_cohort.csv
.\.venv\Scripts\python -m demandos.features --price-evidence data/raw/historical_price_evidence.parquet
```

`--cohort` accepts CSV/Parquet with exactly `store_id,item_id`, unique non-null
keys and valid metadata/history by the selection cutoff. Missing or oversized
cohorts fail rather than silently shrink. `--inference-only` writes under
`data/features/inference/` and omits training matrices/labels; the reusable
inference API also works with future actual sales completely absent.

## Canonical grain and information boundary

Feature-table key: **store_id, item_id, origin_date, horizon**. Each requested
store-item-origin receives horizons 1..28 exactly once, and target date is
origin plus h calendar days. Origin represents end-of-day after that day's
sales are available. Dates are day labels, not claims about intraday ingestion
latency. Static store/product hierarchy is treated as immutable origin-known
metadata; M5 does not establish actual product-launch or inventory histories.

`sales_at_origin = y_origin`; `origin_lag_1 = y_(origin-1)`, not y_origin.
The 7-day rolling mean is the mean of y_(origin-6)..y_origin. Including the
origin is valid under the declared end-of-day contract. Source-date audit
columns record exact lag dates and maximum sales source date. Missing/gapped,
duplicate, nonfinite, negative or fractional sales in the required daily history
are rejected before feature use. Actual zeros are never treated as missing.

Lags: **1, 7, 14, 28, 56**. Windows: **7, 14, 28**, with mean, sum, population
standard deviation (ddof=0), max, zero-sales fraction and positive-sales count.
Each has an observed-day count. Window statistics require complete windows;
short histories yield null statistics and honest counts, not zero-filled sales.
Recent-level features compare the latest seven days to the previous seven;
relative change is null when the prior mean is zero. In all-zero histories,
means/counts are zero and zero fractions one, while relative zero-denominator
features remain undefined. These are observed-sales features, not true demand.

Target features include Monday=0 weekday, month, weekend, occurrence/count and
Cultural/National/Religious/Sporting flags across both calendar event slots,
state-specific SNAP, and sine/cosine weekday/month encodings. Empty event names
mean no listed event; no synthetic event category is invented. Store state picks
only its CA/TX/WI SNAP flag. Calendar dates, scheduled events and SNAP flags are
assumed known by origin; M5 has no announcement archive verifying that assumption.
Missing or duplicated target calendar matches fail instead of multiplying rows.
The warehouse calendar ends April 24, 2016; later targets need an explicitly
provided validated calendar in the reusable interface, not a silent extension.

Correct h=28 examples: origin lag 7, a rolling mean ending at origin, and target
weekday. Leaking examples: target-relative lag 7 uses y_(origin+21); a rolling
mean ending the day before the target uses future actuals. Unknown full-file
target prices, future-fitted scalers, target encodings or segment labels are
also excluded. Predictive columns are an explicit allowlist; audit fields and
supervised labels never enter it.

## Historical price interface

Price features exist but are **null by default on M5**, since its retrospective
weekly prices have no verified as-of availability metadata. Missingness flags
are true, known-week count zero. Missing prices never become zero. Price records
appearing in the current/future week are not enough to declare origin knowledge.

Optional historical evidence must have exactly:
`store_id,item_id,wm_yr_wk,observed_at,available_at`, with one row per weekly key.
These are caller-supplied assertions of actual historical price observation and
availability, not stamps inferred from the full price file. The pipeline joins
to the unique validated weekly price dimension and checks the observation date's
calendar week. Both dates must be on/before the origin for a value to be used.
Evidence validation can check declared provenance and join integrity; it cannot
independently verify a caller's real-world assertion.

Features: last known non-null historical price; prior observed weekly price;
last/prior minus one with a positive prior denominator; last price divided by
the mean of the last four known weekly prices when all four exist and their
mean is positive. Audit columns retain observation/availability dates, and
missingness is explicit. Prior means the previous observed weekly record, not
necessarily a changed price. Zero denominators remain null. No price reduction
is called a promotion; selling prices are not margins or procurement costs.
No known-future-price scenario is implemented or silently enabled.

## Training-only state and temporal design

Default cohort selection and category mapping stop at **2015-10-10**. Both
precede every generated feature origin. Reused Season 4/5 regime/ABCXYZ rules
select a deterministic diverse 13-store-item cohort, including all four regimes,
an unsupported sparse example and all nine ABC/XYZ cells. Labels are descriptive
selection metadata in `cohort.csv`, never model features. Retrospective recorded
training revenue can guide this illustrative selection, not a prospective claim
about financial reporting availability or a representative population sample.

Sorted, per-column ID vocabularies are fitted only on selected-cohort identifiers
observed on positive-sale dates through October 10. Unknown=0; codes have no
ordinal meaning. No dense one-hot matrix, target encoding, global scaler or
future label statistic is fitted. State is saved as `category_mappings.json`.
Raw categorical IDs are preserved: `store_id,state_id,item_id,dept_id,cat_id`.
Season 9 can use those as categorical columns with its library's declared type
conversion, or the saved code columns explicitly marked categorical. It should
choose the representation, not interpret numeric ID codes as ranked quantities.

| Role | Origins | Last eligible target |
| --- | --- | --- |
| Supervised training | 2015-10-11, 2015-11-08, 2015-12-06, 2016-01-03 | 2016-01-31 |
| Evaluation inference | 2016-01-31, 2016-02-28, 2016-03-27 | 2016-04-24 |

No February–April Season 7 evaluation targets fit mappings, selection or
preprocessing. The training-label join requires an explicit cutoff and rejects
any target beyond January 31. Labels are physically separate from features;
no evaluation labels are exported here. Season 7 remains the source of the
existing evaluation design and scoring interfaces.

Reusable interface (`con` provides feature_sales, feature_keys, feature_calendar,
feature_prices relations with the validated warehouse schemas):

```python
from demandos.features.validation import generate

batch = generate(con, origins, config, saved_mapping, inference_only=True)
X = batch.features[batch.predictive_columns]
assert batch.labels is None

training = generate(con, training_origins, config, saved_mapping,
                    inference_only=False, label_cutoff="2016-01-31")
# Join training.labels on canonical keys to obtain y, not an extra X column.
```

`feature_dictionary.json`/`.md` declare every column's type, source grain, role,
availability and null policy. Origin-level columns are broadcast across horizons;
target-level columns depend on target dates. Arrow schemas retain nullable
integers and date32 types consistently even when every price value is null.

## Verified scope and measured artifacts

At **2016-01-31**, the full pass reduces **1,737,930** bounded observations
(57 days × 30,490 series) to **30,490** unique origin rows. The result DataFrame
uses **15,496,142 bytes**; this is not process peak memory. **6,904** relative
seven-day level changes are null because the prior mean is zero; required lag
and rolling histories are complete. Default price values/audit dates are null
for all series due to missing as-of evidence, not missing raw price rows.

Selected-cohort output has **1,456 training rows plus 1,456 separate labels**
(13×4×28), and **1,092 inference rows** (13×3×28). There are **62 declared
predictive columns**, including raw/code categorical representations. All keys,
28-horizon vectors, target dates, source dates and calendar joins validate.
The unknown item/dept codes occur on **84 inference rows** each, because those
categories were absent from the positive-sale mapping fit, not refitted later.

Real-data checks perturb every post-origin sales value at each evaluation
origin and perturb future-week prices using temporary read-only views. The
entire feature frame remains identical; it also matches a generation with all
post-origin actuals removed. Default real price invariance is an exclusion
check; synthetic explicit-price evidence tests prove non-null as-of behavior,
zero denominators, join cardinality and future-price invariance separately.

Unchanged warehouse totals: **58,327,370 rows**, **65,695,409 units**, historical
recorded-price revenue **187,676,570.0200**. Raw CSVs and intermediate/processed
Parquet fingerprints reconcile before/after. DuckDB uses two threads and a
configured 2GB managed buffer limit (reported as 1.8 GiB) with ignored spill
storage. It computes compact origin features before any horizon expansion;
neither all facts nor all facts×28 enter Pandas. No strict total Python-memory
ceiling is claimed. Summary files record measured runtime and result-frame sizes.
The final verified execution took **76.47s** locally; the full-panel origin
reduction took **3.97s**. All **82 Python tests** and **36 fresh real-data dbt
tests** passed. Independent checks reconciled full/selected keys, all 28 horizons,
saved labels against actual sales, state-specific SNAP and a numerical lag/rolling
example against the warehouse.

Default ignored output: `data/features/direct_h28/`:

- `training_features.parquet`, `training_labels.parquet`, and
  `evaluation_features.parquet` (no labels embedded).
- `origin_level/full_origin.parquet`, compact training/evaluation origin tables,
  and `by_origin/` partition files for selected-cohort horizon rows.
- `cohort.csv`, `sample_features.csv` with actual M5 examples, saved category
  mappings, exhaustive feature dictionary, JSON validation and Markdown report.
- `manifest.json` tracks generated files for clean reruns; unrelated files are
  retained. `test_results.json` records the separately executed suite outcomes;
  the CLI itself never pretends those suites ran on each feature generation.

Synthetic tests cover exact lags/rolling/level changes, short and all-zero
histories, missing dates, unknown categories, series isolation, leap/last-date
boundaries, SNAP, event absence, forbidden future fields/fits, labels changing
without feature changes, inference without future actuals, price provenance,
future prices, null/zero denominators and join fanout. Existing regression and
dbt contracts remain in force. This season introduces no new ML/forecast model,
inventory simulation or serving infrastructure.
