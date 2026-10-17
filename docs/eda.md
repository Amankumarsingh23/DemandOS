# Season 4 EDA and sales segmentation

## Reproduce from the verified warehouse

Run from the repository root:

```powershell
.\.venv\Scripts\python -m pip install -c configs/analytics-constraints.txt -e ".[analytics,eda]"
.\.venv\Scripts\python -m demandos.eda
.\.venv\Scripts\python -m unittest discover -s tests -v
```

On macOS/Linux substitute `.venv/bin/python`. Outside the checkout, pass
`--root "C:\path with spaces\DemandOS"`. The CLI accepts `--config <toml>`,
`--start YYYY-MM-DD`, and `--end YYYY-MM-DD`. Defaults use the entire observed
history. For a training-only description, explicitly set `--end` to the training
cutoff and regenerate all labels. Do not use full-history labels as historical
features in earlier forecasting experiments.

The command verifies the successful Season 3 build record against source and dbt
project fingerprints, opens the warehouse read-only, and creates temporary SQL
views filtered to the selected inclusive window. It does not rebuild dbt models,
rerun CSV ingestion, change raw/processed data, or copy 58M facts into Python.
Missing warehouse/source artifacts or an invalid window are blockers, not reasons
to fabricate observations. Run `python -m demandos.warehouse build` separately
only if the warehouse is absent/stale. Do not run warehouse writes concurrently
with EDA against the same DuckDB file.

## Structure and output grains

`src/demandos/eda/` separates configuration, SQL reductions, statistics and
classification functions, temporal summaries, plotting, pipeline orchestration,
and Markdown reporting. `sql/eda/` contains the reusable store-item and complete
weekly-product reductions. `configs/eda.toml` defines all cutoffs and history
safeguards; the quality report records an exact config snapshot.

Generated files are under `data/eda/<start_YYYYMMDD>_<end_YYYYMMDD>/`, ignored by Git:

| Artifact | Grain / purpose |
| --- | --- |
| `tables/store.parquet` | One store over the selected window |
| `tables/state.parquet` | One state over the selected window |
| `tables/category.parquet` | One category over the selected window |
| `tables/product.parquet` | One product across stores over the selected window |
| `tables/store_item.parquet` | One store/item; statistics, ADI, positive-size CV² and regime |
| `tables/product_segments.parquet` | One item; compatible ABC + weekly XYZ and reasons |
| `tables/abc_summary.parquet` | ABC class totals and contribution shares |
| `tables/abc_xyz_matrix.parquet` | Always nine A/B/C × X/Y/Z cells |
| `tables/abc_xyz_unclassified.parquet` | Unsupported labels accounted for separately |
| `tables/regime_summary.parquet` | Store-item regime counts, units/revenue and shares |
| `tables/regime_subgroups.parquet` | State/category/regime with within-group shares |
| `tables/regime_by_store.parquet` | Store/regime with within-store shares |
| `tables/daily.parquet` | Overall date, sales, exposure and calendar metadata |
| `tables/store_daily.parquet`, `state_daily.parquet`, `category_daily.parquet` | Compact segmented trajectories |
| `tables/weekly.parquet`, `monthly.parquet` | Overall week/month with observed-day exposure |
| `tables/weekday.parquet`, `month_of_year.parquet` | Pooled mean daily units and normalized indices |
| `tables/weekday_by_year.parquet`, `store_weekday.parquet`, `category_weekday.parquet` | Stability/comparison of weekly patterns |
| `tables/event_associations.parquet` | Either-slot event indicator versus no named event |
| `tables/named_event_associations.parquet` | Primary event name, exposure and weekday-adjusted mean index |
| `tables/selected_series.parquet`, `selected_anomalies.parquet` | Deterministic high/middle/low-volume and regime examples, unchanged rows plus screens |
| `tables/decomposition.parquet` | Overall + category/date additive STL components, if >=14 continuous days |
| `figures/*.png` | Actual-data exportable figures, labeled with window/provenance |
| `business_report.md` | Facts, interpretation, hypotheses, caveats and figure links |
| `quality_report.json` | Exact source/config provenance, results, reconciliations, versions and measured runtime |

Existing product/day marts are reused in SQL; they are not fetched as a large
Python matrix or needlessly exported again. Temporary reductions do not become
new warehouse models. Previous output remains intact if computation fails before
publication; final files are replaced individually, not as an atomic multi-file
transaction. Treat a run as complete only when its report has `status=passed`.

## Statistical and revenue contracts

The atomic observations are recorded store/item/day sales. Missing observations
are not imputed. Required Season 3 contracts establish a continuous, unique daily
grain. At store/state/category/product levels, distribution statistics summarize
**daily aggregated units**; at store-item level they summarize atomic daily units.
Mean, exact interpolated median/quantiles (10%, 25%, 50%, 75%, 90%, 95%), population
variance, population standard deviation, positive-day counts, and CV are recorded.
CV is undefined for zero mean. Missing values remain distinct from measured zeros.

`zero_sales_count / observation_count` at an aggregated grain refers to atomic
zero-sales observations, not the frequency of zero aggregate daily totals.
`positive_days` in aggregate summaries counts days with a positive aggregate.
`units_per_observed_series_day` divides recorded units by actual source row count,
not verified in-stock exposure. Product-store volume ranks are computed within
stores and within items, with deterministic IDs in exported row order.

Revenue follows Season 3: zero units imply zero recorded revenue even if price
is missing; positive units with missing price imply unknown revenue. Summaries
preserve unknown-revenue counts. ABC excludes products with any unknown positive-
sales revenue from its eligible ranking, rather than pretending a partial value
is complete. No selling price is filled with zero. Historical M5 prices are in
US-dollar context; these values are not profit or verified company financial
outcomes. There is no evidence for inventory turnover, fill rate, or unmet demand.

## Intermittency: store-item grain

- Fixed-window ADI = observed daily periods / strictly positive-sales periods.
- Positive-size CV² = population variance of strictly positive daily quantities
  divided by their squared mean. Zero days do **not** enter this CV².
- Defaults: ADI boundary **1.32**, positive-size CV² boundary **0.49**.
- Smooth: below both boundaries. Intermittent: ADI at/above its boundary only.
  Erratic: CV² at/above its boundary only. Lumpy: at/above both boundaries.
- All-zero, no-observation or fewer than two positive-day histories remain
  unclassified, with explicit reason. Undefined ADI/CV² is never forced to zero.

These conventional thresholds are configurable heuristics, not universal
forecastability laws. Full-window zeros remain in the primary ADI denominator.
`first_positive_date`, `leading_zero_days`, `periods_from_first_positive`, and a
secondary regime computed from that first positive date provide sensitivity
analysis. They are **not** verified launch or availability measurements. Internal
zero stretches may also reflect unobserved supply conditions.

## ABC and XYZ: matching product-across-stores grain

ABC orders eligible known-price revenue descending, with `item_id` ascending to
break equal revenues deterministically. It uses the cumulative share **before**
each item: below 80% → A; below 95% → B; otherwise C. A threshold-crossing item
stays in the earlier class. Equal-revenue items can straddle a boundary, ordered
by ID. Zero-revenue items are C only if total eligible revenue is positive; an
all-zero revenue population remains unclassified. Revenue shares and cumulative
shares use exact Decimal sums before conversion to displayable ratios.

XYZ first sums units across stores at item/date grain, then aggregates into
complete seven-day Walmart weeks using `wm_yr_wk`. Weeks are Saturday–Friday.
Partial weeks at either window edge are excluded. Genuine zero weeks are retained;
missing weeks are never silently filled. Population weekly CV is std/mean.
Defaults: X if CV <=0.5, Y if 0.5<CV<=1.0, Z if CV>1.0. Unsupported cases retain
reasons: no complete weeks, all-zero weekly mean, fewer than eight complete weeks,
fewer than two positive weeks, or mean weekly units below 1.0.

ABC and XYZ are joined **one-to-one on item_id**, requiring equal product coverage.
The nine-cell matrix uses full-population units and known revenue as contribution
denominators; any unsupported product is reconciled in a separate bucket. ABC's
ranking denominator is its eligible known-revenue population, which may differ
from that full population for incomplete-revenue windows. These descriptive labels
must be recalculated at each training cutoff before forecasting use.

## Temporal patterns, decomposition, and outlier screening

Overall, store, state, and category daily trajectories are reduced from the
existing marts. Overall weekly/monthly outputs include observed days and mean
daily units, so incomplete exposures are explicit. Pooled weekday/month-of-year
indices divide group mean daily units by overall mean daily units. Year-specific
weekday indices use that year's mean; store/category indices use their group's
mean. These quantities can mix calendar patterns, trend, assortment and availability.

An event day means either calendar event slot is non-null. Named-event statistics
use only `event_name_1`; the second slot is not exploded into duplicate source
rows. Weekday-adjusted event comparisons divide observed units by the pooled
same-weekday mean. They are associations without trend/promotion adjustment.
Calendar labels are not measured promotional treatment or proof of causation.

Selected series include deterministic highest-, middle-, lowest-positive-volume
store-items and leading-volume representatives of available unusual regimes,
deduplicated and capped at six by default. Global IQR fences flag observations
outside Q1−1.5·IQR and Q3+1.5·IQR. The robust temporal screen compares a day against
the previous eight observations of the **same weekday**, requires four prior
observations, and flags absolute deviation >4.5·1.4826·MAD. Zero MAD is unevaluable,
not an automatic anomaly. No source rows are deleted, corrected, or winsorized.

Prior-28-row averages exclude the current observation. This is a descriptive
one-step baseline; multi-day forecasting must respect available information at
the forecast origin. Additive STL examples use period 7 with robust fitting on
overall and category aggregates. Those fits and global IQR thresholds use the
full selected window; centered decomposition components are not valid historical
forecast features. Positive spikes may be genuine sales, not recording errors.

## Validation and measured results

All five main summary grains preserve source row exposure, units, known revenue,
zero counts, missing-price counts and unknown positive-sales revenue counts.
Matrix coverage and revenue/volume shares reconcile, including unsupported labels.
The full-history run also checks the saved Season 3 report: **58,327,370 rows**,
**65,695,409 units**, **187,676,570.02** known revenue. Monetary sums remain Decimal;
shares use floating-point display precision. Numerical definitions are tested
on synthetic fixtures independently of these actual M5 measurements.

Verified full window: **2011-01-29–2016-04-24**, **273 complete weeks** and **one
partial week excluded** from XYZ. ABC counts: A **1,264**, B **923**, C **862**.

| ABC / XYZ | X | Y | Z |
| --- | ---: | ---: | ---: |
| A | 607 | 535 | 122 |
| B | 295 | 410 | 218 |
| C | 114 | 301 | 447 |

All 3,049 products have supported XYZ labels in this particular full window;
shorter windows may not. Store-item regimes: smooth **980**, intermittent
**23,102**, erratic **497**, lumpy **5,911** (30,490 total). There are **23,511**
histories with leading zeros; **1,277** regime labels change under the first-
positive sensitivity calculation. Do not interpret that count as verified launches.

CA_3 records **11,188,180 units**, or **1.918 units per observed series-day**.
Saturday's mean is **41,546.9 units**, index **1.210**, based on 274 observed
Saturdays. The top **610** revenue-eligible products contribute **59.48%** of known
revenue. The generated business report separates these measurements from
interpretations and at least three questions for further investigation.

The first full-data run took **27.54s** locally, including reductions, analyses
and nine figures before final report publication. Exact store-item quantiles are
computed one store at a time; Python receives at most 30,490 summary rows in the
largest output frame. The existing DuckDB connection uses two execution threads
and a 2GB managed buffer limit; this is not a process-memory ceiling. Runtime
depends on hardware, caches, scheduling and window length. Package versions and
the actual runtime of each regenerated run are saved in its report.
The final verified regeneration took **96.71s**; the earlier 27.54s measurement
is retained as a separate observed run, not a performance guarantee.

All **28 Python tests** and a fresh **36 dbt tests** passed. Tests cover statistical
edge cases, sparse/all-zero histories, positive-size CV², deterministic ABC ties
and boundaries, XYZ guards, matching keys/no fan-out, matrix shares, unmodified
outlier inputs, prior-only moving averages, partial-week exclusion, and a complete
synthetic SQL analysis with incomplete revenue and a training cutoff. Optional
warehouse/plotting integration checks are explicitly skipped if their extras are
absent; install both extras for the full suite. Generated tables, figures and
reports are ignored by Git; only small code/configuration/documentation is tracked.

Technical references: [DuckDB population/quantile aggregates](https://duckdb.org/docs/current/sql/functions/aggregates)
and [statsmodels robust STL](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html).
