# Season 3 warehouse

## Install, refresh, and verify

Run from the repository root after completing Season 2 ingestion:

```powershell
.\.venv\Scripts\python -m pip install -c configs/analytics-constraints.txt -e ".[analytics]"
.\.venv\Scripts\python -m demandos.warehouse build
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m demandos.warehouse examples
.\.venv\Scripts\python -m demandos.warehouse check
.\.venv\Scripts\python -m demandos.warehouse docs
```

On macOS/Linux substitute `.venv/bin/python`. The Python wrapper invokes dbt's
documented Python API in a separate process, so Windows executable launchers are
not required. Use `--root "C:\path with spaces\DemandOS"` when outside the repo.
Missing Season 2 Parquet is a hard error; no data is downloaded or fabricated.

`build` executes **dbt build**, then source/fact/mart verification, all six SQL
examples, timing measurements, and EXPLAIN inspection. It writes
`data/warehouse/analytics_quality_report.json` only after successful verification.
`check` reruns the SQL verification against an existing warehouse and requires
successful tests in the saved build record, matching the source SHA-256 and dbt
project fingerprint. Generating docs cannot replace that record. An unsuccessful
build or changed source/project requires another `build`. `examples` prints the
teaching query results; `docs` runs **dbt docs generate** for local lineage/docs.
Run `build` again after changing source data or SQL, rather than relying on an
old report. Do not run concurrent builds against the same warehouse file.

For direct dbt use in PowerShell:

```powershell
$env:DEMANDOS_PARQUET_PATH = (Resolve-Path data/processed/daily_sales.parquet).Path.Replace('\', '/')
$env:DEMANDOS_WAREHOUSE_PATH = ((Get-Location).Path.Replace('\', '/') + '/data/warehouse/demandos.duckdb')
$env:DEMANDOS_SPILL_PATH = ((Get-Location).Path.Replace('\', '/') + '/data/warehouse/spill')
.\.venv\Scripts\python -c "from dbt.cli.main import dbtRunner; r = dbtRunner().invoke(['build', '--project-dir', 'dbt', '--profiles-dir', 'dbt']); raise SystemExit(0 if r.success else 1)"
```

Create `data/warehouse/` first if not using the wrapper. The committed
`dbt/profiles.yml` contains portable defaults and environment variables, with no
credentials or machine-specific paths. Set `DEMANDOS_MEMORY_LIMIT` to adjust the
default 2GB buffer-manager budget. dbt runs one model/test at a time; DuckDB uses
two threads internally. Process memory can exceed its buffer-manager budget.

## Grains and materializations

All relations are in DuckDB schema `main`.

| Relation | Grain / uniqueness key | Materialization |
| --- | --- | --- |
| `season2.daily_sales` | `(store_id, item_id, date)` | External Parquet source |
| `stg_daily_sales` | `(store_id, item_id, date)`; `d` maps uniquely to date | View |
| `dim_product` | `item_id` | Table |
| `dim_store` | `store_id` | Table |
| `dim_calendar` | `date`; alternate key `d` | Table |
| `dim_price_week` | `(store_id, item_id, wm_yr_wk)` | Table |
| `fact_daily_sales` | `(store_id, item_id, date)` | View |
| `mart_store_daily` | `(store_id, date)` | Table |
| `mart_category_daily` | `(cat_id, date)` across stores | Table |
| `mart_product_daily` | `(item_id, date)` across stores | Table |
| `mart_store_performance` | `store_id` for full observed history | Table |

Lineage: external source → staging → dimensions/fact → marts. Dimensions are
derived only from authoritative columns in the processed source. No brands,
costs, margins, inventory quantities, or availability labels are invented.
Calendar coverage is limited to observed sales days. Weekly prices include only
known prices represented in those days, not future weeks in the raw source.
Missing weekly prices have no price-dimension row. Join weekly prices using
**store, item, and week together**, and use a left join when preserving facts.

Fact rows keep validated hierarchy fields without rejoining dimensions. This
avoids join fan-out and repeated dimension joins on 58M rows. dbt validates
dimension uniqueness, relationships, and metadata consistency. Teaching queries
demonstrate safe joins separately.

## Metric contracts

- `units_sold`: observed units, not uncensored customer demand.
- `sell_price`: historical M5 weekly selling price, cast to `DECIMAL(12,4)`.
  Source tests reject meaningful precision loss, nonfinite, or nonpositive prices.
- Fact `observed_revenue`: zero for zero units, even if price is unknown;
  `units_sold * sell_price` for known prices; NULL for positive units with unknown
  price. This is recorded-price-based sales revenue, not verified accounting
  revenue or profit. No price is imputed.
- Mart `known_revenue`: sum of known fact revenue. Unknown positive-sale revenue
  is excluded. If every contributing fact revenue is unknown, the sum is NULL.
- Mart `revenue`: equals `known_revenue` only when `unknown_revenue_count = 0`;
  otherwise NULL. Never interpret partial known revenue as a complete total.
- `active_selling_items`: distinct items with positive recorded sales in that
  mart's group. It does **not** measure assortment, stock, or product availability.
  It is not additive over time or across stores/categories.
- `selling_store_count`: distinct positive-sales stores in each product/day.
- `zero_sales_count`: zero-unit observations, not a count of stockouts.
- `observation_count`, `priced_observation_count`, `missing_price_count`,
  `price_completeness`, and `unknown_revenue_count` make completeness explicit.

Only additive metrics are reconciled between marts: rows, units, known revenue,
zero counts, missing-price counts, and unknown-revenue counts. Distinct active
item counts and completeness ratios must be recomputed at each aggregation grain.

## Validation and artifacts

dbt tests include required source identifiers, source numerical validity and
price precision, dimension keys/non-null fields, accepted SNAP flags, fact
relationships, metadata consistency, exact daily uniqueness, continuous calendar
coverage, positive known prices, source/fact row-count and unit reconciliation,
mart total reconciliation, and revenue null semantics. The expected current M5
history has **58,327,370 rows and 1,913 days**. The generic daily-grain test groups
by store/item and compares row counts to distinct dates, avoiding a huge output
of daily keys.

Python warehouse tests build the same dbt project on clearly labeled synthetic
Parquet in a temporary directory, overriding only expected rows/days. Fixtures
include positive sales with unknown prices, so NULL revenue behavior is tested
even though that condition does not occur in the current real M5 data. Warehouse
tests skip if optional analytics dependencies are absent; install the extra to
run the full suite.

Generated local artifacts are ignored by Git:

- `data/warehouse/demandos.duckdb`: tables and view definitions.
- `data/warehouse/analytics_quality_report.json`: source SHA-256, actual counts,
  units, known revenue, completeness, mart reconciliations, dbt test outcomes,
  model/query timings, environment, and SQL example results.
- `data/warehouse/dbt_build_record.json`: successful build/test results bound to
  the source and dbt project fingerprints; preserved across docs generation.
- `data/warehouse/representative_explain.txt`: real plan for a filtered store
  aggregation; inspect projection/filter pushdown rather than assuming it.
- `data/warehouse/spill/`: temporary DuckDB spill data.
- `dbt/target/`: manifest, compiled SQL, run results, generated catalog/docs.
- `dbt/logs/`: execution logs.

No source data is loaded in full into Python. dbt performs SQL scans and grouped
computations; Python fetches only summaries and bounded example results. Product
daily aggregates are larger than other marts but still avoid repeating the full
store/item/day grain. Builds replace generated tables; they do not atomically
publish the whole warehouse. A failed build must be corrected and rerun before
the warehouse/report is treated as current. Views refer to the local source
Parquet; rerun `build` after moving the checkout or regenerating that file.

## Verified full-data snapshot

The successful Season 3 run used Python 3.13.15, DuckDB 1.5.6, dbt-core 1.12.6,
and dbt-duckdb 1.11.0 on Windows 11 with 16 logical CPUs. The committed analytics
constraints pin the three tested engine/tool versions, not every transitive
dependency. The quality report records actual runtime versions and settings.

| Model | Actual rows |
| --- | ---: |
| `stg_daily_sales` / `fact_daily_sales` | 58,327,370 each |
| `dim_product` | 3,049 |
| `dim_store` | 10 |
| `dim_calendar` | 1,913 |
| `dim_price_week` | 6,597,201 |
| `mart_store_daily` | 19,130 |
| `mart_category_daily` | 5,739 |
| `mart_product_daily` | 5,832,737 |
| `mart_store_performance` | 10 |

Observed dates cover 2011-01-29 through 2016-04-24. Reconciled totals are
**65,695,409 units** and **187,676,570.0200** in historical selling-price-based
revenue. Missing-price observations number **12,299,413** (21.09%); zero-sales
observations number **39,777,094** (68.20%). There are **zero** positive-sales rows
with unknown revenue. Price coverage by observation count is **78.91%**. This
does not imply physical stock availability or prove zero uncensored demand.

All **36 dbt data tests**, **14 Python tests**, and six teaching SQL files passed.
Python tests include a synthetic dbt build plus deliberate contract violations
for duplicate dimension keys, corrupted mart units, and incorrect revenue nulls.

Measured successful dbt build time was **84.44s**, excluding subsequent Python
verification. Model timings: price dimension **3.48s**, category/day mart
**5.26s**, product/day mart **26.86s**, store/day mart **7.83s**, store performance
**4.04s**. A filtered store aggregation took **0.20s**; the prior-day window example
took **0.0054s**. The saved EXPLAIN plan shows `READ_PARQUET` projecting store and
units columns with the date predicate in the scan. These are local measurements
with two DuckDB threads, one dbt worker, and a 2GB managed buffer budget, not
portable performance guarantees. Repeated queries can benefit from caches.
The figures above are the first uninterrupted successful build snapshot.
Later report runs retain their own actual wall-clock durations, which can
include workstation suspension or long scheduling pauses; compare timing
measurements only under similar execution conditions.

The full-data run exposed two issues fixed before verification: configure the
spill path once at connection creation (not per cursor), and avoid unnecessary
DISTINCT aggregate state at product/day grain. Because daily fact keys are tested,
positive fact-row count equals selling-store count for an item/day; active items
at that grain is a simple 0/1 indicator. This preserves semantics while allowing
the product mart to complete within the chosen memory budget.

## Teaching SQL

`sql/examples/` is separate from transformation models. Each file states a
business question and can execute directly on the built database:

| File | Concepts |
| --- | --- |
| `01_select_where.sql` | SELECT, WHERE, ORDER BY, LIMIT |
| `02_group_having.sql` | GROUP BY, SUM, HAVING |
| `03_joins.sql` | INNER/LEFT JOIN and tested key cardinality |
| `04_null_case.sql` | NULL, CASE, COUNT(*) versus COUNT(column) |
| `05_cte_ranking.sql` | CTEs, DENSE_RANK |
| `06_lag_rolling.sql` | LAG, PARTITION BY, ordered prior-seven-row averages |

The moving average excludes the current row and reports available prior-day
count. Unique continuous daily observations are tested. These are historical
analysis examples, **not** validated forecasting features. Future actual sales
must not enter features at an earlier forecast origin, especially at multi-day
horizons.

Implementation references: [DuckDB adapter configuration and external sources](https://github.com/duckdb/dbt-duckdb),
[dbt Python invocation API](https://docs.getdbt.com/reference/programmatic-invocations),
and [dbt run-result timing artifacts](https://docs.getdbt.com/reference/artifacts/run-results-json).
