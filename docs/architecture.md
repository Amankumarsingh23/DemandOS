# DemandOS architecture

## Implemented scope: Seasons 1–5

The installable Python package includes validated M5 ingestion and a local
DuckDB/dbt analytics warehouse. Season 2 streams CSV sales into long-form
Parquet; Season 3 queries that output through views and builds reusable
dimensions and aggregate marts. Season 4 reduces those relations into descriptive
statistics, temporal summaries, product ABC/weekly XYZ labels, store-item sales
occurrence regimes, actual-data figures, and a reproducible business report.
Season 5 adds small-cohort mathematical diagnostics and forecast-origin temporal
contracts, with training-only segmentation and preprocessing demonstrations.
Forecasting models, inventory optimization, serving, dashboards, and monitoring
are future work.

## Problem and observational unit

The initial forecasting unit is `(store, product, date)`, with observed daily
units sold as the initial target. The implemented M5 validation history contains
58,327,370 observations, with calendar metadata and historical selling prices.

In a simplified setting without replenishment or backorders:

`observed sales = min(customer demand, available stock)`

M5 sales do not supply a complete history of true demand or available inventory.
Do not label observed sales as uncensored demand. Any simulated inventory,
unfulfilled demand, or business costs must be distinguished from measured data.

## Implemented data flow

1. Preserve original inputs in `data/raw/`.
2. Validate and transform data through `data/interim/` to `data/processed/`.
3. Expose `data/processed/daily_sales.parquet` as a dbt external source.
4. Standardize types in `stg_daily_sales`, a view over Parquet.
5. Derive unique product, store, observed calendar, and known weekly-price
   dimensions from the available source fields.
6. Define `fact_daily_sales` as a view with explicit revenue semantics.
7. Persist store/day, category/day, product/day, and store-history marts.
8. Run dbt contracts, source/fact/mart reconciliation, executable teaching SQL,
   and a local analytics-quality report with measured timings and an EXPLAIN plan.
9. Read the verified warehouse without rebuilding it, reduce large facts in SQL,
   and perform bounded Python EDA using `configs/eda.toml`.
10. Export ignored `data/eda/<start>_<end>/` summaries, product-level ABC/XYZ,
    store-item regimes, nine full-history figures, and evidence-backed reports.
11. Construct chronological train/validation/test boundaries, recompute Season 4
    labels only through train_end, and select an illustrative diagnostic cohort.
12. Export guarded correlation/stationarity tests, aligned differences,
    descriptive STL, and audited origin-time demonstrations under ignored
    `data/diagnostics/train_<cutoff>/`; do not fit forecasting models.

The staging and fact views avoid full-size persisted copies of the source.
Dimensions and marts are tables refreshed by `dbt build`. No external database
server, orchestration service, or remote data access is needed. The portable
profile limits DuckDB to two execution threads and 2GB of managed buffer memory;
spill files live under ignored `data/warehouse/`. This is a buffer-manager limit,
not a strict ceiling on total process memory.

See [warehouse operations and contracts](warehouse.md) for exact keys, metric
definitions, commands, limitations, and SQL examples.
See [EDA contracts](eda.md) for analysis windows, compatible classification grains,
cutoffs, revenue/null semantics, observation denominators, and temporal safeguards.

## Evaluation principles

- Use only information available at the forecast origin; avoid future leakage.
- Keep training observations earlier than evaluation periods.
- Evaluate forecast accuracy and uncertainty before relying on predictions.
- Assess inventory costs and service levels under documented assumptions;
  prediction accuracy alone does not establish business value.

## Engineering foundation

Use a `src/` layout so imports resolve through an installed package. Pandas and
PyArrow support ingestion; optional `analytics` dependencies provide DuckDB,
dbt-core, and the DuckDB adapter. The ingestion package is unchanged by Season 3.
Standard-library `unittest` covers installation, ingestion, and a synthetic dbt
warehouse build. dbt data tests separately validate the entire real dataset.
The optional `eda` extra adds Matplotlib and statsmodels. Season 4 leaves ingestion
and dbt transformation code unchanged, uses population statistics, and bounds
exact store-item quantile computation to one store at a time. No full fact table
or product/day matrix is loaded into Python. Temporal components and segmentation
labels are descriptive for their selected window; they are not automatically
valid inputs at an earlier forecast origin.

Season 5 reuses existing dependencies and the read-only warehouse connection.
Broad SQL reductions compute only simple training-period segment statistics;
expensive PACF/ADF/KPSS/STL calculations run on the small selected cohort. Actual
post-origin sales are excluded before feature/scaler computation. The temporal
contract distinguishes target-date historical shifts from origin-relative
features and rejects future sales, future-fitted transforms, centered/descriptive
components and unverified price availability. It checks declared as-of metadata,
not otherwise unavailable real-world reporting delays. See
[time-series contracts and operations](timeseries.md).
