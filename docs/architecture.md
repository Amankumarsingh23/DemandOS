# DemandOS architecture

## Implemented scope: Seasons 1–11

The installable Python package includes validated M5 ingestion and a local
DuckDB/dbt analytics warehouse. Season 2 streams CSV sales into long-form
Parquet; Season 3 queries that output through views and builds reusable
dimensions and aggregate marts. Season 4 reduces those relations into descriptive
statistics, temporal summaries, product ABC/weekly XYZ labels, store-item sales
occurrence regimes, actual-data figures, and a reproducible business report.
Season 5 adds small-cohort mathematical diagnostics and forecast-origin temporal
contracts, with training-only segmentation and preprocessing demonstrations.
Season 6 adds twelve observed-sales point-forecast baselines and a bounded,
fixed-origin holdout experiment. Season 7 adds rolling-origin execution,
expanding/sliding windows, explicit coverage, paired comparisons and train-only
M5-style RMSSE scales. Season 8 adds origin-level sales reductions, direct-horizon
calendar expansion, training-only category state and physically separate labels.
Season 9 fits three pooled, horizon-aware gradient-boosted models with bounded
historical examples, chronological early stopping, origin-specific label
eligibility and native serialization. Season 10 compares pooled, recursive,
direct horizon-specific and independently wrapped vector-output LightGBM, with
an earlier-validation-selected ensemble and audited computational costs.
Season 11 adds observed-sales intermittency diagnostics, global hurdle means,
explicit cold-history simulations and supported priors, plus frozen regime
routes/blends. Inventory optimization, serving,
dashboards, and monitoring are future work.

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
    `data/diagnostics/train_<cutoff>/`.
13. Reuse training-only cohort selection, clip each history at the fixed origin,
    and fit point-forecast baselines on a common trailing training window.
14. Attach actual targets only after forecasting; export explicit fit outcomes,
    raw/nonnegative predictions, coverage, MAE/WAPE, common-sample comparisons,
    and two figures under ignored `data/forecasts/origin_<cutoff>/`.
15. Freeze an origin-safe cohort at the earliest backtest date, reuse the existing
    forecasting interface at each origin/window, and retain every planned target
    with explicit fit status under ignored `data/backtests/<design>/`.
16. Derive nonseasonal training scales after initial-zero trimming, aggregate
    micro/macro losses with visible denominators, and compare only shared
    successful cases by origin, horizon, segment and aggregation grain.
17. Reduce selected historical windows in DuckDB to one store-item-origin row,
    including a representative all-30,490-series pass at one origin.
18. Expand only the selected cohort across horizons, validate target calendars
    and categorical-fit cutoffs, then write inference/training features with
    supervised labels in separate files under ignored `data/features/`.
19. Select a larger historical panel from an early prefix and reuse feature
    generation across a bounded historical-origin schedule, caching only the
    selected daily series in a temporary DuckDB table.
20. Filter every supervised target by its allowed fitting cutoff; fit global
    LightGBM/XGBoost/CatBoost models and fitting-fold categorical state, using
    a strictly later chronological validation block for early stopping.
21. Freeze iteration counts, refit each model using only labels observed by each
    evaluation origin, then join actuals and train-only scales on exact keys.
22. Publish native models, same-case comparisons, eligibility/invariance audits,
    validation-only importance/ablation and reports under ignored `data/boosting/`.
23. Reuse the same historical panel, feature contracts and chronological
    development block to select pooled, one-step and per-horizon iterations.
24. Fit a complete-target independent vector wrapper and select one convex
    pooled/seasonal-naive weight using only pre-evaluation validation losses.
25. Refit all four architectures at each benchmark origin; recursive forecasts
    maintain isolated simulated histories and recompute all sales-history inputs.
26. Publish native children, vector/ensemble state, same-case metrics, resource
    comparisons and actual-versus-simulated traces under ignored `data/strategies/`.
27. Compute observed-prefix ADI/CV² and cold-history statuses without borrowing
    full-history segmentation or fabricating missing observations.
28. Fit binary occurrence and positive-only squared-error size models, then
    freeze December-selected regime routes and conservative convex blends.
29. Reduce pre-origin category priors in SQL with leave-one-item-out exclusion,
    and evaluate actual sales separately for continuity, sparse and cold cohorts.
30. Publish support/probability/route evidence and invariance reports under
    ignored `data/intermittent/`, preserving source fingerprints and warehouse sums.

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

Season 6's `forecasting.models` accepts one ordered complete daily Series and
returns an explicit success/skip/failure result. Maintained statsmodels fits
provide smoothing and ARIMA estimation; small deterministic functions provide
simple and intermittent baselines. The experiment reuses Season 5 extraction
and training-only segmentation, with no ingestion, dbt-model, or EDA changes.
Its full 28-day forecast is generated before holdout observations are attached.
Raw predictions remain available alongside `max(0, raw)` scored predictions.
Fit metadata includes warnings, initialization, parameters, iterations, training
bounds, and runtime. Read-only source access and before/after SHA-256 comparisons
protect raw/interim/processed inputs. Comparisons intersect successful forecast
keys by model set and keep store-item, store, and category scales separate.
No complete fact table is loaded into Python; only the selected series and
reduced training labels are fetched. See [forecasting contracts](forecasting.md).

Season 7's `backtesting` package composes the existing forecasting and temporal
interfaces; no model or warehouse transformation is rewritten. `design` validates
configurable origin/horizon/stride and training bounds. SQL reduces all-store
training labels, while Python receives only the fixed small series cohort and
aggregate histories. Target calendars are frozen and validated before fitting;
fresh model states see only their allowed training prefix. Target vectors and
training scales are cached within each execution, with no reuse of model state.
`metrics` groups point losses and series-origin metrics, keeping undefined WAPE
and RMSSE explicit. Successful intersections include window, series, origin,
target and horizon, so overlapping windows cannot accidentally merge forecasts.
Window sensitivity likewise matches successes across both windows.

Small tested hierarchy and verified-as-of historical-revenue utilities are
building blocks, not an official M5 competition scorer. Real benchmark weighting
remains disabled because price availability has not been established. Manifest
publication replaces generated files and removes only obsolete files previously
tracked by that output manifest. Raw/interim/processed fingerprints are compared
before and after execution. See [backtesting contracts](backtesting.md).

Season 8's `features` package uses the same warehouse views, Season 5 availability
guards, Season 7 origin definitions and manifest publication. Bounded SQL
aggregation validates daily history counts before calculating origin-relative
lags and complete-window rolling statistics. The large fact relation is never
cross-joined to horizons. A compact origin result is materialized once; only a
bounded selected cohort is joined to target calendars and expanded in Python/
DuckDB. Arrow schemas declare every output type, with a dictionary recording
source, grain, role, availability and missingness policy for every column.

Mapping/selection uses a cutoff before every training/evaluation origin.
Categorical state is stable, serializable and target-free; unknown values map
to zero. Predictive columns are allowlisted and exclude audit dates, segments
and labels. The inference API never joins target actuals. Supervised labels need
an explicit eligible cutoff and remain in a separate result/table. Historical
price values require explicit observed/available dates plus a unique join to
the weekly price dimension; default M5 prices remain null rather than inferred
known. Public date/event/SNAP availability is a stated assumption, not verified
announcement metadata. Full-vector future perturbation and prefix-only checks
run against the real selected cohort. See [feature contracts](features.md).

Season 9's `boosting` package composes these feature and evaluation interfaces
without changing earlier pipelines. Historical examples retain their own source
dates; `target_date <= fitting_cutoff` is independently enforced when selecting
labels. Development validation is chronological by target availability, followed
by three later refits with frozen iteration counts. A previously fitted model
may predict later origins, but cannot predict an origin earlier than its fitting
cutoff. The benchmark deliberately refits at each origin so newly observed labels
can enter the expanding historical sample.

Model matrices use an explicit feature allowlist and native categories, omitting
duplicate category codes and all labels/audit fields. The saved vocabulary is
learned only from eligible fitting rows, independently of Season 8's earlier
target-free feature mappings. Native model files and JSON metadata preserve
predictor order, categories, cutoffs, parameters and warnings. Raw forecasts and
nonnegative scored forecasts remain distinct. No full-panel daily matrix is
loaded into Python: selected-series history is cached and aggregate features
expand only across the capped panel/origin/horizon grid. Comparisons reuse exact
common-case metrics and full-prefix training-only scales. Known prior benchmark
outcomes, limited sampling and default price exclusion restrict interpretation.
See [global forecasting contracts](boosting.md).

Season 10's `multihorizon` package adds architecture composition while leaving
the Season 9 native estimator and source feature SQL unchanged. `vector` packs
one origin-history row plus its 28 scheduled calendars and validates a separate
complete observed target vector. `models` fits 28 independent children for direct
and vector strategies; the latter changes sample eligibility and the API, not
LightGBM's scalar objective. `recursive` rebuilds history features in floating
point from observations through issuance plus prior predictions. Internal virtual
origins advance, while canonical forecast origins remain fixed and traces keep
actual-source dates distinct from simulated dates. No future actuals enter the
recursive API. Price state remains frozen at issuance.

`ensemble` validates exact component keys and selects a small convex weight grid
on December validation only. `pipeline` freezes all choices before the three
later refits, verifies all architectures after future perturbations and reloads,
and composes unchanged Season 7 metrics. Canonical prediction aliases and status
fields accompany the earlier evaluation schema. Publication reuses the guarded
manifest workflow, and source fingerprints/warehouse totals remain hard gates.
See [multi-horizon contracts and reproducibility](multihorizon.md).

Season 11's `intermittent` package composes the same feature, native model and
metric contracts. It specializes the native binary objective for occurrence and
reuses squared-error regression on positive-only labels for conditional means.
Historical diagnostics and prior construction filter at issuance independently
of model-fitting label eligibility. Cold simulation item IDs never enter global
training; visible-history restrictions preserve actual values and explicit
missingness. No-history cases retain unavailable ML status, with supported prior
forecasts where possible. SQL prior summaries exclude the target item in every
store, and unsupported priors never become arbitrary global means.

The policy is a few fixed regime rules and blend candidates selected on earlier
validation losses, with minimum sample thresholds and explicit defaults. Actuals
are joined only for offline scoring, cold RMSSE scales use visible history only,
and separate cohorts/common-key comparisons prevent misleading rankings. Real
future perturbations regenerate features, diagnoses and priors before comparing
loaded-model and frozen-policy outputs. No inventory/availability ground truth
is introduced. See [sparse sales and cold-start contracts](intermittent.md).

Season 12's `neural` package reuses bounded cohort selection, warehouse projections,
calendar assumptions and exact-grain metric helpers. Its indexed sequence builder
separates historical predictors, scheduled future covariates and complete target
vectors. Global preprocessing is fitted only on eligible windows; each input scale
uses its own observed prefix. Masks distinguish padding from measured zero sales.
Static vocabularies reserve unknown IDs without learning evaluation categories.

Genuine PyTorch LSTM and GRU models use the same embedding and direct-head design.
December chronological validation selects epochs; later refits consume only full
vectors ending on/before issuance. Saved state dictionaries and JSON preprocessing
are reloaded for forecast checks. Mutation/deletion of future sales and changes to
prices/labels must preserve regenerated tensors and predictions at all origins.
The existing LightGBM/TSB/seasonal-naive implementations supply matched references.
Source hashes and warehouse totals gate manifest-based artifact publication.
See [neural contracts and reproducibility](neural.md).

Season 13's `nbeats` package composes the same daily panel and temporal validators
with a target-only adapter: sales, valid-history mask and an own-origin scale.
Generic dense blocks learn backcast/forecast projections; structured stacks use
fixed polynomial and weekly Fourier bases. Every block subtracts its backcast
from masked history and contributes additively to the future forecast. Interpretable
weights are tied within each stack; generic blocks are independent by default.

Forecast-only normalized MSE and December-selected epochs precede origin refits.
Raw block contributions reconcile with total forecasts; clipping adjustments
reconcile nonnegative scored forecasts separately. Cached Season 12 references
must match source hashes, configs, freshly selected cohorts, fit cutoffs, actuals
and RMSSE scales. Future mutation/removal and serialization checks, followed by
source/warehouse checks, gate manifest-based publication. Nothing injects future
covariates or turns model-attributed components into verified causes. See
[N-BEATS contracts and reproducibility](nbeats.md).

Season 14's `tft` package reuses Season 12's exact historical/future/static
sequence contract and train-only preprocessor. A static VSN feeds four context
GRNs for temporal selection, initial hidden/cell state and enrichment. Separate
past/future VSNs transform and select variables before an LSTM encoder-decoder.
Gated skips, enrichment and shared-value/averaged-head attention fuse temporal
information before a linear point head. Packed recurrence and attention masks
exclude padding independently of future-data availability checks.

Point MSE and December-selected epochs precede the origin-specific refits. TFT
diagnostics preserve variable axes and key dates, with normalized selection/
attention probabilities and exactly zero forbidden weights. Frozen-state future
sales mutation/removal must preserve inputs, forecasts and interpretations.
Reference predictions are checked against Seasons 12/13 provenance, actuals,
scales and shared keys before ranking. No existing model interface or permanent
warehouse table changes. See [TFT contracts and reproducibility](tft.md).

Season 15's `reconciliation` package creates deterministic sparse matrices and
maps for all 12 crossed M5 groupings. Full membership is checked independently
against warehouse aggregates; bounded solvers operate only on an explicitly
named cohort. Existing interpretable N-BEATS checkpoints supply bottoms and
direct observed-history seasonal naive supplies independent aggregate bases.
Top-down/middle-out shares use only origin-prefix sales; historical per-origin
refits generate paired OOS errors before the frozen covariance cutoff. Rank,
support and conditioning gates label covariance fallbacks explicitly. QR and
NNLS produce bottom estimates, followed by sparse aggregation for exact
coherence. Real future-sales/price perturbation and source/warehouse hash checks
gate manifest publication. No earlier pipeline, permanent warehouse relation
or model implementation changes. See [reconciliation contracts](reconciliation.md).

Season 16's `quantiles` package preserves existing model interfaces and adds
native alpha-specific LightGBM objectives plus a generic N-BEATS-based residual
network with a monotone multi-quantile head. Shared contracts validate wide keys,
fitting cutoffs and ordered published thresholds. Shared losses retain raw and
repaired diagnostic views, while exact common-case tables distinguish continuity
from prefix-sparse stress. Frozen January weights, origin-only history/scales
and nine real mutation/deletion/price gates enforce temporal isolation. Source
hashes and warehouse totals gate ignored manifest publication. See [quantile
contracts and limitations](quantiles.md).

Season 17's `conformal` package separates historical model fitting, held-out
forecast ledgers, maturity-aware score selection and interval evaluation.
September validation and an October refit precede all calibration targets;
the same frozen predictors remain in use through evaluation. Core order
statistics/support handling feed split, rolling, weighted and trajectory
thresholds. A date-driven ACI queue consumes only available outcomes before
issuing new intervals. Raw quantiles, score IDs, selected windows and alpha
updates remain auditable. Independent predictor and calibrator perturbations,
source hashes and warehouse totals gate ignored manifest publication. See
[conformal design and empirical limits](conformal.md).

Season 18's `inventory` package consumes source-bound point/quantile/interval
artifacts plus bounded read-only warehouse history. Core discrete economics,
path generation, planning and outcome evaluation have separate APIs. No future
outcomes enter planning. Historical weekday blocks and explicitly assumed
quantile/rank paths generate lead/protection CDFs; conformal width is only a
heuristic dispersion sensitivity. Stock targets and receipt-constrained one-order
execution remain separate, with exact matched cases and explicit hypothetical
state/cost/lead settings. Future mutation/removal, source/forecast/checkpoint
hashes and warehouse totals gate ignored manifest publication. No permanent
warehouse, earlier-model or model-training changes; see [inventory decision
assumptions and results](inventory_decisions.md).

Season 19's `simulation` package adds an event-ordered daily state machine,
immutable observable-state/policy contracts, an order-ID delivery queue and
FIFO backlog cohorts. Supplier streams use calendar issue-day keys rather
than order-count PRNG state. Streaming matched-case Parquet groups keep memory
bounded; cost/service and paired series-origin metrics are independent of policy
logic. Engine conservation plus independent persisted-output SQL, origin-prefix
regeneration and pre-mutation-day state/action tests gate generated publication.
Real 28-day paths, bootstrap sensitivities and shared-tail warm-up runs have
separate identifiers. No earlier model/warehouse or Season 18 interface changes.
See [sequential simulation contracts](inventory_simulation.md).

Season 20's `optimization` package adds sparse LP/MILP allocation with separate
scenario-mismatch and heuristic-benefit objectives. The original Season 19
timeline yields each series' post-demand observation; a coordinator collects
all requests before allocating and committing one order vector. Budget and
supplier ledgers persist across the horizon, while procurement-flow and store
order-authorization quotas have explicit epoch scope. Frozen origin inputs,
candidate sorting, verified integer orders, LP-only dual diagnostics and
independent persisted resource/engine checks gate publication. Source/model
fingerprints and warehouse totals remain protected. See [shared optimization
models and limitations](optimization.md).
