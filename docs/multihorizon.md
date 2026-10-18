# Multi-horizon global forecasting

Season 10 compares four LightGBM architectures and one small ensemble on the
existing known M5 benchmark. It reuses Season 8 features, Season 9 native model
contracts and deterministic training panel, and Season 7 scoring. Earlier
pipelines and dependencies remain unchanged.

## Reproduction

Prepare the existing verified warehouse, then use the Season 9 tested CPU
environment (Python 3.12+, tested 3.13.15):

```powershell
.\.venv\Scripts\python -m pip install -c configs/analytics-constraints.txt -c configs/boosting-constraints.txt -e ".[analytics,eda,boosting]"
.\.venv\Scripts\python -m demandos.multihorizon --root . --config configs/multihorizon.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,output,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

`configs/multihorizon.toml` references the unchanged Season 9 configuration.
The experiment uses LightGBM 4.7.0, squared-error regression, learning rate 0.05,
31 leaves, depth 6, minimum child size 20, L1 0/L2 1, 0.9 row/column sampling,
seed 42 and two fitting threads. Each child has at most 160 rounds and 15-round
validation patience. No random search, test-selected feature family, new model
library or residual correction is introduced. Development plus three evaluation
folds fit exactly **232 native models**, sequentially, within an explicit cap.

## What the architectures learn

Let `X_t` be sales history available after day t, `C_(t+h)` the assumed scheduled
calendar, and `H=28`.

| Architecture | Forecast rule | Native models per origin | Cross-horizon sharing |
| --- | --- | ---: | --- |
| Pooled | `yhat_(t+h|t) = f(X_t, h, C_(t+h))` | 1 | Same trees across horizons |
| Recursive | `yhat_(t+h|t) = g(simulated_history_through_(t+h-1), C_(t+h))` | 1 | Same one-step trees; earlier predictions feed later inputs |
| Direct | `yhat_(t+h|t) = f_h(X_t, C_(t+h))` | 28 | Shared feature code and hyperparameters; separate trees/targets |
| Independent multi-output wrapper | `F(origin_row) = [f_1(...), ..., f_28(...)]` | 28 | Shared complete sample keys and vector API; separate trees/objectives |
| Ensemble | `w*pooled + (1-w)*seasonal_naive` | Reuses pooled | Fixed convex combination after earlier validation |

The vector wrapper is **independent multi-output**, not joint output modeling.
Its one-row-per-origin feature table stores history once and scheduled calendar
features for all 28 days. Its label table has columns `y_01` through `y_28`.
At prediction, it returns an actual `(number_of_origin_rows, 28)` array. Each
independent child projects that horizon's calendar from the vector and uses the
same 57-predictor Season 9 allowlist. It does not inspect other output labels or
other horizons' calendars when fitting a child. No shared vector loss, shared
tree splits, or learned output covariance is claimed.

Direct models can use a partially observed historical origin for horizons whose
targets are already observed. The vector wrapper drops that entire sample
unless all 28 labels are available, and applies the same complete sample set to
every child. No missing-target masks or target imputation are supported. Thus
direct and vector approaches can differ through label eligibility, not through
an invented jointly trained LightGBM objective. Native LightGBM regression's
scalar target contract is documented in its
[regression API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html).

## True recursive feature updating

Only observed history through the issuance origin E enters the recursive API.
It rejects a history argument containing later dates. A separate prefix helper
filters before examining values, so changes to excluded future observations
cannot change the accepted prefix. Daily uniqueness, contiguous history,
nonnegative finite observed integer sales and series isolation are checked.

For step h, the virtual origin is `E+h-1`. Recompute `sales_at_origin`, lags
1/7/14/28/56, 7/14/28-day mean/sum/population-standard-deviation/max/zero-fraction/
positive-count, rolling observation counts, prior-seven-day mean and level/
relative changes from a bounded simulated history. Predict one day, append
**max(0, raw_prediction)**, and repeat. Fractional predictions remain floating
point; they are neither rounded nor cast into the observed-sales integer schema.
Incomplete windows stay null; a zero previous mean leaves relative change null.

Scheduled target calendar rows are fixed at issuance and taken from original
horizon h. Static hierarchy and category state are unchanged. Historical price
state is frozen at issuance, including its knowledge timestamps; no synthetic
future prices arrive. Default retrospective M5 price features therefore remain
null with missingness flags, as in Season 8. These are the only stateful
covariates deliberately held fixed: all sales-history predictors are updated.

The one-step model's internal input uses `horizon=1` and the virtual origin.
Its source-date fields describe the **simulated** sequence, not newly observed
future sales. The trace separately records `actual_source_max=E`, the virtual
origin, `prior_predictions=h-1`, feedback values and rolling statistics. Canonical
output retains **origin_date=E**, horizon h and target date `E+h`. The first
step's feature matrix must match Season 8's original h=1 matrix. Output keys,
finite raw values and the clipping policy are checked before every feedback
step, including step 28. Series histories are separately keyed by store/item.

Feedback can amplify or damp an earlier error; increasing error at every horizon
is not assumed. Unlike direct models, later recursive history inputs contain
predictions while fitting inputs contain observations: this distribution change
is a declared limitation. No holdout actuals are used to repair it.

## Temporal protocol and fair scoring

The training panel remains **96 series** selected through **2014-01-04**.
There are **58** historical fortnightly origins from **2014-01-05** through
**2016-03-13**, and **154,560** eligible pooled rows through March 27. Historical
features respect each example's origin; labels independently require
`target_date <= fitting_cutoff`. The wrapper additionally requires
`origin_date + 28 days <= fitting_cutoff` and complete unique target vectors.

Development fitting labels end **2015-12-06**. The common validation origin is
December 6, with **96 × 28 = 2,688** targets on **December 7–January 3**.
Pooled early stopping sees that full block; recursive early stopping sees its
96 h=1 targets; each direct/vector child sees its own 96 targets. Chronological
validation and eligible sample sets remain explicit rather than pretending
every architecture has the same optimization problem. Each selected round count
is frozen before evaluation. Full 28-day validation scores are also reported
for all four architectures, without selecting architecture on test scores.

The shared historical-origin schedule is fortnightly. Consequently, recursive
h=1 training sees one next-day label per series per sampled origin, not a dense
daily one-step training table. Pooled training shares many more horizon tasks;
direct children each see fewer rows. Different label density and validation
objectives accompany the architecture changes here, so results do not isolate
the feedback mechanism alone. A denser one-step schedule could change results.

Ensemble weights are preregistered as **0, 0.5 and 1** on pooled LightGBM, with
the complement on seasonal naive. Minimum common-block validation MAE selects
one weight, ties preferring the smaller pooled weight. The choice and validation
end are saved before evaluation and never updated from January–March scores.
There are no learned local residual corrections.

| Issuance origin | Evaluation targets |
| --- | --- |
| 2016-01-31 | 2016-02-01 through 2016-02-28 |
| 2016-02-28 | 2016-02-29 through 2016-03-27 |
| 2016-03-27 | 2016-03-28 through 2016-04-24 |

Each fold refits origin-appropriate state from eligible labels with frozen
iterations. Later folds may incorporate earlier outcomes once observed, without
using their scores to select settings. Category vocabularies are learned from
each child's eligible fitting rows only, with an explicit unknown token.
Numerical missingness remains native NaN. Targets, audit dates and segments are
excluded from predictors. Scheduled calendars and immutable hierarchy retain
the stated Season 8 availability assumptions; future price availability remains
unverified and excluded by default.

The same **13 benchmark series × three origins × 28 days** yield **1,092 targets
per method**. Actuals are attached after forecasting by validated one-to-one
keys. Comparisons intersect successful cases before ranking; fanout and missing
successful forecasts are hard failures. MAE, pooled WAPE, per-series WAPE where
defined, macro RMSSE, paired losses, per-horizon/band/origin and regime/ABC-XYZ
metrics reuse Season 7. Training-only scales use full-prefix histories after
first-positive trimming: 38 valid series-origin cases, one undefined case.
Statistical methods retain their original full-prefix inputs. These are not
official M5 WRMSSE scores.

Only seven evaluation identities overlap the independent training panel. These
previously examined Season 7/9 outcomes are a known retrospective benchmark,
not an untouched confirmation period. Small segment cells, correlated targets,
the sampling scheme and censored sales limit interpretation. No causal effect,
business saving or universal winner is claimed.

## Verified real-data findings

All four architectures and the ensemble cover the same 1,092 targets each;
including naive, seasonal naive, moving average and TSB gives **9,828** rows
and **351** complete successful series-origin-method cases. All **232** native
and **156** baseline fits succeed. There are no skips, failures, fallbacks,
nonfinite forecasts or clipped raw evaluation forecasts. Native model metadata
retains **214** warning entries about no further positive-gain splits; no Python
warnings were captured. LightGBM's tree growth may stop before the nominal
budget when no useful split remains.

| Strategy | MAE | Pooled WAPE | Macro RMSSE |
| --- | ---: | ---: | ---: |
| Pooled | 12.898 | 44.695% | 0.725 |
| Recursive | 18.360 | 63.624% | 0.904 |
| Direct | 14.167 | 49.095% | 0.837 |
| Independent multi-output | 14.187 | 49.164% | 0.838 |
| Selected ensemble | 12.898 | 44.695% | 0.725 |
| Seasonal naive | 13.132 | 45.506% | 0.789 |
| Moving average | 13.076 | 45.313% | 0.742 |
| TSB | 13.015 | 45.102% | 0.729 |
| Naive | 18.780 | 65.080% | 0.931 |

The ensemble's internal-validation MAE is **3.901107** at pooled weight 1,
**4.479799** at 0.5 and **5.566220** at 0. It therefore selects **100% pooled**,
and its evaluation forecasts are identical to the pooled reference. This is a
reported negative ensemble result, not a claimed improvement. The original
Season 9 pooled predictions reproduce exactly (maximum difference **0**).

Pooled selects **141** rounds and recursive **45**. Direct children select
**3–158** rounds; vector children **3–158**. The full per-horizon counts and all
native fit metadata are saved. Recursive and direct h=1 use the same rows and
45 rounds; subsequent recursive forecasts differ through feedback. Direct and
vector predictions on horizons **15–28** coincide because complete-vector
eligibility removes no additional rows for those horizons under the fortnightly
schedule, and their chosen iterations also coincide. Horizons 1–14 can use
additional partially observed origins in direct fitting, producing differences.

| Origin | Pooled rows | One-step rows | Complete vector samples |
| --- | ---: | ---: | ---: |
| 2016-01-31 | 143,808 | 5,184 | 5,088 |
| 2016-02-28 | 149,184 | 5,376 | 5,280 |
| 2016-03-27 | 154,560 | 5,568 | 5,472 |

Development has **133,056** pooled labels and **4,704** complete vectors.
The latest partial origin supplies direct labels only where target dates have
arrived; it is excluded entirely from vector training. Early-stopping budgets
remain fixed rather than increased to repair weak evaluation scores.

Horizon-band MAE:

| Strategy | Days 1–7 | Days 8–14 | Days 15–28 |
| --- | ---: | ---: | ---: |
| Pooled | 9.677 | 12.815 | 14.549 |
| Recursive | 12.301 | 18.330 | 21.404 |
| Direct | 10.001 | 12.768 | 16.950 |
| Independent multi-output | 9.896 | 12.953 | 16.950 |
| Seasonal naive | 11.692 | 14.073 | 13.381 |

Direct is slightly better than pooled on days 8–14, worse overall, and worse at
long horizons. Recursive error is substantially larger in later bands in this
run; this observation does not establish monotonic error propagation generally.
Direct improves the selected erratic example versus pooled (MAE **14.775 vs
16.039**) while losing on smooth (**31.905 vs 26.272**) and intermittent
(**18.668 vs 15.515**) examples. Recursive has slightly lower MAE on the eight
selected lumpy cases (**10.047 vs 10.342**) despite poor overall performance.
Tiny segment samples do not establish a general architecture-selection rule.

| Architecture | Native models per origin | Eval refit time, three folds | Eval prediction time, three folds | All four folds' native files |
| --- | ---: | ---: | ---: | ---: |
| Pooled | 1 | 22.78s | 0.32s | 1.59 MB |
| Recursive | 1 | 1.18s | 33.08s | 0.52 MB |
| Direct | 28 | 37.19s | 9.57s | 17.76 MB |
| Independent multi-output | 28 | 27.51s | 14.26s | 18.73 MB |

Development fitting adds **2.16 / 0.12 / 3.35 / 3.28s** in the same order.
These timings include concurrent local workload and do not establish intrinsic
speed ratios. Recursive inference is slower here because it makes 28 sequential
prediction/feature-update calls. Vector assembly adds overhead to independent
children. Model file totals above use decimal MB, exclude metadata/contracts,
and count development plus three evaluation folds. Ensemble mixing reuses
components and is not independent training. Total measured execution is
**554.06s**, through dataset preparation **29.73s**. The historical pool frame
occupies **103,110,972 bytes**, not total process peak memory.

The complete regression suite passes **105 Python tests**, including **12** new
synthetic architecture tests, with no skips or failures. Fresh real-source dbt
validation passes **36/36** tests. All **15** architecture/ensemble-origin
future-perturbation checks pass; source fingerprints and warehouse totals remain
unchanged. Native save/reload probes, per-model cutoff/category/digest checks,
recursive feedback traces and exact Season 7/9 comparison reconciliation provide
separate evidence beyond the small synthetic fixtures.

## APIs, serialization and generated artifacts

`GlobalModel` continues to implement pooled and one-step fit/predict/save/load.
`HorizonModels` groups 28 native children. `IndependentMultiOutput.fit` accepts
wide origin features and separate complete target vectors; `predict_vector`
returns raw/scored arrays and `predict` returns canonical long forecasts.
`VectorLayout` validates pack/unpack schemas and shared origin-history values.

All native models use the existing text format and JSON state. Direct children
are `h01` through `h28`; the vector wrapper also saves `wrapper.json`, declaring
its layout, independence and unsupported partial targets. Every native fit has
an origin-valid save/load probe; complete architecture forecasts are checked
again after reload. No pickle or hidden fallback is used.

Outputs are ignored under **`data/strategies/global_h28/`**:

- `models/development/` and `models/origin_<YYYYMMDD>/` contain all four
  architectures, native model states and vector contracts.
- `ensemble.json` stores the selected weight, complement, candidate grid and
  pre-evaluation validation end.
- `tables/forecasts.parquet` contains store/item/origin/horizon, target date,
  architecture/model IDs, raw/scored predictions, offline actuals and statuses.
  `y_pred` aliases the scored prediction; `y_true` is evaluation-only.
- `fit_status.parquet` records all native samples, cutoffs/digests, rounds,
  warnings, fit/probe times, serialization checks and native/metadata sizes.
  Baseline outcomes have their own table. No failures are silently substituted.
- Common/available metrics, exact membership, paired losses, horizon/band/
  origin/series/segment tables, coverage and training-only scales are Parquet.
- Resource tables distinguish per-origin model counts, fit time, strategy
  inference time, artifact sizes and ensemble mixing time. Ensemble end-to-end
  cost also requires its pooled and seasonal-naive dependencies. Native model
  byte totals exclude metadata/reports; those bytes are separately observable.
- `recursive_trace.parquet`, training eligibility and leakage checks provide
  explicit actual-versus-simulated provenance.
- Five figures show overall horizon error and smooth/intermittent forecast
  paths plus their horizon errors. `architecture_report.md`, `summary.json`
  and the generated-file manifest record actual measured results.

Source and processed SHA-256 hashes and warehouse totals are checked before/
after real execution. Every architecture and the ensemble must retain its
forecasts under future-sales/price perturbation with frozen model state.
Future-label changes must leave eligible rows and complete vectors unchanged.
The pooled reference must reproduce the saved Season 9 forecasts when available.
Tests run separately from the experiment CLI; the final ignored verification
record binds their results to reviewed source/artifact state. A later experiment
removes that stale record through the manifest publisher.

Model count is 1/1/28/28 per origin, not a claim of equal compute: pooled fits
many more rows per model; independent children may be cheaper each, while model
loading, feature assembly and recursive feedback add costs. Fitting is sequential
with capped native threads. DuckDB retains its two-thread/2GB managed buffer
budget, not a hard total process-memory limit. Fixed seeds improve repeatability
without promising bit identity across versions, platforms or thread settings.
