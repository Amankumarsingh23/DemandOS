# Season 16: observed-sales quantiles

`demandos.quantiles` adds three native LightGBM quantile objectives, a joint
N-BEATS-based neural extension and a historical empirical baseline. The existing
ingestion, warehouse, point models and reconciliation interfaces are preserved.
The target is **recorded daily observed sales**, not uncensored latent demand.
Missing price is not verified stockout evidence.

## Meaning of the outputs

A squared-error point predictor targets a conditional mean; an absolute-error
predictor targets a median. A quantile threshold is defined by the inverse CDF,
`Q_tau = inf{y: F(y) >= tau}`. These can differ greatly for zero-inflated counts.
For example, 80% probability of zero and 20% probability of 100 gives mean 20
and median zero.

The output grid is **P10/P50/P90**, with neural shape **[B,28,3]** from historical
**[B,56]** sales/masks and observed-input mean scales. Published wide tables use
unique `(store_id,item_id,origin_date,horizon,target_date,model)` keys and
`q10,q50,q90`, with `0 <= q10 <= q50 <= q90`. Horizon is 1..28 and target date
is origin plus horizon. Quantiles are continuous thresholds; learned outputs
need not be integer counts. Empirical inverse-CDF quantiles here are observed
integer values. No rounding is applied to learned thresholds.

The P10-P90 interval is nominally an 80% central interval. Its observed coverage
need not be 80%, particularly with discrete atoms, misspecification or shifting
sales. Neither one series nor a small retrospective cohort establishes exact
finite-sample conditional calibration.

## Loss, calibration and interval arithmetic

For error `e = y-q`, pinball loss is `max(tau*e, (tau-1)*e)`.

| tau | Actual | Prediction | Loss |
|---:|---:|---:|---:|
| .9 | 30 | 20 | 9 |
| .9 | 10 | 20 | 1 |
| .1 | 30 | 20 | 1 |
| .1 | 10 | 20 | 9 |

P50 pinball is **half MAE**, not MAE. Ties have zero pinball loss. The shared
NumPy implementation is cross-checked against scikit-learn, and differentiable
joint PyTorch loss is checked against the same arithmetic. Mean pinball averages
the three quantiles equally, then the observed target rows equally. Neural
optimization uses own-history-normalized units; reported metrics use original
sales units and therefore differ from normalized training/validation losses.

Cumulative quantile coverage is **`mean(actual <= predicted_quantile)`**.
The non-strict comparison matters: if 50% of observations are zero and P10 is
zero, inclusive P10 coverage is already at least 50%. Such a jump can be
consistent with the inverse-CDF definition. Reports retain success counts,
nominal levels, signed deviations, exact ties and Wilson 95% intervals. Wilson
intervals are illustrative iid-binomial summaries; serial dependence,
overlapping forecast horizons and shared cross-series factors invalidate a
general independent-trials interpretation. They are not simultaneous or
conditional calibration guarantees.

For ordered endpoints `[l,u]`, the alpha=.2 interval score is
`u-l + 10*max(l-y,0) + 10*max(y-u,0)`. Thus `[10,30]` scores 20 when actual is
20, and 70 when actual is 35. Lower score rewards useful narrow intervals while
penalizing missed observations. Width alone or higher coverage alone cannot
establish a better interval. Coverage includes both endpoints; it is not
generally `C90-C10` when endpoints have ties.

Raw forecasts retain all negative values and crossings. Three pairwise crossing
flags and their union are measured before postprocessing. Operational repair
clips negative values at zero and sorts the three values, without consulting
actual outcomes. Raw and repaired pinball/coverage are both scored on every
matched target. Reversed raw P10>P90 endpoints do not define a valid interval:
only their interval metrics are excluded, with explicit denominator counts.
There were no such reversed endpoints in the actual run, although 28 raw
P10>P50 crossings occurred.

## Models and fixed chronological design

The three actual LightGBM models use **`objective="quantile"`** with alpha
**.10/.50/.90**, native categorical vocabularies fitted only on eligible fitting
rows, and native missing-number handling. They reuse the established origin-safe
feature builder. The native model uses **57 predictors**: the earlier 62-column
predictive contract excluding five ordinal `*_code` columns, while retaining
the five original categorical IDs. Labels, audit timestamps, future actual sales
and retrospective future prices are excluded from predictors.

The **generic N-BEATS-based monotone joint-quantile extension** retains the genuine
multi-block backcast/residual chain. Each block forecasts three latent streams;
these accumulate across blocks before a softplus-and-cumulative-sum head gives
ordered P10/P50/P90. This nonlinear final head makes it an extension, rather than
unchanged canonical point N-BEATS or an interpretable trend decomposition.
Latent block contributions sum to the latent total, not directly to final
quantiles. There are **42,384 parameters**, hidden width 64, three generic blocks,
three hidden layers per block and no dropout. Training uses AdamW, deterministic
seed 42, two CPU threads, finite loss/parameter/gradient checks and gradient
clipping. Final-backcast-only parameters without forecast supervision are
explicitly identified by the existing architecture contract.

History scaling uses only each example's observed 56-day history, with a floor
of one and valid-history masks. Missing supported dates are errors; padding is
allowed before known observation support. All-zero or short observed histories
have explicit scale/support semantics. No global future-fitted normalizer or
future labels enter inference.

The **historical empirical baseline** takes inverse-CDF quantiles of each series'
last 56 observed days, including zeros, and repeats them for all 28 horizons.
It records history start/end, observation count, complete/short/all-zero status
and quantile method. No observed history is an explicit unavailable error;
missing daily records are not invented as zeros. Actual data had no unavailable
histories, skips or fallback substitutions.

| Stage | Cutoff and rows |
|---|---|
| Training cohort selection | 96 established series, prefix through 2014-01-04 |
| Development fitting | Labels through 2015-12-06; 133,056 tabular rows or 4,704 complete neural vectors |
| Chronological validation | Origin 2015-12-06, targets through 2016-01-03; 2,688 tabular rows or 96 neural vectors |
| Frozen evaluation fit | Labels through 2016-01-31; 143,808 tabular rows or 5,088 complete neural vectors |
| Evaluation inputs | Current observed histories at each issue origin; learned weights/vocabularies remain frozen |

Validation selected **68/47/160** LightGBM rounds for P10/P50/P90 within the
160-round cap. Their native validation pinball scores were **0.653187 / 1.910232 /
1.426715**. Neural validation selected **epoch 8/8**, with normalized joint
pinball **0.194442**. P90 and the neural model reached their fixed budgets; no
budget expansion or parameter retuning followed evaluation. Six LightGBM fits
and two neural fits produce **eight genuine saved development/final models**.

For pooled rows, eligibility requires `origin+horizon <= fit_cutoff`; neural
eligibility requires `origin+28 <= fit_cutoff`. Selection/epoch decisions end
before the first evaluation origin. Unlike prior rolling point-model refits,
this season deliberately freezes learned fitting labels at January 31 for all
three origins, satisfying the request to exclude benchmark target outcomes from
training. Later observations update input histories and the historical baseline
only. These differences in objective and fitting policy prevent an architecture-
only comparison with prior point results.

| Issue origin | Exact 28-day targets |
|---|---|
| 2016-01-31 | 2016-02-01 through 2016-02-28 |
| 2016-02-28 | 2016-02-29 through 2016-03-27 |
| 2016-03-27 | 2016-03-28 through 2016-04-24 |

The original **13-series continuity** cohort yields **1,092 cases per method**.
A separate **12-series prefix-sparse stress** cohort yields **1,008 cases per
method**. It selects the lowest historical positive-day fractions among the
96 training series through 2014-01-04, excludes continuity IDs and breaks ties
by store/item. All stress series are known training entities evaluated at later
dates; they are not held-out entities. Prefix sparsity does not guarantee future
sparsity: across 36 stress series-origin cases, 32 are intermittent and four
smooth by current trailing-year ADI/CV². Their current zero-day fractions range
from 8.22% to 94.79%. Origin-safe regime diagnostics are reported separately
from evaluation-only observed-zero/positive target strata.

## Actual matched M5 results

The three methods cover all required cases: **6,300 raw forecast rows** and
**6,300 repaired rows**, with no failed models or skipped targets. Cohorts are
scored separately. These are retrospective observed-sales results, not official
M5 probabilistic scores or a pristine prospective test.

Repaired outputs, original sales units:

| Cohort | Model | P10 pinball | P50 pinball | P90 pinball | Mean pinball | C10 | C50 | C90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Continuity | LightGBM | 2.726806 | 8.606644 | 4.381317 | 5.238256 | 23.26% | 33.24% | 79.03% |
| Continuity | Joint neural | 2.482732 | 6.233704 | 4.318541 | 4.344992 | 22.89% | 37.64% | 83.88% |
| Continuity | Historical | 2.434982 | 7.873626 | 4.786447 | 5.031685 | 25.27% | 45.33% | 83.52% |
| Prefix-sparse stress | LightGBM | 0.113119 | 0.461438 | 0.266781 | 0.280446 | 50.79% | 55.16% | 88.29% |
| Prefix-sparse stress | Joint neural | 0.119953 | 0.467763 | 0.276182 | 0.287966 | 50.79% | 55.75% | 88.39% |
| Prefix-sparse stress | Historical | 0.112103 | 0.442460 | 0.260119 | 0.271561 | 50.79% | 63.89% | 94.05% |

| Cohort | Model | 80% interval coverage | Mean width | Mean interval score |
|---|---|---:|---:|---:|
| Continuity | LightGBM | 72.99% | 36.143196 | 71.081234 |
| Continuity | Joint neural | 60.99% | 40.143558 | 68.012730 |
| Continuity | Historical | 78.94% | 36.179487 | 72.214286 |
| Prefix-sparse stress | LightGBM | 87.00% | 2.518921 | 3.798998 |
| Prefix-sparse stress | Joint neural | 37.60% | 2.505352 | 3.961355 |
| Prefix-sparse stress | Historical | 94.05% | 2.750000 | 3.722222 |

Neural quantiles have the lowest mean pinball in the continuity cohort; the
historical baseline has the lowest in the stress cohort. The neural lower
endpoint gives a substantial discrete-zero coverage failure despite favorable
continuity pinball. Its softplus head approaches zero but does not parameterize
exact zero ties: even a tiny positive P10 excludes actual zero from the interval.
This consequence is explicitly unit-tested. Neural intervals cover **none of
the 245 continuity zeros or 512 stress zeros**. On positive stress observations,
P90 coverage is only **76.41%** for neural and **76.21%** for LightGBM, compared
with **87.90%** for the baseline. Inclusive pooled P90 coverage therefore masks
undercoverage during positive-sales bursts. No post-hoc correction was fitted.

The historical baseline retains repeated zero quantiles in **280 continuity**
and **616 stress** cases, including **112 zero-width intervals in each cohort**.
Both learned models produce no repeated-zero pairs in the actual repaired run.
These counts, exact ties, denominator accounting and observed-zero strata remain
visible rather than being hidden by an aggregate coverage score.

LightGBM had **28 raw crossings** (2.56% of continuity cases, all P10>P50) and
negative quantiles on **292 continuity + 61 stress rows**. Repair leaves **zero
crossings, negative values or nonfinite predictions**. Continuity raw versus
repaired mean pinball is **5.240148 → 5.238256**; stress is **0.280507 → 0.280446**.
Neural and empirical outputs are unchanged by repair. Raw P10>P90 counts are zero,
so every actual raw interval has valid ordered endpoints.

Continuity mean pinball by origin:

| Origin | LightGBM | Joint neural | Historical |
|---|---:|---:|---:|
| 2016-01-31 | 6.108227 | 4.631413 | 6.121978 |
| 2016-02-28 | 4.351299 | 3.651419 | 3.921245 |
| 2016-03-27 | 5.255242 | 4.752146 | 5.051832 |

Continuity mean pinball by horizon band:

| Horizons | LightGBM | Joint neural | Historical |
|---|---:|---:|---:|
| 1-7 | 4.234760 | 3.309598 | 3.991697 |
| 8-14 | 5.829474 | 4.761744 | 5.409280 |
| 15-28 | 5.444395 | 4.654314 | 5.362882 |

Tables retain all **36 origin**, **336 horizon**, **36 band**, **42 regime**,
**24 historical-zero-band** and **24 observed-zero/positive** raw/repaired result
rows, including calibration deviations, widths, scores and counts. Horizon-
weighted pinball and cumulative coverage reconcile with pooled metrics.

Median-point diagnostics are distinct from distribution scores. On continuity,
neural P50 MAE is **12.467408**, WAPE **43.2039%**; historical is **15.747253** /
**54.5697%** and LightGBM **17.213289** / **59.6500%**. The earlier interpretable
N-BEATS point result **12.357747 / 42.8239% / 0.712729 macro RMSSE** remains
historical context with a different objective/fitting policy, not a quantile
calibration benchmark. No RMSSE-based distribution ranking is claimed.

## Hierarchical quantiles are not additive

For independent A and B with probabilities .9 at zero and .1 at ten, inverse-CDF
P90(A)=P90(B)=0. Their sum has probabilities .81/.18/.01 at 0/10/20, so
P90(A+B)=10. Summing marginal P90 values gives zero and fails. The deterministic
test verifies the exact boundary convention.

A separate real diagnostic reuses the verified Season 15 **102-bottom cohort**
maps. On the past 56 observed days at January 31, summed historical marginal
P90 values equal **2,023**, while the direct observed COHORT_TOTAL P90 equals
**1,497**; later origins give **2,113 vs 1,704** and **2,194 vs 1,712**. These
are historical empirical diagnostics, not future aggregate forecasts or claims
of probabilistic coherence/calibration. The 102-bottom hierarchy is not mixed
with the 13/12-series forecasting score tables. No scenario generator, copula
or probabilistic reconciliation is implemented.

## Verification, resources and reproduction

At each issue origin, frozen models/vocabularies/scaling rules are tested against
three independent changes: mutation of every future sale in the inference
projection, removal of all such sales, and mutation of post-origin retrospective
prices. The SQL feature matrix, history/mask/scale tensors, raw neural/tree
quantiles, empirical baseline and resulting endpoints remain unchanged. Eligible
fitting labels are also unchanged. All **nine real gates** pass.

All **216 Python tests**, including 23 new quantile tests, and **36 fresh real-data
dbt tests** pass. Pinball/interval math, quantile/date/fit provenance, native
objectives/vocabularies, missing numbers, neural residual gradients/serialization,
zero/short histories, matched keys and non-additivity have deterministic checks.
Raw/interim/processed fingerprints and the Season 15 hierarchy remain unchanged.
The warehouse retains **58,327,370 observations** and **65,695,409 units**.

The actual bounded run took **177.80 s** before report rendering/publication,
with **1,275.0 MiB** peak process working set. It trained eight models, produced
zero training warnings, and had no failed/skipped cases. CPU threads remain
bounded at two; DuckDB has its existing 2 GB execution limit. Predictions are
bounded projections of the warehouse, never the entire 58-million-row panel
loaded into a DataFrame.

Requires the existing verified warehouse, real local M5 sources and Season 15
hierarchy artifacts. Existing optional dependencies suffice; no new dependency
or infrastructure is added.

```powershell
.\.venv\Scripts\python -m pip install -e ".[analytics,eda,boosting,neural,reconciliation]"
.\.venv\Scripts\python -m demandos.quantiles --root . --config configs/quantiles.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,warehouse,env=settings(Path('.')); invoke_dbt(root,env,'test')"
```

Ignored output: `data/quantiles/global_h28/`:

- `models/lightgbm/{p10,p50,p90}/{development,frozen_evaluation}/`: native models,
  alpha/objective/iteration/config/version metadata and training-only vocabularies.
- `models/neural/{development,frozen_evaluation}/`: native state dictionaries,
  architecture/scaling/quantile/cutoff metadata and chronological training curves.
- `tables/raw_forecasts.parquet`, `forecasts.parquet`: exact wide keys, raw versus
  repaired values, observed targets, fit cutoffs and origin-safe cohort/regime data.
- `tables/metrics_*.parquet`: pinball, calibration, Wilson/tie counts, intervals,
  crossing/negative/zero diagnostics and explicitly labeled median metrics.
- `tables/baseline_provenance.parquet`, `origin_diagnostics.parquet`,
  `hierarchy_diagnostic.parquet`, `leakage_checks.parquet`, `inference_audit.parquet`.
- `feature_mapping.json`, `summary.json`, `verification.json`, `manifest.json`,
  `quantiles_report.md` and two interval/calibration figures. The manifest owns
  38 generated files; repeated runs replace only manifest-owned artifacts.

Previously examined benchmarks limit prospective claims. No calibration offsets
are fitted to these targets, and no conformal prediction, safety-stock decision,
inventory simulation, API or dashboard is included.

Season 17 extends these interfaces with a separate nested historical fit,
genuine held-out calibration ledger, corrected split ranks, rolling/weighted
CQR and delayed-feedback bounded ACI. It preserves these Season 16 results as
different-checkpoint references, rather than retrospectively treating January
training errors as calibration errors. The same October-frozen model is used
on both sides of primary split calibration. See [conformal provenance, observed
zero inclusion and empirical limitations](conformal.md).

## References

- [scikit-learn mean pinball loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_pinball_loss.html).
- [LightGBM parameters, quantile objective and alpha](https://lightgbm.readthedocs.io/en/latest/Parameters.html).
- [Forecasting: Principles and Practice, prediction intervals for aggregates](https://otexts.com/fpp2/aggregates.html).
