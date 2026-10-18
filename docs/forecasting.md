# Season 6: point forecasting contracts and experiment

## Reproduction and source

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python -m pip install -c configs/analytics-constraints.txt -e ".[analytics,eda]"
.\.venv\Scripts\python -m demandos.forecasting --root . --config configs/forecasting.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m demandos.warehouse check
```

Use `.venv/bin/python` on Unix. No new dependency is required: the existing
`eda` extra supplies statsmodels and plotting; `analytics` supplies DuckDB/dbt.
The existing verified `data/warehouse/demandos.duckdb` reads the unchanged
`data/processed/daily_sales.parquet` (58,327,370 source observations).
Missing or stale warehouse/source build evidence raises an error; no download,
fabricated rows, automatic ingestion, or warehouse rebuild occurs.

## Library interface and safety

```python
from demandos.forecasting import forecast

# history: one chronological pd.Series with complete daily DatetimeIndex
result = forecast(history, 28, "seasonal_naive", seasonal_period=7)
if result.status == "ok":
    predictions = result.predictions
    raw = result.raw_predictions
else:
    reason = result.reason  # skipped or failed; no substitute forecast
```

`at_origin(frame, origin, lookback)` clips before numerical validation/fitting,
rejects mixed identifiers, and requires an observation at the origin. The library
rejects missing/duplicate/unordered/non-midnight dates, gaps, nonfinite or negative
sales, invalid horizons and unsupported parameters. Horizons are bounded to
1..366 and seasonal periods to 2..31. Successful results contain exactly h finite
values. Every result records status, reason, runtime, warnings and training bounds.
Nonnegative scoring uses `max(0, raw)`; raw values and clipping counts remain
available. Actual observations are never clipped or filled.

Constant and all-zero histories are skipped by numerical smoothing/ARIMA models,
with no implicit naive fallback. Simple methods handle them directly. Intermittent
methods return a documented zero rate for no positive observations; exactly one
positive event is explicitly skipped. Small positive-count histories are not
automatically excluded from all other models: their fits still face numerical
and convergence checks, and sparse data may be poorly supported statistically.

## Models and deterministic choices

| Model | Rule / default parameters |
| --- | --- |
| naive | Repeat the last observation |
| seasonal_naive | Repeat the last m=7 observations cyclically, including h>m |
| moving_average | Repeat mean of the last k=28 observations; skip if n<k |
| ses | Estimated initial level and optimized level smoothing |
| holt | Estimated level and additive linear trend |
| holt_damped | Estimated level, additive trend and damping |
| holt_winters | Estimated level, additive trend and weekly additive seasonality |
| arima | One predeclared ARIMA(1,0,0) candidate: short persistence |
| sarima | One predeclared ARIMA(1,0,0)(0,1,1)[7]: weekly differencing and seasonal shock |
| croston | Positive size / interarrival interval, alpha=.1 for both |
| sba | Croston rate multiplied by (1-alpha/2) |
| tsb | Positive size times daily occurrence probability, alpha=.1 / beta=.1 |

Smoothing uses statsmodels Holt-Winters classes, estimated initialization by
default, L-BFGS-B, no brute initial grid, no Box-Cox transform and no bias removal.
SES also accepts known initial level and fixed smoothing for analytic checks;
Holt/additive seasonal models expose corresponding known initial states and
smoothing parameters. The known state precedes the first observation. General
fits need at least three observations; known-level SES can start earlier when
nonconstant. Additive seasonal fits require at least two complete seasons plus
one observation. No multiplicative seasonal/trend model is used.

ARIMA and SARIMA use statsmodels' state-space fit and L-BFGS. Both default to
stationarity/invertibility enforcement. ARIMA's default constant is included
when undifferenced; no constant is included when seasonal differencing is used.
Each configured order component is bounded to 0..2; seasonal period must match
the wrapper's m. Default maximum optimizer iterations: 100; configurable 1..500.
There is no grid search or selection by holdout accuracy. The experiment runs
one configurable ARIMA candidate on the cohort and one configurable SARIMA
candidate on a capped subset. Candidate history, cohort size and iteration caps
control workload; they are not hard wall-clock timeouts. Nonconverged optimizers
are failures and are not scored. All caught numerical warnings remain in status
artifacts, including warnings accompanying otherwise successful fits.

Croston initializes size to the first positive value and interval to its
one-based index (including leading zeros). At each later positive event, size
and interval update with alpha, using the exact number of periods since the last
positive event. Zeros leave Croston/SBA state unchanged. TSB starts probability
at the inverse initial interval; from then on it updates probability every
observed day using beta and a 0/1 occurrence indicator. Size updates only on
positive days. Repeated zeros therefore decay TSB's forecast. There is no
unobserved-day imputation, future-event lookup or fabricated positive initial
rate. These methods estimate observed sales rate, not recovered customer demand.

A trailing moving average averages past observed units. The MA(q) term of ARIMA
instead represents a linear combination of past innovation/forecast-error shocks.

## Fixed-origin experiment

The existing Season 5 cohort extraction recomputes regimes at store-item grain
and product ABC/XYZ using training data through **2016-02-28** only. Selection is
deterministic by training volume/regime/ABCXYZ and stable keys, not holdout scores.
It includes **13 store-item series** spanning smooth/intermittent/erratic/lumpy,
all nine ABC/XYZ cells where supported, plus an explicitly unclassified sparse
case, **CA_3** and **FOODS/HOUSEHOLD/HOBBIES** aggregates. Labels fit on the entire
training prefix; every model fits the same trailing **730 daily observations**.

The forecast vector has **28 days: 2016-02-29 through 2016-03-27**. Model fitting
receives only prefix observations. Holdout targets are joined afterward with a
validated many-to-one join and complete target-calendar check. The subsequent
2016-03-28–2016-04-24 test block is unused. No future prices, transforms fitted
on targets, future segment labels, or holdout-driven hyperparameters enter fits.

SARIMA chooses the first stable-ID smooth series and intermittent series, then
aggregates by stable ID, then remaining series, capped at four. This chooses
CA_3/FOODS_3_090, CA_1/FOODS_3_090, category/FOODS and category/HOBBIES. The cap
and exact membership are recorded. Unsupported fits remain explicit; this is an
illustrative sample, not a representative population experiment.

## Metrics and measured results

MAE is mean absolute error in units. WAPE is pooled absolute error divided by
pooled actual units, and is null when the denominator is zero (even perfect zero
predictions). Metric rows include observation/series counts and both sums. Never
average series WAPE percentages into an unlabeled global figure. Store-item,
store and category scales are reported separately.

Three comparison panels intersect successful series/date/horizon keys: all three
simple methods; all 11 non-SARIMA methods; all 12 including capped SARIMA. Unequal
available-coverage summaries are explicitly not rankings. The common store-item
11-model panel excludes both smooth cases because additive seasonal fits failed;
this selection effect limits comparisons further. Common-sample segment tables
and exact membership make that exclusion visible.

Actual **eight-series common store-item panel**, 224 daily forecasts per method:

| Model | MAE | WAPE |
| --- | ---: | ---: |
| naive | 15.290 | 51.127% |
| seasonal_naive | 11.362 | 37.991% |
| moving_average | 12.099 | 40.458% |
| ses | 13.448 | 44.967% |
| holt | 13.734 | 45.925% |
| holt_damped | 13.761 | 46.015% |
| holt_winters | 11.671 | 39.027% |
| arima | 13.302 | 44.478% |
| croston | 12.947 | 43.293% |
| sba | 12.842 | 42.940% |
| tsb | 11.986 | 40.078% |

The **all-12 panel has only one store-item series**, CA_1/FOODS_3_090: SARIMA
MAE **11.195**, WAPE **20.718%**; seasonal naive **12.964**, **23.992%**;
Holt-Winters **11.736**, **21.718%**. These numbers cannot be compared to the
eight-series table as if they had the same cohort.

In the common three-category panel, Holt-Winters MAE/WAPE is **1,024.056/7.314%**,
seasonal naive **1,283.679/9.169%**, naive **3,369.702/24.068%**. In the common
CA_3 aggregate panel, Holt-Winters is **268.519/4.154%**, seasonal naive
**269.607/4.171%**. These are single-origin observations, not robust winner claims.

Segment examples on identical individual series: the selected smooth AX case has
SBA MAE **14.929** versus naive **27.857**; the selected intermittent AY case has
seasonal naive **12.964** versus naive **19.321**. A lumpy BX case has zero total
holdout sales: naive/seasonal naive MAE **0**, SBA **17.868**, TSB **0.557**;
its WAPE is undefined. No method dominates every selected segment.

Of **204** model/series combinations: **184 succeed, 16 skip, four fail**. Skips:
13 outside the SARIMA cap, three insufficient-positive intermittent fits. Four
Holt-Winters nonconvergence warnings/failures remain visible. There are **5,152
forecast rows** and **52 negative raw predictions** retained before clipping.
The final verified regeneration took **8.73s** locally; each JSON report records
its own measured runtime. All
**54 Python tests** and a fresh **36 real-data dbt tests** passed.

## Artifacts and verification

Ignored `data/forecasts/origin_20160228/` contains:

- `summary.json`: config, versions, SHA-256 input fingerprints, unchanged-input
  result, origin/targets, cohort, outcomes, metric summaries and runtime.
- `forecast_report.md`: actual metrics, explicit comparison denominators,
  segment tables, coverage, failures, warnings and limitations.
- `tables/forecasts.parquet`: unique series/model/horizon, origin/target date,
  raw/scored prediction, actual and absolute error for every successful vector.
- `tables/fit_status.parquet`: every attempted/capped model-series pair, status,
  reason, warnings, fitted parameters, clipped count, bounds and runtime.
- Coverage, available/common metrics, common/available segment metrics,
  h=1/7/14/28 metrics, comparison membership, training observations, cohort and
  all training-derived labels in reusable Parquet tables.
- `figures/smooth.png` and `figures/intermittent.png`: recent training history,
  fixed cutoff, shaded holdout and observed sales/forecast comparison.

Deterministic synthetic tests check exact multi-horizon rules, SES recursion,
Holt/damped trend, additive seasonality, bounded ARIMA/SARIMA output and failure,
Croston event gaps/size, SBA correction, TSB zero-run decay, invalid inputs,
constant/zero/short cases, finite outputs, raw clipping, warning retention,
future-perturbation invariance, rejected future label metadata, series isolation,
metric zeros, comparison membership, forecast/status reconciliation, complete
targets, and raw/processed fixture immutability. Existing training-label leakage
tests remain in force. Real execution checks SHA-256 before/after for available
raw CSVs and both input Parquet files, with unchanged results.

## Season 7 evaluation extension

Rolling-origin execution, expanding/sliding sensitivity and train-only RMSSE
are now implemented separately in [backtesting](backtesting.md), reusing this
library without changing these Season 6 results. The January-first benchmark
refits its selection labels at its earliest origin rather than borrowing
February-fitted labels. The former final 28-day Season 6 block is now evaluated
by Season 7, so it is no longer an untouched final test. Formal model selection,
uncertainty/probabilistic forecasts and inventory decisions remain deferred.

## Season 9 global model extension

The separate `demandos.boosting` package now implements CPU LightGBM, XGBoost
and CatBoost through consistent fit/predict/save/load interfaces. Each model
pools store-item series and direct horizons; the Season 6 statistical interface
continues to fit one history at a time. Earlier experiment results above remain
specific to their original cohort and origin.

The new benchmark compares seven methods on the exact Season 7 store-item grid:
13 series, three origins, 28 days. Models see only historical feature values
available at each example's origin and labels observed by the fitting cutoff.
December 2015 chronological validation selects iteration counts before January–
March refits. Later evaluation targets never drive early stopping or feature
selection. Reusing known Season 7 outcomes makes this an established benchmark,
not a pristine final holdout. Global fitting uses a larger independent panel and
a growing 2014-onward label sample; the local baselines retain full-prefix
histories. Only forecast keys, actuals and scoring scales are identical.

All seven methods cover 1,092 common targets each. LightGBM achieves measured
MAE/WAPE/macro RMSSE **12.898 / 44.695% / 0.725**, seasonal naive
**13.132 / 45.506% / 0.789**, moving average **13.076 / 45.313% / 0.742**.
RMSSE averages 38 valid series-origin cases, with one undefined case. No method
wins consistently across origins, horizon bands and regimes. See
[global forecasting contracts, complete measured results and commands](boosting.md)
for category/missing-value policies, serialization, validation-only importance
and ablation, safety evidence, outputs and limitations.
