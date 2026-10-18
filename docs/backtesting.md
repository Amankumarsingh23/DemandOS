# Season 7: rolling-origin contracts and measured evaluation

## Repeat the benchmark

Reuse the existing analytics/EDA extras and verified Season 3 warehouse:

```powershell
.\.venv\Scripts\python -m pip install -c configs/analytics-constraints.txt -e ".[analytics,eda]"
.\.venv\Scripts\python -m demandos.backtesting --root . --config configs/backtesting.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,output,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

On Unix use `.venv/bin/python`. Inputs are the unchanged verified DuckDB/dbt
warehouse and underlying `data/processed/daily_sales.parquet`, with **58,327,370
source observations**. There is no download, CSV rewrite, model rewrite, or
warehouse rebuild. Successful warehouse-build evidence must still match the
source and dbt project fingerprints.

`configs/backtesting.toml` fixes the benchmark before fitting. `[backtesting]`
controls origin bounds, daily horizon/stride, expanding/sliding windows, minimum
history, cohort sizes and ARIMA/SARIMA caps. `[model]` supplies the existing Season
6 parameters, including seasonal period, moving window, smoothing rates, orders
and optimizer iteration limit. Horizon can be 1..28, at most 12 origins are
allowed, and the cohort is bounded to 30 store-items, ten stores, three categories,
16 ARIMA and eight SARIMA series. Defaults are smaller. Unknown or invalid
configuration fails clearly. Preserve identical configuration, source hashes,
versions and earliest-origin label definitions to repeat a benchmark.

## Default evaluation design

| Origin | Target start | Target end | Expanding observations | Sliding observations |
| --- | --- | --- | ---: | ---: |
| 2016-01-31 | 2016-02-01 | 2016-02-28 | 1,829 | 730 |
| 2016-02-28 | 2016-02-29 | 2016-03-27 | 1,857 | 730 |
| 2016-03-27 | 2016-03-28 | 2016-04-24 | 1,885 | 730 |

Expanding histories start 2011-01-29. Sliding histories end at each origin and
contain exactly 730 complete daily observations. Forecasts retain every day
1..28, plus bands 1–7, 8–14 and 15–28. Daily calendars, midnight origins,
origin observations and complete targets are validated; no gaps are imputed.
The calendar ends 2016-04-24; later target designs are rejected, not truncated.

The three target blocks do not overlap. A configurable stride shorter than the
horizon produces overlapping forecasts; each origin/target/horizon pair counts
as its own prediction event. Repeated target dates consequently receive repeated
weight in pooled metrics, and dependence precludes interpreting those events as
independent observations. Even the non-overlapping design shares training
history. Origin variation is described with paired differences, range and
population SD, without significance tests or independence-based intervals.

### Fixed training-only cohort

The Season 5/6 `training_labels` and `select_cohort` rules are reused through
**2016-01-31**, before any evaluated targets. Labels and cohort remain fixed at
all later origins. They are origin-known descriptive strata and eligibility
metadata, not forecasting regressors. No full-history labels, future-fitted
normalization, prices or seasonal indices enter the models. The selection uses
training volume, all four occurrence regimes, ABC/XYZ cells and stable keys.

The actual cohort has **13 store-item series**, **CA_3**, and the three category
aggregates **FOODS, HOUSEHOLD, HOBBIES**. It retains 12 of 13 Season 6 store-item
identities; the sparse example becomes `CA_1/FOODS_3_595`, because selection now
stops in January. The cohort includes two smooth, one intermittent, one erratic,
eight lumpy and one unclassified store-item case, spanning all nine ABC/XYZ cells.
This small illustrative sample is not representative of all 30,490 store-items.
Broad-cohort model fitting is deliberately deferred to keep this run bounded.

All 12 unchanged models run at both windows/three origins. ARIMA eligibility is
capped at eight identities and SARIMA at four. Both caps select stable-ID smooth,
intermittent, then aggregate and remaining identities using earliest-origin
metadata. Their exact membership is in `summary.json`; caps never depend on fit
success or target performance. Every origin/window fits fresh model state.
Existing minimum-history, all-zero, convergence, warning and nonnegative-output
policies remain intact. Model orders and tuning parameters are not selected
using these target blocks. The last block was Season 6's untouched test block;
after this season all three blocks are evaluation data, not a fresh final test.

## Metric definitions and denominators

- **MAE:** sum absolute error / number of evaluated point forecasts, in observed
  units. Micro MAE weights point events equally.
- **WAPE:** pooled absolute error / pooled nonnegative actual sales. Zero actual
  total gives null and `wape_status=zero_actual_total`; a perfect zero forecast
  does not change that denominator. Per-series-origin WAPE is retained.
- **Macro MAE/WAPE:** equal-weight means of series-origin metrics within the
  reported group. Undefined WAPE cases are excluded only from macro WAPE, with
  valid count recorded. Macro and micro can differ when case lengths differ.
- **RMSSE scale:** take the permitted training history, begin at its first
  positive observation, then mean the squared lag-one differences within that
  trimmed sequence. Exclude the transition from the pre-introduction zero
  stretch to the first positive observation. Denominator is `trimmed_n-1`.
- **RMSSE:** sqrt(evaluation MSE / training scale), computed per series-origin
  for its full vector or requested horizon subset. All-zero, only-one-point-after-
  trimming, and constant trimmed histories have explicit undefined scale status;
  no arbitrary epsilon or replacement denominator is used. Aggregate scales are
  computed from their aggregate daily history, not averaged bottom-series scales.
- **Macro RMSSE:** equal-weight mean of valid series-origin RMSSE values; valid
  and undefined case counts remain visible. **Micro scaled RMSE** is separately
  sqrt(mean of valid per-point squared errors / their own case scale), not the
  same aggregation. No population-wide or official score label is attached.

Expanding RMSSE uses all permitted historical observations at each origin.
Sliding RMSSE uses only that method's sliding history and its own first-positive
trimming. The latter is a documented local sensitivity metric, not a competition
score. Identical forecasts can have different RMSSE across windows solely from
changed scales; use matched MAE/WAPE to isolate direct forecast-error changes.
An RMSSE below one references historical naive first-difference loss, not a claim
that seasonal naive was beaten on the evaluation targets.

### Official M5 versus these cohort results

Official M5 accuracy evaluation covers **42,840 series across 12 hierarchy
levels**. RMSSE uses nonseasonal training differences after initial zeros.
Competition WRMSSE combines series RMSSE with sales-value weights from the last
28 training days, normalized within each level, then averages across levels.
See the [organizers' guide](https://github.com/Mcompetitions/M5-methods/blob/master/M5-Competitors-Guide.pdf)
and [competition results paper](https://doi.org/10.1016/j.ijforecast.2021.11.013).

`hierarchy.aggregate_hierarchy` implements the 12 key combinations for bounded
fixtures/inputs and validates metadata and bottom-grain uniqueness. Tests prove
each level reconciles to bottom-level units. `historical_revenue_weights`
requires a complete daily trailing window and verified `price_available_at`
metadata for positive-sales prices available by the origin. Future rows cannot
change historical weights. Missing/unavailable positive-sales prices fail,
zero revenue yields undefined weights, and `cohort_weighted_rmsse` never silently
renormalizes missing positive-weight scores. It checks exact series membership.
These utilities do not build the complete official scorer.

Real-data revenue weighting is disabled because the repository has no verified
historical availability metadata for M5 recorded prices. No equal-price proxy or
future price is substituted. All reported metrics are explicitly selected-cohort
MAE, WAPE, RMSSE or micro scaled RMSE, **not official M5 WRMSSE**.

## Shared successful cases and coverage

Forecast keys include **window, series ID, model, origin, target date and horizon**.
Every planned key is retained, with null raw/scored predictions for failed/skipped
fits. Actual targets are validated before any model runs. Successful forecast
vectors are finite and have exactly h entries, using the existing `max(0, raw)`
scoring policy and retained raw output. Status rows record configuration, allowed
history, timing, convergence warnings and reason.

Comparison panels: `simple` (naive, seasonal naive, moving average), `core_10`
(all except ARIMA/SARIMA by default), `arima_capped`, `sarima_capped` (each capped
model against naive/seasonal naive), and `configured` (all requested models).
Each intersects successful keys across its own model set and validates shared
actuals/scales. MAE/WAPE/macro-RMSSE ranks are within that panel, grain and
window only. Origins, horizon days/bands and origin-safe regimes/ABCXYZ have
their own matched summaries. Available-coverage scores are explicitly not ranks.

Paired absolute/squared losses subtract seasonal naive's error on identical
keys; negative differences favor the candidate. Exact matches and per-origin
sample counts are exported. Successful-only comparisons have selection bias,
especially when difficult seasonal fits fail. Window sensitivity intersects
successes across both windows for each individual method; it cannot support
ranking different methods on unequal samples. Sparse/zero-target cells and
small origin counts constrain every interpretation.

## Verified measured findings

There are **1,224** fit cases (17 series × 12 models × three origins × two
windows) and **34,272** planned daily target rows. **1,060 successful fits** emit
**29,680** predictions; **142 skip**, and **22 additive Holt-Winters fits fail
convergence with 22 captured warnings**. Skips are **132 capped cases plus ten
all-zero numerical-fit cases**. No fallback is scored. Training scales cover
102 series-origin-window cases, including two explicit no-positive histories.

Common three-simple-method expanding panel: **13 store-items × three origins**,
**39 cases / 1,092 target events** per method; **38 valid RMSSE cases**:

| Model | MAE | Pooled WAPE | Macro RMSSE |
| --- | ---: | ---: | ---: |
| naive | 18.780 | 65.080% | 0.931 |
| seasonal_naive | 13.132 | 45.506% | 0.789 |
| moving_average | 13.076 | 45.313% | 0.742 |

The small overall moving-average/seasonal-naive difference changes with origin:
seasonal naive MAE **12.451** versus moving average **13.359** at January 31;
**10.989 vs 10.979** at February 28; **15.956 vs 14.891** at March 27.
Moving-average MAE is lower in days 1–7 and 8–14; seasonal naive is lower in
days 15–28 (**13.381 vs 13.943**). In these simple comparisons, seasonal naive
has lower MAE on the selected smooth/intermittent strata, while moving average
has lower MAE on erratic/lumpy strata. Several strata contain only one series.

On `core_10`'s different **28 successful cases / 784 observations**, TSB has MAE
**12.860**, WAPE **48.429%**, macro RMSSE **0.712**; moving average **12.886**,
**48.527%**, **0.726**; seasonal naive **13.043**, **49.121%**, **0.787**. This
panel excludes unsupported or failed cases and is not the same as the simple
panel. On SARIMA's **two store-items / six cases / 168 observations**, SARIMA
has **20.882 / 25.815% / 0.483**, seasonal naive **23.048 / 28.491% / 0.517**.
SARIMA's paired MAE difference ranges from **-7.710 to +5.340** across origins;
it is lower at two of three origins. No universal winner is established.

Across all three category aggregates and origins on shared core cases,
Holt-Winters has MAE **1,007.036**, WAPE **7.178%**, macro RMSSE **0.595**;
seasonal naive **1,372.806 / 9.785% / 0.794**. At CA_3 aggregate:
**327.496 / 5.118% / 0.437** versus **393.238 / 6.146% / 0.524**. Aggregate
units and bottom-level unit rates are never pooled into a ranking.

Matched expanding/sliding SARIMA store-item WAPE is **25.815% vs 21.097%**,
whereas macro RMSSE is **0.483 vs 0.575**, illustrating both changed forecasts
and training scales. This is sensitivity evidence, not tuning a final window
using those evaluation targets. No configuration is switched based on results.

## Outputs, runtime and verification

Ignored `data/backtests/20160131_20160327_h28_s28/` contains `summary.json`,
`backtest_report.md`, `manifest.json`, three PNG figures and Parquet tables:

- Cohort, earliest-origin training labels, training observations and six-row
  origin calendar.
- All 34,272 planned forecasts and 1,224 fit statuses, including null predictions
  and warning reasons; 102 train-only scale records.
- Available metrics, per-series-origin, origin, daily horizon and horizon-band
  metrics; common-case ranks, regime/ABCXYZ summaries and exact membership.
- Coverage counts/fractions, pointwise paired losses, paired per-origin metrics
  and descriptive variation, plus matched window sensitivity/membership.

SQL extraction uses two DuckDB threads and configured 2GB managed-buffer limit
(reported by DuckDB as 1.8 GiB), with ignored spill storage. Only the selected
series and aggregate histories enter Python; model fitting is sequential and
iteration/candidate/cohort caps bound workload. No strict total Python-memory
ceiling or wall-clock timeout is claimed. Summary files record actual runtime,
extraction time, resource settings, input SHA-256s, package versions, origin
config and exact expensive-model eligibility. Training prefixes/scales and target
vectors are reused in memory; fits are never reused across origins.
The verified run took **31.75s** locally. All **70 Python tests** and **36
fresh real-data dbt tests** passed. Independent artifact checks reconciled
34,272 planned rows, 29,680 predictions, unique keys, February 1–April 24
targets and an analytically recomputed exported RMSSE/WAPE case.

Reruns replace generated files and remove only obsolete paths from the previous
generated-file manifest; unrelated files are preserved, and escaping paths are
rejected. Before/after fingerprints verify available raw CSVs, interim and
processed Parquet are unchanged. The original Season 6 experiment remains
separate under `data/forecasts/`; its code/results are preserved.

Tests cover numerical definitions, first-positive trimming, undefined scales,
macro/micro distinctions, calendar/leap/overlap boundaries, future invariance,
series isolation, exact pairing, no false ranking when all fits fail, coverage,
hierarchy reconciliation, strict as-of weights, horizon bands, clean publication
and unchanged sources. Test fixtures are deterministic synthetic data and do
not substitute for real M5 findings. Relevant real-data dbt tests are run
separately. No new statistical/ML model, feature engineering, inventory policy
or serving infrastructure is introduced.
