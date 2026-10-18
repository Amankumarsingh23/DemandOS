# Global gradient-boosted forecasting

Season 9 implements pooled CPU LightGBM, XGBoost and CatBoost observed-sales
point forecasts. One model per algorithm learns across store-item identities
and horizons 1–28. Local statistical methods still fit each series separately.
Source ingestion, warehouse transformations and earlier model interfaces are
unchanged. These are forecasts of recorded sales, not uncensored demand.

## Reproduce the experiment

First prepare the existing M5 Parquet inputs and verified warehouse using the
[README](../README.md). The command does not download data or rebuild ingestion.
From the repository root, with Python 3.12+:

```powershell
.\.venv\Scripts\python -m pip install -c configs/analytics-constraints.txt -c configs/boosting-constraints.txt -e ".[analytics,eda,boosting]"
.\.venv\Scripts\python -m pip check
.\.venv\Scripts\python -m demandos.boosting --root . --config configs/boosting.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,output,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

The exact tested stack is Python **3.13.15**, LightGBM **4.7.0**, XGBoost
**3.4.1**, CatBoost **1.2.10**, scikit-learn **1.9.1**, pandas **3.0.6**,
NumPy **2.5.4** and DuckDB **1.5.6**. Tested constraints require Python 3.12+
because of the pinned XGBoost version. The base package remains Python 3.11+;
the optional boosting ranges permit older compatible XGBoost versions but those
combinations were not tested here. Existing analytics constraints remain intact.

`configs/boosting.toml` declares one fixed parameter set per library: seed 42,
two fitting threads, learning rate 0.05, depth 6, at most 160 iterations and
15-round validation patience. LightGBM uses 31 leaves, minimum child size 20,
L1 0/L2 1, and 0.9 row/column sampling. XGBoost uses CPU histogram trees,
minimum child weight 5, L1 0/L2 1, 128 bins and 0.9 row/column sampling.
CatBoost uses CPU RMSE, L2 3 and Bernoulli 0.9 row sampling.
There is no random search or evaluation-driven parameter sweep. Configuration
rejects more than four fitting threads, depth outside 2–10, or iterations beyond
500; explicit row/series/origin caps also guard generation and fitting.
Defaults cap the pool at 200,000 rows, 96 series and 64 historical origins.

DuckDB uses the existing two-thread, 2GB managed buffer setting (reported as
1.8 GiB). This is not a total process-memory ceiling. Daily observations for
the union of training/evaluation keys are cached once in a temporary relation;
only compact reductions are expanded across historical origins and horizons.
The verified run cached **91,494 daily observations** and used **103,110,972
bytes** for the historical pool DataFrame. No 58-million-row Python frame is
created. The final run took **126.80s**, including **27.16s** through dataset preparation;
timings vary with machine load and caches.

## Historical examples and temporal contracts

The canonical key is `(store_id, item_id, origin_date, horizon)`, with
`target_date = origin_date + horizon` and a nonnegative integer
`target_units_sold` label. Origins mean end-of-day after that day's sales arrive.
Overlapping examples for one target day at different origins are intentional.

A deterministic **96-series** training panel is selected using only sales
through **2014-01-04**, combining early-prefix regime/ABC-XYZ diversity,
high-volume store examples and seeded identity hashes. It does not force the
later benchmark identities into the sample. The schedule starts **2014-01-05**
at a **14-day stride**, includes the validation origin **2015-12-06**, and ends
**2016-03-13**: **58 historical origins**. The complete planned grid has
155,904 examples; **1,344** labels after March 27 are ineligible and excluded,
leaving **154,560** labeled examples. The final historical origin therefore has
only its first 14 horizons in the labeled pool, without inventing observations.

Two independent conditions are enforced:

1. Predictors for an example may use only observations available at that
   example's origin. Source/lag dates, price knowledge stamps and feature-mapping
   cutoff are audited, reusing Season 8 generation and validation.
2. Its label must be observed by the model's allowed fitting cutoff:
   **`target_date <= allowed_cutoff`**. Filtering on origin date alone is invalid.

The development fit ends **2015-12-06** and has **133,056** rows. The validation
origin is December 6, with **2,688** targets on **December 7–January 3** across
the 96 series. Its targets follow all fitting targets and end before the first
evaluation origin. There is no random horizon-row split or shared target date
across the fit/validation boundary. Early stopping selects LightGBM **141**,
XGBoost **131** and CatBoost **133** rounds. These counts are frozen.

| Evaluation origin | Eligible training rows | Training series | Latest allowed label |
| --- | ---: | ---: | --- |
| 2016-01-31 | 143,808 | 96 | 2016-01-31 |
| 2016-02-28 | 149,184 | 96 | 2016-02-28 |
| 2016-03-27 | 154,560 | 96 | 2016-03-27 |

Each algorithm refits separately at each origin using its frozen count, with no
evaluation-period early stopping. Earlier benchmark outcomes can enter a later
fit once observed; their scores never select settings. Model reuse at later
origins is valid but deliberately not used here; prediction at an origin earlier
than a model's allowed fitting cutoff is rejected.

The **13-series** evaluation cohort comes from the established Season 7
January-selected benchmark. Only **7** identities overlap the independent
training panel. Its selection and earlier outcomes are already known: this is
a retrospective benchmark, not a pristine final test or representative M5
sample. March 27 outcomes do not select features, rounds or parameters. Public
target-date calendar/event/SNAP data are assumed scheduled and known, as in
Season 8; announcement-time archives do not verify this assumption.

## Model API, columns and category state

```python
from demandos.boosting.models import GlobalModel, TARGET

# training/validation include keys, provenance, explicit features and TARGET.
# feature_columns is the experiment's explicit predictive allowlist.
model = GlobalModel("lightgbm", feature_columns, config, feature_config)
model.fit(training, allowed_cutoff="2015-12-06",
          validation=validation, validation_cutoff="2016-01-03")
predictions = model.predict(validation.drop(columns=TARGET))
model.save(output_directory)
restored = GlobalModel.load(output_directory)
```

The same interface works for `xgboost` and `catboost`. No targets or audit dates
may enter the predictor list. All models require horizon, unique daily keys and
valid source provenance. Extra audit columns remain available for validation;
the matrix follows the saved feature order even if input columns are reordered.
`predict` rejects actual/target labels, duplicate keys, future-fitted models,
nonfinite predictions and infinite numerical features. No fallback replaces
errors. Raw predictions are preserved; scoring uses **max(0, raw)** without
modifying actuals. The real run clipped zero rows.

The primary model uses **57** Season 8 predictors, excluding five duplicate
integer category-code columns from the 62-column allowlist. Raw store/item/
department/category/state identifiers remain native categories. Audit dates,
selection segments and labels are excluded. Historical price features remain
null without verified as-of evidence; zeros remain valid sales values.

Each fit builds a sorted vocabulary only from its eligible fitting rows, never
from validation or inference identities. Missing/unseen categories map to the
reserved `__DEMANDOS_UNKNOWN__` token; actual identifiers containing that token
are rejected. This saved model vocabulary is separate from the earlier,
target-free Season 8 feature mapping, whose cutoff remains January 4, 2014.

| Backend | Numerical missingness | Categorical representation | Saved model |
| --- | --- | --- | --- |
| LightGBM | float32 NaN, default missing-value handling; zero stays zero | Unordered pandas categoricals with fixed fit-only category list and native categorical flags | `model.txt` |
| XGBoost | float32 NaN via DMatrix | Same fixed pandas categoricals, `enable_categorical=True`, CPU `hist` | `model.ubj` |
| CatBoost | float32 NaN, default `nan_mode=Min` | Explicit categorical columns containing strings, including missing/unseen token | `model.cbm` |

LightGBM uses squared-error regression; XGBoost uses `reg:squarederror`; CatBoost
uses RMSE. CatBoost explicitly uses **Plain** boosting with **has_time=True**.
Training rows are ordered by target availability, then origin/identity/horizon.
Its internal category statistics stay within eligible fitting data; this does
not by itself prove per-example time-series causality. Predictor availability
and outer-fold label restrictions provide the temporal safeguards.

Native files have a neighboring `metadata.json` with library/version, predictor
order, vocabulary, feature/config state, chosen rounds, fitting/validation
cutoffs, sample counts, native parameters, logs, warnings and elapsed fitting
time. XGBoost retains category metadata in UBJSON; reload uses the saved rounds
so post-best early-stopping trees do not enter predictions. Importance also
restricts trees to the selected rounds. Save/reload predictions are checked for
all **13** saved global fits, to floating-point tolerance. Fixed seeds and
LightGBM deterministic mode improve repeatability; cross-platform/version or
multithreaded bit identity is not promised.

Backend references: [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html),
[XGBoost categories](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html),
[XGBoost serialization](https://xgboost.readthedocs.io/en/release_3.1.0/tutorials/saving_model.html),
and [CatBoost training parameters](https://catboost.ai/docs/en/references/training-parameters/common).

## Measured same-case benchmark

All seven methods share **13 series × three origins × 28 days**, or **1,092
targets each**. Actuals are attached after predictions by validated exact keys.
Local baselines use full-prefix histories; global models use the growing
2014-onward historical example sample. Their input histories differ, while
scored cases and scale definitions match exactly. Full-prefix training-only
RMSSE scales use Season 7 first-positive trimming. WAPE is total absolute error
divided by total actuals; zero denominators remain undefined. Macro RMSSE
averages **38 valid series-origin cases**, with **one** undefined zero scale.

| Model | MAE | Pooled WAPE | Macro RMSSE |
| --- | ---: | ---: | ---: |
| LightGBM | 12.898 | 44.695% | 0.725 |
| XGBoost | 13.561 | 46.994% | 0.823 |
| CatBoost | 14.516 | 50.301% | 0.748 |
| Moving average | 13.076 | 45.313% | 0.742 |
| TSB | 13.015 | 45.102% | 0.729 |
| Seasonal naive | 13.132 | 45.506% | 0.789 |
| Naive | 18.780 | 65.080% | 0.931 |

LightGBM reduces pooled WAPE by **0.812 percentage points** relative to seasonal
naive and **0.619** points relative to moving average. Comparisons use the
seven-method successful intersection; pairwise tables preserve exact shared
keys. Coverage is complete, so intersections retain all 39 series-origin cases.
These are cohort scores, not official M5 42,840-series WRMSSE.

Paired candidate-minus-seasonal-naive MAE by origin (negative favors candidate):

| Model | January 31 | February 28 | March 27 |
| --- | ---: | ---: | ---: |
| LightGBM | +2.711 | -0.808 | -2.606 |
| XGBoost | +3.045 | +0.039 | -1.796 |
| CatBoost | +6.879 | -0.350 | -2.379 |

LightGBM wins two of three origins but loses January. Its MAE for horizons
1–7/8–14/15–28 is **9.677/12.815/14.549**, versus seasonal naive
**11.692/14.073/13.381**: longer-horizon gains are not established. By regime,
LightGBM has lower MAE for the selected erratic and lumpy cases, higher MAE for
the selected intermittent and smooth cases. Cells contain only 1–8 series;
they do not establish population or causal effects. Origins and horizon rows
are dependent, so paired differences are descriptive, without significance
claims or invented confidence intervals.

All **169** fitting attempts succeed: **13** pooled fits (three development,
one validation ablation, nine evaluation refits) and **156** local baseline
fits. All **273** model/series/origin prediction cases succeed; **7,644** rows
are written. No skips, failures, nonfinite forecasts or fallbacks occur. Five
LightGBM fit records retain the warning that no further positive-gain split is
available. This is a recorded tree-growth warning; it did not cause fit failure.
Full logs and unique native warning lines are retained in model metadata.

## Predictive importance and controlled ablation

Development models provide LightGBM gain/splits, XGBoost total gain/split count,
and CatBoost PredictionValuesChange. Their scales are not interchangeable.
Recent seven-day sales means and item identity lead LightGBM/XGBoost native
importance; CatBoost ranks lag 56 highest. These are predictive associations.

Limited permutation checks use only **1,024 seeded internal-validation rows**
per model and the five highest-ranked numerical non-key predictors. Identities
and horizon are excluded from permutation to preserve legal forecast keys.
One shuffle per feature is a sensitivity probe, not an uncertainty interval.
Shuffling the seven-day mean increases MAE by **2.775** for LightGBM and
**3.129** for XGBoost; shuffling lag 56 increases CatBoost MAE by **1.772**.
Overlapping/correlated rolling summaries limit attribution.

The controlled LightGBM ablation removes the six event predictors, retaining
the same development/validation rows, all other settings and **141 rounds**.
It improves internal-validation MAE **3.901107 → 3.866181**, WAPE
**48.9368% → 48.4987%**. The primary feature specification remains fixed;
the ablation never selects a variant on March outcomes. Validation uses a
different, larger series panel, so its scores are not directly comparable to
the benchmark table above.

## Verification and artifacts

The full Python suite passes **93 tests**, including eleven focused synthetic
boosting tests. They cover forbidden targets/audits, cutoff isolation, fit-only
categories, numerical missingness, unseen identities, column/row order,
serialization, raw clipping/nonfinite errors, future-label exclusion, output
grain/fanout, metrics, chronological validation and resource caps. Existing
Season 8 tests cover future prices with non-null synthetic as-of evidence.
Fresh full-source dbt validation passes **36/36** tests. Dependency consistency
checks pass. Synthetic fixtures remain explicitly distinct from actual M5.

Real-data checks cover all **nine** model/origin combinations: change future
actual sales and future-week prices, regenerate inference rows, require exact
feature equality, and compare saved-model forecasts within 1e-12. Altering
post-origin pool labels leaves eligible fitting inputs unchanged. Real price
features are unavailable by default, so that check demonstrates exclusion;
non-null verified-as-of behavior is separately tested with synthetic inputs.
Raw CSV and interim/processed Parquet SHA-256 fingerprints remain unchanged.
Warehouse totals still reconcile to **58,327,370 rows**, **65,695,409 units** and
**187,676,570.02** historical selling-price-based revenue.

All generated artifacts are ignored under `data/boosting/global_h28/`:

- `historical_features.parquet`, separate `historical_labels.parquet`,
  `inference_features.parquet` and `feature_mapping.json`.
- `models/<library>/development/` and `origin_<YYYYMMDD>/`: native models and
  JSON state; LightGBM also has `ablation_no_events/`.
- `tables/forecasts.parquet`: planned keys, origin/target/horizon, actual,
  raw/scored prediction, status, regime, loss and scale.
- Metric tables by model, series-origin, origin, horizon, band, regime and
  ABC/XYZ; exact comparison membership, paired losses, coverage, fit statuses,
  unknown-category counts, training eligibility and leakage checks.
- Native/permutation importance and internal-validation ablation metrics.
- `figures/horizon_mae.png`, `figures/forecast_origins.png`, `summary.json`,
  `experiment_report.md` and `manifest.json`.

The manifest publisher replaces generated files and removes only obsolete files
owned by the prior manifest. Tests are run separately from the experiment CLI;
the final local `test_results.json` records the reviewed test/dbt/dependency and
saved-artifact checks, and is included in the output manifest. Training-row audit
digests normalize date and integer storage representations; regression tests and
real saved-artifact checks verify they survive Parquet reloads. A new experiment
removes this stale verification record until checks are run again.

## Season 10 extension

The separate [multi-horizon experiment](multihorizon.md) now compares this
unchanged pooled LightGBM reference with true recursive one-step inference,
28 horizon-specific global models, an explicitly independent 28-output wrapper,
and a validation-selected pooled/seasonal-naive ensemble. It retains the same
training panel, feature allowlist, chronological cutoffs and scoring contracts.
The pooled reference is checked against these saved Season 9 predictions when
available. Vector samples require the entire target vector observed by cutoff;
recursive history uses predictions after issuance. These architecture experiments
do not change the Season 9 results above. No neural forecasting, quantiles,
inventory simulation, serving or business-savings claim is included.
