# Season 5: mathematical diagnostics and temporal validity

## Reproduce

From the repository root, with Season 3's verified warehouse present:

```powershell
.\.venv\Scripts\python -m pip install -c configs/analytics-constraints.txt -e ".[analytics,eda]"
.\.venv\Scripts\python -m demandos.timeseries
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Use `.venv/bin/python` on macOS/Linux. `--root` accepts a checkout path, including
spaces; `--config` accepts an alternative TOML. Defaults live in
`configs/timeseries.toml`. No additional libraries beyond existing analytics/EDA
extras are required. The pipeline reads the existing warehouse and verifies its
successful build record, input fingerprint, full row count and unit total. It
does not modify ingestion, warehouse SQL, raw CSV or processed Parquet.

## Split and cohort selection

Default data boundaries are inclusive and nonoverlapping:

| Role | Start | End |
| --- | --- | --- |
| Training | 2011-01-29 | 2016-02-28 |
| Validation targets | 2016-02-29 | 2016-03-27 |
| Test targets | 2016-03-28 | 2016-04-24 |

The final 28 days are test targets and the preceding 28 are validation targets.
Training has 1,857 days. Calendar dates must be unique and continuous; there is
no random split. Origin means the latest fully observed, available sales day.
This demonstration assumes daily sales are available by that origin; M5 does
not verify operational reporting latencies.

Full-history Season 4 labels must **not** select a forecasting training cohort
as though they were available earlier. We reuse Season 4's functions and
`configs/eda.toml` thresholds, but recompute store-item regimes and product ABC/
weekly XYZ using only the training prefix. Simple SQL reductions cover all
30,490 series without running expensive diagnostics on them. Label output has
`segment_fit_end`; the exact segmentation config and dates are saved.

Selection is deterministic and intentionally illustrative:

1. Highest training-volume store-item in each available smooth/intermittent/
   erratic/lumpy regime, with store/item IDs breaking ties.
2. Lowest training-positive fraction, with IDs breaking ties, to demonstrate
   sparse-series guardrails.
3. Highest-volume eligible store-item for each available product-level ABC/XYZ
   cell; deduplicate keys and retain all reasons, capped at 14 store-items.
4. Highest training-volume store aggregate and three category aggregates.

The actual cohort has **13 distinct store-items plus CA_3, FOODS, HOUSEHOLD and
HOBBIES aggregates: 17 series**. Aggregates have explicit `kind`/series IDs;
they are not fictitious store-product source records. This is not a random or
statistically representative sample. All original diagnostics use training only.
Held-out values are extracted only for a feature-invariance demonstration and
are removed before fitting/feature calculations.

## Temporal utilities

`daily_panel` canonicalizes store/item/date order on a copy, rejects duplicate
keys, missing daily dates, missing observations and nonfinite/negative recorded
sales, and never imputes missing dates as zero. Grouped shifts/rolling windows
preserve store-item boundaries.

- `descriptive_lags`: target-relative `target_lag_k(u) = y_(u-k)`, with initial
  missing values preserved. Prior rolling statistics exclude the current row.
  These historical shifts are not automatically valid at a multi-day origin.
- `origin_features`: fixes origin t for all direct horizons; `sales_at_origin`
  is y_t, and `origin_minus_k` is y_(t-k). Each lag records its source date.
  Rolling origin mean/population variance include the latest observed y_t and
  up to 28 days ending there; available history count is explicit.
- `fit_location_scale`: computes per-series population mean/std on the prefix
  through fit_end. Origin features reject fitted parameters from later dates;
  constant scale yields an undefined scaled value rather than invented variance.
- `FeatureEvidence` / `validate_feature`: check source date, availability date,
  transformation fit cutoff, kind and descriptive/centered flags. Future actual
  sales, unknown/late availability, future-fitted transforms and centered or
  descriptive components are rejected. Calendar target weekday is deterministic
  and known ahead; retrospective prices require independently verified as-of
  availability and are **excluded** from these demos.
- `chronological_split`, `origin_schedule`: deterministic boundaries and expanding/
  fixed-365-day history examples with consecutive nonoverlapping 28-day targets.
  There are two default schedule rows. These are boundary examples, not a full
  backtesting framework. Future benchmarks must refit preprocessing/segmentation
  at each origin, and treat earlier validation outcomes as history only when they
  have actually become available at the advanced origin.

The h=7 naive target lag-1 points to 2016-03-05 when origin is 2016-02-28 and
target is 2016-03-06. It is deliberately rejected; no future actual value is put
in the corrected feature row. Corrected origin features are unchanged when all
post-origin observations are removed or altered. The provenance validator checks
truthfully supplied metadata; it cannot establish unrecorded real-world as-of
availability on its own. No forecast model, recursion or benchmark is implemented.

## Diagnostics and estimator choices

Weak stationarity concerns constant expected mean, finite constant variance,
and covariance depending on lag rather than calendar position; a constant plot
is neither required nor sufficient. These descriptive diagnostics investigate
those assumptions without claiming to prove them.

Default eligibility requires >=84 finite observations, >=20 nonzero observations
and nonzero fraction >=0.10, with explicit skips for all-zero, near-constant,
missing/nonfinite, short or extremely sparse inputs. Nonzero applies to signed
differences too. A computational failure returns a failed/inconclusive status
and reason, never fabricated coefficients or p-values.

| Diagnostic | Convention |
| --- | --- |
| ACF | statsmodels; adjusted=False; FFT; common full-sample centered denominator |
| PACF | Yule-Walker maximum-likelihood convention, `method='ywm'` |
| Lags | Up to 56, capped strictly below half transformed sample size |
| Intervals | alpha=0.05; ACF Bartlett, PACF library asymptotic intervals |
| ADF | regression `c` by default; AIC autolag, maximum 14 within sample bound |
| KPSS | regression `c`; automatic bandwidth `nlags='auto'` |
| Rolling plots | Trailing 28-day mean and population variance, including current observation |
| STL | Robust additive, weekly period 7, regular eligible training series |

`ct` can be configured for trend specifications. ADF's null is a unit root under
the selected deterministic specification; KPSS's null is level stationarity for
`c` or trend stationarity for `ct`. P-value boundary warnings are retained (the
actual p-value may lie outside KPSS's tabulated range). The implementation
explicitly requests tuple/array return conventions when supported by the library
to avoid a pending result-object default change in statsmodels.

Paired outcomes: reject ADF only → compatible with stationarity under the test
assumptions; reject KPSS only → compatible with nonstationarity; reject both →
conflicting; reject neither → inconclusive. Intervals and tests are exploratory,
uncorrected for multiple comparisons and sensitive to seasonality, structural
changes, deterministic specification and sparse counts. They are not proofs of
mathematical stationarity, causal effects or automatic ARIMA order choices.

Correlation tables retain confidence intervals centered on each coefficient;
figures subtract the coefficient to draw zero-centered uncertainty bands.
Highlighted lags are 1/7/14/28 where supported.

First differences lose one initial observation, lag-7 seasonal differences lose
seven, optional second differences lose two. Original dates and undefined initial
values remain aligned; only that deterministic prefix is removed for testing.
Negative differences are valid. Differencing can leave nonstationarity or induce
unnecessary noise; no automatic transformation recommendation is made.

STL components reconstruct the original training observations and are tagged
`descriptive_only`. Even a training-only fit uses later training data in centered
smoothing of earlier components. The API refuses a forecasting-feature usage
mode. A prefix-at-origin fit may be inspected descriptively, but is not silently
reused as historical features at earlier origins.

## Outputs and verified observations

Local outputs: `data/diagnostics/train_20160228/`, entirely Git-ignored.

| Artifact | Grain / purpose |
| --- | --- |
| `summary.json` | Config, split, source fingerprint, versions, cohort, outcomes, warnings, runtime |
| `diagnostic_report.md` | Actual results, interpretation caveats, availability demonstration, figures |
| `tables/cohort.parquet` | One diagnostic series with kind, training segments and selection reasons |
| `tables/training_store_item_labels.parquet` | 30,490 training-only store-item labels |
| `tables/training_product_labels.parquet` | 3,049 training-only product labels |
| `tables/diagnostics.parquet` | One series/variant with ADF/KPSS, selected ACF/PACF, status and reasons |
| `tables/correlations.parquet` | Series/variant/lag coefficients and intervals |
| `tables/transformations.parquet` | Series/date observed sales, aligned differences and trailing plots |
| `tables/stl.parquet`, `stl_status.parquet` | Descriptive components/fit dates and explicit skips |
| `tables/historical_lags.parquet` | Training-only store/item/date target-relative lags |
| `tables/origin_features.parquet` | Store/item/origin/horizon inputs with source dates; no target actuals |
| `tables/training_scale.parquet` | Per-series training-only mean/std and fit cutoff |
| `tables/expanding_origins.parquet`, `rolling_origins.parquet` | History/target boundaries, no trained models |
| `figures/*.png` | Nine selected diagnostic panels, aggregate STL, weekly profiles, origin timeline |

On real data, **68** original/first/seasonal/second combinations yield **64 valid
ACF/PACF and paired stationarity results**, with **four intentional sparsity
skips** for CA_3/HOUSEHOLD_1_020; no unexpected numerical failures. **16 STL fits**
and one sparsity skip are recorded. Original-series paired outcomes: **13
conflicting**, **two compatible with nonstationarity**, **one compatible with
stationarity**, **one skipped**. These are selected-series exploratory results.

CA_3's training lag-7 ACF is **0.7599**, versus lag-1 **0.6811**; HOUSEHOLD's lag-7
ACF is **0.8814**. This is compatible with weekly dependence, with trend/structural
effects still possible. Numeric before/after results are retained rather than
claiming that differences always fix stationarity. There are **52** corrected
origin-feature rows (13 series × four horizons). Training ends before all
validation/test targets; future perturbation leaves features and fitted
parameters unchanged.

The first real-data run took **34.27s**, including SQL reductions, cohort-sized
diagnostics and 12 figures before final report publication. Python receives only
small label/cohort arrays; it does not load the full fact or product/day matrices.
DuckDB uses the existing two-thread/2GB managed-buffer configuration; native
statistical libraries may manage separate threads/memory. Actual runtimes depend
on hardware, caches and pauses and are recorded per run. Publication replaces
files individually after successful computation; it is not a multi-file atomic
transaction.
The final verified regeneration took **33.35s**. It retained **61** KPSS
tabulation-bound warnings across eligible combinations and no ADF warnings;
undefined/skipped estimates remain null rather than fabricated.

**41 Python tests** passed, including order/gaps/duplicates, four lag alignments,
series boundaries, differences, exact ACF convention, safe skips and simulated
numerical failures, non-brittle synthetic stationary-like/trending checks, STL
alignment and usage, origin availability and future-fitted/centered leaks, split
and target boundaries, and training-segment/cohort invariance under changed future
actuals. A fresh **36 dbt tests** passed without rebuilding warehouse models.
Raw/processed fingerprints and generated-output exclusions are verified before
publication. No ARIMA/ETS/Croston, ML, deep learning or inventory model is fitted.

Implementation references: [ACF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html),
[PACF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html),
[ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html),
[KPSS](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html)
and [STL](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html).
