# Season 17: chronological conformal intervals

`demandos.conformal` implements absolute-residual split conformal, signed and
nonnegative CQR, chronological rolling CQR, a recency-weighted heuristic,
bounded delayed-feedback ACI and a complete-trajectory max-score diagnostic.
The target is recorded observed sales. Earlier model interfaces, datasets and
warehouse relations are preserved.

## Genuine held-out forecasts and the model-version distinction

Season 16's evaluation checkpoints were fitted through 2016-01-31. Their earlier
training errors cannot become held-out calibration errors by changing a label.
Likewise, its December selection-validation errors are not an independent
calibration sample. Applying historical errors from another fitted model while
calling the result ordinary same-model split conformal would be misleading.

Season 17 therefore fits separate historical versions of the same architectures:

| Stage | Dates | Actual fitting examples |
|---|---|---|
| Fixed training cohort | 96 series selected through 2014-01-04 | Established prefix-only selection |
| Nested development fit | Labels through 2015-09-13 | 116,928 tabular rows; 4,128 complete neural vectors |
| Nested validation | Origin 2015-09-13, targets 2015-09-14 through 2015-10-11 | 2,688 tabular rows; 96 neural vectors |
| Final historical refit | Labels through 2015-10-11 | 122,304 tabular rows; 4,320 complete neural vectors |
| Split calibration | Targets 2015-10-12 through 2016-01-31 | Four nonoverlapping 28-day issue blocks |
| Evaluation | Origins 2016-01-31, 2016-02-28, 2016-03-27 | Same original 13 continuity and 12 prefix-stress series |

Validation chooses **127/90/153** LightGBM rounds for P10/P50/P90, with validation
pinball **0.870572 / 1.888649 / 1.012882**. Neural validation chooses **epoch 7/8**,
normalized joint pinball **0.159819**. Three development/final tree pairs and one
development/final neural pair make **eight genuine fitted models**. Settings,
architecture, cohort/band definitions and calibrator budgets are fixed before
viewing this experiment's evaluation results.

The final October-11 checkpoints remain frozen for **both calibration and
evaluation**. Weekly reconstructed issues run from **2015-10-11 through
2016-03-27**, giving **35,000 held-out forecast-observation rows** across two
families, 25 series, 25 issue dates and 28 horizons. These are retrospective
historical forecasts, not a claim of deployment in 2015. Every score target
follows both the fitting cutoff and model-selection-label cutoff.

The ledger records store/item, issue/origin date, target date, horizon, family,
model ID, fitting/label/selection cutoffs, raw and repaired quantiles, point
prediction, actual sales, scores and observation availability. It assumes daily
observed sales are available **at the end of their target date**; M5 does not
provide independently verified production arrival timestamps. Delayed-availability
handling is implemented and tested separately.

Unchanged Season 16 final LightGBM/neural outputs and its historical baseline
are separate reference methods. They share target keys but use different fitted
versions. **No pre-January-31 split calibration is claimed for those final
checkpoints**, because that would require genuine held-out history they do not
have. Primary calibration comparisons instead hold the October model fixed.

## Scores, finite-sample rank and known support

Point intervals use the frozen model's P50 as a point predictor, with
`s = abs(y - P50)`. This is a median-point baseline, not a newly fitted mean
forecast. CQR uses repaired nonnegative P10/P90 endpoints:

`s_signed = max(P10-y, y-P90)`

`s_nonnegative = max(0, P10-y, y-P90)`

Scores and corrections are in original observed-sales units. For `[10,30]`, an
actual of 35 gives signed score 5; actual 20 gives -10 or nonnegative score 0.
Raw independent quantiles remain preserved; the same deterministic repair
precedes score construction and evaluation. No actual target determines repair.

For `n` eligible scores and miscoverage alpha, use
`k = ceil((n+1)*(1-alpha))`, with **one-based** kth sorted score. Decimal rank
arithmetic preserves integer boundaries. There is no interpolated NumPy
percentile substitution. With n=9, alpha=.2 and `[1,2,3,4,5,6,7,8,10]`, k=8
and correction=8: a point forecast 20 produces `[12,28]`. Ties retain the kth
order statistic. If k=n+1 (including no calibration observations), the explicit
policy is an extended infinite correction and interval `[0,+infinity)`.
Such intervals have infinite width/interval score and are counted explicitly;
none occurred in the real matched experiment.

CQR forms `[P10-c, P90+c]`. A negative signed correction can make this set
empty. The target-free policy uses `c_effective=max(c,-(P90-P10)/2)` and an exact
midpoint singleton when the shrinkage reaches that limit. This enlarges an
empty set and cannot remove an outcome from it. Final lower bounds are clipped
at zero, intersecting with known nonnegative observed-sales support without
removing a valid nonnegative outcome from an otherwise valid interval. No
midpoint enlargement was needed in the actual run.

Under exchangeable calibration/test scores and a fixed fitted predictor, the
corrected split rank supports finite-sample **marginal** coverage. It does not
promise conditional coverage for each SKU, origin or horizon. M5 time dependence,
scale heterogeneity, shared store effects and repeated/overlapping targets do
not establish exchangeability. The inherited benchmark cohorts were selected
and examined in earlier seasons; these are not pristine prospective tests.

## Split, rolling and weighted policies

Groups are predeclared **cohort × horizon band**: 1-7, 8-14 and 15-28, separately
for each frozen model family. No outcome-optimized regime routing is fitted.
All three benchmark origins use the same fixed split origins:

| Split issue | Its calibration targets |
|---|---|
| 2015-10-11 | 2015-10-12 through 2015-11-08 |
| 2015-11-08 | 2015-11-09 through 2015-12-06 |
| 2015-12-06 | 2015-12-07 through 2016-01-03 |
| 2016-01-03 | 2016-01-04 through 2016-01-31 |

Each family has 2,800 split-score rows: **1,456 continuity + 1,344 stress**.
The corresponding bandwise `(n,k)` are **(364,292), (364,292), (728,584)** for
continuity and **(336,270), (336,270), (672,539)** for stress. Exact band date
ranges differ: band 1-7 ends January 10, band 8-14 January 17, and band 15-28
January 31. Threshold tables preserve these dates and all selected score IDs.
Four issue blocks and repeated series do not make these row counts independent
replications. Signed/nonnegative split CQR produced identical actual thresholds
here because the selected signed order statistics were nonnegative.

Rolling CQR takes the most recent **512 mature pointwise scores per group**.
Eligibility requires both target date and observation-available date at or
before the current issue close. It does not use forecast issue date alone.
Sorting is availability, target date, issue, store, item, horizon; deterministic
ties can cut through a date block at the fixed-size boundary. This is an honest
recent-score rule, not an exact time-series exchangeability guarantee.

Weighted CQR uses the same 512 scores with weight
`2 ** (-age_days/28)`, where age is measured from observation availability.
The **28-day half-life** is fixed a priori to the forecast period, not chosen
from evaluation outcomes. We normalize weights to sum to one, sort scores
stably and select the first cumulative weight reaching `1-alpha`. There is
**no standard finite-sample rank correction or test-point mass**; the method is
explicitly a recency-weighted empirical heuristic. Weight sums, normalized
weights and Kish effective size are recorded; Kish size measures weight
concentration, not independent statistical information.

At later origins, rolling/weighted rules can use earlier evaluation outcomes
only after they mature. Models stay frozen and no setting is retuned on those
outcomes. Changing a future calibration candidate cannot change an earlier
threshold.

## Bounded ACI with precise feedback timing

One alpha state per family/cohort/band begins at **.2 on 2016-01-03**, using
mature October-onward scores for its initial rolling window. At each day close:

1. Process previously issued interval outcomes whose observation-available date
   is today, ordered by original issue, store, item, horizon and family.
2. For each binary miss, update
   `alpha_next = clip(alpha_current + .001*(.2-miss), .02, .5)`.
3. If today is a weekly issue date, use the updated state in the corrected rank
   of the current mature 512-score window, then issue all 28-day intervals.

Each queued event retains its issued endpoints; later corrections never rewrite
past endpoints or use not-yet-observed labels as completed feedback. Events
can overlap in forecast origin/horizon and are correlated. Calendar arrival
delays, ordering and long-run-prefix invariance are tested.

The real experiment issued **18,200 ACI intervals** and processed **14,700 mature
feedback events**, from **2016-01-04 through 2016-03-27**. Another 3,500 issued
outcomes remain pending at the last issue close and are not consumed to produce
those forecasts; they can still be scored retrospectively when observed later.
Alpha ranged **0.1064-0.2602**, with zero bound hits. Every before/after update,
miss, issue/target/feedback date, threshold and effective score window is saved.
Delayed overlapping feedback, bounded alpha and this event-wise scheme differ
from unbounded theoretical ACI; no exact conditional or arbitrary-shift guarantee
is asserted.

## Actual same-case daily results

There are **19 methods/reference systems**, each with **1,092 continuity** and
**1,008 stress** target cases: **39,900 intervals**. Coverage, widths, 80%
interval scores, lower/upper misses, zero inclusion and sample counts use exact
identical keys. No method is scored as available without its required history.
All mandatory comparisons and the pathwise diagnostic are available here.

| October-frozen method | Continuity coverage | Width | Interval score | Stress coverage | Width | Interval score |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM base | 70.79% | 35.9321 | 70.3859 | 85.02% | 2.4013 | 3.9464 |
| LightGBM absolute split | 72.34% | 26.1388 | 94.3145 | 78.67% | 1.8578 | 4.9220 |
| LightGBM split CQR | 76.10% | 37.1265 | 69.3707 | 85.02% | 2.4013 | 3.9464 |
| LightGBM rolling CQR | 77.66% | 38.4721 | 69.3948 | 85.22% | 2.4029 | 3.9467 |
| LightGBM weighted heuristic | 77.84% | 38.5293 | 69.4086 | 85.22% | 2.4022 | 3.9465 |
| LightGBM ACI | 79.40% | 40.7720 | 70.3493 | 85.22% | 2.4022 | 3.9465 |
| Neural base | 62.36% | 42.9579 | 68.4865 | 36.41% | 2.3943 | 4.0202 |
| Neural absolute split | 77.84% | 27.5671 | 68.9617 | 76.79% | 1.7772 | 5.1122 |
| Neural split CQR | 81.96% | 43.2083 | 68.4200 | 79.76% | 2.4154 | 4.0034 |
| Neural rolling CQR | 79.21% | 43.3730 | 68.4686 | 84.03% | 2.4690 | 3.9839 |
| Neural weighted heuristic | 79.30% | 43.3428 | 68.4725 | 83.73% | 2.4579 | 3.9872 |
| Neural ACI | 77.75% | 43.3085 | 68.4945 | 82.84% | 2.4843 | 3.9861 |

The separately retained January-fitted Season 16 references reproduce continuity
coverage **72.99% LightGBM / 60.99% neural**, and stress **87.00% / 37.60%**.
The historical empirical baseline remains **78.94% continuity / 94.05% stress**.
Do not attribute differences between the January and October base checkpoints
to conformal adjustment. Within the fixed October neural model, stress split
CQR changes coverage from **36.41% to 79.76%**, while mean width changes only
**2.3943 to 2.4154**. Small positive lower-bound adjustments can include a large
atom of genuine zero sales: zero-target coverage rises from **0% to 84.96%**;
rolling achieves **93.16%**, weighted **92.58%**, ACI **90.82%**. This is observed
sales calibration, not evidence that zeros reflect stockouts or latent demand.

Origin variation remains material. Neural split-CQR continuity coverage is
**79.40% / 87.09% / 79.40%** for January/February/March; stress is
**77.68% / 78.57% / 83.04%**. LightGBM ACI continuity is **65.66% / 94.51% /
78.02%** despite a pooled value close to 80%. Neural rolling stress coverage
is **79.76% / 83.33% / 86.51%** by horizon band, while split CQR is **80.56% /
80.95% / 78.77%**. A pooled number does not establish per-origin or per-horizon
calibration. Larger/wider intervals are not automatically better: coverage and
proper interval score must be considered together.

The report retains all **114 origin**, **1,064 horizon**, **114 band** and **133
origin-safe regime** result rows. Count-weighted horizon coverage, width, score
and miss rates reproduce pooled metrics. Wilson intervals are illustrative
binomial summaries only; dependence invalidates a general iid interpretation.

## Whole-trajectory calibration is a separate objective

The pathwise score is the maximum signed CQR violation over all 28 targets of
one store-item/origin trajectory. Every target must be observed before the
trajectory is eligible. At least three distinct issue blocks and ten complete
trajectories are required; insufficient support returns an explicit unavailable
status. These are feasibility gates, not proof of statistical independence.

The four disjoint split blocks yield **52 continuity / 48 stress trajectories
per family**, with corrected ranks **43 / 40**. A single groupwide correction
is applied across all 28 horizons. The resulting whole-path results differ from
daily coverage:

| Family/cohort | Whole-path coverage | Daily coverage | Mean width | Daily interval score |
|---|---:|---:|---:|---:|
| LightGBM continuity | 89.74% | 98.99% | 96.0714 | 99.2243 |
| Neural continuity | 82.05% | 97.53% | 81.8587 | 88.1977 |
| LightGBM stress | 75.00% | 98.71% | 5.0738 | 5.2073 |
| Neural stress | 69.44% | 98.81% | 5.0951 | 5.2092 |

Even the pathwise procedure does not empirically attain 80% on the stress
sample. For ordinary neural split CQR, whole-path coverage is only **20.51%
continuity / 8.33% stress** despite approximately 80% daily coverage. Only 39
continuity and 36 stress evaluation trajectories exist; small samples, shift
and dependence preclude strong conditional claims.

## Reproduction and integrity

```powershell
.\.venv\Scripts\python -m demandos.conformal --root . --config configs/conformal.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,warehouse,env=settings(Path('.')); invoke_dbt(root,env,'test')"
```

Use the existing analytics/boosting/neural/reconciliation environment, verified
warehouse and Season 16 source-bound reference artifacts. No dependency or
external infrastructure addition is needed. The actual run took **290.18 s**,
with **1,299.6 MiB** peak process working set, two CPU threads and the existing
2 GB DuckDB limit. It loads bounded series projections, not the 58-million-row
fact panel into RAM. There were zero training warnings, failed/skipped cases,
infinite real intervals or signed empty-set enlargements.

All **239 Python tests** (23 conformal tests) and **36 fresh dbt tests** pass.
Nine real predictor mutation/deletion/price gates and nine real calibrator gates
verify unchanged eligible histories, models' predictions, score windows,
thresholds, issued intervals and ACI states. Future calibration candidates are
filtered before inspecting their labels; unavailable future labels can be
removed without changing earlier state. ACI delay, ordering, prefix replay and
rank indexing are explicitly tested. Original CSV/interim/processed hashes,
Season 16 hashes and warehouse totals remain unchanged: **58,327,370 rows**,
**65,695,409 recorded units**.

Ignored output root: `data/conformal/m5_h28/`:

- `models/`: eight nested development/final native checkpoints and metadata.
- `tables/forecast_score_ledger.parquet`: 35,000 genuine held-out forecasts,
  scores, exact dates/IDs and raw/repaired quantile provenance.
- `tables/intervals.parquet`, `thresholds.parquet`: evaluated intervals,
  corrections/ranks, exact score IDs/date ranges and weighted window states.
- `tables/aci_issued_intervals.parquet`, `aci_feedback.parquet`: every issued
  ACI interval and every mature before/after update, including warm-up issues.
- `tables/path_availability.parquet`, `trajectory_cases.parquet`,
  `metrics_trajectory.parquet`: full-trajectory maturity/support and coverage.
- `tables/metrics_*.parquet`, `predictor_checks.parquet`,
  `calibrator_checks.parquet`, cohort and inference-audit tables.
- `summary.json`, `verification.json`, `conformal_report.md`, two figures and
  `manifest.json`. All 39 manifest-owned artifacts exist; repeat publication
  replaces only manifest-owned files.

No inventory optimization or later-season decision policies are implemented.

## References

- [Angelopoulos and Bates, Conformal Prediction: A Gentle Introduction](https://mlanthology.org/ftml/2023/angelopoulos2023ftml-conformal/).
- [Romano, Patterson and Candes, Conformalized Quantile Regression](https://papers.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html).
- [Gibbs and Candes, Adaptive Conformal Inference Under Distribution Shift](https://papers.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html).
