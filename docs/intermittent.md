# Intermittent observed sales, cold starts and frozen segment policies

Season 11 reuses the validated warehouse, native LightGBM interface, historical
training panel, feature contracts and backtest metrics. It adds historical
diagnostics, a two-part hurdle mean model, explicit cold-history simulations,
leave-one-item-out priors, and a small validation-selected regime policy.
Earlier pipelines and model interfaces are unchanged.

## Reproduce

Use the existing Season 9 tested CPU environment and prepared M5 warehouse:

```powershell
.\.venv\Scripts\python -m pip install -c configs/analytics-constraints.txt -c configs/boosting-constraints.txt -e ".[analytics,eda,boosting]"
.\.venv\Scripts\python -m demandos.intermittent --root . --config configs/intermittent.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,output,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

No new dependency or automatic data download is introduced. Python 3.13.15 and
LightGBM 4.7.0 are the tested versions; pinned boosting installation requires
Python 3.12+. `configs/intermittent.toml` references the unchanged Season 9
training configuration, caps cohort sizes and sets the diagnostic/prior windows,
shrinkage strength and validation-evidence thresholds.

## Sales are not uncensored demand

In the simplified illustration `sales=min(demand, availability)`, a zero can
arise from no demand, no availability, or other measurement circumstances. M5
does not provide verified inventory, availability, stockout, lost-sales or item
introduction ground truth. Missing selling prices, zero runs, high ADI or level
changes do not identify actual stockouts. No supervised stockout detector,
uncensoring model, exact lost-sales estimate, inventory policy or business saving
is implemented. No synthetic latent-demand/censoring rows are mixed into M5.

Cold experiments use **actual M5 products and actual future observed sales**,
while deliberately withholding histories/metadata. The information restriction
is simulated; it is not evidence that those products really launched then.

## Historical diagnostics

For a permitted daily prefix with N observed days and K positive days:

`ADI=N/K`, `CV²=population_variance(positive_sizes)/mean(positive_sizes)²`.

Diagnostics default to the last **365** observed calendar days; a separate
**56-day** summary describes recent regime changes. Export observed/positive
counts, zero/positive fractions, positive mean/population variance, ADI/CV²,
days since the last observed positive sale, status and trailing indicators.
Visible-history length is reported separately from the diagnostic window size.

| ADI | Positive-size CV² | Regime |
| --- | --- | --- |
| <1.32 | <0.49 | Smooth |
| <1.32 | >=0.49 | Erratic |
| >=1.32 | <0.49 | Intermittent |
| >=1.32 | >=0.49 | Lumpy |

Threshold equality belongs to the high side, matching the existing heuristic.
No observations and all-zero observations are different statuses; both have
undefined ADI/CV². A single positive observation has population variance zero
but CV² is withheld as insufficient evidence for regime classification.
No full-history Season 4 labels enter prediction-time routing. Prefix filtering
happens before checking values, so corrupt/changed future rows cannot influence
the current diagnostics. Missing or duplicate days inside visible history fail
validation; absent observations are never filled with fabricated zeros.

## Global hurdle mean model

The model estimates:

`p=P(recorded sales>0 | origin features)`

`m=E[recorded sales | recorded sales>0, origin features]`

`expected recorded sales=p*m`.

Binary LightGBM with binary log loss learns occurrence targets `1(y>0)` on all
eligible fitting examples. A second LightGBM uses untransformed squared-error
regression **only on positive labels**, targeting a conditional mean. No median,
uncorrected log transformation or latent-demand interpretation is used. The
size score is projected to at least 1, consistent with a positive integer sales
target's conditional mean; its raw estimate is also retained. Overall expected
sales can be fractional and are nonnegative. Probabilities must lie in [0,1].

This is a two-part **point-mean** model, not a fitted full count distribution.
A distributional count hurdle would model a positive distribution truncated at
zero. A zero-inflated Poisson instead combines an extra-zero mixture component
and a Poisson component that can itself emit zeros. Such mixtures do not assign
causal stockout labels. The optional ZIP comparator is omitted; high zeros alone
do not justify its dispersion assumptions. See the
[LightGBM objective documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
and [distributional hurdle reference](https://www.statsmodels.org/stable/examples/notebooks/generated/count_hurdle.html).

Degenerate occurrence partitions use the observed constant class probability.
Fewer than 20 positive labels use their observed positive mean, explicitly
marked as a small-partition fallback. With no positive labels, return zero
occurrence/expected sales and declare that a positive-size model was not learned.
Constant fallbacks are not represented as trained native models. Unknown
categories use fit-only vocabularies/unknown tokens; numerical missingness stays
native NaN. Native files and JSON state support reload-equivalent inference.

Croston, SBA and TSB reuse earlier implementations: Croston smooths positive
size and event intervals, SBA applies its bias adjustment, and TSB updates the
occurrence probability even on observed zero days. They estimate recorded sales
behavior. No positive history yields an explicit zero rate; one positive event
is an explicit insufficient-events skip under the existing contract.

## Cold history and priors

Detect no history, an existing item in a new store, fewer than 7/28 visible
periods, historically observed all-zero sales, and unknown category metadata.
The real experiment uses three distinct training-unseen items for each mode:
`new_product_0`, `new_store_0`, `short_3`, `short_7`, `short_28`,
`unknown_metadata_0`. All 18 cold item IDs are excluded from the entire global
training pool, including other stores; global models therefore cannot memorize
hidden item outcomes. Metadata is assumed available for the named simulations
except the explicit unknown-category case.

History is restricted relative to each issuance, preserving original recorded
values inside that window. The standard feature schema requires an actual
observation at origin, so zero-history pooled/hurdle rows are explicitly skipped.
No artificial all-zero ML feature row is manufactured. Short histories use
genuinely missing lags and incomplete-window indicators. Baselines retain their
own minimum-history/positive-event requirements.

Priors aggregate **only the preceding 56 days through issuance** in DuckDB:
store-category first, then category across stores, with at least 56 observed
donor rows. The target item is excluded from every donor store in both levels.
This is stricter than excluding only future item outcomes and prevents hidden
item history from contaminating new-product priors.

`shrunk_mean=(n*visible_observed_mean+kappa*prior_mean)/(n+kappa)`

Default `kappa=28`; n counts all visible daily observations, including recorded
zeros, not the 365-day diagnostic window. At n=0 use the supported prior. If a
prior is unavailable but observations exist, use their explicitly labeled mean.
If neither is available, report unavailable; never silently borrow a global
mean. Export donor rows/items, excluded item ID, origin cutoff, prior source,
observed n/mean and fallback status. A missing category is not assumed to be an
existing category or assigned fabricated metadata.

M5's calendar-wide records do not verify pre-introduction availability. Recorded
zeros remain observed zeros, while the deliberately withheld dates in a cold
simulation remain unavailable. First positive sale is not an introduction label.

## Fixed temporal experiment and policy selection

Training remains **96 early-prefix-selected series**, **58 fortnightly origins**,
and **154,560** eligible pooled rows through March 27. Development fits use
**133,056** labels observed by **2015-12-06**; validation has **2,688** targets
on December 7–January 3. Every historical feature respects its own row origin
and every fitting target must satisfy `target_date<=allowed_cutoff`. Occurrence
and size stopping use their respective eligible validation partitions. All
iteration counts, routes and blend choices freeze before evaluation.

Evaluation origins remain **2016-01-31, 2016-02-28 and 2016-03-27**, each with
28 target days, including February 29. Origin-specific refits may incorporate
earlier outcomes once observed, without choosing settings from their scores.
Native CPU settings retain seed 42, two fitting threads, maximum 160 rounds,
15-round patience and the Season 9 feature allowlist. Prices remain unavailable
without verified as-of evidence; scheduled calendar/SNAP/event availability
remains the earlier stated assumption.

The continuity cohort has **13** existing benchmark series. A separate stress
selection uses only the first origin's preceding 365 days: up to four real
series each for all-zero, one-positive, intermittent, lumpy, smooth and erratic
conditions. Sparse regime examples prefer fewer positive days with seeded
identity-hash tie breaks; other groups use stable hashes. Cold examples are
18 distinct unused item IDs sampled by stable hash, independently of holdout
outcomes. Actual group counts are reported rather than padded with fake series.

Router candidates are pooled, hurdle, TSB and seasonal naive. Each regime needs
at least **five distinct validation series / 140 common target rows** for a
specialist. Minimum validation MAE chooses the route. The conservative blend
grid is pooled alone or 50/50 pooled with hurdle, TSB or seasonal naive.
Nonnegative weights sum to one. Insufficient/unseen regimes default to pooled.
Histories shorter than the configured 28 days override to cold priors. Missing
requested candidates trigger an explicit logged pooled/prior default, or an
unavailable status if neither exists. There is no per-item optimized weight or
retrospective per-series winner selection.

The manifest stores policy version, validation end, candidate losses, sample
counts and evidence SHA-256. Every decision records requested/effective weights
and fallback reason. Current regimes update from permitted observed prefixes;
the rule mapping itself never updates from evaluation losses.

Early stopping and policy selection reuse this same December tuning block; the
policy's validation losses are not an independent assessment of its selection
procedure. This is a small fixed-budget experiment, not nested model selection.

These January–March outcomes were already examined in earlier seasons. They
remain a retrospective known benchmark, not untouched confirmation data.

## Measured real-data findings

All 55 series use actual M5 targets: 13 continuity, 24 sparse stress and 18
cold simulations. Ten methods over three origins and 28 days produce **46,200
planned rows**, with **38,976 successful / 7,224 explicitly skipped**. There are
no failed global fits. All **12 native models** (four pooled plus four binary
and four positive-size children) serialize/reload correctly; 11 recorded native
warning entries describe no further positive-gain splits. The 825 local baseline
attempts contain **648 successes / 177 skips**, reflecting minimum observation/
positive-event requirements and missing cold histories.

Selected rounds are **141 pooled / 160 occurrence / 125 positive size**. The
occurrence model reached the declared budget; it was not expanded after looking
at evaluation. Positive-size fitting rows are **75,791** in development and
**81,513 / 84,897 / 88,295** at the three refits, a subset of **143,808 /
149,184 / 154,560** full fitting rows respectively.

On the identical 13-series, three-origin continuity grid (**1,092 targets each**):

| Method | MAE | Pooled WAPE | Macro RMSSE |
| --- | ---: | ---: | ---: |
| Pooled | 12.898 | 44.695% | 0.725 |
| Hurdle | 13.382 | 46.372% | 0.746 |
| TSB | 13.015 | 45.102% | 0.729 |
| Seasonal naive | 13.132 | 45.506% | 0.789 |
| Cold prior/shrinkage | 15.447 | 53.529% | 0.828 |
| Frozen router | 13.320 | 46.159% | 0.731 |
| Frozen ensemble | 12.960 | 44.910% | 0.715 |
| Moving average | 13.076 | 45.313% | 0.742 |
| Croston | 13.809 | 47.853% | 0.756 |
| SBA | 13.635 | 47.249% | 0.755 |

These ten methods happen to share all continuity keys. Macro RMSSE has 38 valid
cases and one undefined zero-history scale. Pooled reproduces Season 9 exactly.
The frozen ensemble slightly improves macro RMSSE but loses MAE/WAPE to pooled;
the router and hurdle do not win overall. Ensemble MAE is lower at days 1–7
(**9.636 vs 9.677**) and 8–14 (**12.711 vs 12.815**) but higher at days 15–28
(**14.746 vs 14.549**). These are descriptive small-cohort comparisons.

On **1,960 shared sparse-stress targets / 70 cases**, TSB has WAPE **56.745%**,
hurdle **60.033%**, pooled **64.814%**, router **63.750%**, ensemble **63.948%**
and seasonal naive **66.719%**. Two sparse cases lack TSB's required positive
events and are excluded from that primary intersection, while their available
forecasts remain visible. A 365-day all-zero diagnostic window does not imply
the entire earlier M5 history was all zero; older observations can yield valid
full-visible-history scales.

The cold seven-method intersection has **504 targets / 18 cases**, all from
supported 7/28-day histories. Hurdle WAPE is **87.924%**, TSB **89.004%**, pooled
**106.295%**. Low target totals make large WAPE possible. No-history cases are
not included in that ranking. Prior/router support covers **1,260** cold targets;
pooled/hurdle support covers **756**, so their available-case scores must not be
compared as matched rankings. Unknown-category/no-history simulations provide
no supported prior or learned model and explicitly skip all methods.

Occurrence Brier scores are **0.163264** on 1,092 continuity targets,
**0.124317** on 2,016 sparse targets and **0.215825** on 756 supported cold
targets. Calibration is imperfect: continuity predictions in the lowest bin
average **0.145**, while positives occur at **0.404**; supported cold bins also
underpredict occurrence. No test-period recalibration is applied. Conditional
positive-size MAE is **12.489** on 847 realized-positive continuity targets,
**2.222** on 920 sparse positives and **1.011** on 317 supported cold positives.
These are conditional populations, not unconditional point-error rankings.

December policy evidence selects TSB routes for erratic/lumpy, pooled for
intermittent and hurdle for smooth. Selected blends are pooled/TSB 50/50 for
erratic, pooled/seasonal-naive 50/50 for lumpy, pooled/hurdle 50/50 for smooth,
and pooled alone for intermittent. The all-zero regime has only one validation
series, so it retains the predefined pooled default. Rules are not changed to
repair later losses. Minimum evidence can prevent a tempting specialist route.

The final experiment took **246.12 seconds** locally. Four-fold pooled fitting
took **13.71s** and the four two-child hurdle fits **47.60s**. Prior construction,
all candidate predictions and policy application took **9.94 / 9.89 / 11.22s**
at the three evaluation origins, excluding feature generation and invariance
replay. These machine-specific measurements are not intrinsic model speed ratios.
Native files total **5,321,204 bytes**, excluding JSON and reports. Only
**258,255 selected real daily rows** enter Python; full-panel prior reductions
remain in SQL. DuckDB retains its two-thread/2GB managed buffer setting, not a
hard ceiling on total process memory.

## Evaluation and artifacts

Primary comparisons use pooled, hurdle, TSB, seasonal naive, cold prior, router
and ensemble on **identical successful keys**, separately by cohort. Additional
Croston/SBA/moving-average results preserve their own support and paired pooled
comparison keys. Available-case numbers are not all-method rankings. Zero-
history cold cases can have only prior/router support; empty shared comparisons
are not disguised as complete experiments.

MAE, pooled/per-series WAPE, training-only first-positive-trimmed RMSSE, explicit
undefined denominators, origin/horizon/band/regime/cold-status coverage and paired
losses reuse Season 7 metrics. Cold RMSSE scales use **only visible history**,
never the withheld observations. Occurrence reports Brier score and five fixed
calibration bins. Conditional-size MAE uses realized positive target observations
among supported hurdle cases, with this conditioning stated explicitly.

Ignored outputs live under `data/intermittent/global_h28/`: native pooled and
hurdle states, `policy.json`, exact forecasts and probability/size components,
diagnostics, cold priors, policy decisions, validation evidence, fit/coverage/
metric tables, four figures, `summary.json`, `experiment_report.md` and a file
manifest. The experiment CLI does not run the test suite itself; reviewed
Python/dbt/dependency/artifact checks are recorded separately afterward.

Raw/interim/processed hashes and warehouse totals are hard gates. Real checks
change every post-origin sale and future-week price, regenerate features,
diagnostics and priors, and require frozen loaded model probabilities, sizes,
forecasts, routes and weights to agree. Future-label changes must preserve
eligible fitting rows. Default price tests establish exclusion rather than
verify unavailable price-announcement metadata. Synthetic fixtures test edge
cases; they are clearly separate from actual M5 evaluation observations.

No official M5 WRMSSE, actual inventory saving or true-demand recovery is claimed.
No Season 12 neural forecasting or unrelated infrastructure is included.

Final regression evidence: **118 Python tests** (13 new Season 11 synthetic
tests), no skips/errors, and **36 fresh full-source dbt tests** passed. Native
serialization, same-case metrics, cold scales, policy evidence, prior arithmetic
and all-origin future perturbations have separate real-artifact checks. The
ignored `test_results.json` records the reviewed code fingerprints and outcomes;
it is removed as stale by the manifest publisher on a subsequent experiment.
