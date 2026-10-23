# Scientific evidence and comparability

Season 24 recomputes selected arithmetic from saved Parquet records. It does not
claim to rerun every training study. The machine-readable
[evidence registry](evidence_registry.json) supplies artifact SHA256s, source
fingerprint, checkpoint, cutoffs, population, origins, grain, units, assumptions,
outcome label and comparison key. Runtime copies also live in `evidence.json`.

## Artifact-backed historical evidence

| Evidence and population | Recomputed result | Interpretation/source |
| --- | --- | --- |
| S23, 25 series × 3 origins × 28 days, S16 native checkpoint, fit 2016-01-31 | MAE 9.393891; WAPE 60.4349%; mean pinball 2.858507; coverage 79.7143% | [Serving](production_serving.md); recorded daily sales; maturity ledger, 2,100 final cases |
| Same S23 cases, S17 older native checkpoint, fit 2015-10-11 | MAE 8.414804; WAPE 54.1360%; pinball 2.685633; coverage 77.6190% | Different fitted ensemble/windows: no isolated method improvement |
| S17 same neural checkpoint, 12 stress series, 3 origins, 1,008 daily targets | Raw 36.4087%; split CQR 79.7619%; rolling CQR 84.0278% coverage | [Conformal](conformal.md); repaired base versus calibrated intervals; temporal dependence remains |
| S20 continuity, 39 series-runs, balanced economics, budget .7, fixed lead 7 | Expected-mismatch MILP mean simulated cost 55,317.50 / fill 67.1109%; proportional 57,291.23 / 69.8623% | [Optimization](optimization.md); lower modeled cost accompanies lower immediate service |
| S21, 25-series pool, 3 day-zero origins, alpha .75 | Mean planning expected/CVaR 73,732.33 / 155,529.33; mean+2CVaR 74,673.67 / 152,934.67 | [Risk](robust_optimization.md); recomputed loss from scenario demand and frozen allocations |
| S21 baseline operating worlds, 8 worlds/origin, 3 origins, whole pool | Mean operating expected/CVaR 726,107.53 / 772,612.77; risk policy 729,076.29 / 774,309.37 | Different cost definition and sequential policy: planning CVaR improvement does not establish operating CVaR improvement |
| S22 paired baseline, 24 conditional whole-pool world/origin pairs | CVaR-minus-mean simulated cost +3,008.89; fill +0.3176 percentage points | [Causal evaluation](causal_evaluation.md); model-world intervention, no real treatment effect |
| S22 known-propensity ridge, repetition 0, deterministic mean target, 192 held-out context/world observations | IPS −328,355.35; SNIPS −314,571.51; DM −301,915.85; DR −317,327.77 reward | Negative simulated cost over 14 days; different estimator behavior, not a winner claim |

S16 validation ends 2016-01-03; S17 validation ends 2015-10-11. The S23 ledger
includes early and mature monitoring windows: final mature windows alone enter
these comparisons, avoiding duplicate counting of the same forecast.

## Direct comparison rules

| Pair | Directly comparable? | Conditions/boundary |
| --- | --- | --- |
| Within S17 neural base/split/rolling calibration | Yes, descriptive interval comparison | Same checkpoint, stress keys, dates, target, nominal level; calibration feedback must be mature |
| S16 vs S17 via S23 native replay | Same-case descriptive accuracy only | Different fit/validation windows and checkpoint; cannot attribute differences to calibration or a newer method |
| S18 one-shot vs S19 sequential cost | No | One protection-period mismatch cost differs from repeated procurement/holding/shortage accounting |
| Within S20 methods | Yes within frozen scenario | Match whole resource pool, cohort, costs, origins, lead, capacities, horizon and aggregation |
| S20 vs S21 or S21 planning vs operating | No generic cost ranking | Risk loss, scenario distribution and operating accounting differ; preserve separate comparison keys |
| Within S22 paired interventions | Conditional model-world comparison | Match complete shared-resource pool, path and supplier shocks; interference rules out independent item treatment assumptions |
| S22 synthetic OPE vs real M5 treatment effects | No | Actual logging propensities, assignment, inventory and causal identification are absent |
| S24 bounded demo vs all 30,490 series | No population generalization | Six predetermined series and one origin demonstrate integration and invariants |

## Scientific claim matrix

| Label | Permitted interpretation | Unavailable claim |
| --- | --- | --- |
| REAL_OBSERVED_SOURCE | Recorded M5 sales, calendar, observed prices | Uncensored demand or observed stockouts |
| HISTORICAL_FORECAST_EVALUATION | Retrospective eligible-checkpoint sales error/coverage | Official full M5 competition ranking |
| SIMULATED_INVENTORY_OUTCOME | Assumed stock/order/cost state trajectory | Measured Walmart inventory or savings |
| SIMULATED_OPTIMIZATION_OUTCOME | Feasible allocation under stated assumed resources | Verified operational procurement improvement |
| SYNTHETIC_POLICY_LOG | Known-propensity synthetic assignments/rewards | Real retailer propensities |
| MODEL_WORLD_CAUSAL_DEMO | Paired or identified effects inside the declared model world | Verified real-world causal uplift |
| LOCAL_SERVING_FAULT_EXERCISE | Isolated failure blocking/reuse and local recovery | Live service reliability or supplier delivery guarantees |

Every direct result additionally needs the same outcome, evaluation grain,
availability, support and economics. There is no cross-season grand leaderboard.
The independent capstone report distinguishes the current replay from these
saved-record reconciliations. See [quickstart](capstone_quickstart.md) and
[architecture](architecture.md).
