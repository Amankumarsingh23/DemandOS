# DemandOS interview defense

## Thirty seconds

I built DemandOS to connect retail sales forecasts with explicit, testable
inventory decisions. It validates 58.3 million M5 observations, evaluates models
without future-data leakage, and uses a stateful simulator and shared-resource
optimizer to study cost/service trade-offs. Its native-model batch replay checks
provenance, retries and failures. Sales evaluation is historical; inventory,
supplier costs and causal outcomes are simulated rather than measured Walmart
results.

## Two minutes

The data unit is one store, item and date. I stream the wide CSV into Parquet,
check calendar/price joins, and expose dimensions and marts through DuckDB/dbt.
Origin-safe features separate known future calendar fields from unavailable sales
and price history. Models include baselines, pooled boosting and bounded neural
experiments, with chronological validation and fitted-through metadata.

Probabilistic forecasts are useful only when their evaluation is well defined.
I inspect pinball loss, empirical coverage and width, and calibrate from mature
scores. I do not add item quantiles to claim an aggregate quantile. The inventory
side starts with newsvendor reasoning, then uses a stateful receive/serve/order
timeline. A shared MILP prevents the same budget or supplier capacity being
spent repeatedly by independent item policies. Risk experiments show why better
planning-tail loss need not improve operating-tail cost.

The final replay fixes six series and one origin, loads an eligible native
checkpoint, produces complete 28-day forecasts, and independently checks simulated
stock, order identities, costs, persistent resources and temporal perturbations.
Scientific evidence is digest-linked and comparisons keep population, horizon,
economics and checkpoint provenance explicit. No real inventory state, randomized
retailer policy log or deployed supplier system is available, so causal and
business claims remain restricted to their declared model worlds.

## Ten-minute architecture walkthrough

Spend minutes 0–2 on the data grain, streaming reshape and SQL join conservation;
2–4 on origin contracts, chronological fits and global model families; 4–5 on
hierarchy/quantiles/calibration; 5–7 on newsvendor, the state machine and shared
MILP; 7–8 on CVaR and synthetic OPE; 8–9 on registry/publication/monitoring;
9–10 on the [evidence table](capstone_results.md), reproduction and limitations.
Use the [current diagram](architecture.md), not a mandatory chain through every
model. Distinguish actual source measurements, saved-record reconciliation,
current integration replay and proposed deployment architecture.

## Technical questions and worked challenges

| Domain | Repository-grounded answer and worked challenge |
| --- | --- |
| Data grain and joins | `data/` validates keys before chunked melt and many-to-one joins. 30,490 × 1,913 = 58,327,370. A weekly price reused by seven daily facts is valid; two price rows with the same store/item/week key can double those facts. Reject duplicate dimension keys before joining. |
| Missing demand/inventory | M5 has recorded sales, not verified latent demand or stockout labels. If sales are 0, demand may be 0 or unavailable stock may hide positive demand; without inventory neither explanation is identified. Keep missing price as missing, not zero. |
| Temporal features and validation | `features/` and `serving/features.py` require event and assumed availability no later than origin, with labels separate. At E=Jan31, a 7-day mean ends Jan31; a Feb1 realized price is unavailable. An example whose target date is Feb28 cannot train a checkpoint fitted through Jan31 even if its issue date is earlier. |
| LightGBM vs deep models | `boosting/`, `neural/`, `nbeats/`, `tft/` implement different bounded studies. Global pooling shares parameters; neural windows encode temporal structure but need train-only preprocessing and chronological tuning. A lower error on a different cohort/fit is not evidence of superiority. Match origins, targets, compute and fit eligibility first. |
| Intermittent sales | `intermittent/` separates occurrence and positive-size modeling. If probability of a sale is .2 and conditional mean size is 5, the unconditional mean is 1; a positive-only forecast of 5 is wrong for average daily sales. A zero can still be censored; a hurdle model does not identify hidden demand. |
| Hierarchy | `reconciliation/` uses grouped sums/coherence. Two complementary Bernoulli items can each have P90=1 while their sum is always 1; adding P90 gives 2, not the aggregate P90. Reconcile point forecasts or explicitly model joint sample paths. |
| Quantiles and calibration | `quantiles/` repairs crossings; `conformal/` uses eligible mature scores. With 9 exchangeable calibration scores and alpha .2, the split conformal rank is ceil(10×.8)=8. Temporal dependence/shift can invalidate the exchangeability guarantee; descriptive rolling coverage is not a universal promise. |
| Newsvendor and safety stock | `inventory/` distinguishes one-shot mismatch economics from operating cost. With underage=40 and overage=40, the critical fractile is .5. For lead-time mean 70 and standard deviation 10, a stated normal .95 heuristic gives target about 86.45; normality, independence and uncensored demand require justification. Do not replace a joint lead-time distribution by a sum of marginal quantiles. |
| State conservation | `simulation/engine.py` receives, serves old backlog, serves new demand, then orders. Start with stock 10, receive 3, demand 15: serve 13, lose 2, end stock 0. Ordering 5 after service puts 5 in the queue; it cannot serve today's demand. Verify receipts once by order identity. |
| MILP resources | `optimization/` coordinates a complete pool. Two requests of 10 units at price 60 with a 5 activation fee need 1,210, exceeding budget 1,000. Integer pack/MOQ and activation must be enforced alongside cash, supplier, flow and store limits; splitting a shared constraint across workers changes feasibility. |
| Risk | `risk/metrics.py` handles fractional boundary mass in upper-tail CVaR. For losses 0 with probability .8 and 100 with probability .2, CVaR at .75 is 80, because the worst quarter includes all .2 high-loss mass and .05 zero-loss mass. It is not 100. Planning mismatch loss differs from operating cost; the audited operating result can worsen. |
| Causal/OPE | `causal/` uses synthetic known propensities and chronological nuisance fits. If target action matches a reward 10 logged with probability .5, that observation contributes 20 to IPS before averaging over all observations. DR adds q(target)+weight×(reward−q(logged)); zero propensity cannot be fixed by an outcome model. Real M5 supplies neither propensities nor assignment ignorability. |
| Interference and policy effects | Items share cash and supplier quotas, so an item's allocation affects others. Compare whole-pool policy trajectories on paired paths; do not treat 25 item outcomes as independent randomized treatment units. S22's +3,008.89 modeled cost difference is not measured retailer treatment effect. |
| Registry and idempotency | `serving/registry.py` checks native file digests, dependency/feature contracts, cutoff and approval history. Same request with same content reuses its published batch; same identity with changed content blocks. A local atomic directory does not make an external order API exactly-once. |
| Drift and mature labels | `serving/monitoring.py` uses fixed reference bins for PSI and descriptive KS. At monitoring date Feb10 with 2-day lag, a Feb9 outcome is immature and masked; Feb8 may be scored. A drift warning can be informational rather than proof of accuracy loss. |
| Deployment and safety | No live supplier/inventory integrations exist. [System design](system_design_defense.md) proposes transactional reservations, outbox, acknowledgement reconciliation, identity/access controls and kill switches. Two workers each reserving 700 against the same 1,000 budget must coordinate; local independent feasibility is insufficient. |

## Defend the evidence

The S16/S17 served checkpoints have different fit/validation windows. The older
checkpoint's lower MAE and lower coverage on identical cases do not isolate a
method effect or justify automatic promotion. Within S17, the same neural
checkpoint's calibration comparison is more controlled. Within S20, lower modeled
cost comes with lower immediate fill. Cost definitions differ across one-shot,
planning and sequential studies. The [machine-readable registry](evidence_registry.json)
records those boundaries, and the [quickstart](capstone_quickstart.md) states exact
data/model prerequisites rather than promising weights in a fresh Git clone.

Engineering completion does not grade the learning assessments. Written
understanding assessments for Seasons 21–24 remain separate from these code and
scientific verification results; no assessment pass is asserted here.
