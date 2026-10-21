# Season 22: causal and off-policy evaluation

All new assignments, rewards, effects and intervals are **SIMULATED / SYNTHETIC
LOGGING / MODEL-WORLD**. M5 has recorded sales, not Walmart policy assignments,
logging propensities, inventory decisions, verified costs or uncensored demand.
This implementation does not identify actual Walmart treatment effects/savings.

## Architecture and scientific questions

Forecasting predicts sales; optimization selects orders under assumptions;
simulation advances hypothetical inventory. Causal claims additionally need a
defined intervention, target population and defensible identification assumptions.
Better forecast error or planning CVaR does not establish real policy impact.

`demandos.causal` separates estimator math, known-randomization logging,
chronological nuisance fitting, controlled demonstrations, bounded source access,
single-allocation simulations, complete-controller interventions, independent
verification and reporting. It reuses Seasons 19–21 without changing their APIs,
defaults or earlier models. Existing NumPy/SciPy/Pandas/scikit-learn are sufficient.

## Track A: a single contextual decision

The intervention unit is the **whole 25-series resource pool**. Action 0 is the
mean-mismatch allocation; action 1 is mean+2CVaR(.75). Each action is a complete
feasible integer vector with shared budget, supplier, procurement-flow and store
quotas. The builder retains activation/pack/MOQ semantics; this catalogue uses
pack 1/MOQ 0. There is no independent item randomization followed by clipping.
If two algorithms produce the same vector, their potential outcomes must agree.

At each origin, 12 predeclared contexts cross initial stock 0/10/30, budget
factors .10/.70 and flow factors .50/.80. Existing prepaid pipeline is three
units/item. Targets and eight planning points are regenerated from eligible
history and verified against Season 21's frozen manifest. Pool historical mean,
zero rate/CV, inventory, resources and action descriptors are known before
assignment. Future sales/prices, supplier events, later stock and rewards are
not context or propensity features.

Issue one vector at E with seven-day lead; run the original engine for 14 days
with **no further orders**. Reward is negative procurement/activation/holding/
lost-sales cost. Because the engine treats day-zero orders as prepaid, add new
vector procurement/activation cash exactly once; keep previous pipeline prepaid.
No Cu/Co mismatch expense is added to operating cost. Terminal credit is zero;
the outcome matures E+14. This is not a complete 28-day controller estimand.

For each pool context/world, define potential reward `Y_i(a)`,
`V(pi)=mean_i sum_a pi(a|X_i)*Y_i(a)`, and
`ATE=mean_i[Y_i(1)-Y_i(0)]`.
Positive reward ATE favors CVaR on cost. ATT conditions on treated units; the
audit reports the finite actual-logged-treated oracle ATT and a normalized
odds-weighted control estimate. These noisy finite targets need not equal ATE.

## Logging and identification

`z=1.5*(1-stock/30)+(.7-budget)/.6+.5*(.8-flow)/.3-1`;
`mu(1|X)=.2+.6*sigmoid(z)`, `mu(0|X)=1-mu(1|X)`.
Uniforms are SHA-keyed by seed/context/world/replication in a different domain
from demand sampling. These are synthetic known probabilities, not Walmart logs.

All-action outcomes stay in `oracle/`. Estimator inputs contain only selected
reward, exact propensity/full vector, eligible actions, predecision context,
maturity and provenance. Nuisance/estimator APIs have no oracle path/table
argument. Extra future/oracle columns are rejected; changing evaluation rewards
cannot change predictions. Oracle joins occur only after nuisance fitting.

Identification assumes consistent treatment versions, exchangeability from the
known randomization, positivity on target support, explicit interference
boundaries, observable selected outcomes and mature feedback. Within-pool
competition is part of the intervention; separate resets eliminate cross-episode
carryover by construction. Real observational data do not inherit these
properties simply because a regression model was fitted.

## IPS, SNIPS, DM and DR

With `w=pi(A|X)/mu(A|X)`:

- `IPS=mean(w*R)`.
- `SNIPS=sum(w*R)/sum(w)`; zero denominator is unavailable, not silently zero.
- `DM=mean(sum_a pi(a|X)*m_hat(X,a))`.
- `DR=mean(sum_a pi(a|X)*m_hat(X,a)+w*(R-m_hat(X,A)))`.
- `ESS=(sum w)^2/sum(w^2)`; no sample matches gives ESS zero.

Weight ESS counts equally weighted equivalents in that logged sample, not
independent real resource pools or demand-world clusters.

Actions [1,0,1,0], rewards [10,4,8,6], uniform logger, always-1 target give
weights [2,0,2,0], IPS 9 and ESS 2. One matching reward 10 in four uniform
assignments gives IPS 5 versus SNIPS 10. A=1, R=12, mu=.25 and m(1)=10 give
DR contribution 18. With a target/action mismatch the correction vanishes but
the direct component remains. SNIPS is not generally unbiased.

Targets are always mean, always CVaR, a 50/50 mixture and a fixed risk rule:
CVaR probability .8 for z>=0, otherwise .2. Vectors must normalize within
absolute tolerance 1e-12. Rewards/actions must be finite and aligned.

Missing true target support returns **NONIDENTIFIABLE_MISSING_SUPPORT**, including
for DM; prediction/extrapolation is not identification. Positive estimated
probabilities cannot repair zero true support. Finite sample zero matches are
different: IPS zero is computable, SNIPS is undefined. Cap-2 clipping is a
separate biased sensitivity and never repairs positivity.

## Chronological nuisance learning and repeated logging

Development is January 31, 2016: its 14-day outcomes mature February 14, before
February 28/March 27 evaluation. Each assignment replication has 96 development
and 192 held-out evaluation units; context clusters cannot overlap. Scalers
fit development only. These origins were repeatedly studied in earlier seasons;
they are retrospective benchmarks, not prospective untouched test data.

Outcome models are Ridge(alpha=10), a 64-tree forest (depth 5, minimum leaf 4,
one CPU job), and a deliberately misspecified pooled intercept. Action-feature
interactions and catalogue descriptors are predecision information. A separate
standardized logistic model estimates propensities from development actions,
never rewards. It only approximates the bounded true logging function. True
versus estimated propensity results, training/maturity cutoffs, selected-data
hashes and features remain separate and auditable.

Eight eligible-history seven-day-block worlds per origin are shared across
context strata/actions. Tracks A/B and planning have distinct sampling streams.
Thirty-two assignment replications reuse the **same finite potential-outcome
table**; repeated rows are not extra independent demand worlds. Estimates,
bias/variability/RMSE are scored against exact finite model-world oracle values.
Counterfactual nuisance RMSE is an audit of realized outcomes, which include
world noise; it does not prove conditional-mean correctness.

## Controlled confounding and misspecification

Independent synthetic units have X~Bernoulli(.5), common epsilon~N(0,1),
`Y(a)=10X+2a+epsilon`, and `mu(1|X)=.1+.8X`. Population V(always1)=7 and
ATE=ATT=2. Correct nuisances are known structural conditional functions; wrong
ones omit X (uniform propensity and constant outcome prediction 5). This
isolates the four double-robustness conditions separately from learned M5 models.

Two hundred replications of 2,000 units report every estimator's bias, standard
deviation, RMSE and nominal Wald coverage. SNIPS uses its ratio influence
approximation. Coverage under misspecification is a diagnostic, not a valid
promise. Double robustness is conditional/asymptotic, not finite-sample accuracy
or a remedy for missing support/unmeasured confounding. The naive confounded
difference is compared with an independent randomized benchmark and adjustment.

## Track B and sequential OPE

Complete mean versus mean+2CVaR controllers run for 28 days from freshly reset
stock, pipeline, budgets and supplier commitments, under identical exogenous
worlds. Baseline and joint surge/receiving-closure each include observed-sales
proxy paths and eight additional bootstrap worlds per origin: 108 complete
policy runs. Paired cost/fill/lost-unit differences, delay exposure in unit-days,
store allocation disparity, cohort service and operating VaR/CVaR are reported.
The queue retains delayed order identities/units and committed cash. Full reset,
future-day prefix and hidden-unmet-demand invariance are hard gates.
Demand is exogenous to policy within this model: customer response, substitution
and uncensored demand are not identified from M5 recorded sales.

A separate bounded stateful binary-action demonstration uses complete weights
`W_T=exp(sum_t(log pi(A_t|H_t)-log mu(A_t|H_t)))` for T=1/7/14/28.
Ratio 2 for 28 steps gives **268,435,456**. Diagnostics include ESS, matching,
variance, support/no-match statuses and complete-return IPS/SNIPS against an
on-policy Monte Carlo benchmark. This is illustrative, not inventory-controller
reliability evidence. Daily one-step DR is never labeled general sequential OPE.

## Uncertainty and real evidence

Primary intervals resample eight complete world panels, carrying all context
strata/origins together. Nuisances remain fixed in bootstrap draws; fitting
uncertainty is omitted. Controller contrasts use the same block principle.
Individual items and 2,100 daily targets are not independent policy experiments.
One real historical resource pool and three overlapping/dependent origins do
not support reliable real-world coverage/significance. Eight synthetic blocks
give conditional descriptive intervals; observed-path comparisons decline a
real CI. Controlled IID structural experiments report separate nominal coverage.

The [future cluster-randomized trial plan](causal_experiment_design.md) specifies
missing logging, compliance, randomization and carryover evidence. No trial or
real policy deployment occurred.

## Reproduction and artifacts

```powershell
.\.venv\Scripts\python -m demandos.causal --root . --config configs/causal.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,output,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

Run dbt before/after the runner closes its warehouse connection. No full-panel
materialization, new dependency or earlier-model refit. Ignored output root:
`data/causal/m5_h28/`. It contains contexts/catalogue, selected logging ledger,
propensity/ESS diagnostics, nuisance predictions/provenance, all OPE estimates
and repeated errors, oracle values/ATE/ATT, confounding/misspecification tables,
stateful ledgers/paired metrics, sequential diagnostics, future checks and report.
`oracle/` separately holds all-action outcomes and simulator ledgers. An
unverified diagnostic copy is never a published successful result. Independent
metric/conservation/temporal audits, unchanged source/model fingerprints and
exactly 58,327,370 warehouse observations gate manifest publication.
Fixed random streams do not guarantee identical time-limited MILP incumbents
across hardware; manifests retain actual vectors, solver status and gaps.

## Verified bounded M5-informed run

All numbers here are **SIMULATED / SYNTHETIC LOGGING / MODEL-WORLD**. The actual source is 25 M5 series (13 continuity / 12 sparse-stress) at January 31, February 28 and March 27, 2016: 75 observed 28-day paths / 2,100 daily target rows. The unchanged warehouse has 58,327,370 records across 30,490 series. No actual policy decisions, propensities or true lost demand are supplied by M5.

Across 36 predecision contexts and eight worlds, 576 single-allocation interventions produced 201,600 daily rows. Thirty-two assignment replications produced 9,216 selected-reward logs, with 96 development / 192 held-out evaluation units each. There are 9,312 audited estimates including 96 separately labeled cap-2 sensitivities. Repeating assignments does not create new independent demand worlds.

The following are 32-replication means using the bounded forest and known propensities over the 192 held-out units; rewards are negative assumed operating cost. The oracle is exact for this finite world/context distribution.

| Target | Oracle | IPS | SNIPS | DM | DR |
|---|---:|---:|---:|---:|---:|
| cvar | -311927.71 | -313986.33 | -312587.45 | -306064.66 | -312148.47 |
| mean | -312293.03 | -310146.92 | -311470.96 | -304081.01 | -311876.90 |
| mixture | -312110.37 | -312066.62 | -312010.54 | -305072.84 | -312012.68 |
| risk_rule | -312173.71 | -313995.07 | -312508.45 | -305276.47 | -312385.81 |

Empirical DR assignment-replication errors versus the same finite oracle:

| Target | Bias | Std | RMSE |
|---|---:|---:|---:|
| cvar | -220.76 | 1782.80 | 1768.56 |
| mean | 416.13 | 2476.69 | 2472.95 |
| mixture | 97.69 | 664.96 | 661.74 |
| risk_rule | -212.10 | 844.68 | 858.00 |

The first assignment replication has action counts **[87,105]**, known probabilities **0.271522 to 0.728478**, 100% target support and zero positivity violations. Weight ESS / maximum weight: mean **81.78 / 3.683**, CVaR **100.75 / 2.767**, mixture **178.06 / 1.841**, risk rule **163.50 / 1.600**, each out of 192 logged units. These weight ESS values are not independent cluster counts. A deliberate zero-support example returns nonidentifiable even with clipping or outcome predictions.

CVaR-target nuisance sensitivities, over the same 32 assignment replications:

| Outcome model | Propensity | DM | DR | DR bias | DR RMSE |
|---|---|---:|---:|---:|---:|
| forest | estimated | -306064.66 | -313332.69 | -1404.98 | 3216.93 |
| forest | known | -306064.66 | -312148.47 | -220.76 | 1768.56 |
| misspecified | estimated | -303102.06 | -315263.97 | -3336.26 | 8391.03 |
| misspecified | known | -303102.06 | -312637.99 | -710.28 | 2916.90 |
| ridge | estimated | -305376.35 | -313341.21 | -1413.50 | 3087.89 |
| ridge | known | -305376.35 | -312139.79 | -212.08 | 1478.95 |

The pooled one-shot oracle reward ATE is **+365.32**; the first-replication actual-treated oracle ATT is **-114.71**, versus noisy normalized IPW ATT **4,217.88** and naive difference **8,996.82**. These finite targets and estimators differ; a positive ATE is not evidence of a favorable complete-controller cost effect. Post-fit all-action realized-outcome RMSE is Ridge **22,927.68**, forest **23,905.16**, intercept **42,128.72**, including demand-world noise.

In the separate controlled structural experiment (200 replications x 2,000 independent units), DR behaved as follows; nominal coverage is a finite diagnostic, with no guarantee even under a correct nuisance.

| Correct propensity | Correct outcome | DR bias | Std | RMSE | Nominal coverage |
|---|---|---:|---:|---:|---:|
| False | False | 4.0091 | 0.1727 | 4.0128 | 0.000 |
| False | True | 0.0111 | 0.1243 | 0.1245 | 0.910 |
| True | False | 0.0119 | 0.1900 | 0.1899 | 0.960 |
| True | True | 0.0090 | 0.1297 | 0.1297 | 0.925 |

True controlled ATE is 2. The naive confounded difference averaged **9.9973**, randomized benchmark **2.0347**, known IPW **1.9399**, correct DM **2**, correct DR **2.0004**, omitted-confounder DR **9.9982**. Correctness of code alone does not eliminate confounding.

Track B completed **108 full pool runs / 75,600 daily rows**, plus 36 replay interventions. The following differences are CVaR-controller minus mean-controller over 24 paired bootstrap worlds per family (eight panels carrying all three origins):

| Family | Cost difference | Conditional 95% world-block interval | Fill difference (pp) | Lost units difference | Delay unit-days difference | Allocation disparity difference |
|---|---:|---|---:|---:|---:|---:|
| baseline | 3008.89 | [2017.38, 3854.08] | 0.318 | -34.83 | 0.00 | -0.117753 |
| joint | 3850.87 | [2604.43, 5031.36] | 0.245 | -39.38 | 335.42 | -0.129951 |

Mean simulated operating cost rose **727,584.08 to 730,592.98** in baseline and **958,198.47 to 962,049.33** under joint disruption. Average origin-specific operating CVaR(.75) rose **775,773.83 to 779,586.60** and **1,033,396.13 to 1,038,532.37**, respectively; these are not pooled-origin CVaRs. Lower store allocation-share disparity is a descriptive allocation metric, not an equal-service guarantee. Group-level service and risk results remain in the saved tables.

On the three dependent observed-sales proxy paths, baseline cost rose **5,182.33** / mean paired fill **0.731 pp**; joint cost rose **4,915.80** / fill **0.293 pp**. No real confidence interval is reported. Synthetic world-panel intervals condition on fixed nuisances; eight synthetic panels and one actual resource pool cannot establish real-world coverage or significance.

In the separate 2,048-trajectory sequential illustration, deterministic-target ESS fell **968 to 10 to 0 to 0** at horizons **1/7/14/28**, with no matching trajectories at 14/28 despite structural support. For target P(1)=.75, ESS fell **1,620.94 to 481.47 to 136.35 to 36.10**. Complete-return weights use log arithmetic; the golden ratio is **2^28=268,435,456**. Neither result validates one-step DR for dependent controller days.

All **364 Python tests**, **36 fresh dbt tests**, **38 independent saved-output gates**, **234 future-source checks**, **108 full-vector/propensity perturbations** and **30 reset/temporal checks** pass. Raw/processed inputs, earlier model artifacts and warehouse totals are unchanged. Runtime **765.23 s**, peak process memory **788.66 MiB**. Catalogue solves: **284 optimal / 4 feasible not proven optimal** (including perturbation repeats); controller solves: **431 optimal / 1 feasible not proven optimal**. All feasible incumbents pass independent constraints; maximum relative gap across both sets **0.00483%**. No experiment cells were skipped.

The generated `data/causal/m5_h28/causal_report.md` contains all four estimators' repeated errors/variability, propensity/ESS tables, outcome/propensity sensitivities, paired service/delay/disparity and sequential diagnostics. Detailed primary intervals and all 9,312 estimates are in Parquet. The future real experiment plan uses resource-pool randomization, ITT, measured power inputs and carryover/override logging; it deploys no policy.

## References

- [Hernán and Robins, official What If book page](https://miguelhernan.org/whatifbook). The supplied older Harvard PDF URL now redirects to the author's profile.
- [Wang, Agarwal and Dudík, contextual bandit OPE](https://proceedings.mlr.press/v70/wang17a.html).
- [Dudík, Langford and Li, doubly robust policy evaluation](https://arxiv.org/abs/1103.4601).
