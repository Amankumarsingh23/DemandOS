# Season 20: shared supply allocation optimization

Season 21 adds [finite-scenario minimax, regret, CVaR and probability-ambiguity
objectives](robust_optimization.md) using this same sparse feasible set and an
optional coordinator allocator callback. Existing Season 20 defaults remain.

All economics, supplier limits, budgets, volumes and outcomes are
**SIMULATED / SCENARIO-ASSUMED**. Recorded M5 sales serve only as a declared
demand proxy. They do not reveal uncensored demand, acquisition costs,
actual supplier capacities, purchasing budgets, physical storage, stockouts,
lost sales or Walmart savings. No purchase cost is inferred from selling prices.

## Architecture and unchanged simulation contracts

`demandos.optimization` separates sparse mathematical models, independent
business-unit feasibility checks, coordinated execution, experiment/provenance
orchestration, saved-output verification, exact toy examples and reporting.
It uses installed **SciPy 1.18.1**, `linprog(method="highs")`, `milp`,
`LinearConstraint`, `Bounds` and sparse CSR matrices; no commercial solver or
new dependency is needed. Use the existing analytics/EDA/reconciliation extras.

Season 19's original stock, delivery, FIFO backlog and cost loop now also
exposes `steps(...)`. It yields the immutable post-demand observation and
accepts the approved `Action`. Its existing `simulate(...)` API drives that
same generator with its policy; earlier policy/CLI/result contracts remain.
This is a small interface extraction, not a second implementation of inventory
transitions. All earlier tests and the exact three-day golden example are retained.

At each date, the coordinator advances **every** series to its post-demand
observation, collects all candidate requests, approves one shared order vector,
then commits each approved action. No series is simulated to completion before
the others propose. The original day-open receipt / old-backlog / current-demand /
day-close order timeline is preserved; an issue on t with lead L arrives at
t+L opening. Only approved units enter the original delivery queues.

Decision keys include origin, simulation day, `SELECTED_COHORT_DC` allocation
group and scenario-config ID. Candidate/allocation keys additionally identify
store, item, policy/method and deterministic request ID. Original store/item
IDs, current inventory position, pack/MOQ, supplier/volume coefficients,
forecast source, fit/availability cutoff and scenario IDs remain auditable.

## LP, MILP, bounds and statuses

The continuous LP maximizes `sum benefit_i*x_i` with nonnegative quantities,
request bounds and budget, flow-volume, shared supplier and configured store
receiving-order constraints. Per-unit benefit is a **heuristic assumption**:
`Cu * empirical_probability(protection_demand > current_IP)`. It is constant
over a candidate's requested units, not a validated diminishing marginal value
or an operating-profit estimate. The implementation minimizes its negative.

Integer orders use case count n and binary activation z:

```text
q_i = case_pack_i * n_i
0 <= n_i <= floor(min(request_i,item_supplier_cap_i)/case_pack_i)
n_i <= U_i*z_i
n_i >= max(1,ceil(MOQ_i/case_pack_i))*z_i
```

U is the derived finite case bound, not an arbitrary giant big-M. Items whose
request/cap cannot meet one case and MOQ receive zero. Fractional candidate
requests are rejected rather than silently changing units. Unconditional
minimum commitments are separate from conditional MOQ; incompatible minima,
resource caps and fairness can make a model infeasible.

Procurement resource use includes `purchase*q + fixed_fee*z`. Default unit
purchase is hypothetical INR 60 and positive-order fee 5. Per-item caps,
shared supplier caps, case packs, MOQ and store-order quotas are independently
verified in original units after solving. LP activations/quantities may be
fractional and are diagnostic only: they are never rounded into live orders.

Solver settings: presolve on, time limit **3 seconds per solve**, requested
relative MIP gap **zero**, and independent feasibility/integrality acceptance
tolerance **1e-7**. LP primal/dual tolerances are 1e-7. SciPy's documented MILP
options do not expose all internal HiGHS tolerances; independently checked
business feasibility remains mandatory. Statuses distinguish `OPTIMAL`,
`FEASIBLE_NOT_PROVEN_OPTIMAL`, `INFEASIBLE`, `UNBOUNDED`,
`TIME_LIMIT_NO_INCUMBENT` and `NUMERICAL_ERROR`. A feasible limited incumbent
does not imply optimality. Invalid fractional/infeasible incumbents are rejected.

If no valid incumbent exists, the live coordinator may explicitly approve zero
only if zero passes **all hard constraints**. Soft preferences can have declared
slack; resources are never relaxed. Hard forced minima/fairness may prevent
this fallback and then execution stops. Diagnostics include solver message,
minimum forced budget/flow commitments and cheaply detected violating families.

Candidates and matrix rows use stable sorted identifiers. Deterministic HiGHS
traversal for identical installed inputs is the tie policy; tests and temporal
replays verify repeatability. No objective perturbation is hidden, no unique
economic optimum is claimed, and identical tie choices across solver versions
are not guaranteed. Matrix manifests record sparse shape/nnz, coefficient/
objective/RHS hash, row names, derived case bounds and objective type. Inputs
needed to rebuild them are saved in candidates, resources and frozen scenarios;
the exact two-stage toy matrix is also exported as sparse NPZ.

## Hand checks, duals and integrality

The required LP maximizes `8*A+5*B` with `4*A+2*B<=20`, `A+B<=8`.
It returns **A=2/B=6/value=46**, matching brute-force vertex enumeration.
Both constraints bind. Economic RHS shadow prices are **1.5 and 2**, with
dual value `20*1.5+8*2=46`; small RHS perturbations verify the signs/values.
Adding B<=4 gives **A=3/B=4/value=44**.

`linprog` minimizes, so economic maximization shadow prices are the negative
of returned minimization RHS marginals. Real model artifacts retain the signed
minimization marginals and constraint names. These values are local to an LP
relaxation and active basis; degeneracy, a changing basis and numerical
tolerances limit interpretation. They are not MILP shadow-price guarantees.

Binary knapsack costs 6/5/4, benefits 10/9/7, budget 10 selects A+C for
**benefit 17**, matching exhaustive enumeration. For A/B alone, the integer
optimum is **10** versus LP value **17.333333**, gap **7.333333**. Rounding both
fractional selections upward spends 11 and is infeasible. Raising the integer
budget from 10 to 11 changes value from 10 to 19: discrete finite-difference
improvement is distinct from an LP marginal. The mandatory forced orders
5+5 with total flow capacity 8 report **INFEASIBLE**, without fabricated orders.

## Scenario objective and nonanticipativity

There is one first-stage integer order vector q shared by every forecast
scenario. Scenario-specific shortage/excess variables are accounting recourse:

```text
short_iw >= demand_iw - current_IP_i - q_i; short_iw >= 0
excess_iw >= current_IP_i + q_i - demand_iw; excess_iw >= 0
minimize sum_iw p_w * (Cu_i*short_iw + Co_i*excess_iw)
```

Cu/Co reuse Season 18's explicit p/c/v/penalty assumptions: balanced 40/40,
shortage-expensive 180/20 and overage-expensive 10/50. Procurement is **not
added again** to this mismatch objective. Procurement/fixed fees still consume
the hard budget. The realized simulation separately books its procurement,
ordering, holding and lost/backlog costs under Season 19's definitions. This
proxy does not optimize the full dynamic receipt-constrained operational ledger.

Each item uses the first 16 deterministic paths from its verified 512-path
Season 18 generator, frozen at E. Seven-day historical blocks are the main
source; a sensitivity uses Season 16 assumed quantile paths. Series bootstrap
draws are independent, while the hypothetical random lead support is shared
across series using deterministic inverse-CDF strata. This is an explicit
cross-series scenario assumption, not a joint distribution identified by
P10/P50/P90 or conformal intervals. For the additive expected-cost objective,
cross-series dependence does not create a chance-constraint guarantee.

For random L=2/4/7 (.25/.50/.25), sixteen strata give 4/8/4 lead realizations.
Protection demand sums coherent paths through L+7. Scenario probabilities must
be nonnegative and sum to one. Targets, dependence assumptions, models and
samplers stay frozen; only observed inventory/pipeline state evolves. This is
first-stage allocation with scenario accounting, **not** adaptive multistage
stochastic purchasing or daily model reforecasting.

The EVPI toy has two items and equally probable demand vectors (0,10)/(10,0),
shared ten-unit budget, Cu=9/Co=1. A common first-stage plan costs **50** in
expectation; separate perfect-information plans cost **zero**, so **EVPI=50**.
Both benchmarks have identical resources/cost rules. Hindsight is unimplementable,
not attainable policy savings.

At E, non-committed day-zero snapshots use initial IP and frozen scenarios.
They are reproduced after future-sales mutation/removal and future-price mutation.
They are diagnostics and are **not extra orders** in the live simulation, whose
first decision follows day-one demand. During simulation, changing day k's
demand requires all series' pre-k states/orders/allocations/budget balances to
remain unchanged. Later decisions may respond legitimately. Hidden lost demand
never reaches policies. Tests and real checks cover these information boundaries.

## Resources, units, time windows and fairness

All 25 selected series share **one cohort allocation pool**; supplier groups
follow actual department metadata as an assumed supplier assignment. They do
not represent actual Walmart suppliers or the full stores/states. Group mapping
uses static source metadata and no evaluated future sales.

Resource normalization is frozen independently of the competing model/cost
regime: B_i is the original balanced historical-fractile target at E. With four
scheduled reviews, the declared baseline cash envelope is
`4*60*sum(B_i) + 4*N*5`. Tight/medium/loose horizon budgets are **.35/.70/1.40**
times this envelope. Model and cost sensitivities retain the same reference
resource vectors, avoiding a changing-capacity comparison.

- Budget: **entire 28 days**, reduced at order commitment, never refunded on
  receipt and never reset daily.
- Supplier units per department: **entire 28 days**, `4*sum(B_i in group)`;
  scarcity sensitivity multiplies by .25. Capacity is shared across stores.
- Warehouse **procurement-flow volume per decision epoch**: `.8*sum(v_i*B_i)`;
  scarcity uses .25 instead. Assumed v is FOODS=1, HOUSEHOLD=2, HOBBIES=3.
  This is not physical on-hand storage and does not constrain combined future
  receipts or carried inventory. No such physical-storage claim is made.
- Store receiving-**order authorization** per decision epoch: `sum(B_i in store)`
  with a minimum one-unit quota. It is not a verified delivery-dock limit on
  actual daily arrivals.

Cash includes fees for every activated order. Initial stock/open orders are
prepaid; the original simulator charges new procurement at order, including
terminal pipeline, with no default terminal credit or invented backlog tail cost.
Resource conservation is checked after every epoch, including non-review days.

Fairness is a configurable allocation preference, not a legal/ethical claim.
For positive store requests, share is aggregate approved/requested quantity.
Zero-request stores are excluded. Two linear families support minimum shares
and maximum pairwise relative-share disparity. Hard versions may be infeasible.
Live sensitivities use **soft** minimum .20 or disparity .10, with explicit
penalty **1000 per unit of each dimensionless share-constraint slack**. Their
slacks, achieved shares/disparity and objective changes are reported; no hard
physical resource is relaxed. Proportional/greedy heuristics do not claim to
optimize those preference penalties.

## Real experiment and reproduction

The grid is fixed before scoring: 25 series (13 continuity, 12 stress), three
origins Jan 31 / Feb 28 / Mar 27, 2016, and their complete 28-day recorded-sales
paths, including Feb 29. There are **75 unique paths / 2,100 unique daily targets**.
Main cases are three cost regimes × three budgets × fixed-seven-day lead.
Seven sensitivities cover supplier scarcity, flow scarcity, minimum/disparity
fairness, pack=6/MOQ=12, Season 16 scenarios and random lead. Initial stock is
10, initial on-order 3, no backlog; periodic reviews are day 1/8/15/22.

Each case compares an **unconstrained reference** (not resource-feasible),
feasible proportional floor-to-pack allocation, feasible benefit-ranked greedy,
deterministic benefit MILP and expected-mismatch MILP. Both MILPs retain LP
relaxation diagnostics; LP results are never implemented as integer orders.
All constrained methods share costs/resources, demand, initial queues, calendar
supplier shocks, event order and terminal accounting. Paired per-series-origin
costs use the proportional method reference; days are not independent replications.

```powershell
.\.venv\Scripts\python -m demandos.optimization --root . --config configs/optimization.toml
.\.venv\Scripts\python -m unittest discover -s tests
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; r,s,w,e=settings(Path.cwd()); invoke_dbt(r,e,'test')"
```

Run dbt separately from the optimization CLI: DuckDB's dbt writer must not
overlap the final read-only integrity check. Required real source and prior
forecast artifacts must exist; none are downloaded/fabricated. Extraction loads
only the bounded 25-series history/targets, not a dense 58-million-row panel.

Ignored outputs under `data/optimization/m5_h28/` include candidates/approvals,
resource/cash/supplier balances, original-engine daily/order ledgers, scenarios
and provenance, origin snapshots/perturbations, solver/gap/LP marginal records,
sparse matrix manifests, toy/EVPI evidence, cohort cost/service/paired results,
independent feasibility JSON, configuration, report and runtime/memory summary.
Correctness, hashes and warehouse totals gate manifest-owned publication.
Generated data, matrices, caches and credentials stay out of Git. Season 21
features are outside this implementation.

## Verified real results

The final run completed **240 coordinated pool runs**, **6,000 series runs**
and **168,000 daily rows**, retaining 75 unique recorded-sales paths. Main grid:
135 pool runs / 94,500 daily rows; sensitivities: 105 / 73,500. Unconstrained
references account for 33,600 daily rows and are excluded from constrained
feasibility claims. Every method shares 48 origin/config cases, each containing
all 25 series, with continuity/stress strata preserved.

All **384 main/sensitivity integer allocations** and their **384 LP relaxations**
were optimal. Additional origin/toy/temporal diagnostics are separate.
Maximum reported MIP gap was **0**; there were no fallbacks, solver failures or
skipped cases. Recorded MILP allocation time was **5.77 seconds**, including
model construction/checks; total final execution was **127.90 seconds**, peak
process memory **972.13 MiB**. Overall execution also includes LPs, original
engine trajectories, counterfactuals, source hashing and persisted verification.

Medium-budget balanced/fixed-seven-day results, hypothetical INR operating
cost per 28-day series run and demand-weighted immediate fill:

| Method | Continuity cost | Fill | Stress cost | Fill |
|---|---:|---:|---:|---:|
| Unconstrained reference — resource limits not enforced | 57315.62 | 70.07% | 1622.49 | 80.44% |
| Feasible proportional | 57291.23 | 69.86% | 1622.12 | 79.56% |
| Feasible greedy | 57458.80 | 66.13% | 1614.22 | 79.03% |
| Deterministic benefit MILP | 57245.02 | 69.82% | 1621.97 | 80.35% |
| Expected-mismatch MILP | 55317.50 | 67.11% | 1578.73 | 76.90% |

Each cell contains 39 continuity or 36 stress series-origins, not independent
days. The expected-mismatch method's lower operating cost accompanies lower
service. This is not a uniformly better policy or real-world savings. Balanced
operating lost penalty is 40 versus unit purchase 60; cost-only rankings can
prefer less purchasing, and do not substitute for a service requirement or
profit objective. Paired per-origin/config differences are retained against
proportional allocation, with no daily-IID inference claim.

The historical reference gives exact medium horizon budgets / epoch flow limits:

| Origin | Budget, hypothetical INR | Epoch procurement-volume limit |
|---|---:|---:|
| 2016-01-31 | 887894.00 | 5248.0 |
| 2016-02-28 | 898478.00 | 5296.8 |
| 2016-03-27 | 895790.00 | 5286.4 |

Exact department/store limits and before/after balances are in `resources`.
Across the three origins, stochastic balanced tight-budget utilization is
99.9894–99.9980%; medium utilization 64.7629–66.2050%, loose 32.3814–33.1025%.
No delivery refunds or daily resets occur. Tight budget reduces continuity/
stress fill to 64.73%/76.02%; medium and loose both give 67.11%/76.90%, showing
that increasing a nonbinding budget alone does not overcome other limits or
change this proxy policy's decisions.

Expected-mismatch sensitivity results, with other medium/balanced conditions fixed:

| Sensitivity | Continuity cost | Fill | Stress cost | Fill |
|---|---:|---:|---:|---:|
| Supplier factor .25 | 41426.96 | 47.44% | 1132.44 | 69.20% |
| Epoch flow factor .25 | 48210.25 | 44.21% | 1103.53 | 49.56% |
| Soft minimum store share .20 | 55809.46 | 67.35% | 1585.22 | 77.35% |
| Soft disparity target .10 | 56280.31 | 68.50% | 1585.06 | 77.52% |
| Case pack 6 / MOQ 12 | 56090.35 | 66.84% | 1355.63 | 66.73% |
| Season 16 assumed scenarios | 47494.90 | 55.20% | 1724.97 | 84.16% |
| Random 2/4/7-day lead | 51622.87 | 74.71% | 1430.69 | 82.83% |

The minimum-share study achieves at least **21.05%** among eligible positive
store requests. The disparity study's mean maximum share difference is
**0.13893**, with mean excess **0.03896** above target .10: its soft target is
not guaranteed. Cost/benefit objectives, actual shares and slack remain separate.
Small pack-ineligible requests are explicitly blocked. Blocked request totals
are per decision epoch and can repeat across reviews; they are not observed
lost sales. Allocation objective contributions are state diagnostics at an
epoch, not costs to sum across all simulation days.

All **313 Python tests** (289 prior plus 24 new) and **36 fresh dbt tests** pass.
The run passes **225** frozen-prefix perturbations, **nine** coordinated sequential
counterfactuals and baseline/mutated day-zero allocation vector comparisons at
each origin (300 saved scalar allocation records). **18 independent saved-file
gates** validate resource feasibility, cash/supplier persistence, pack/MOQ,
source cutoffs, nonanticipativity, lifecycle/receipt flows, daily stock/backlog,
operating costs, case keys, probabilities and shared demand/supplier shocks.
The **58,327,370-row / 30,490-series** warehouse, **65,695,409 units**, and
**187,676,570.0200** known recorded revenue remain intact; source/model/Season 19
artifact fingerprints are unchanged. This revenue is only an integrity total.

Official solver references: [SciPy milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html),
[SciPy linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).
Installed SciPy is recorded separately from the version of the online manual.
