# Season 21: robust optimization and decision risk

All costs, budgets, supplier events, demand shifts and outcomes here are
**SIMULATED / SCENARIO-ASSUMED**. M5 contains recorded sales, not verified
uncensored demand or actual inventory, acquisition costs, supplier failures,
resource limits, lost sales or business savings.

## Contracts and model reuse

`demandos.risk` adds finite-scenario objectives to Season 20's sparse MILP
builder. Its shared first-stage variables remain integer case counts and binary
order activations. Positive orders satisfy MOQ; purchase plus activation fees
consume cash. Every method uses identical requests, resources, cost definitions
and fairness rules in a matched experiment cell. No post-hoc clipping occurs.

The coordinator accepts an optional allocator callback; existing methods and
CLI defaults retain their behavior. All series propose before the shared vector
is committed to Season 19's original `steps` delivery queue. A policy observes
stock, backlog, outstanding orders and fulfilled sales, never hidden unmet
customer demand or realized future prices/demand/disruptions.

Decision records identify origin, allocation group, decision day, method,
planning/evaluation family, configuration, demand-path fingerprint, calendar-keyed
supplier stream, risk level/penalty, resource configuration and provenance.
Planning scenarios, evaluation manifests, solver records and operating ledgers
are separate artifacts. Earlier forecast checkpoint hashes, training and
calibration cutoffs remain in the source-bound provenance; no model is retrained.

## Objective definitions

For order vector q, current observable inventory position s and joint protection
demand D[i,w], the economic mismatch loss is

`L[w] = sum_i Cu*max(D[i,w]-s[i]-q[i],0) + Co*max(s[i]+q[i]-D[i,w],0)`.

Shortage and excess are continuous accounting auxiliaries, not scenario-specific
purchase decisions. Soft fairness preferences, when enabled, add the same
deterministic preference cost to each scenario. That preference remains a
separate reported component. No procurement overlay is added to mismatch costs.

- Mean: minimize `sum_w p[w]*L[w]`.
- Minimax: minimize t, with `L[w] <= t` for **every supported scenario**, including
  zero-probability stress points. This is finite-support protection only.
- Regret: solve each scenario's perfect-information oracle `Lstar[w]` using the
  **same feasible set and cost/fairness convention**, then minimize t subject to
  `L[w]-Lstar[w] <= t`. Every oracle must be proven optimal; otherwise the regret
  decision is unavailable. Oracles are hindsight benchmarks, not policies.
- Mean–CVaR: minimize `E[L] + lambda*CVaR_alpha(L)`, lambda >= 0. The configured
  penalties are .5 and 2; the lambda=0 test agrees with the compatible Season 20
  expected-mismatch model up to mathematically valid ties.
- Epsilon constraint: minimize `E[L]` subject to a **hard** CVaR ceiling. Static
  origin diagnostics use 0, 10,000, 100,000 and 1,000,000 assumed mismatch-cost units. An
  infeasible ceiling has no allocation or simulated result; it is never relaxed.

For normalized, finite, nonnegative weights (absolute sum tolerance 1e-12),
`VaR_alpha = inf{x : P(L<=x)>=alpha}`, with `0<alpha<1`. Zero-mass points do not
affect distributional VaR/CVaR. One floating-point predecessor is used at the
CDF boundary to handle representational roundoff, not a different quantile rule.

CVaR integrates the **worst 1-alpha probability mass**, taking fractional mass
at its boundary. It is not the naive average of all losses >= VaR. Its epigraph is

`eta + sum_w p[w]*z[w]/(1-alpha)`, with `z[w]>=0`, `z[w]>=L[w]-eta`, eta unrestricted.

Sorted-tail integration, a separate fixed-loss LP and the returned integer
allocation all crosscheck the risk objective. For `[10,12,14,16,18,20,22,24,30,100]`
with equal weights: alpha .9 gives VaR 30/CVaR 100; alpha .8 gives VaR 24/CVaR 65.
For `[10,20,30,100]`, alpha .75 gives VaR 30/CVaR 100. Eta 30 with excesses
`[0,0,0,70]` attains 100; all eta in [30,100] are also optimal in that example.

Newsvendor golden: D={5,10,15}, Cu=4, Co=1. Enumeration of Q=0..20 and the MILP
both give minimax Q=13, worst cost 8. Probabilities [.8,.1,.1] give mean cost 6
for every Q=5..10; the inverse-CDF convention selects the smallest, Q=5. Stable
HiGHS traversal need not select that smallest allocation in a tied MILP.
The regret matrix A=[0,100], B=[20,40], C=[50,20] selects B, maximum regret 20.

## Finite probability ambiguity and uncertainty sets

Four normal and four adverse joint planning points form the fixed support.
Within-family weights are equal. Nominal adverse-family mass is .10; shifted
probability sensitivity uses .20. The ambiguity set lets that mass range over
[.05,.20], preserving conditional weights. For a fixed order the expectation is
affine in p, so its maximum is the larger endpoint expectation. The DRO MILP
uses one epigraph row per endpoint. A 100 normal / 1,000 shock cost has worst
expectation 280 at p=.20. This is an exact finite endpoint result, with no
Wasserstein, distribution-free or unmodeled-distribution guarantee.

The uncertainty-box artifact records per-series min/max bounds containing the
support. It is a diagnostic; arbitrary combinations of these coordinate bounds
are not all optimized. No continuous box/Gamma robust counterpart is claimed.
Gamma-budget optimization is outside this bounded implementation.

## Predeclared scenario and resource assumptions

The population remains 13 continuity and 12 sparse-stress series, at origins
2016-01-31, 2016-02-28 and 2016-03-27. Before replication there are exactly 75
recorded-sales paths / 2,100 daily targets, including leap day. Each origin's
365-day eligible history produces weekday-aligned seven-day block paths.
Planning uses four normal 14-day sums and four assumed 1.5x 21-day sums. The
adverse support interprets 14-day supplier lead plus seven-day review. Series
blocks are independently sampled but the adverse-family indicator is shared.
This coupling is an assumption; daily P10/P90 are not treated as joint paths.

Evaluation families are baseline, 1.5x demand surge, shared days 8–14 threefold
burst, 14-day rather than seven-day lead, receiving closure on days 8–14, joint
surge/closure, and changed planning probabilities. Severity/timing are declared
in `configs/risk.toml`/`risk.scenarios`, not selected from 2016 outcomes. Eight
additional complete, distinct prefix-bootstrap worlds per origin are evaluated
for baseline and joint stress. Their random streams are disjoint from planning
streams. All methods see the same realized world and exogenous supplier schedule.

The nominal horizon supplier multiplier is 1; both .75 and 1 are assumed possible,
so all comparisons use the robust hard authorization bound .75. It remains a
persistent **commitment** limit, not a guarantee of timely delivery. Horizon cash
is .70 of the Season 20 reference; procurement-flow capacity is .80 per epoch;
store order quotas are per epoch. Daily no-order dates do not reset horizon
cash/supplier commitments. Separate baseline observed-path cells use shortage
cost/.35 budget, overage cost/1.40 budget, soft minimum share .20, soft maximum
disparity .10, and case pack 6/MOQ 12. This is a balanced subset, not a full grid.

Procurement-flow volume is not physical on-hand storage. The earlier engine has
no verified shelf capacities and none are invented. Inventory is conserved and
nonnegative; no claim of actual physical storage feasibility is made.

## Delivery disruption and conservation

The receiving-closure stress has capacity zero on days 8–14 and unlimited
receiving capacity otherwise. At each closed opening, every order **due today**
is re-timed to day 15. Outstanding committed orders retain their full quantity,
identity and cash/supplier commitment. Its new arrival and lead time, previous
arrival and re-timing day are recorded. The callback is evaluated only at today's
opening; future realized closure information is not an optimizer input.

Partial delivery and cancellations are disabled. Residual units are the entire
order until receipt; after the scored horizon they remain explicitly outstanding.
No stock disappears, no phantom receipts or procurement refunds occur. Opening
receipts still precede old-backlog and current-demand service. Unit tests cover
both lost-sales and FIFO-backlog modes; the real mandatory grid uses lost sales.

## Operating risk, fairness and comparisons

The operating ledger includes procurement at order, activation fees, holding,
lost-sales/backlog penalties and terminal on-hand credit (zero in this grid).
Initial pipeline is prepaid; terminal outstanding orders remain paid and queued.
No mismatch cost is added to this ledger. Operating CVaR is computed on **that
entire identical operating-cost definition**, independently of planning losses.

Operating distributions use eight genuinely distinct complete pool worlds
conditional on origin/family, with separate total, continuity, stress and store
scores. At alpha .75, empirical CVaR averages the worst two worlds. These small
finite samples do not establish tail coverage. Neither 28 days nor 25 series
are treated as independent experimental worlds. Paired cost and demand-weighted
fill differences are recorded per pool-origin/cohort; no unsupported significance
or independent-origin claim is made.

Positive-request store denominators define shares; zero-request stores are
excluded explicitly, and zero-demand fill is undefined (`null`). Hard fairness
is enforced or infeasible. Soft preferences retain measured shortfall/disparity
and penalties; they do not relax resource constraints. Per-store operating
CVaR/service and allocation slacks are separate: system CVaR does not guarantee
group service. The frontier marks nondominated points among the tested integer
allocations only. Weighted sums may miss unsupported efficient integer points;
the hard-ceiling diagnostics provide additional tested points, not a complete
Pareto proof.

## Solver and reproducibility

Installed stack: SciPy **1.18.1**, bundled HiGHS **1.12.0**; `milp`, sparse
CSR/CSC, `Bounds`, `LinearConstraint`, and fixed-loss `linprog(method="highs")`.
Each MILP is bounded to three seconds, presolve enabled, relative gap target zero.
Native `mip_feasibility_tolerance=1e-7` is forwarded through SciPy (only its
forwarding notice is filtered). Independent original-unit feasibility remains
1e-7. The default 1e-6 MIP tolerance produced a rejected 3.3e-7 residual in the
initial run. Currency normalization and exact reconstruction of continuous
shortage/excess/fairness/epigraph auxiliaries remove numerical degeneracy without
changing the integer order vector or any resource limit. Raw and reconstructed
residuals are retained; quantities are independently verified before reconstruction.
The final gap is checked against the solver dual bound as well.
HiGHS 1.12 also has a presolve transformation failure on some degenerate regret
instances. Status 4 triggers one retry of the **identical** sparse model with
presolve disabled, within the original three-second total solve budget. Both
attempts, statuses and runtimes are logged. This does not permit an infeasible
fallback or change the economic objective.
Statuses and achieved gaps are retained: time-limited incumbents are never
called optimal, and nonoptimal oracles cannot support exact regret claims.

```powershell
.\.venv\Scripts\python -m demandos.risk --root . --config configs/risk.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,output,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

Run dbt after the experiment closes its read-only warehouse connection.
Generated outputs stay ignored under `data/risk/m5_h28/`: risk/box/scenario
manifests, shock ledger, static epsilon solves, Pareto table, decision proofs,
solver variables/status/gaps, scenario losses/oracle regrets, operating and
paired/group risk tables, fairness slacks, complete daily/order/resource
ledgers, temporal/future checks, `verification.json`, `summary.json`, and
`risk_report.md`. Publication reuses the guarded manifest-based writer, preserving
unowned files. Raw/interim/processed inputs and previous checkpoints are hashed
before/after; warehouse totals must remain exactly 58,327,370 observations.

## Measured real M5-informed results

The fixed grid produced **504 pool runs / 12,600 series runs / 352,800 daily
rows**, plus 72 temporal/hidden-demand pool replays. There remain only **75
original observed-sales paths / 2,100 store-item-date observations**. All
economics and disruptions below remain simulated assumptions.

The following are balanced-cost, origin-issued allocation diagnostics, averaged
over the three origins at alpha .75. Costs are **forecast mismatch**, not the
procurement-inclusive operating ledger.

| Method | Ordered units | Expected cost | CVaR | Worst cost | Maximum regret |
|---|---:|---:|---:|---:|---:|
| Mean | 4447.00 | 73732.33 | 155529.33 | 345493.33 | 35506.67 |
| Mean + .5 CVaR | 4454.67 | 73899.00 | 153476.00 | 345186.67 | 33720.00 |
| Mean + 2 CVaR | 4493.00 | 74673.67 | 152934.67 | 343653.33 | 36666.67 |
| Minimax | 4747.00 | 82901.67 | 157966.67 | 333493.33 | 44426.67 |
| Minimax regret | 4265.33 | 75583.67 | 162102.67 | 352760.00 | 24626.67 |
| Probability ambiguity | 4447.00 | 73732.33 | 155529.33 | 345493.33 | 35506.67 |

At these origins the DRO allocation matches mean-cost allocation; every tested
static allocation's adverse conditional cost is higher, so the worst probability
endpoint is .20. This finite ambiguity set does not demonstrate general
distribution-shift protection. The minimax policy lowers the worst supported
loss at a higher expected cost; minimax regret minimizes a different criterion.

Operating results below average three origin-specific, eight-world pool panels.
Each empirical CVaR is computed on complete procurement-inclusive operating
costs at alpha .75, with whole-pool demand-weighted immediate fill.

| Evaluation family | Method | Mean operating cost | Operating CVaR | Fill |
|---|---|---:|---:|---:|
| Baseline | Mean | 726107.53 | 772612.77 | 65.97% |
| Baseline | Mean + .5 CVaR | 727338.88 | 772924.63 | 66.00% |
| Baseline | Mean + 2 CVaR | 729076.29 | 774309.37 | 66.06% |
| Baseline | Minimax | 752649.65 | 798281.83 | 68.54% |
| Baseline | Minimax regret | 721466.05 | 768148.30 | 65.69% |
| Baseline | Probability ambiguity | 726008.11 | 772536.83 | 65.93% |
| Joint surge/closure | Mean | 950014.77 | 1010475.57 | 34.01% |
| Joint surge/closure | Mean + 2 CVaR | 953047.44 | 1011899.63 | 34.30% |
| Joint surge/closure | Minimax regret | 942147.89 | 1002066.87 | 34.50% |

These are paired on the same complete worlds/resources; differences can reflect
both purchasing and service. Reduced planning CVaR did **not** reduce simulated
operating CVaR here. No actual business savings or statistically reliable tail
ranking follows from eight assumed worlds. Full cohort/store/world and paired
tables remain available in the generated report.

There were **5,660 MILP backend executions**: 5,639 optimal, 18 infeasible static
ceilings, one feasible incumbent not proven optimal, and two initial presolve
errors recovered on the identical model. All **3,216 scenario-wise oracles** were
optimal. The **2,016 committed decision solves** comprise 2,015 optimal and one
time-limited feasible regret decision: January 31 origin, soft-disparity study,
day 22, gap **0.0557305%**. All its resource/integrality/cost gates pass. Timed
incumbents may differ on another machine; they are never presented as optimal.
All 24 epsilon diagnostics ran: 18 infeasible at ceilings 0/10,000/100,000;
six feasible at 1,000,000. No ceiling was relaxed or given fabricated outcomes.

All **340 Python tests** and **36 fresh real-warehouse dbt tests** pass, with no
skips or dbt warnings. The saved-output audit passes **26 independent gates**,
including 2,016 independent fixed-loss CVaR LP crosschecks. There are **225
future-sales/removal/price mutations**, **1,800 origin allocation records**
across the four perturbations, and **54 full-pool temporal/hidden-demand checks**.
All **2,749 order re-timings** preserve units and identities; stock, cash, supplier,
store-quota and pack/MOQ gates pass. Soft minimum-share targets were met; the
soft disparity study has measured violations, with maximum slack **0.33465046**
beyond its .10 preference. These remain explicit preferences, not hard guarantees.

The run took **381.69 seconds**, peaking at **1,226.40 MiB**. Raw/interim/processed
and prior forecast/model fingerprints match their pre-run values; warehouse
totals retain exactly **58,327,370 observations**. No experiment cell was skipped.
The infeasible epsilon points and the one unproven incumbent remain explicit.

## References

- [SciPy MILP interface and status semantics](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).
- [HiGHS feasibility tolerances](https://ergo-code.github.io/HiGHS/dev/guide/kkt/).
- [Rockafellar and Uryasev, general loss distributions](https://www.sciencedirect.com/science/article/pii/S0378426602002716).
- [Bertsimas and Sim, The Price of Robustness](https://pubsonline.informs.org/doi/10.1287/opre.1030.0065).
