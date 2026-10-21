# Future real-world policy experiment: design only

This proposed design is **not a deployed trial or measured Walmart effect**.
The simulator supplies no real assignment probabilities, operational variances,
inventory costs or verified effect sizes for a power calculation.

## Estimand, eligibility and randomization

Compare approved mean-based and mean+2CVaR replenishment controllers over a
registered complete follow-up period. Primary estimand: intent-to-treat (ITT)
operating-cost difference per eligible resource-pool episode, with reward
advantage defined as its negative. Agree beforehand on acquisition, holding,
ordering, transport, emergency transfers, returns, terminal inventory/pipeline
and backlog valuation. Service/fairness are prespecified secondary outcomes
and guardrails, not substitutes chosen after reviewing results.

Randomize independently budgeted/supplied pools, such as validated distribution
catchments. Stores/items sharing binding budgets, suppliers or transfers belong
in one intervention unit. Inspect the dependency graph; enlarge clusters or
measure spillovers before assuming between-pool independence. Do not treat
competing SKUs or their item-days as independent experiment units.

Require reliable inventory/cost/event logs, executable feasible actions, stable
catalogue definitions and sufficient follow-up. Prespecify exclusions for data
outages, closures and unrecorded transfers. Do not selectively drop adverse
post-assignment outcomes. Use a registered, baseline-stratified 1:1 pool
assignment, balancing only pre-randomization volume, geography and service/risk.
Record the complete eligible action set, exact assignment vector and allocation
list/seed. Authorization for any real trial is separate from this educational task.

## Required logging and compliance

Record intended assignment, policy/configuration version, actual execution,
proposals/approved vectors, resources/binding constraints, purchase orders,
supplier commitments, receipts/partial deliveries/delays/cancellations, transfers,
returns and terminal pipeline. Preserve identities, availability/event times,
time zones, forecast/calibration cutoffs and corrections for as-of reconstruction.

Outcome measurement requires on-hand and shelf availability, sales, credible
lost-demand/backorder observation, acquisition/holding/order/transport costs,
promotions, exceptional closures and terminal valuation. Zero recorded sales
alone cannot distinguish zero demand from stockout. Selling price is not
acquisition cost. Validate reconciliation and maturity before releasing outcomes.

Allow safe manager overrides but retain original assignment, execution and
override time/reason. Primary analysis remains ITT; compliance is separately
reported. Per-protocol/as-treated effects need extra identification assumptions
and cannot silently replace ITT. Preserve rejected/no-order actions and logs.

## Carryover, power and analysis

Prefer parallel clusters where inventory carryover makes crossover problematic.
Measure starting stock/pipeline rather than assuming equality. Follow committed
orders through agreed terminal accounting. Any crossover washout must reflect
measured lead times, replenishment cycles, residual stock and seasonality; no
universal fixed washout or destructive stock disposal is prescribed. Avoid
overlapping episodes that reuse inventory without accounting for dependence.

Collect blinded baseline between-pool variance, serial dependence/ICC, cluster
sizes, compliance, meaningful minimum effect, desired type-I error/power,
attrition/outages, multiplicity and independent-supplier counts. Then simulate
the registered cluster design to select duration/sample size. Eight assumed
worlds and simulator variances are not reliable real-world power inputs; no
guaranteed sample size is invented.

Prespecify cluster/episode-level ITT inference, baseline adjustment, weighting,
missingness/maturity handling and multiplicity/interim-monitoring rules. Few
independent clusters need cautious randomization/cluster inference, not daily
pseudo-replication. Preserve allocation concealment until assignment and report
protocol deviations with complete adverse outcomes.

## Guardrails, rollback and governance

Operational owners must set numeric service, fairness, supplier and cash
thresholds from real requirements before enrollment. Independently enforce hard
order feasibility. Predeclare pause/rollback triggers for reconciliation/resource
failures, service degradation, supplier exposure, data outages or incident review;
specify decision owners and fallback policy. Rollback retains tracking and
follow-up of open orders, delayed receipts and mature outcomes.

Minimize customer/employee data, restrict access, keep audit trails, protect
credentials, and agree retention/deletion and approved aggregate reporting.
Register policy versions and the analysis plan. Simulation motivates further
investigation; only a properly executed real experiment can support the
registered real-world causal claim. This document deploys nothing.
