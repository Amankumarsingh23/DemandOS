# Season 19: sequential inventory simulation

Every result is **SCENARIO-BASED / SIMULATED**. Recorded M5 sales are used as a
customer-demand proxy, not verified uncensored demand. M5 does not establish
actual stock, stockouts, supplier delays, backlogs, purchasing costs or savings.
Simulated fulfillment and shortages remain distinct from recorded source sales.
Season 18's single-order experiments and all earlier models remain unchanged.

## State, event timeline and observation boundary

`demandos.simulation` separates immutable policy/observable-state contracts,
the daily event engine, operational metrics and paired comparisons, experiment
orchestration, streaming Parquet publication and independent SQL verification.

Each run has canonical keys `(series_id, origin_date, policy_id,
cost_scenario_id, lead_scenario_id, demand_scenario_id, replication_id)`.
Store/item identifiers are preserved verbatim; `series_id` is their JSON pair.
Policy IDs include the experiment namespace so initialization/warm-up variants
cannot collide. Daily keys add simulation date and one-based day index. A
separate calendar-day index anchors reviews/shocks to the original origin.

State comprises integer on-hand, FIFO backlog cohorts, an outstanding-order
dictionary keyed by order ID, scheduled arrival day/quantity/lead/shock ID,
on-order total, inventory position and cumulative fulfillment/cost ledgers.
Policies receive an immutable `Observation` with stock, observable backlog,
scheduled pipeline, calendar information and fulfilled **new** sales history
through that day's close. They do not receive demand labels, future prices,
unserved lost demand, simulator queues they can mutate, or future shocks.
Their immutable `Policy` configuration contains the frozen forecast-derived
ROP/target. They return quantity, reason, threshold/target and timestamp.

The daily timeline is fixed:

1. Receive **all** orders due at opening, exactly once; remove them from pipeline.
2. Fulfill existing backorders FIFO, if using backlog mode.
3. Observe evaluator demand and serve current demand from remaining stock.
4. Record unmet units as lost sales or new backlog.
5. Build the permitted post-demand observation and compute inventory position.
6. Review the policy and place its order at day close.
7. Queue the new order for opening of `issue_day + lead_time`.
8. Book costs and closing state; assert daily and cumulative balances.

An order on day t with L=1 arrives at **t+1 opening**, never on its issue day.
An initial open order is assumed issued at local day zero, with its ETA known
at initialization; it was paid before the scoring window. The main initial
three-unit order's ETA uses the shared supplier day-zero shock. These are
assumed states, not reconstructed operational M5 records. Season 18 used its
own explicitly different single-order receipt boundary; its prior results are
preserved, not reinterpreted as this sequential backtest.

## Conservation and fulfillment modes

For opening stock I, receipts A, new demand D and existing backlog B:

```text
old_fulfilled = min(B, I+A)            # zero in lost-sales mode
immediate = min(D, I+A-old_fulfilled)
unmet = D-immediate
ending_stock = I+A-old_fulfilled-immediate

lost sales: lost=unmet; new_backlog=0; ending_backlog=0
backorders: lost=0; new_backlog=unmet
            ending_backlog=B-old_fulfilled+new_backlog

I+A = old_fulfilled + immediate + ending_stock
D = immediate + lost + new_backlog
B+D = old_fulfilled + immediate + lost + ending_backlog
initial_pipeline + cumulative_orders = cumulative_receipts + ending_pipeline
```

All states/flows are nonnegative integers. Lost-sales mode never accumulates
backlog. Old backlog fulfillment is not counted again as new demand or
immediate service. FIFO cohorts measure completed backorder waiting unit-days;
unfulfilled backlog remains censored at the horizon. End-of-day backlog
unit-days include pending units; this differs from completed-order waiting.

Golden tests verify initial 10, initial order 3 arriving day 3, demand `[6,5,4]`,
and a day-1 order of 8 with lead 2: receipts `[0,0,11]`, immediate service
`[6,4,4]`, losses `[0,1,0]`, closing stock `[4,0,7]`. Total demand 15,
served 14, lost 1; initial stock 10 plus receipts 11 equals served 14 plus
final stock 7. Queue tests cover simultaneous arrivals, crossing orders,
unique receipt identity and terminal outstanding orders.

## Frozen targets and policy rules

Origin-safe saved point, Season 16 quantile and Season 17 rolling-width
assumptions are reused through the verified [Season 18 adapter](inventory_decisions.md).
Every origin's saved ROP/target is reproduced before simulation. Original
365-day historical prefixes, 512 scenarios and seven-day weekday blocks remain
unchanged. There is no model retraining or retrospective scenario tuning.

Four policy sources are historical cost-fractile, saved N-BEATS point-mean
heuristic, Season 16 assumed quantile paths and Season 17 assumed rolling-width
paths. Their fixed thresholds use the declared cost fractile and lead/protection
distributions; point forecasts use expected-sum heuristics instead. They are
**stationary targets frozen at E**, not daily rolling forecasts. Using the
first L / L+R forecast days as a fixed planning distribution is an explicit
approximation. These legacy lead/protection targets are not claimed to be
exactly optimal for the new event timeline, daily undershoot, fixed fees or
finite-horizon operational cost objective.

```text
IP = closing_on_hand + outstanding_on_order - closing_backlog
(r,Q): at each day close, if IP<=r order fixed Q
(r,S): at each day close, if IP<=r order max(0,S-IP)
(R,S): on scheduled review days only, order max(0,S-IP)
```

The first two are **daily continuous-review approximations**, not instant-event
continuous review. Default periodic review is days 1,8,15,22, anchored to E+1;
no new order is placed at day zero. Positive orders round upward to the
configured multiple/minimum; defaults are multiple 1, minimum 0, fixed Q=20.
Triggers and quantities remain separate. R=3 sensitivity recomputes S from
L+3 scenarios before evaluation. Three quantiles do not identify joint demand,
and conformal interval width has no inherited inventory service guarantee.

## Exogenous supplier streams and fair comparisons

Fixed lead times 1/7 days and random 2/4/7 days (probabilities .25/.50/.25)
reuse Season 18 assumptions. A separate sensitivity adds two days to weekend
issues, allowing time dependence and order crossing; planning retains the
baseline lead distribution, explicitly measuring unmodeled delay exposure.

Supplier uniforms are deterministically keyed by fixed seed, series, original
origin, replication and **calendar issue day**. Neither policy, cost regime
nor order sequence enters the key. Policies ordering on different days do not
desynchronize a mutable PRNG; policies ordering on the same day share the
supplier shock. Quantity/order ID do not influence lead time. Initial pipeline
uses the same day-zero convention across competing policies.

Each matched comparison holds demand, initialization, mode, costs, supplier
stream convention, scoring window, review assumptions and terminal valuation
fixed. Twelve policies are four sources × three rules. Daily and run-level
cost differences use historical-fractile periodic RS as the named reference.
Supplier replications are averaged within **series-origin clusters** before
computing descriptive standard errors. Days within a path are never treated
as independent replications. Repeated series/origins remain dependent;
these standard errors do not establish population or causal confidence.

## Operating costs and service definitions

`configs/simulation.toml` declares hypothetical INR costs, independently of M5
selling prices. Procurement uses the explicit Season 18 purchase assumption 60.

| Regime | Holding / unit-day | Lost penalty / unit | Backlog / unit-day | Order fee | Purchase / unit |
|---|---:|---:|---:|---:|---:|
| Balanced | 0.2 | 40 | 2 | 5 | 60 |
| Shortage expensive | 0.2 | 180 | 9 | 5 | 60 |
| Overage expensive | 2 | 10 | 0.5 | 5 | 60 |

Daily cost = closing-stock holding + lost penalty **or** closing-backlog carry
+ positive-order fee + purchasing cost of new orders - optional final stock
credit. Season 18 Cu/Co determine planning targets; its one-period mismatch cost
is **not added** to this operational ledger. There is no revenue or profit claim.
The hand test stock=7/lost=1/order=8, h=2/lost penalty=30/fee=50/purchase=10
gives **14+30+50+80=174**.

Purchasing is booked at **order**, including deliveries due after the horizon.
Initial open orders are prepaid. Default terminal stock value is zero; a
separate sensitivity credits ending on-hand at assumed salvage 20. Outstanding
orders get no terminal credit. No future backlog-tail penalty is fabricated;
ending backlog is reported. These finite-horizon accounting choices affect
rankings and do not define an optimized steady-state policy.
Starting on-hand is also a supplied prepaid state: the stock-100 sensitivity
does not charge its earlier acquisition capital. Its lower scored-window cost
is not a claim that extra inventory is free or improves total investment return.

Immediate unit fill = immediately served **new** demand / total new demand.
All-zero demand fill is undefined (`null`), with no-demand cases counted.
Demand-weighted pooled fill and unweighted per-run fill are reported separately.
Daily no-shortage fraction counts days without new unmet units; it is **not CSL**.
No replenishment cycle segmentation is implemented, so replenishment CSL is
explicitly undefined. Reports retain holding/lost/backlog/ordering/procurement
components, cost per demanded/immediately-served unit, stock exposure, backlog
unit-days, completed waiting, order/arrival timing and terminal state.

## Evaluation, sensitivity and warm-up design

The recorded-sales evaluation reuses 25 series × three origins: January 31,
February 28 and March 27, 2016. Targets are Feb 1–28, Feb 29–Mar 27 and
Mar 28–Apr 24. There are **75 unique recorded-sales paths / 2,100 daily targets**,
with continuity/stress strata of 13/12 series and 1,092/1,008 daily observations.
Every main run scores all 28 days from on-hand 10, on-order 3, no backlog;
**no evaluation warm-up is discarded**. Two supplier replications are explicitly
identified; fixed-lead replications duplicate outcomes and do not add independent
demand observations.

Nine separate sensitivities use balanced costs/fixed-seven-day lead/replication
zero across all 75 paths: backlog instead of lost sales, initial stock 0/100,
initial open orders zero, review interval 3, weekend delay, terminal stock credit,
and historical seven-day-block versus independent-day bootstrap demand. Bootstrap
paths are explicitly separate from recorded-sales backtests and use sample zero
of the frozen origin-safe scenario distribution; they are not extra observed M5
days. No combinations are selected by evaluation performance.

Warm-up experiments use one preselected series per cohort at the first origin,
two supplier/demand replications and independent/block bootstrap families.
A 112-day synthetic path is generated from origin-prefix history only. Burn-in
0/28/56 days precedes **the identical final 56 scored days** of each path.
Calendar review phase and supplier keys stay aligned; state carries through
burn-in and warm-up costs remain separate from scored cost. Initialization
never uses future M5 sales. This bounded sensitivity does not assert convergence
to stationarity.

## Hard gates and reproduction

All three origins undergo future-sales mutation/removal and future-price
mutation before regenerating frozen plans: 225 checks. Sequential checks change
demand at day 10/20 and require **only pre-k** states/actions/placed-order
schedules to stay identical. Later decisions may legitimately respond to
changed realized stock/backlog. Separate exhausted-stock paths change hidden
lost demand without changing observed fulfillment or decisions. These tests
cover all 75 paths, three policy rules, and both fulfillment modes where relevant.

The engine checks every daily/cumulative stock, new-demand, backlog and queue
balance. Independent DuckDB queries recheck persisted state carryover, complete
canonical daily keys, dates/leap day, costs, individual order lifecycles,
receipt/order totals, unique exogenous scenarios, matched shocks/demand and
daily-to-run paired cost sums. All earlier model/source fingerprints and final
warehouse totals must remain unchanged. Correctness failures prevent publication.

```powershell
.\.venv\Scripts\python -m demandos.simulation --root . --config configs/simulation.toml
.\.venv\Scripts\python -m unittest discover -s tests
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,warehouse,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

No new dependencies are required. Genuine source-bound Seasons 15–18 artifacts
and the existing analytics/EDA environment are required; unavailable/mismatched
inputs fail explicitly. No downloads, invented forecasts, warehouse/model
changes, optimization constraints or Season 20 features are included.

Ignored outputs under `data/simulation/m5_h28/` include daily inventory/action
ledgers, order and arrival events, unique exogenous demand/supplier scenarios,
run/stratum/origin/regime cost-service metrics, daily/run paired comparisons,
warm-up/sensitivity cases, temporal/origin checks, conservation/verification JSON,
source/model/config provenance, a trajectory figure, results report and manifest.
Parquet is streamed by matched experiment groups. DuckDB extraction is bounded
to 25 series with two threads/2 GB; persisted-output verification uses 1 GB.
Only manifest-owned generated files are replaced. Raw data and prior artifacts
remain immutable; generated data/figures/caches remain outside Git.

Inventory terminology is consistent with [MIT's stochastic-demand inventory
lecture](https://ocw.mit.edu/courses/15-772j-d-lab-supply-chains-fall-2014/resources/mit15_772jf14_lec14/).
The event order, costs, initial state, supplier streams, frozen-target policies
and finite-horizon experiments here are explicit project assumptions.

## Verified real execution results

The completed experiment produced **16,200 main runs / 453,600 main daily
rows**, **8,100 sensitivity runs**, and **72 separate bootstrap warm-up runs**:
**24,372 total runs / 686,448 daily rows**. Twelve policies share 1,350 main
cases each (75 series-origins × three costs × three leads × two supplier
replications). Main observed demand still comprises only 75 paths / 2,100
unique records. There are 293,092 order-event records, including initial,
placed, received and terminal-outstanding event types; these are not all new
orders. All 1,125 sequential temporal checks, 225 frozen-origin checks and
19 independent persisted-output SQL gates pass.

Balanced costs, fixed-seven-day lead, periodic RS, hypothetical INR operational
cost per 28-day run and demand-weighted immediate fill:

| Frozen planning source | Continuity cost | Fill | Daily no-shortage | Stress cost | Fill | Daily no-shortage |
|---|---:|---:|---:|---:|---:|---:|
| Historical cost-fractile | 57315.62 | 70.07% | 71.43% | 1622.49 | 80.44% | 89.88% |
| Saved point mean heuristic | 55856.38 | 67.11% | 74.73% | 1801.77 | 86.64% | 93.45% |
| Season 16 assumed quantile paths | 48516.02 | 57.37% | 69.05% | 1771.45 | 85.31% | 93.65% |
| Season 17 assumed rolling-width paths | 51146.59 | 61.12% | 71.61% | 1750.98 | 83.72% | 93.15% |

Each continuity/stress cell uses 78/72 runs with identical cases and accounting.
These include duplicate fixed-lead replications; descriptive paired clusters
number 39/36 series-origins. Relative to historical RS, Season 16 RS costs
8799.60 less per continuity run (cluster SE 2591.29) but 148.96 more per stress
run (SE 40.60). The continuity lower-cost policy also serves less demand; no
uniform cost-and-service winner or causal savings claim follows.

Historical source policy comparison at the same balanced/fixed-7 settings:

| Rule | Continuity cost | Fill | Mean orders | Stress cost | Fill | Mean orders |
|---|---:|---:|---:|---:|---:|---:|
| Daily rQ, Q=20 | 42734.37 | 30.49% | 16.74 | 2128.39 | 86.46% | 1.58 |
| Daily rS | 63674.63 | 73.11% | 3.62 | 1783.68 | 82.65% | 2.50 |
| Periodic RS | 57315.62 | 70.07% | 3.38 | 1622.49 | 80.44% | 2.56 |

Historical RS ends with mean stock/pipeline 64.26/162.90 units in continuity,
and 4.39/5.47 in stress, with simulated lost units 241.82/6.14 per run.
Daily rS carries more closing stock (153.18/6.72); fixed Q=20 severely limits
high-volume continuity service. Full daily trajectories and all cost components
are retained; the figure uses the first identifier, not a favorable case.

Separate historical-RS sensitivity findings (balanced/fixed-7 reference):

- Backorders yield immediate fill 62.60%/68.14%, ending backlog 47.51/1.78,
  and cost 63574.24/1751.06. Delayed backlog fulfillment is reported separately;
  zero lost units here does not imply perfect immediate service.
- Starting stock zero changes fill to 69.09%/64.69%; stock 100 to 76.02%/100%,
  with the prepaid-initial-capital limitation above. Initial open-order zero
  gives 69.84%/78.41% fill.
- R=3 gives cost 59659.43/1672.11 and fill 68.51%/78.58%; more frequent review
  is not automatically better under frozen targets and the cost ledger.
- Weekend delay gives fill 69.86%/79.20%. Terminal salvage credit changes cost
  to 56030.49/1534.72 without changing physical trajectories or service.
- Block versus independent bootstrap demand produces continuity fill
  66.81%/71.95% and stress fill 88.44%/86.62%, on simulated prefix-derived paths,
  not additional observed M5 outcomes.

In the selected continuity warm-up series, historical RS independent-day
bootstrap fill rises from 83.62% with no burn-in to 95.77% with 28 or 56 days;
block-bootstrap fill rises from 78.44% to 86.90%. These use the same scored
tail, supplier dates and review phase. Stress warm-up fill is unchanged within
each family (89.92% independent, 90.58% block), despite changed scored costs.
Equality at 28/56 days for these selected cases is not general stationarity proof.
Season 16 RS stress intermittent versus smooth groups achieve 89.37%/75.88%
fill on 64/8 fixed-lead balanced runs; these are small, unbalanced strata.

All **289 Python tests** (263 prior plus 26 new) and **36 fresh dbt tests** pass.
The 58,327,370-row / 30,490-series warehouse retains 65,695,409 units and
187,676,570.0200 known recorded revenue. All source/model fingerprints remain
unchanged. The scenario execution took **920.76 seconds**, peak process memory
**931.50 MiB**; the independent post-run audit took **22.55 seconds**, peaking
at **1657.99 MiB**, including DuckDB/Python overhead beyond its 1 GB query-buffer
cap. No cases were skipped. Explicit assumption/finite-horizon warnings remain
in the manifest and report. No earlier source/model pipeline was modified.
