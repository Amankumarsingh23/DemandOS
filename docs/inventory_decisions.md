# Season 18: scenario-based inventory decisions

Every result in this season is **SCENARIO-BASED / SIMULATED**. M5 contains
observed sales, not verified uncensored demand, on-hand stock, open orders,
supplier lead times, historical stockouts, backorders or lost sales. Subsequent
recorded sales serve as demand **only under the sales-as-demand scenario**.
Unserved units and monetary costs below are simulated, not measured Walmart
losses, purchases, profit or savings. This is a bounded single-decision study;
there is no multi-step replenishment/backtesting engine from Season 19.

## From forecasts to decisions

A forecast describes possible future observations. A decision also needs cost,
lead-time, inventory-state and timing assumptions. This implementation reuses
genuine saved forecasts without fitting or selecting any models on evaluation
outcomes:

- Season 15 `independent_bases`: interpretable N-BEATS native-checkpoint bottom
  forecasts for all 25 selected series. Checkpoint metadata and weight hashes
  are verified against its saved provenance.
- Season 16: repaired LightGBM P10/P50/P90, fitted through 2016-01-31, frozen
  for the three evaluation origins.
- Season 17: October-11-frozen LightGBM base quantiles and the corresponding
  rolling CQR intervals. The same checkpoint is required for base versus
  width-inflated scenarios. Its saved model/selection/calibration provenance
  is retained; Season 16 and 17 checkpoints are distinct.

Forecast/source/model artifact hashes, fit dates and complete keys are checked.
The forecast adapter drops outcomes before planning. Historical scenario
generation consumes only observations dated at or before each issue origin.
No future recorded selling prices enter costs or decisions.

## Hypothetical decision parameters

`configs/inventory.toml` declares monetary amounts in **hypothetical INR**.
They are not estimated from M5 selling prices:

| Cost regime | Selling price p | Purchase c | Salvage v | Shortage penalty | Overage penalty | Cu | Co | Fractile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced | 100 | 60 | 20 | 0 | 0 | 40 | 40 | 0.5 |
| Shortage expensive | 100 | 60 | 40 | 140 | 0 | 180 | 20 | 0.9 |
| Overage expensive | 70 | 60 | 10 | 0 | 0 | 10 | 50 | 1/6 |

The three lead-time settings are fixed one day, fixed seven days, and an
independent random lead time of 2/4/7 days with probabilities 0.25/0.50/0.25.
Each uses a seven-day review period; protection periods are at most 14 days,
fully covered by each saved 28-day forecast. Each series-origin resets to
10 on-hand units, three units on order, no backlog, and lost sales. This
deliberately common state is not claimed to be realistic for every series.
It especially constrains high-volume series before delivery. A separate
continuous fixed-quantity policy orders 20 units when triggered. Default
order multiple is one, minimum order zero; positive orders round upward to
the declared multiple/minimum, and orders never become negative.

## Discrete economics and service

Underage cost `Cu = p - c + shortage_penalty`; overage cost
`Co = c - v + overage_penalty`. Parameters must be finite and nonnegative,
with `Cu >= 0`, `Co >= 0`, and `Cu + Co > 0`. Compatible overage penalties
are single-period costs, not a second count of purchase cost.

For an integer stock target Q and integer period demand D:

```text
cost(Q,D) = Cu * max(D-Q,0) + Co * max(Q-D,0)
tau = Cu / (Cu+Co)
Q* = smallest nonnegative integer q with F_D(q) >= tau
```

`Distribution` validates weights, combines duplicate support, removes zero
probability support, exposes the CDF and uses its left inverse (empirical
inverted-CDF quantile, not interpolated sample quantiles). At tau=0 choose
zero; at tau=1 choose the largest positive-probability support. Machine
precision CDF tie tolerance is eight floating-point epsilons. Full finite
support cost curves verify each scenario optimum independently. Means and
variances are probability-weighted population moments.

Tests hand-check p=100/c=60/v=40: Cu=40, Co=20, tau=2/3;
D=30/Q=25 costs 200 and Q=35 costs 100. For P(D=0)=0.8/P(D=100)=0.2,
equal Cu/Co select Q=0 despite positive mean; Cu=90/Co=10 selects Q=100.
No policy silently chooses P90 regardless of costs.

For stock available for a whole single period, `CSL=P(D<=Q)` and
`fill_rate=1-E[(D-Q)+]/E[D]`. All-zero demand has CSL=1 for nonnegative
stock; fill rate is undefined (`null` in JSON), not silently one. Nine
fully served 100-unit cycles and one 90-unit-served cycle have CSL=90%
and unit fill=99%. One-order execution measures no-stockout over the
whole cycle and total served units. Reported fill pools units across
matched cases, rather than averaging ratios; CSL averages cycle indicators.

## Path assumptions and dependence

Three marginal quantiles do **not** identify a complete daily CDF, much less
a joint multi-day distribution. The saved conformal upper endpoint is not
assigned any arbitrary one-sided quantile probability. We expose assumptions:

1. Historical scenarios resample weekday-aligned, fully observed contiguous
   seven-day blocks, with replacement from the last 365 observed days ending
   at the origin. No block wraps beyond the prefix. Four blocks make each
   28-day path. This preserves within-block observed-sales patterns and
   breaks dependence across blocks. There are 512 paths per series-origin.
2. A separate weekday-aligned independent-day bootstrap changes block length
   to one for dependence sensitivity. Neither procedure identifies latent
   uncensored-demand dependence.
3. Forecast-informed paths assume a piecewise-linear inverse CDF with
   probability knots `(0,.1,.5,.9,1)` and anchors
   `(0,q10,q50,q90,q90+max(q90-q50,1))`. The two tails are explicit choices.
   Randomized empirical historical ranks couple these assumed daily
   marginals across horizons. Tie randomization is recorded through stable
   seeds. Half-up rounding after nonnegative clipping creates integer paths
   and can shift their empirical marginal quantiles. Even zero P10/P50/P90
   do not force an all-zero assumed tail. This is a sensitivity mechanism,
   not the uniquely implied predictive distribution.
4. Season 17 rolling-width paths use the **same October base quantiles** as
   their uninflated comparison. Their deviations around the base median
   scale by `max(1, interval_width/max(base_q90-base_q10,1))`. This heuristic
   never contracts widths; it has no inherited conformal service guarantee.
   It is not the cost-relevant quantile read directly from a conformal bound.

Seeds derive from SHA256 of the fixed seed, store, item and origin, not Python
hash randomization or outcomes. All policies share the same underlying sampled
paths/ranks. Cost and lead settings are fixed before evaluation. Exact synthetic
families are separately labeled `EXACT_SYNTHETIC_TOY`: independent daily demand,
shared positive shocks, repeated weekly patterns, clustered bursts, smooth demand,
and all-zero demand. Real M5 artifacts use `SIMULATED_SALES_AS_DEMAND`.

For fixed lead L, `D_L=sum(D_h,h=1..L)`; periodic protection demand sums
through `L+R`. We sum coherent daily paths first and then take their quantile.
Summing daily P90s is not a cumulative P90. For IID daily mean/variance,
`E[D_L]=L*mu`, `Var(D_L)=L*sigma^2`; generally variance is the sum of all
covariance-matrix entries. With independent random L and IID daily demand,
`Var(D_L)=E[L]*sigma^2+Var(L)*mu^2`. Exact finite joint distributions verify
these formulas. Normality is **opt-in only**: the hand example mu=20/sigma=5,
L=4, CSL=.95 gives 96.4485 continuous units and integer ROP=97.

## Policies and receipt timing

```text
ROP = quantile_tau(D_L)
signed_safety_offset = ROP - E[D_L]
nonnegative_buffer = max(0, signed_safety_offset)
IP = on_hand + on_order - backorders
S = quantile_tau(D_(L+R))
periodic order = max(0,S-IP), rounded to order constraints
continuous trigger = (IP <= ROP)
continuous order = periodic quantity OR fixed quantity, only when triggered
```

The safety offset may be negative for intermittent demand; the nonnegative
buffer is a separate descriptive quantity. The economic fractile is a target
probability under the assumed distribution, not a measured service guarantee.
The point and median heuristics use their summed daily estimates instead of
claiming those sums are cumulative quantiles. Random-lead targets/ROPs are
optimized against the probability mixture **before** lead time is realized.
The three non-distributional mean/median heuristics use the common historical
paths for their reported expected costs; that reference is named explicitly
in the decision ledger. It is not a fitted forecast-error distribution.

E is the day-close origin. One decision is placed then. Existing on-hand serves
days 1..L. Both the assumed open order and new replenishment arrive at **day L+1
open**, before its consumption; the cycle ends at day L+R close. Unserved units
are lost immediately, never backlogged or retroactively served. There are no
further orders. Stock exposure is the sum of daily closing stock (unit-days).
`sequential_demo.parquet` hand-checks four days: initial 3, open order 2, new
order 5, L=2/R=2, demand `[2,2,4,1]`; shortages=1, leftover=2, exposure=6.

**Two objectives stay separate.** The stock-target newsvendor comparison
assumes Q is available for the whole protection period. The executed one-order
comparison respects delivery timing. A newsvendor target is not claimed to
optimize this receipt-constrained sequential policy. Own-distribution regret
compares target cost to the optimum of that same assumed CDF. The common
historical CDF permits an additional shared expected-cost target comparison.
Executed regret uses a clairvoyant one-order lower-bound benchmark that knows
future demand and realized lead: it cannot undo pre-receipt lost sales or
dispose of initial overhang, but can choose the needed nonnegative order.
Default unit orders make that benchmark feasible. If minimum/multiple constraints
are enabled, this remains a **relaxed-order-constraints lower bound**, not
necessarily a feasible constrained policy. Benchmark future information never
enters policy decisions.

## Verified M5-informed experiment

Origins are 2016-01-31, 2016-02-28 and 2016-03-27; the February targets start
on leap day. Original 13-series continuity and disjoint 12-series stress
cohorts retain **1,092 and 1,008 daily forecast observations** respectively.
They produce 75 series-origins, not 2,100 inventory cycles. Three leads × three
costs produce **675 matched decision cases per method/action**: 351 continuity
and 324 stress. Seven methods yield **4,725 target plans**; three order actions
yield **14,175 single-order outcomes**. Every policy uses identical keys/costs
and future-sales windows. No incomplete windows or skipped cases occurred.

Balanced costs and periodic order-up-to, mean realized **simulated** mismatch
cost per case (hypothetical INR), across the same three lead settings:

| Method | Continuity cost | CSL | Unit fill | Stress cost | CSL | Unit fill |
|---|---:|---:|---:|---:|---:|---:|
| Historical mean | 8986.58 | 22.44% | 63.59% | 296.11 | 68.75% | 81.57% |
| Saved N-BEATS point heuristic | 8472.22 | 21.58% | 63.21% | 305.93 | 70.37% | 83.87% |
| Sum of daily median heuristic | 7581.11 | 16.88% | 46.21% | 321.11 | 65.51% | 74.04% |
| Historical cost-fractile | 9090.51 | 22.44% | 64.09% | 300.56 | 67.82% | 80.12% |
| Season 16 assumed quantile paths | 6833.76 | 19.23% | 57.89% | 303.70 | 70.83% | 82.39% |
| Season 17 assumed base paths | 7255.98 | 21.58% | 60.30% | 313.89 | 70.60% | 79.98% |
| Season 17 rolling-width heuristic | 7253.08 | 21.58% | 60.49% | 313.89 | 70.60% | 79.98% |

There are 117 continuity and 108 stress cases per row for this cost regime.
The common ten-unit initial stock makes pre-receipt shortages pronounced for
larger series. Lower cost does not imply better fill; the assumed forecast
distribution and holding/shortage tradeoff matter. These are descriptive
comparisons, not new model/scenario choices selected by test outcomes.
Rolling-width inflation produced no change in these aggregate balanced stress
metrics. Improving daily marginal coverage does not guarantee better inventory
decisions, and the Season 16/17 rows use different checkpoint vintages.

For historical cost-fractile decisions under balanced costs, continuous up-to
versus periodic up-to costs are 9114.44 versus 9090.51 (continuity), and 340.00
versus 300.56 (stress); their trigger rates are 65.81% and 5.56%. Mean periodic
order quantities are 299.15 and 2.43 units. Fixed-20 continuous orders cost
11274.96 and 333.70 respectively. Shortage-expensive historical periodic targets
raise mean orders to 443.27/5.66 units and achieve fill 67.43%/89.48%; balanced
and shortage-expensive costs have different coefficients and are not directly
comparable monetary savings.

Origin-safe regimes comprise continuity all-zero 1, erratic 6, intermittent 15,
lumpy 9, smooth 8; stress intermittent 32 and smooth 4 series-origins.
The stress cohort is sparse by selection, not uniformly intermittent at every
origin. With Season 16 balanced periodic decisions, stress intermittent cases
have 86.90% fill/78.13% CSL versus 71.61% fill/12.50% CSL for its smooth cases;
these small unbalanced groups do not support causal claims.

Dependence matters even with similar daily marginal means: for fixed seven-day
lead plus seven-day review, historical seven-day blocks versus independent-day
resampling yield average protection mean 209.482/209.368, variance
17876.381/5120.259, and shortage-expensive target 298.840/263.453 units across
75 series-origins. This is an assumption sensitivity, not verified latent-demand
covariance. Detailed expected and realized target costs are saved separately.
For that same fixed-seven-day, shortage-expensive comparison, realized
aggregate-target cost is 2922.67 versus 3108.27 hypothetical INR/case; this is
not the receipt-constrained one-order cost or actual measured inventory loss.

## Verification and reproduction

```powershell
.\.venv\Scripts\python -m demandos.inventory --root . --config configs/inventory.toml
.\.venv\Scripts\python -m unittest discover -s tests
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,warehouse,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

Use the existing analytics/EDA/reconciliation environment; no new dependencies
are added. Required genuine prior artifacts must exist; missing/mismatched
inputs fail rather than being fabricated or downloaded. DuckDB is read-only,
two threads and 2 GB memory cap; only 25 series across the bounded history and
target range are extracted, never a dense 58-million-row panel.

At all three origins, every series replays with future sales mutated, future
sales removed, and future prices mutated: **225 checks** compare exact paths,
their distributions, quantities, ROPs, triggers and provenance. Costs, state,
forecasts and seeds remain frozen. Complete dates, matched cases, exact support
cost minimizers, scenario labels and source hashes are hard gates.
The **58,327,370-row**, **30,490-series** warehouse, **65,695,409 units**, and
**187,676,570.0200 known recorded revenue** remain unchanged. That revenue is
a source-integrity total, not a hypothetical inventory cost parameter.

All **263 Python tests** (239 prior plus 24 new) and **36 freshly executed dbt
tests** pass with no failures/skips. The final recorded scenario run took
**66.93 seconds**, peaking at **723.16 MiB** process working set and extracting
**11,225** observed rows. All five raw/interim/processed source fingerprints
and 75 prior forecast/model/provenance fingerprints are unchanged. An
independent saved-output audit reproduces all **14,175** one-order outcomes
using prefix-sum formulas instead of the daily execution loop, with maximum
cost/quantity difference **zero**. It also checks all discrete probabilities,
labels, complete common cases, manifest files and final warehouse totals.
There are no execution skips or training warnings; the declared scenario
limitations above remain applicable.

Generated ignored artifacts: `data/inventory/m5_h28/`, including `config.json`,
`summary.json`, `manifest.json`, `inventory_report.md`, a comparison figure,
and Parquet tables for paths, discrete period distributions, cost curves,
decisions, single-order evaluations, matched cost/lead/regime/origin metrics,
dependence sensitivity, exact toy scenarios, the sequential demo, provenance
and future-data checks. Publication occurs only after correctness gates pass
and only replaces manifest-owned generated files. Raw files, prior forecasts,
models and warehouse definitions are preserved. None of these generated or
private files belong in Git.
The local `independent_verification.json` records the separate post-publication
audit and fresh dbt result fingerprint; it is not a replacement for running tests.

## References

Season 20 now adds [shared-resource allocation](optimization.md), retaining
these forecast/scenario assumptions. Its expected mismatch objective and the
simulator's procurement-inclusive operating costs remain distinct.

Season 19 now extends these frozen planning inputs with a separate
[sequential simulator](inventory_simulation.md), multiple orders, FIFO backlog,
explicit operational costs and daily state transitions. It preserves this
season's single-order results and their own receipt convention.

The objective and policy terminology follow the inventory topics in
[MIT D-Lab: Supply Chains](https://ocw.mit.edu/courses/15-772j-d-lab-supply-chains-fall-2014/),
especially [Lecture 14: Stochastic-demand Inventory Models](https://ocw.mit.edu/courses/15-772j-d-lab-supply-chains-fall-2014/resources/mit15_772jf14_lec14/).
Forecast distributions and prediction-interval interpretation are discussed in
[Forecasting: Principles and Practice](https://otexts.com/fpp3/),
[section 5.5](https://otexts.com/fpp3/toolbox.html#distributional-forecasts-and-prediction-intervals).
Our tail interpolation, historical rank coupling, cost parameters, receipts
and dispersion sensitivity are project scenario assumptions, not empirical
facts established by those sources.
