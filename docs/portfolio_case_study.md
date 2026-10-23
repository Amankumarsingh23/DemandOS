# DemandOS: forecasting recorded retail sales and testing inventory decisions

Retail planning needs both a forecast and an explicit way to act under uncertainty.
DemandOS implements a local research-to-serving workflow around the M5 validation
history: 58,327,370 daily observations across 30,490 store-item series. It turns
raw CSVs into validated Parquet and a DuckDB/dbt warehouse, constructs origin-safe
features, evaluates forecasting families, and tests hypothetical replenishment
policies under shared resource limits.

Recorded sales can be smaller than customer demand when stock is unavailable.
M5 does not identify the missing demand, actual inventory, supplier capacities,
economic costs or policy assignment. Forecast evaluation uses recorded sales;
decision outcomes use deliberately labeled assumptions. This distinction is
central to the project's design and claims.

## Technical contributions

The ingestion layer streams wide sales into long-form daily rows and checks
schemas, uniqueness, joins, nulls and row conservation. SQL dimensions/marts have
independent reconciliation. Forecasting uses chronological selection and
forecast-origin contracts: observed history stops at issue time, future calendars
are separate from unknown future sales and prices, and native model metadata
records fitted-through dates.

Implemented model families include statistical baselines, pooled boosting,
intermittent/hurdle routes, LSTM/GRU, N-BEATS and compact TFT. Hierarchical and
probabilistic experiments remain explicit alternatives. Calibration experiments
consume mature scores rather than unobserved labels. The serving demonstration
loads existing native quantile models through a digest-bound registry.

The decision layer connects newsvendor/protection-period reasoning to a stateful
receive/serve/approve/order simulator. A shared MILP coordinates budget, supplier,
flow and receiving entitlements without independently allocating the same cash
to every item. Risk and off-policy studies use declared hypothetical worlds.
The local publication layer stages and audits immutable batches, rejects conflicts
and reuses duplicate requests; it does not claim distributed exactly-once orders.

The [architecture diagram](architecture.md) separates serving, simulated decisions
and offline research. [Source modules](../src/demandos/) and their season-specific
guides preserve the implementation details.

## Evidence and experimental design

The [scientific results](capstone_results.md) and [registry](evidence_registry.json)
link every highlighted historical result to exact local artifact digests and
population definitions. On 2,100 matched historical cases, the S16 native model
has MAE 9.393891 and interval coverage 79.7143%; the older S17 checkpoint has MAE
8.414804 and coverage 77.6190%. Different fits prevent an isolated method claim.
Within one S17 neural checkpoint, stress-cohort coverage changes from 36.4087%
raw to 79.7619% split CQR and 84.0278% rolling CQR. This is descriptive temporal
calibration evidence, not a universal coverage guarantee.

The S20 constrained continuity comparison lowers mean simulated cost from
57,291.23 to 55,317.50 while immediate fill falls from 69.8623% to 67.1109%.
S21 planning CVaR improves while simulated operating CVaR worsens in the audited
baseline. These trade-offs prevent a blanket winner claim. S22 causal/OPE
results are synthetic/model-world, not actual financial returns.

The Season 24 golden path fixes six keys before examining their future outcomes,
issues 28-day forecasts at 2016-01-31, then independently audits costs, stock,
order receipts, resources, temporal perturbations and local fault handling.
It preserves raw source/checkpoint hashes and the 58,327,370-row warehouse.
Current replay measurements live in its immutable runtime bundle, described in
the [quickstart](capstone_quickstart.md); historical table reconciliation is
reported separately from actual rerunning or training.

## Limits and deployment requirements

The demo is CPU-first and bounded. Assumed costs, stock, capacity and supply do
not estimate Walmart operations. Physical storage uses a conservative entitlement
ceiling; partial deliveries, real stockout sensing, supplier dispatch, distributed
transactions, multi-user access and live telemetry are absent.

A deployment would require reliable inventory and availability events,
contracted costs/capacities, supplier order acknowledgements, durable idempotency
and reconciliation, access control, monitored service objectives and a safe
human approval/kill-switch process. Those are [proposed design extensions](system_design_defense.md),
not claimed implementation. [Interview notes](interview_defense.md) explain the
main technical decisions and their limits.
