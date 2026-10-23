# DemandOS system design defense

## Implemented local architecture

The [current architecture](architecture.md) describes code that exists: validated
CSV-to-Parquet, read-only DuckDB queries/dbt marts, chronological features and
model evaluation, native-model registry, historical batch forecasts, isolated
monitoring/fault exercises and a simulated shared-resource decision controller.
The [capstone command](capstone_quickstart.md) loads bounded rows and three saved
native LightGBM quantiles; it does not retrain all model families.

One process publishes a complete directory only after persisted-record audits.
A local exclusive writer file prevents cooperating simultaneous writers.
Same-filesystem rename exposes the completed directory atomically. SHA256 binds
files to a manifest; a changed request/content cannot reuse the same identity.
This supplies local retry safety. It does not ensure delivery of an external
order, prevent all malicious tampering, or implement a cross-service transaction.
Conservative stale-lock handling requires operator investigation after a crash.

The real source has 58.3 million rows; SQL handles aggregation while the capstone
materializes only a 25-series loader snapshot and a six-series replay. Native
inference runs on CPU. Solver limits are short and hard feasibility must pass;
a forecast cannot authorize an infeasible order. Timing and peak process memory
are measured per invocation, not inferred into fleet capacity or service SLOs.

## Proposed scale-out design — not implemented

Partition ingestion by immutable source snapshot and validated date ranges.
Record event time, arrival time, correction lineage and watermarks; quarantine
late/invalid records. Publish versioned columnar partitions only after row/key
and source-total checks. Keep missing price distinct from zero and missing
observations distinct from recorded zero sales.

Forecast partitions can run by store/item shard with a frozen model/feature
contract and explicit as-of cutoff. Static calendars have a different availability
contract from sales or prices. Store signed native artifacts and metadata in
versioned object storage, resolve approved aliases at batch start, pin exact
digests throughout that batch, and preserve retired/rollback history. Artifact
hash integrity alone is not proof of trusted training or authorization.

Shared budget/supplier allocation cannot be solved independently per shard with
the full budget repeated in each worker. Use a central pool coordinator or
preallocated, transactionally conserved resource leases; combine demand summaries,
solve the declared pool objective, verify physical limits, then atomically reserve
cash and supplier entitlements. Reconcile reservations with acknowledgements,
cancellations and partial receipts. Solve decomposition changes optimality and
must report its bounds and operational trade-offs.

Separate forecast publication from an order outbox. Persist a globally unique
business order key and payload digest in a transactional database alongside
resource reservations. A dispatch worker uses supplier-supported idempotency
tokens, retries, acknowledgement state and dead-letter handling. Reconcile
supplier receipts to order identities and avoid creating a second purchase when
a timeout hides the first acceptance. Without supplier idempotency, human
reconciliation may be necessary; an atomic local directory cannot solve this.

## Monitoring, rollback and safety — deployment requirements

Track source freshness/completeness, forecast contract failures, model digest,
prediction coverage, backlog, order receipt discrepancies, pool reservation
balance and solver incumbent/bound. Drift PSI/KS are descriptive signals; score
accuracy and calibration only after labels mature, with segment/sample-size
context. Real business monitoring needs inventory, censored-demand indicators,
economic definitions and incident telemetry not present in M5.

Rollback should pin a prior eligible model plus compatible feature mapping and
decision policy. Keep audit history immutable. A compatible historical baseline
may provide forecasts, but final resource feasibility remains mandatory. An
incompatible fallback must block. Prefer a safe hold/manual review if no validated
decision is available; do not loosen budget or supplier limits silently.

Introduce least-privilege accounts, secrets management, supplier network isolation,
approved economic configuration, human order approval, a dispatch kill switch,
durable backups and recovery drills. These controls are required future design,
not current production capabilities. Bound staleness tolerance by business impact;
measure capacity with representative cohorts and concurrent workloads before
setting latency/availability objectives.

## Worked system challenge

Two workers each see a 1,000-unit shared cash budget and independently reserve
700. Both local plans can be feasible while total reservations reach 1,400.
Forecast sharding does not fix the violation. A serializable reservation of the
whole vector, or disjoint leases whose sum is at most 1,000, prevents overspend.
If one supplier acknowledgement is lost, retry its original business key and
reconcile state before releasing/reissuing its resources. This transaction and
supplier protocol is proposed, whereas the current coordinator demonstrates the
conservation arithmetic inside one simulated process.
