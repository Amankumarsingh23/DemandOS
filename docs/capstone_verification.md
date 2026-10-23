# Season 24 acceptance evidence

This record separates the current bounded execution from earlier saved-artifact
reconciliation. Reproduce it with the [quickstart](capstone_quickstart.md).

## Executed golden path

The native S16 LightGBM P10/P50/P90 ensemble was loaded without retraining, with
fit cutoff 2016-01-31 and earlier development validation ending 2016-01-03.
The origin was 2016-01-31. Three keys per established continuity/stress cohort
were selected by sorted identifiers, without looking at their evaluation outcomes.

| Store | Item | Frozen cohort |
| --- | --- | --- |
| CA_1 | FOODS_1_051 | sparse_stress |
| CA_1 | FOODS_3_090 | continuity |
| CA_1 | FOODS_3_137 | sparse_stress |
| CA_1 | FOODS_3_595 | continuity |
| CA_2 | HOUSEHOLD_1_283 | sparse_stress |
| CA_3 | FOODS_3_090 | continuity |

There are 168 forecasts, six noncommitted origin previews, and three complete
168-row simulated state trajectories: historical recorded-sales proxy, a change
of +1,000 units on day 14 for the first series, and a receiving closure on days
8–14 that retimes whole orders. No preview order is double-booked into the
complete sequential controller's procurement ledger.

| Current replay measurement | Value |
| --- | --- |
| Recorded-sales MAE | 22.139290 units/day |
| WAPE | 81.1388% |
| Mean P10/P50/P90 pinball loss | 5.522123 |
| P10–P90 empirical coverage | 77.3810% |
| Mean interval width | 29.194444 units |
| Primary simulated operating cost | 214,449.60 assumed currency units |
| Primary simulated immediate fill | 24.3237% |
| Recorded-sales demand proxy / immediately served / simulated lost units | 4,584 / 1,115 / 3,469 |
| Procurement and activation committed / initial cash limit | 75,270.00 / 165,396.00 |
| Initial complete invocation | 42.86 seconds, 571.77 MiB peak process memory |
| Identical-input complete rerun | 75.27 seconds, 570.86 MiB peak; immutable bundle reused |

The small fixed cohort performs differently from the 25-series, three-origin
historical checkpoint comparison. These populations must not be merged into a
single improvement claim. Costs and lost units above are simulated, not observed
Walmart outcomes. Runtime includes source hashing and independent replay; the
staged-only timing in `runtime.json` has a narrower scope.

## Independent gates and isolated faults

The audit recomputed forecast keys/dates/quantiles, current native inference,
source/model digests, mature-label scoring arithmetic, stock/backlog/queue
continuity, receipt identities and dates, activation/integer/pack/MOQ feasibility,
persistent cash and supplier entitlements, procurement flow and receiving quotas,
and conservative assumed physical-storage occupancy ceilings. All recorded
states/actions before day 14 matched after the future-demand perturbation.
Post-origin sales and prices did not change issued features or their digest.

Five isolated fault cases passed: future-ineligible features, incompatible
fallback contract, invalid hard allocation and corrupt synthetic artifact bytes
were blocked; a duplicate publication reused its batch with no new actions.
The original raw inputs and native checkpoints were never modified. The previous
Season 23 registry entries/state hashes were also preserved.

The full warehouse remained **58,327,370 rows**. All **414 Python tests** passed,
including 22 capstone tests and all 392 earlier tests. A fresh real-data dbt
invocation passed **36 tests**, with zero errors/warnings/skips. Documentation
scope/internal-link checks and Git ignore checks passed. Generated data,
checkpoints, local environments and credentials are excluded from publication.

## Exact provenance and output locations

The acceptance bundle is:

```text
data/capstone/m5_h28/runs/ca99ad7a532da581200d/
```

Its full request identity is
`ca99ad7a532da581200d18107e94750896608afec2421a56bbdbbffafa6dec1e`.
It binds the inspected Git base `e3e2aa8be105b981bb560d67ae1b560026f6696c`
plus the exact implemented source/config/dbt code digest
`54333df420cc871d90ef3a80e8c5302eb267778017853b25ef3cb0f995b7b6d4`.
The full-source snapshot fingerprint is
`02844756610e074f866ce2ea4fc137916b61c5aa2881de6da0e385e40d1e871b`.
This was a pre-publication acceptance run: its base SHA alone is not a claim that
the capstone source already existed in that earlier commit. Exact file hashes
bind the reviewed implementation that is subsequently committed.

The manifest lists every source/model file digest, environment version, selected
key, feature contract, cost, opening stock, lead/review period, resource limit,
seed and table outcome label. `completed.json` seals persisted bytes;
`verification.json` records independent arithmetic; `faults.json` records the
isolated dispositions. `last_invocation.json` records successful equivalent reuse.
Changed Git commit or code/input/environment creates a new identity on a later
execution; `latest_verified.json` locates the most recent complete run.

The [15-record evidence registry](evidence_registry.json) recomputes selected
historical calibration, constrained inventory, planning/operating risk, native
serving and synthetic OPE results from saved records. The
[comparability and claim matrices](capstone_results.md) explicitly separate
their populations, economics and measured-versus-simulated status. No full
cross-season retraining or live production test is claimed.

## Remaining limits

Physical storage is a conservative assumed entitlement ceiling, not a study of
binding real storage capacity. Supply delays preserve whole-order quantities;
partial deliveries are not implemented. There is no external order dispatch,
distributed order transaction, real inventory/propensity telemetry, identified
retailer causal effect, official full M5 score or verified financial savings.
[System design](system_design_defense.md) describes deployment requirements as
proposals. Written learning assessments remain separate and ungraded here.
