# Season 13: target-only N-BEATS

DemandOS now implements generic and interpretable N-BEATS in `demandos.nbeats`,
using the existing daily-grid validators, origin schedule, cohort selection,
PyTorch seed/loader/gradient utilities and exact-case evaluation. It predicts
recorded observed sales, not verified uncensored demand. No new dependencies
are required beyond the existing neural and analytics environment.

## Backcast, forecast and double residual stacking

Each fully connected block reads the remaining normalized history, runs three
64-unit ReLU layers, and produces separate backcast and forecast coefficients.
The backcast spans the 56 historical positions; the forecast spans 28 future
positions. The backward path removes the block's backcast from the residual;
the forward path adds its forecast to the accumulated forecast:

```text
r = X * valid_history_mask           # [B,56], oldest to newest
F = zeros(B,28)
for block in blocks:
    b, f = block(r)
    r = (r - b) * valid_history_mask
    F = F + f
```

This is a residual forecasting architecture with a backcast path, not just an
MLP with a 28-unit output. Blocks learn jointly through the future forecast
loss. There is **no supervised backcast reconstruction loss**. Earlier backcast
heads receive gradients through later blocks. An unshared final backcast head
cannot affect the future loss because no subsequent block consumes its residual;
its missing gradients are expected and explicitly recorded. Any other missing
gradient, nonfinite loss/gradient/parameter or invalid prediction fails the run.

| Variant | Applied blocks | Sharing | Forecast basis |
|---|---|---|---|
| Generic | 3 generic blocks | None by default | Separate learned 8-coefficient backcast/forecast linear projections |
| Interpretable | 2 trend, then 2 seasonal | One shared block within each stack; no cross-stack sharing | Fixed degree-2 polynomial and weekly Fourier bases |

Generic sharing and interpretable sharing are configurable. Sharing aliases the
same block parameters across applications rather than copying their initial
values. Parameter counts count unique parameters. Generic contributions are
labelled by block index, without assigning unsupported trend/seasonal meanings.

## Structured bases and component meaning

Trend uses constant, linear and quadratic terms on continuous signed grids:
past days `[-55,...,0]/56`, future days `[1,...,28]/56`. Both basis matrices
have 3 rows; historical/future widths are 56/28. The block learns independent
backcast and forecast coefficients, so continuity is not imposed by tying them.

Seasonality uses a constant plus cosine/sine pairs at harmonics 1, 2 and 3 of a
**7-day period**, in daily units. Historical phases use days -55 through 0;
future phases use days 1 through 28. Bases have 7 rows. Fixed bases are registered
buffers, not optimizer parameters; coefficient networks learn their amplitudes.
Trend is applied before seasonality. Trend and seasonal forecast contributions
sum to the raw interpretable forecast. Both include constant terms, so their
allocation is not uniquely identified. They are model-attributed components,
not proven economic signals or causal explanations.

The original method is described in
[Oreshkin et al., ICLR 2020](https://arxiv.org/abs/1905.10437).
The [ServiceNow reference](https://github.com/ServiceNow/N-BEATS) provides the
residual/basis architecture and shares interpretable weights within each stack.
DemandOS independently implements a smaller CPU experiment. Compared with the
released reference, it keeps chronological input ordering, initializes forecasts
at zero as required here, uses a continuous origin-centred trend grid and explicit
weekly harmonics, and uses a low-rank learned generic basis. Its normalization,
MSE/AdamW training, reduced block sizes and single-seed experiment differ from
the original paper's large ensembles. That paper did not benchmark DemandOS.

## Target-only sequence, scale and mask contract

Inputs are `X [B,56]`, binary history mask `[B,56]`, and scale `[B]`.
Training adds a separate `Y [B,28]`; inference has no labels. Identifiers remain
in canonical dataset and forecast keys, but no store/item embeddings, prices,
calendar, SNAP or event values enter the model. The common Panel stores calendar
metadata for compatibility; the target-only adapter reads only its indexed sales
arrays and known source-support boundary. Changing calendar feature values does
not change target-only tensors. This feature-set difference means comparison
with covariate-enriched LSTM/GRU and LightGBM is not architecture-only isolation.

For origin E, inputs span E-55 through E and targets E+1 through E+28. The
existing daily validator rejects duplicates, unordered dates, noninteger or
negative observed sales, and internal source gaps. Only leading history before
known source support can be padded. Padding has value zero and mask 0; a genuine
observed zero has mask 1. Invalid positions remain masked through every residual
update. No-history, short-history and observed-all-zero states remain distinct.
Source support is not claimed as a verified product introduction date.

Scale is `max(1, mean(abs(observed history)))` at that example's own origin.
Padding is excluded from numerator and denominator. No global statistics or
category vocabularies are learned; no future-series statistics are fitted.
Normalization configuration and fitting metadata are persisted with each model.
All-zero and no-history scales are 1. This input/label scale differs from the
full-training-prefix first-difference denominator used for RMSSE scoring.

Training minimizes unweighted normalized future MSE. Raw signed block forecasts
are added without Softplus or per-block clipping. Predictions are inverse-scaled
and clipped at zero **only for inference/scoring**. Raw forecasts and every block
contribution are retained. Interpretable `trend + seasonality = raw_prediction`;
`raw_prediction + clipping_adjustment = prediction`. Normalized tests use PyTorch
floating-point tolerances; original-unit component reconciliation uses
`atol=1e-4, rtol=1e-6`. Forecast reload/perturbation checks use `atol=rtol=1e-6`.

## Chronological experiment and optimization

Reuse the deterministic 96-series training panel selected using history ending
2014-01-04, with historical origins from 2014-01-05 every 14 days. A full training
vector is eligible only when `origin + 28 <= fit cutoff`, and every label exists.
The earliest input date is 2013-11-11; first labels begin 2014-01-06.

| Fit cutoff | Complete training vectors | Normalized label positions |
|---|---:|---:|
| 2015-12-06 development | 4,704 | 131,712 |
| 2016-01-31 | 5,088 | 142,464 |
| 2016-02-28 | 5,280 | 147,840 |
| 2016-03-27 | 5,472 | 153,216 |

Windows overlap in source dates; positions are not distinct observations.
December validation has 96 vectors / 2,688 labels from 2015-12-07 through
2016-01-03. It selects the best epoch within a predeclared **15-epoch budget,
patience 3**, restoring that checkpoint. Architecture/settings are fixed before
evaluation. Later refits use the frozen selected epochs with each origin's own
eligible vectors; evaluation outcomes do not tune settings.

The optimizer is AdamW, learning rate 0.001, weight decay 0.0001, batch 128,
gradient norm limit 1, dropout 0, seed 42 and two CPU threads. Windows DataLoader
workers are 0; mixed precision is disabled. CPU is the tested default. Explicit
CUDA requires a compatible existing PyTorch build and deterministic device support.
The cached daily projection contains only the 102 train/evaluation series; tensor
caches are capped at 8,192 windows. Process-wide peak memory is not sampled;
actual cached-tensor bytes are reported instead.

The 13 evaluation series use the known January 31 / February 28 / March 27, 2016
origins, with target intervals February 1–28 / February 29–March 27 /
March 28–April 24. Seven methods share exactly 1,092 targets each. MAE and WAPE
use original units, while macro RMSSE averages defined series-origin scores.
Undefined scales/WAPE denominators remain visible. Origin, horizon, horizon-band,
regime/ABC-XYZ, coverage and signed paired-loss tables reuse existing metrics.
Small segment strata cannot support broad conclusions. These are not official
M5 WRMSSE or untouched prospective results.

## Reproduce and inspect outputs

Use the existing Season 12 installation, warehouse and generated results:

```powershell
# If the verified Season 12 artifacts are absent:
.\.venv\Scripts\python -m demandos.neural --root . --config configs/neural.toml
.\.venv\Scripts\python -m demandos.nbeats --root . --config configs/nbeats.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,output,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

Existing Season 12 GRU/LSTM/LightGBM/TSB/seasonal-naive forecasts are reused only
after source hashes, boosting configuration, freshly selected cohorts, origins,
model cutoffs, output keys, real labels and full-prefix RMSSE scales reconcile.
Earlier timing evidence is retained as reference timing, not presented as newly
measured fitting work. Missing/stale references stop with a rerun instruction;
data and checkpoints are never silently downloaded or fabricated.

Ignored output directory: `data/nbeats/global_h28/`.

- `models/{generic,interpretable}/{development,origin_YYYYMMDD}/`: state-dict
  weights and JSON architecture/scaling/fit metadata.
- `tables/`: forecasts, raw block contributions, attributed components and
  clipping adjustments, all comparison metrics, coverage/paired losses,
  eligibility, tensor audits, curves, fit logs and reference timings.
- `figures/`: train/validation losses, per-horizon MAE and three real examples
  selected deterministically from distinct historical regimes. Each example
  shows observed history/future, both forecasts and attributed components.
- `summary.json`, `leakage_verification.json`, `nbeats_report.md`, `manifest.json`:
  real experiment, versions/resources, checks and owned artifact list.

Each origin freezes weights/scaling, then changes or removes all future sales
and mutates future prices/label columns. Regenerated inference tensors must be
exactly identical; raw/clipped/block forecasts must agree within tolerance.
Eligible training tensors must also be identical. Original data hashes and
warehouse totals are checked before/after; output publication occurs only after
all gates pass. No fallback silently replaces a failed neural fit. Prior model
implementations and artifacts remain unchanged.

Sparse spikes, short histories, a limited basis and tiny training cohorts can
limit accuracy. Mask support alone is not validated cold-start accuracy.
Components are descriptive, prices remain unverified as-of, and M5 lacks true
stockout/inventory/lost-sales labels. No inventory savings are inferred.

## Measured real-data results

The run used PyTorch 2.14.1+cpu, NumPy 2.5.4, Pandas 3.0.6 and DuckDB 1.5.6.
It cached 91,392 daily rows across the 102 train/evaluation series. The largest
target tensor cache occupies **3,086,208 bytes** (2.94 MiB); warehouse/query/model
memory is additional. The initial complete bounded experiment took **93.75 s**.

| Variant | Parameters | Unique/applied blocks | Development epochs run/selected | Best validation MSE | Weights bytes |
|---|---:|---:|---:|---:|---:|
| Generic | 41,040 | 3/3 | 5/2 | 1.602318 | 174,711 |
| Interpretable | 25,236 | 2/4 | 5/2 | 1.651043 | 113,755 |

Each origin refit then used two epochs, with no tuning on evaluation targets.
Generic fits took **0.49–0.60 s** and interpretable fits **0.41–0.50 s**;
364-point inference took **1.77–2.16 ms** and **1.47–2.20 ms**, excluding tensor
construction. December development included early stopping and took 3.91/3.19 s.
The measured run recorded no training warnings. Prior neural/tree reference
timings are retained separately, rather than relabelled as new measurements.

| Model | MAE | Pooled WAPE | Macro RMSSE |
|---|---:|---:|---:|
| Interpretable N-BEATS | 12.357747 | 42.8239% | 0.712729 |
| Generic N-BEATS | 12.527406 | 43.4118% | 0.712214 |
| GRU | 12.892712 | 44.6777% | 0.723991 |
| Pooled LightGBM | 12.897550 | 44.6945% | 0.724525 |
| TSB | 13.015273 | 45.1024% | 0.729239 |
| LSTM | 13.115248 | 45.4489% | 0.732582 |
| Seasonal naive | 13.131868 | 45.5065% | 0.788598 |

All **7,644 forecasts** succeeded, exactly 1,092 per method / 39 series-origin
cases, with no failed/skipped forecasts. RMSSE is defined for 38 cases and
undefined for one training scale; unscaled metrics keep that case.

Interpretable has the best pooled MAE/WAPE; generic has the best macro RMSSE.
Their January/February/March WAPE is **42.534/37.885/48.340%** and
**46.469/37.349/46.673%** respectively. LightGBM wins February (34.339%) and
generic beats it in March (46.673% versus 47.568%). There is no uniform origin
winner. First-week LightGBM WAPE (37.025%) beats generic/interpretable
(41.273/40.662%); days 15–28 favor generic/interpretable (44.195/43.590%) over
LightGBM (49.335%).

For the two smooth series, generic/interpretable WAPE is **21.601/20.681%**,
versus LightGBM's 26.856%. On the eight lumpy series it is **86.477/85.882%**,
worse than LightGBM's 79.477%. These tiny training-defined strata are descriptive,
not stable population conclusions. The original paper makes no claim about this
experiment, and these scores do not establish universal N-BEATS superiority.

All eight checkpoints reload, and both variants pass future-sales mutation and
deletion, price/label exclusion and exact input/training-tensor invariance at all
three origins. Forecast perturbation difference is **0**. Maximum block-sum
reconciliation error in original units is below **0.000011**, within the documented
floating-point tolerance. Trend/seasonality sum to raw forecasts; clipped totals
reconcile using their separately exported clipping adjustment. All raw/interim/
processed source hashes and the **58,327,370-row** warehouse totals are unchanged.
All **150 Python tests** (including 16 N-BEATS tests) passed without skips;
all **36 fresh real-data dbt tests** passed without warnings or failures.
