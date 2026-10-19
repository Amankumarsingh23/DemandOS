# Season 12: global LSTM and GRU observed-sales forecasts

The optional `demandos.neural` package trains genuine PyTorch recurrent models
on the existing training-selected 96-series panel. Earlier ingestion, warehouse,
features, metrics and models remain unchanged. The comparison uses the known
13-series retrospective benchmark, not an untouched prospective test or a
full-panel M5 result. Recorded sales do not establish unconstrained demand,
stockouts or inventory savings.

## Model and tensor contract

A feedforward network maps a fixed input to outputs. An RNN also carries a hidden
state through time, sharing parameters across daily steps. Backpropagation through
time differentiates the forecast loss through those steps. Long chains can make
gradients vanish or explode. LSTM gates maintain a separate cell state; GRU gates
use a simpler hidden-state update with fewer parameters. Gating helps information
flow but does not guarantee accuracy. Gradient clipping limits large updates;
finite-loss, finite-gradient and finite-parameter checks fail explicitly.

Both models use one batch-first encoder layer with hidden size 32 and five static
embeddings of dimension 4: store, state, category, department and item. Unknown IDs
map to reserved index 0. Vocabularies contain only entities in eligible fit windows.
The final hidden output, concatenated embeddings and each target day's known
calendar row feed a shared 32-unit ReLU head, dropout 0.1 and a Softplus output.
The encoder type is the controlled architectural difference. A single recurrent
layer has recurrent dropout **0**, while head dropout is active during training.
The static tensor is `[B,5]`, with each column supplying one `[B]` embedding input.

| Tensor | Default shape | Contents |
|---|---|---|
| Past | `[B,56,10]` | Scaled sales, observed mask, unavailable price, price availability mask, six calendar values |
| Future known | `[B,28,7]` | Weekday/month sine and cosine, scheduled event flag, state SNAP, horizon/28 |
| Static | `[B,5]` | Training-vocabulary categorical IDs |
| Labels / predictions | `[B,28]` | Complete observed target vector / direct forecast |

For origin E, history is exactly E-55 through E and targets are E+1 through E+28.
Daily keys must be unique, ordered, midnight and contiguous. A missing internal
or trailing source record is a hard error. Before the first actual source record,
left padding carries zero internally plus an observed mask of 0. An observed zero
carries mask 1. Both channels are consumed by the recurrent encoder. No-history,
short-history and all-zero statuses and observed counts remain visible. The first
record means source support, not verified product introduction; first-positive
sale is never substituted for introduction.

Calendar schedules are assumed known ahead under the Season 8 contract. Historical
SNAP uses the series' state. The M5 price table has no verified as-of timestamps,
so historical price inputs are unavailable (numeric 0, availability mask 0).
Realized future prices and target actuals never enter inference tensors. The
predictor builder and label reader are separate APIs; inference datasets contain
four tensors, labelled datasets contain a fifth target tensor.

## Scaling and chronological eligibility

Each window uses its own observed historical mean with floor 1 as its unit scale.
It never uses its labels or a full-history series mean. On eligible training
windows only, the preprocessor fits a global mean/std of history divided by those
window scales; std has floor 1e-6. Only the sales channel is standardized, and
padded positions remain zero with mask 0. Bounded calendar/mask channels retain
their prescribed scale. State includes statistics, vocabularies, fit cutoff,
window count and last historical source date.

Targets are divided by their own origin's scale. Training minimizes normalized
MSE using AdamW (learning rate 0.001, weight decay 0.0001), batch 128, gradient
clip norm 1.0 and deterministic seed 42. Softplus enforces nonnegative normalized
outputs, which are multiplied by the historical scale **before** metrics. It can
produce a small positive forecast for an observed zero; it is not a count
distribution or hurdle model. Input scales are separate from the reused
train-prefix first-difference RMSSE denominators.

Training eligibility requires **origin + 28 <= fit cutoff**, with all targets
present. Incomplete vectors are excluded by schedule, never partially trained.
The panel was chosen using history ending 2014-01-04, independent of evaluation.
Historical origins start 2014-01-05, stride 14 days, with the existing 64-origin cap.
Unknown embeddings cover unseen identifiers.

Development fits use labels through 2015-12-06 and validate the origin of
2015-12-06 on targets 2015-12-07 through 2016-01-03. Best validation MSE selects
each encoder's epoch, with patience 3 within the fixed 12-epoch budget. That
checkpoint is saved. Later refits use that frozen epoch count, with preprocessing
refitted only on that origin's eligible training vectors. Their own evaluation
targets do not select settings or early stopping.

| Fit cutoff | Forecast target interval |
|---|---|
| 2016-01-31 | 2016-02-01 through 2016-02-28 |
| 2016-02-28 | 2016-02-29 through 2016-03-27 |
| 2016-03-27 | 2016-03-28 through 2016-04-24 |

LightGBM is refitted using its existing pooled-row eligibility and December
iteration-selection policy. It can use partially observed historical horizons;
neural models require whole vectors. This is a documented difference between
the established baseline and neural training contract. All evaluation keys,
actuals and RMSSE scales are identical. TSB and seasonal naive reuse existing
implementations. Comparisons include pooled MAE/WAPE, macro case RMSSE, explicit
undefined scales, per-origin/horizon/band/segment tables, coverage and paired
loss differences against seasonal naive. Segment labels are descriptive
benchmark strata, never neural predictors; tiny strata do not support broad claims.

## Reproduce

Use the verified analytics/EDA/boosting installation and warehouse. The only new
model dependency is PyTorch. The CPU wheel tested here is 2.14.1+cpu:

```powershell
.\.venv\Scripts\python -m pip install torch -c configs/neural-constraints.txt --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -e ".[analytics,eda,boosting,neural]" -c configs/analytics-constraints.txt -c configs/boosting-constraints.txt -c configs/neural-constraints.txt
.\.venv\Scripts\python -m demandos.neural --root . --config configs/neural.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,output,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

CPU is the default and verified device; no GPU is required. Explicit
`device = "cuda"` requires a compatible CUDA PyTorch installation and deterministic
CUDA support. Mixed precision is disabled. DataLoader workers are 0 on Windows,
tensor batches and CPU threads are bounded, and no unbounded prefetch is used.
A small indexed daily projection is cached; generated windows are bounded at
8,192, instead of expanding 58 million fact rows into tensors. Tensor-cache bytes
are reported. Total process peak memory is not sampled or claimed. Reproducibility
across library/device versions is not promised; CPU seeded runs are tested.

Ignored outputs live under `data/neural/global_h28/`:

- `models/{lstm,gru}/{development,origin_YYYYMMDD}/`: state-dict `weights.pt`,
  metadata and preprocessing JSON; separate native LightGBM reference models.
- `tables/`: predictions, comparison membership, metrics, paired losses,
  coverage, train curves, fit/timing logs, tensor audits, leakage checks,
  history-mask diagnostics and selected cohorts.
- `figures/`: December loss curves and identifier-selected example forecasts.
- `summary.json`, `neural_report.md`, `manifest.json`: measured experiment,
  source hashes, warehouse totals, versions, warnings and owned artifact list.

Artifacts are first built in an ignored temporary directory and published only
after all hard gates pass. A failed fit or leakage check aborts publication;
statistical fallbacks are not silently substituted. Existing comparison helpers
exclude failed/skipped cases from shared rankings, and coverage remains explicit.

At each evaluation origin, frozen checkpoints and preprocessing are tested after
both mutation and complete removal of future sales, mutation of future prices,
and insertion of forbidden future-label columns. Inference tensors must be
exactly equal; forecast/save-load comparisons use atol=rtol=1e-6 in original
units. Eligible training tensors also remain exactly equal after perturbation.
Five 0/3/7/28/56-day history diagnostics verify masks and finite preprocessing;
these **do not validate cold-start forecast accuracy**. Raw/interim/processed
fingerprints and warehouse totals are checked before/after the real experiment.

The implementation follows official PyTorch contracts for
[batch-first LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html),
[GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) and
[state-dict serialization with weights-only loading](https://docs.pytorch.org/docs/stable/notes/serialization).

## Verified real M5 experiment

Executed with Python 3.13.15, PyTorch 2.14.1+cpu, NumPy 2.5.4, Pandas 3.0.6,
DuckDB 1.5.6 and LightGBM 4.7.0. The bounded SQL projection contains 91,392 daily
rows across the 102 distinct train/evaluation series, rather than the full fact
table. Training tensors at the largest refit occupy **17,400,960 bytes** (16.6 MiB).
No process-wide peak memory measurement is claimed.

| Fit cutoff | Eligible windows | Label values | Excluded incomplete vectors |
|---|---:|---:|---:|
| 2015-12-06 development | 4,704 | 131,712 | 864 |
| 2016-01-31 | 5,088 | 142,464 | 480 |
| 2016-02-28 | 5,280 | 147,840 | 288 |
| 2016-03-27 | 5,472 | 153,216 | 96 |

These come from 58 scheduled historical origins × 96 series, filtering whole
vectors at each cutoff. They overlap in dates; label counts are training positions,
not distinct source observations. December validation has 96 vectors / 2,688
labels. The earliest historical input date is 2013-11-11; the first training
target is 2014-01-06. Latest eligible historical origins are 2015-11-08 for
development and 2016-01-03 / 01-31 / 02-28 for the evaluation refits.

LSTM has **8,001 parameters** and a 36,284-byte weights file; GRU has **6,593**
and a 30,652-byte weights file. December selected **12 LSTM epochs** (best MSE
1.546367) and **3 GRU epochs** (best MSE 1.558581). GRU development ran 6 epochs
before patience was exhausted, restoring epoch 3. All six evaluation refits use
the selected counts without evaluation-target early stopping. No training
warnings or nonfinite losses/gradients were recorded. All checkpoints reloaded.

| Model | MAE | Pooled WAPE | Macro RMSSE |
|---|---:|---:|---:|
| GRU | 12.892712 | 44.6777% | 0.723991 |
| Pooled LightGBM | 12.897550 | 44.6945% | 0.724525 |
| TSB | 13.015273 | 45.1024% | 0.729239 |
| LSTM | 13.115248 | 45.4489% | 0.732582 |
| Seasonal naive | 13.131868 | 45.5065% | 0.788598 |

All methods have 1,092 successful forecasts / 39 series-origin cases, with no
failed or skipped predictions in this continuity experiment. RMSSE uses 38
defined cases; one has an undefined training scale. Coverage and available-case
scores retain that case for unscaled metrics. These are not official WRMSSE.

GRU's pooled WAPE advantage over LightGBM is just **0.0168 percentage points**.
Its January/February/March WAPE is 44.448% / 36.060% / 54.018%, versus LightGBM's
52.539% / 34.339% / 47.568%. GRU wins the pooled comparison by a tiny margin but
loses at two origins. Its 1–7 / 8–14 / 15–28 day WAPE is 39.736% / 45.561% /
46.414%, versus LightGBM's 37.025% / 42.276% / 49.335%. Neither encoder wins
uniformly across horizons or strata. The two smooth series favor both encoders
(LSTM 20.773%, GRU 23.770%, LightGBM 26.856% WAPE), while the eight lumpy series
favor LightGBM (79.477% versus GRU 81.760% and LSTM 84.219%). Those small
training-defined benchmark strata do not establish population-level effects.

Evaluation fit times span **17.06–18.65 s for LSTM**, **6.96–7.86 s for GRU**;
13-window / 364-point inference spans **3.2–5.0 ms** and **6.1–6.4 ms**,
respectively, excluding sequence construction. LightGBM fits take 3.25–3.62 s,
with prediction 0.118–0.129 s. These local timings are not a general throughput
benchmark. The full initial bounded experiment took **373.4 s**, including
warehouse/feature preparation, verification, baselines and scoring.

Future-sales modification and deletion, price/label perturbation, exact tensor
invariance and checkpoint equivalence passed at all three origins for both
encoders; maximum forecast difference was 0 within the specified tolerance.
Source hashes and the 58,327,370-row warehouse totals remained unchanged.
The full suite contains **134 Python tests**, including 16 new neural tests;
the real warehouse has **36 dbt tests**. Final execution evidence is retained
locally alongside the ignored artifacts.

## Season 13 extension

[N-BEATS](nbeats.md) reuses the full-vector schedule, cohort, seed/loader and metric
contracts with a separate target-only adapter. It uses no recurrent state or
future covariates: dense backcast/forecast blocks subtract masked residuals and
accumulate forecasts. Original LSTM/GRU implementations and artifacts remain
unchanged. Target-only versus covariate-enriched comparisons do not isolate
architecture alone.
