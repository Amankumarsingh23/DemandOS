# Season 14: compact global point Temporal Fusion Transformer

`demandos.tft` independently implements the TFT mechanisms in PyTorch, reusing
the Season 12 sequence/preprocessing contract and existing cohort/metric helpers.
It forecasts recorded observed sales. No verified stockout, lost-demand,
inventory-savings or price-elasticity conclusion follows from this experiment.

## Research and implementation boundary

The architecture follows the GRN, variable-selection, static-context,
encoder-decoder and interpretable-attention formulas in
[Lim et al., Temporal Fusion Transformers, IJF 2021](https://doi.org/10.1016/j.ijforecast.2021.03.012).
The [public paper](https://arxiv.org/abs/1912.09363) and
[Google Research implementation](https://github.com/google-research/google-research/tree/master/tft)
were inspected. This implementation preserves all seven core mechanisms; it is
neither a bare TransformerEncoder nor an LSTM with a single attention add-on.
The original research did not benchmark this DemandOS experiment.

Explicit differences and resource choices:

- One linear point head and normalized MSE replace the original quantile heads
  and quantile loss. There is no prediction-interval pipeline.
- Hidden size 32, four heads, one encoder/decoder LSTM layer, dropout 0.1,
  four-dimensional static embeddings projected to model width, and a fixed
  eight-epoch budget keep CPU work bounded. No hyperparameter sweep is used.
- Attention computes only the 28 future-query rows against all 84 key positions.
  Since later processing is pointwise, omitted historical-query outputs are not
  needed to generate future forecasts.
- Leading padding is skipped with packed recurrent sequences and excluded as
  attention keys. Zero-history examples retain static initial hidden/cell states.
- Attention probability dropout is applied during training; exported weights
  are the normalized probabilities before dropout. This regularization placement
  differs from the reference implementation's head/output dropout. At inference,
  dropout is disabled and the shared-value averaged-head formulation applies.
- AdamW, seed 42, an earlier December development split and the fixed selected
  cohort replace the original paper's broader experimental setup.

## Tensor and feature availability contract

| Input/output | Shape | Content |
|---|---|---|
| Past | `[B,56,10]` | Origin-scaled/standardized sales, observed mask, unavailable price placeholder, price availability mask, weekday/month sine/cosine, event flag, state SNAP |
| Future known | `[B,28,7]` | Weekday/month sine/cosine, scheduled event flag, state SNAP, horizon/28 |
| Static IDs | `[B,5]` | Store, state, category, department, item; separate `[B]` embedding inputs |
| Labels/predictions | `[B,28]` | Complete future observed-sales vector / direct point forecast |

The example grain is `(store_id,item_id,origin_date)`. History spans E-55 through
E, targets E+1 through E+28, including February 29, 2016. Daily grids are ordered,
unique and complete. Missing-source gaps are hard errors. Static metadata is
treated as the existing immutable entity contract, not a source of future sales.
Vocabularies contain training-eligible entities only; unknown IDs map to index 0.

Past selling prices have no verified as-of timestamps in M5 and remain explicitly
unavailable: numeric placeholder 0 plus availability mask 0. Future realized
prices, sales, labels and inventory never enter the decoder, VSNs or attention.
Scheduled events/SNAP are assumed known ahead under the Season 8 calendar
contract; the retrospective dataset alone does not prove that availability.
No arbitrary numeric NaN is silently replaced: missing-source/nonfinite inputs
fail, while unavailable prices and leading padding have explicit masks.

## GRN, GLU and variable selection

The gating primitive is `linear_value(x) * sigmoid(linear_gate(x))`, with dropout
before the projections. A GRN applies input/context projections, ELU, a second
hidden projection, GLU, a residual skip (projected when dimensions differ), and
LayerNorm. Static context is optional only for GRNs declared without context;
temporal selection/enrichment require their static contexts.

Static embeddings feed a static VSN. Separate past and future VSNs first project
each scalar into model width, transform each variable with its own GRN, and
learn instance/time-specific softmax selection scores from the flattened variable
representations plus static context. The softmax axis is **variables**, not time.
The transformed variables are combined as their weighted sum. Past/future/static
VSNs have distinct parameters.

The selected static representation feeds four separate GRNs producing contexts
for temporal VSN selection, initial LSTM hidden state, initial cell state, and
static temporal enrichment. Static information thus conditions multiple parts
of the model instead of merely being appended to an output vector.

## Local recurrence, temporal fusion and Q/K/V attention

The encoder processes 56 selected historical steps. The decoder consumes only
the 28 selected future-known rows, initialized from the encoder's hidden/cell
states. There is no teacher forcing or feedback of future actual sales. A gated
skip plus LayerNorm joins recurrent outputs with selected inputs. A contextual
GRN enriches those 84 temporal representations with static information.

Queries represent forecast positions; keys represent available temporal
positions; values carry the content to aggregate. Each head learns its own Q/K
projection with depth `32/4 = 8`. Scores are `Q_h K_h^T / sqrt(8)`, then masked
and softmaxed over **key positions**. The mask permits valid history and
future-known positions up through the query's own date; it excludes padded
history and all later keys. Its inclusive diagonal matches the inspected
reference mask: a forecast step may use its own known calendar context.

Unlike conventional attention with independent values and concatenated heads,
TFT uses **one shared 32-to-8 value projection**. Head outputs are averaged,
then projected from 8 back to 32. Fused Q/K weight matrices have separate learned
row slices for each head; the value matrix is shared. Only allowed keys receive
nonzero probability. A zero-history first query still has its own future-known
key, avoiding an all-masked softmax.

After attention, a gated residual/LayerNorm joins the query's enriched input.
A positionwise GRN follows, then another gated skip to the local recurrent
representation, LayerNorm and the linear point head. The LSTM supplies temporal
ordering, so no additional positional encoding is introduced.

## Scaling, padding and chronological training

The reused preprocessor fits global sales-channel mean/std only on eligible
training-window histories. Each sample also uses its own origin's observed
historical mean with floor 1 as a unit scale. Labels are divided by that scale;
predictions are multiplied by it before external scoring. This is distinct from
the train-prefix first-difference RMSSE denominator. No future target fits a
normalizer or vocabulary. Versioned JSON saves preprocessing alongside weights.

Leading padding has value zero and observed mask 0; observed sales zero has mask
1. Before variable projection, padded temporal inputs are cleared. After VSN,
padded selected inputs are cleared again. Packed sequences skip padded recurrent
steps, then restore calendar positions; attention masks padded keys. No-history,
short-history, all-zero and missing-source states remain distinct. Padding support
does not establish cold-start forecast accuracy.

The raw linear head trains against unweighted normalized MSE. Original-unit raw
forecasts are retained; scoring uses `max(0,raw)` after inverse scaling. Clipping
is not a quantile estimate or a fitted lost-demand correction.

The 96 training series are selected deterministically using history ending
2014-01-04. Historical origins start 2014-01-05 at 14-day spacing. Complete
training eligibility requires `sample_origin + 28 <= fit cutoff` with all labels
present. Windows overlap in dates; label positions are not distinct source rows.

| Fit cutoff | Eligible vectors | Label positions |
|---|---:|---:|
| 2015-12-06 development | 4,704 | 131,712 |
| 2016-01-31 | 5,088 | 142,464 |
| 2016-02-28 | 5,280 | 147,840 |
| 2016-03-27 | 5,472 | 153,216 |

December validation has 96 vectors / 2,688 targets from 2015-12-07 through
2016-01-03. Best validation MSE selects epochs within the eight-epoch budget,
patience 2. That checkpoint is restored. Later refits use the frozen count and
preprocessing fitted only on each origin's eligible training samples. Final
evaluation targets do not choose settings, early stopping or an ablation winner.

AdamW uses learning rate 0.001, weight decay 0.0001, batch 128 and gradient-norm
limit 1. CPU is the default, two threads, DataLoader workers 0, with no mixed
precision. Explicit CUDA requires a compatible installed PyTorch build and
deterministic operation support; CPU is the verified device. Tensor cache caps
remain 8,192 windows. The daily projection covers only 102 distinct train/eval
series. Actual tensor bytes are reported; process-wide peak RAM is not sampled.
Nonfinite loss, gradient, parameter or output and missing parameter gradients
abort the run. Same-environment seeded runs are tested; cross-version/device
bitwise reproducibility is not promised.

## Interpretation outputs and limits

Exported arrays contain per-head attention `[B,4,28,84]`, static VSN weights
`[B,5]`, past weights `[B,56,10]`, future weights `[B,28,7]` and the allowed-key
mask `[B,28,84]`. Case keys and date-aligned interpretation tables preserve their
association with forecasts. Only evaluation cases are exported, not every
training window. Historical VSN aggregation excludes padded positions. Historical
attention-lag summaries retain negative relative-day indices -55 through 0;
future-known keys use 1 through 28. Historical/future-known attention masses sum
to one jointly, not separately.

VSN and attention weights are **internal model diagnostics**, not causal effects,
identified consumer behavior, true stockouts or price elasticities. A constant
unavailable-price input can still receive VSN weight through learned projection
biases; such weight is not evidence that actual prices were observed or useful.
Mean weights can conceal case variation, so case tables and representative
plots are preserved. Small smooth/lumpy strata have only 2/8 series. Paired-case
loss standard deviations/ranges are descriptive variation, not confidence
intervals for independent observations or proof of statistical superiority.

## Reproduce and inspect artifacts

Use the existing neural environment and verified warehouse/Seasons 12–13 outputs:

```powershell
# Only if prerequisite references are absent or stale:
.\.venv\Scripts\python -m demandos.neural --root . --config configs/neural.toml
.\.venv\Scripts\python -m demandos.nbeats --root . --config configs/nbeats.toml
.\.venv\Scripts\python -m demandos.tft --root . --config configs/tft.toml
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -c "from pathlib import Path; from demandos.warehouse.cli import settings,invoke_dbt; root,source,output,env=settings(Path.cwd()); invoke_dbt(root,env,'test')"
```

No new dependencies are needed beyond the existing PyTorch/analytics extras.
Prior predictions are reused only after source hashes, configs, freshly selected
cohorts, fit cutoffs, exact keys, actuals, RMSSE scales and recomputed metrics
reconcile. Missing/stale references stop explicitly; nothing is downloaded or
fabricated. Prior fitting timings are historical reference evidence, not newly
measured work. TFT has extra static/known-future inputs versus target-only N-BEATS,
so these are whole-system comparisons, not an architecture-only ablation.

Outputs remain ignored under `data/tft/global_h28/`:

- `models/development` and `models/origin_YYYYMMDD`: state-dict weights, metadata,
  schema version and preprocessing JSON.
- `tables/`: forecasts, metrics, coverage, paired losses, descriptive segment
  variation, eligibility, curves, timings/fit status, unknown IDs, tensor audits,
  VSN summaries/cases and date-aligned attention summaries/cases.
- `interpretation_arrays.npz`: bounded full per-head evaluation diagnostics and IDs.
- `figures/`: losses, VSN summaries, attention-by-horizon, representative forecast/
  attention cases and horizon MAE.
- `summary.json`, `leakage_verification.json`, `tft_report.md`, `manifest.json`:
  measured experiment, resources, warnings, checks and owned artifact list.

At all three origins, frozen weights/preprocessing are tested after both mutation
and deletion of all future sales and mutation of future prices/label columns.
Inference and eligible training tensors must be exactly equal; raw/scored
forecasts, VSNs and attention must match with `atol=rtol=1e-6`. Independent tests
also change later future-known rows and check earlier horizons remain unchanged.
Attention/VSN probability axes must sum to one; forbidden attention weights must
be exactly zero. Source fingerprints and warehouse totals gate atomic artifact
publication. Failed fits are not silently replaced with fallback predictions.

This is the repeatedly examined 13-series, three-origin retrospective benchmark,
not an untouched prospective test or official M5 WRMSSE. It does not establish
universal model superiority, recovered latent demand or real inventory savings.

## Verified real M5 results

The run used PyTorch 2.14.1+cpu, NumPy 2.5.4, Pandas 3.0.6 and DuckDB 1.5.6.
All expected counts were verified: 96 training series, 4,704 development and
96 validation vectors, then 5,088/5,280/5,472 complete vectors at the three refits.
The selected epoch was **5**, after 7 development epochs, with best normalized
validation MSE **1.550650**. The model has **184,335 parameters**, and each
weights file is **858,847 bytes**. All four checkpoints reloaded successfully.

| Model | MAE | Pooled WAPE | Macro RMSSE |
|---|---:|---:|---:|
| Interpretable N-BEATS | 12.357747 | 42.8239% | 0.712729 |
| Generic N-BEATS | 12.527406 | 43.4118% | 0.712214 |
| GRU | 12.892712 | 44.6777% | 0.723991 |
| Pooled LightGBM | 12.897550 | 44.6945% | 0.724525 |
| TSB | 13.015273 | 45.1024% | 0.729239 |
| LSTM | 13.115248 | 45.4489% | 0.732582 |
| TFT | 13.297411 | 46.0801% | 0.742327 |
| Seasonal naive | 13.131868 | 45.5065% | 0.788598 |

There are **8,736 successful forecasts**, exactly 1,092 per method, with no
failed/skipped forecast rows. Macro RMSSE uses 38 defined series-origin scales;
one scale is undefined and remains included in unscaled metrics.

TFT has the highest pooled MAE/WAPE of these eight systems. Its macro RMSSE
beats seasonal naive but not the other systems. This is a completed experiment
with weaker accuracy, not a reason to tune on the benchmark or mislabel a
correctness failure. The small predefined cohort/budget and richer architecture
do not establish TFT's general performance.

TFT January/February/March WAPE is **42.665/41.015/54.942%**, versus
interpretable N-BEATS **42.534/37.885/48.340%**. TFT's horizon-band WAPE is
**40.352/49.141/47.045%** for days 1–7/8–14/15–28; interpretable N-BEATS is
**40.662/43.196/43.590%**. The small first-week advantage does not reverse the
pooled result. LightGBM's first-week WAPE is still lower (37.025%).

For two smooth series, TFT WAPE is **24.188%** versus N-BEATS's **20.681%**;
for eight lumpy series, **86.914%** versus **85.882%**. The paired-case mean
absolute-loss difference versus interpretable N-BEATS is +3.431 units in the
six smooth series-origin cases, with case range -4.980 to +8.004; for 24 lumpy
cases it is +0.134, range -1.439 to +2.488. These ranges describe variation and
small support, not independent confidence intervals or significance claims.

Mean static VSN weight is largest for category (**0.437**); historical selection
is largest for scaled sales (**0.607**); future-known selection is largest for
horizon fraction (**0.375**) and weekday sine (**0.253**). Six evaluation item IDs
are unknown at each origin; store/state/category/department IDs are known. These
weights do not prove category causality or the uselessness of a low-weight input.
Price availability remains 0 regardless of its nonzero internal VSN weight.

Head-averaged attention concentrates mostly on historical representations:
mean historical mass declines from **0.994** at horizon 1 to **0.900** at horizon
28. The largest mean historical key weights are at relative days -1 and -7.
All static/past/future weights normalize over variables; attention normalizes
over allowed keys. Forbidden/padded-key probability is exactly zero. Heatmaps
and case tables are internal diagnostics, not evidence of causal lag effects.

Development fitting took **173.25 s**; the three five-epoch refits took
**132.98 / 138.81 / 144.64 s**. Diagnostic-enabled 364-point inference took
**38.3–45.2 ms**, excluding sequence generation. The initial complete experiment
took **814.48 s** (13.6 minutes), including cohort/preprocessing, reference checks,
training, invariance tests and metrics; references were not retrained. It used
two CPU threads, zero DataLoader workers and no mixed precision. The largest
input/label tensor cache was **17,400,960 bytes** (16.6 MiB), excluding models,
activations, Python dataframes and warehouse memory. No end-to-end process peak
RAM measurement is claimed. No training warnings were recorded.

All three real perturbation checks preserve inference and eligible-training
tensors exactly. Raw/clipped forecasts, VSNs and attention match after reload,
future-sales mutation/removal and future-price/label changes; maximum forecast
difference is **0**, within `atol=rtol=1e-6`. Source hashes and the
**58,327,370-row** warehouse totals remain unchanged. The independent saved-model
verification recomputes preprocessing and rechecks the persisted forecasts and
metrics. Generated artifacts remain local and ignored.
All **167 Python tests**, including 17 TFT tests, passed without skips. All
**36 fresh real-data dbt tests** passed without warnings or failures. Independent
saved-model verification recomputed the three origin preprocessors, reproduced
saved forecasts/interpretations, reconciled metrics and confirmed all **50**
manifest-listed artifacts are present.
