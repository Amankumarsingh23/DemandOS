"""Readable, bounded interpretation of actual diagnostics, not forecasting claims."""

import pandas as pd
from demandos.eda.report import markdown_table


def write_report(summary, path):
    rows = pd.DataFrame(summary["diagnostics"])
    original = rows[rows.variant.eq("original")]
    cohort = pd.DataFrame(summary["cohort"])
    display = cohort.merge(original[["series_id", "correlation_status", "acf_1", "acf_7", "pacf_7", "interpretation"]], on="series_id", validate="one_to_one")
    split = summary["split"]
    lines = ["# DemandOS Season 5 — diagnostics and temporal validity", "",
             "These are actual M5 observations from the existing Season 3 warehouse. No forecasting models are trained.",
             f"Input SHA-256: `{summary['source']['sha256']}`. Full source: {summary['source_rows']:,} observations, {summary['source_units']:,} recorded units.", "",
             "## Information boundaries", "",
             f"- Training: **{split['train_start'].date()} through {split['train_end'].date()}**.",
             f"- Validation targets: **{split['validation_start'].date()} through {split['validation_end'].date()}**.",
             f"- Test targets: **{split['test_start'].date()} through {split['test_end'].date()}**.",
             "- All cohort selection labels, diagnostics, rolling plots, ADF/KPSS inputs, and fitted scale parameters use training observations only. ABC/XYZ and regime labels are recomputed with Season 4 definitions on this prefix, not borrowed from full-history labels.",
             "- The cohort intentionally illustrates regimes, product ABC/XYZ diversity, a sparse case, and high-volume aggregates. It is not a random sample or a population stationarity estimate.", "",
             f"## Actual cohort: {len(cohort)} series", "",
             markdown_table(display, ["series_id", "regime", "abc_xyz", "acf_1", "acf_7", "pacf_7", "interpretation"]), "",
             "## Measured diagnostic outcomes", "",
             "Original-series paired ADF/KPSS interpretations:", "",
             markdown_table(original.interpretation.value_counts().rename_axis("interpretation").reset_index(name="series"), ["interpretation", "series"]), "",
             f"Across {len(rows)} original/transformed combinations: {int(rows.correlation_status.eq('ok').sum())} valid ACF/PACF results; {int(rows.correlation_status.eq('skipped').sum())} skipped; {int(rows.correlation_status.eq('failed').sum())} failed. Exact reasons and test warnings are in the tables and JSON.",
             "A large weekly ACF relative to neighboring lags is compatible with weekly dependence, but may reflect trend or other nonstationarity; inspect original and differenced estimates together. It does not establish causation or select an ARIMA order.", "",
             "## Estimators and hypotheses", "",
             "ACF: statsmodels, adjusted=False, FFT, Bartlett confidence intervals. This uses centered cross-products with a common full-sample denominator. PACF: Yule-Walker maximum-likelihood convention (`ywm`). Maximum lag is capped below half the transformed sample length. Highlighted lags are 1, 7, 14, 28 where supported.",
             f"Intervals use alpha={summary['config']['alpha']}; they are approximate exploratory per-lag intervals, not simultaneous significance guarantees. Coefficient tables retain library intervals centered at the estimates; plots show the corresponding zero-centered uncertainty bands.",
             f"ADF uses regression={summary['config']['regression']}, AIC autolag, maxlag capped at {summary['config']['adf_maxlag']}. Its null is a unit root under that deterministic specification. KPSS uses the same deterministic specification and automatic bandwidth; its null is level stationarity for `c`, trend stationarity for `ct`. Bounded-tabulation p-value warnings are retained.",
             "Rejecting ADF only is compatible with stationarity under these assumptions; rejecting KPSS only is compatible with nonstationarity. Both rejected means conflicting evidence; neither rejected is inconclusive. No single p-value proves mathematical stationarity. There is no multiple-testing correction or automatic model selection.", "",
             "## Differences, rolling variability and STL", "",
             "First differences lose one initial observation; seasonal lag-7 differences lose seven; optional second differences lose two. Dates and missing prefixes remain aligned in transformation tables. Negative changes are valid. Diagnostics are repeated after transformation, without assuming differencing improves stationarity; over-differencing may add noise/dependence.",
             "Rolling plot means/variances are trailing and include the current observation; they are descriptive, unlike the separate prior-row lag demonstration. Variance is population variance. Constant, all-zero, missing, short, and extremely sparse samples are explicitly skipped before classical diagnostics.",
             "Additive robust STL uses period 7 on eligible regular histories; components reconstruct observed units. It is fitted only through train_end, but its centered smoothing still uses later training observations for earlier components. All outputs are tagged descriptive_only and the feature contract forbids their use as historical prediction-time inputs. Nothing is deleted or imputed.", "",
             markdown_table(pd.DataFrame(summary["stl"]), ["series_id", "status", "reason"]), "",
             "## Origin-time demonstration", "",
             f"Verified direct horizons: **{summary['horizons_verified']}**. Each uses the same origin, {split['train_end'].date()}.",
             "`sales_at_origin` is y_t; `origin_minus_k` is y_(t-k), independent of horizon. Trailing origin statistics end at t and fitted normalization parameters stop no later than t. Historical `target_lag_1(u)` is a separate target-date-aligned calculation, not automatically valid for h>1.",
             f"For h=7 the naive target lag-1 references **{summary['leakage_demonstration']['naive_target_lag_1_source'].date()}**, after origin, and was **rejected**. Corrected examples reference the origin or earlier dates.",
             "Origin feature values and fitted scale parameters were independently recomputed after removing every post-origin actual; the results are identical. Only deterministic target weekday is known ahead. Recorded prices are not origin features because their retrospective presence does not establish as-of availability.",
             "The provenance validator rejects future actual sales, after-origin fitted transformations, centered/descriptive components, unavailable covariates, and unverified prices. It checks declared provenance; it cannot prove unknown real-world data availability. Expanding and 365-day rolling schedules illustrate nonoverlapping 28-day target blocks, without fitting forecasts or implementing a backtest engine. Future experiments must refit transformations/segments at each origin.", "",
             "## Artifacts, performance and limitations", "",
             f"- Runtime before final Markdown/JSON publication: **{summary['runtime_seconds']:.2f}s**; cohort-sized arrays only, with DuckDB reductions for selection. Source CSV/Parquet and warehouse transformations remain unchanged.",
             "- tables/diagnostics.parquet contains numerical before/after results and warnings; correlations.parquet contains lag coefficients and intervals. Additional tables store cohort membership, training labels, aligned transformations, STL components/status, historical lags, fit parameters, origin features and example schedules.",
             "- summary.json records config, split, source fingerprint, package versions, actual results, skips and limitations. No trained models, predicted targets, benchmark claims or verified inventory/demand estimates are produced.", "",
             "## Figures", ""]
    for name in summary["figures"]:
        lines.extend([f"![{name}](figures/{name})", ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
