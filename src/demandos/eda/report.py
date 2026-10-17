"""Evidence-backed generated Markdown, with facts separated from hypotheses."""

import math


def markdown_table(frame, columns):
    def cell(value):
        if isinstance(value, float):
            return f"{value:.4f}" if math.isfinite(value) else "undefined"
        return str(value).replace("|", "/")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(row[column]) for column in columns) + " |" for _, row in frame.iterrows())
    return "\n".join(lines)


def write_report(frames, report, path):
    baseline = report["baseline"]
    store = frames["store"].sort_values(["units_sold", "store_id"], ascending=[False, True]).iloc[0]
    weekday = frames["weekday"].sort_values("weekday_index", ascending=False).iloc[0]
    eligible = frames["product_segments"][frames["product_segments"].revenue_share.notna()]
    n = math.ceil(len(eligible) * .2)
    top_share = eligible.head(n).revenue_share.sum()
    revenue_text = f"{baseline['known_revenue']:,.2f}" if baseline["known_revenue"] is not None else "unknown"
    lines = [
        "# DemandOS Season 4 — recorded-sales EDA", "",
        f"Analysis window: **{report['window_start']} through {report['window_end']}**, inclusive.",
        "Source: existing, verified DuckDB/dbt facts and marts over M5 validation-history Parquet.",
        f"Source SHA-256: `{report['source']['sha256']}`. This report contains actual-data results, not synthetic fixture values.", "",
        "## Measured facts", "",
        f"- Observations: **{baseline['daily_rows']:,}**; observed days: **{baseline['observed_days']:,}**; units: **{baseline['units_sold']:,}**.",
        f"- Known recorded-price revenue: **{revenue_text}**, in historical M5 US-dollar price context; this is not profit or verified accounting revenue.",
        f"- Zero-sales observations: **{baseline['zero_sales_count']:,} ({baseline['zero_sales_count']/baseline['daily_rows']:.2%})**.",
        f"- Missing-price observations: **{baseline['missing_price_count']:,}**; unknown positive-sales revenue observations: **{baseline['unknown_revenue_count']:,}**.", "",
        "## Evidence-backed findings and interpretation", "",
        f"1. **Store comparison:** {store.store_id} has the highest recorded volume, **{store.units_sold:,} units**, or **{store.units_per_observed_series_day:.3f} units per observed store-item-day**. This normalizes recorded exposure, not verified in-stock exposure; volume alone does not establish management performance.",
        f"2. **Weekday association:** {weekday.weekday} has the highest pooled daily mean, **{weekday.mean_daily_units:,.1f} units**, with weekday index **{weekday.weekday_index:.3f}** over **{int(weekday.observed_days)} observed days**. Check year/store/category profiles before assuming a universal weekly pattern.",
        f"3. **Revenue concentration:** the top **{n} of {len(eligible)} revenue-eligible products** (approximately 20%) contribute **{top_share:.2%}** of eligible known-price revenue. This prioritizes investigation by historical value, not margin or optimal inventory policy.",
        f"4. **Observation-window sensitivity:** **{report['leading_zero_series']:,} store-item series** start with zero-sales periods. Replacing the fixed-window ADI denominator with days from the first positive sale changes **{report['first_positive_sensitivity_changed']:,} regime labels**. First positive sale is only a sensitivity proxy, not a verified launch date.", "",
        "## Follow-up hypotheses and questions", "",
        "- Is the observed strongest-weekday association stable within each year, store, and category, and after controlling for event dates and trends? The exported subgroup tables permit this check; no causal effect is established.",
        "- Do high-value, high-weekly-variability products benefit from different forecasting baselines or uncertainty estimates? Recompute classifications using training data only before comparing future holdouts.",
        "- How much apparent intermittency reflects product introduction or unobserved availability rather than customer demand? Seek independent availability/launch evidence before trimming leading zeros or changing inventory policy.", "",
        "## Product-level ABC", "",
        "Eligible products have complete recorded-revenue information. Rank known-price revenue descending, break ties by item_id ascending. Classify using the cumulative share **before** the item: below A cutoff → A; below B cutoff → B; otherwise C. A threshold-crossing item stays in the earlier class. Zero-revenue items are C only when total eligible revenue is positive; otherwise classifications are unsupported.", "",
        markdown_table(frames["abc_summary"], ["abc_class", "products", "units_sold", "known_revenue", "revenue_share"]), "",
        "## Compatible product-level ABC × weekly XYZ", "",
        f"XYZ uses **{report['complete_weeks']} complete Saturday-Friday weeks**; **{report['partial_weeks_excluded']} partial weeks excluded**. Population standard deviation divided by mean weekly units includes genuine zero weeks. X/Y thresholds and sparse/low-mean safeguards are in the config snapshot. Unsupported labels stay separate.", "",
        markdown_table(frames["abc_xyz_matrix"], ["abc_xyz", "products", "revenue_share", "volume_share"]), "",
        "Matrix shares use all observed known revenue and units; unsupported products are not silently discarded.", "",
        markdown_table(frames["abc_xyz_unclassified"], ["abc_xyz", "products", "revenue_share", "volume_share"]), "",
        "## Store-item sales-occurrence regimes", "",
        "ADI = all observed days / positive-sales days. CV² uses **strictly positive daily sales sizes**, population variance, and at least the configured positive-day count. At equality, ADI/CV² boundaries enter the higher-sparsity/higher-variability class. All-zero or insufficient histories remain unclassified.", "",
        markdown_table(frames["regime_summary"], ["regime", "series", "volume_share", "revenue_share"]), "",
        "## Outliers, events, and decomposition", "",
        f"On **{len(frames['selected_series'])} selected store-item series** ({report['selected_anomaly_rows']:,} observations), global IQR screening flags **{report['iqr_flags_selected']:,}** rows and prior-same-weekday robust-MAD screening flags **{report['robust_flags_selected']:,}**. These overlap and are not population-wide anomaly estimates. No rows are deleted. Zero MAD or too little history makes the robust screen unevaluable.",
        "Event flags use either calendar event slot; named-event summaries use event_name_1 only, explicitly. Weekday-adjusted indices compare units with the pooled same-weekday mean, without controlling for trends or promotions. M5 supplies event labels, not measured promotional treatment effects.", "",
        markdown_table(frames["event_associations"], ["any_event", "observed_days", "mean_daily_units", "mean_weekday_adjusted_index"]), "",
        "Additive STL examples use period 7 and robust fitting on overall/category daily aggregates, with at least 14 continuous days. Trend + seasonal + remainder reconstructs observed sales. The fit uses the complete selected window and is descriptive; its centered components and global IQR bounds are **not** forecasting inputs. Prior-row averages exclude the current day but still require forecast-origin discipline at multi-day horizons.", "",
        "## Reconciliation, provenance, and limitations", "",
        f"- {report['reconciliation']}; ABC/XYZ joins are one-to-one. Additive segment shares reconcile; distinct counts and ratios are not summed across incompatible grains.",
        "- Daily group means/medians/quantiles/std/CV use one aggregate per observed date; store-item statistics use one observation per observed day. Standard deviations are population quantities. Missing observations are not imputed as zeros; mean-zero CV is undefined.",
        "- Store normalized volume divides by observed series-days, not inventory availability. Full-window zeros remain included; first-positive sensitivity does not assert launch dates.",
        "- Monthly totals record observed-day exposure; pooled calendar-month daily means can mix trend and seasonality. Partial weeks are plotted separately through exposure tables and excluded from XYZ.",
        f"- Measured runtime before Markdown publication: **{report['runtime_seconds']:.2f}s**. SQL reduces facts before Python; largest output summary has {max(report['table_rows'].values()):,} rows. Runtime depends on hardware, caches and scheduling.",
        "- Thresholds, input fingerprint, package versions, warning semantics and table sizes are recorded in `quality_report.json`; all output is local and Git-ignored.", "",
        "## Figures", "",
    ]
    for name in report["figures"]:
        lines += [f"![{name.removesuffix('.png').replace('_', ' ')}](figures/{name})", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
