"""Concise measured feature evidence, assumptions and an exhaustive dictionary."""

import pandas as pd
from demandos.eda.report import markdown_table


def write_dictionary(dictionary, path):
    lines = ["# Feature dictionary", "", "Predictive inputs are explicitly allowlisted; labels and audit dates are excluded."]
    for title in ("origin", "features", "labels"):
        lines += ["", f"## {title}", "", markdown_table(pd.DataFrame(dictionary[title]),
                  ["name", "type", "role", "grain", "source", "availability", "null_policy"])]
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def write_report(summary, samples, path):
    validation = summary["evaluation"]
    full = summary["full_origin_pass"]
    nulls = pd.DataFrame([{"column": name, "null_rows": number} for name, number in validation["null_counts"].items() if number])
    lines = ["# Season 8: origin-safe direct-horizon features", "",
             f"Runtime: {summary['runtime_seconds']:.2f}s. Cohort: {summary['cohort_series']} store-item series; "
             f"{validation['rows']} evaluation inference rows; {summary['training_labels']} supervised training labels.",
             (f"Full origin pass: {full['rows']:,} series at {full['origin']}; {full['bounded_history_rows']:,} bounded history rows, "
              f"{full['dataframe_bytes']:,} result DataFrame bytes, {full['seconds']:.2f}s. This computes compact origin features, not all-panel horizon expansion."
              if full["executed"] else "Full-panel pass disabled; selected-cohort features only."),
             "", "## Contract and temporal safety", "",
             "Grain: store_id, item_id, origin_date, horizon (1..28). Target date = origin + horizon calendar days. Origin is end-of-day after realized sales are available. origin_lag_1 = y_(origin-1); sales_at_origin = y_origin. Rolling windows end at the origin and require complete observed daily windows; population std uses ddof=0.",
             "Predictive inputs are listed in feature_dictionary.json. Source/audit dates, mapping provenance and target labels are excluded from that allowlist. Training labels are physically separate; evaluation/inference features never query target actual sales. Codes use training-only sorted ID vocabularies with unknown=0; static IDs are also preserved for native categorical models.",
             f"Selection/mapping cutoff: {summary['cohort_fit_end']}. Training label cutoff: {summary['label_cutoff']}. No Season 7 evaluation targets fit mappings or segment selection. Segments stay in cohort metadata, not predictive inputs.",
             f"Invariance checks: {summary['invariance']}. All 28 feature rows per selected series-origin match after changing every future actual and future-week price; they also match with future actuals entirely absent. Default price exclusion makes the real-price check an exclusion check; non-null historical as-of behavior is tested separately on explicit synthetic evidence.",
             "", "## Null and price policies", "",
             summary["historical_price_policy"],
             "Retrospective weekly price rows alone do not establish knowledge by an origin. Last/prior price, change and four-week relative-price features stay null by default, with explicit missing flags and zero known weeks. No missing price becomes zero. The optional historical evidence interface joins unique store-item-week keys to verified weekly prices, requires both observation and availability dates <= origin, and checks week/date mappings. No planned future price scenario or promotion classification is inferred.",
             markdown_table(nulls, ["column", "null_rows"]) if not nulls.empty else "No missing feature values.",
             "", "## Known-ahead and observational assumptions", "", summary["known_ahead_calendar_assumption"],
             "Store state chooses the appropriate CA/TX/WI SNAP flag. Missing event names mean no listed event. Identifiers/hierarchy are treated as immutable metadata; category vocabularies use only positive-sale identifier observations through the fit cutoff. Historical M5 zeros remain observed zeros, not estimates of inventory or uncensored customer demand.",
             "", "## Actual M5 sample rows", "",
             markdown_table(samples, ["store_id", "item_id", "origin_date", "horizon", "target_date", "sales_at_origin", "origin_lag_7",
                                     "rolling_mean_7_at_origin", "target_weekday", "target_snap", "last_known_price", "item_id_code"]),
             "", "## Resources and reproducibility", "", f"Resources: {summary['resources']}. DataFrame byte counts are observations of bounded result tables, not total process peak memory. DuckDB limits apply to managed buffers, with ignored spill storage; there is no strict total-process ceiling.",
             "SQL scans only required sales columns and bounded history for origin calculations; it never expands the 58.3m daily fact rows across 28 horizons. Compact origin Parquet is materialized once, then selected-cohort calendar expansion is bounded and partitioned by origin. Raw/interim/processed SHA-256s and warehouse row/unit/revenue totals reconcile before/after.",
             "Command: `.\\.venv\\Scripts\\python -m demandos.features --root . --config configs/features.toml`. Add `--inference-only` to omit supervised tables, `--cohort` for explicit keys, or `--price-evidence` for verified historical-price knowledge stamps. See tests recorded separately in test_results.json; a rerun does not claim those suites ran again.",
             "No forecasting or ML model is trained. Season 9 can join training_labels.parquet to training_features.parquet on the canonical keys and use only the saved predictive-column allowlist, with temporal train/evaluation boundaries preserved."]
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
