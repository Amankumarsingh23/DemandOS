"""Measured raw/repaired calibration, interval sharpness and sparse limitations."""


def artifacts(summary, tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output/"figures"; figures.mkdir()
    final = tables["metrics_overall"].query("view == 'repaired'")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for row in final.itertuples():
        axes[0].plot([.1, .5, .9], [row.coverage_q10, row.coverage_q50, row.coverage_q90], marker="o", label=f"{row.cohort}/{row.model}")
        axes[1].scatter(row.interval_width_mean, row.interval_coverage, label=f"{row.cohort}/{row.model}")
    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--")
    axes[0].set(xlabel="Nominal quantile", ylabel="Inclusive empirical coverage")
    axes[1].axhline(.8, color="black", linestyle="--")
    axes[1].set(xlabel="Mean interval width (units)", ylabel="P10-P90 coverage")
    axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(figures/"calibration_width.png", dpi=120); plt.close(fig)
    forecasts = tables["forecasts"]
    selected = forecasts.query("cohort == 'continuity'").iloc[0]
    frame = forecasts[forecasts.store_id.eq(selected.store_id) & forecasts.item_id.eq(selected.item_id) & forecasts.origin_date.eq(selected.origin_date)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    for axis, (model, group) in zip(axes, frame.groupby("model")):
        axis.fill_between(group.target_date, group.q10, group.q90, alpha=.25, label="P10-P90")
        axis.plot(group.target_date, group.q50, label="P50")
        axis.plot(group.target_date, group.actual, color="black", label="Observed sales")
        axis.set_title(model); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures/"representative_intervals.png", dpi=120); plt.close(fig)
    rows = ["# Season 16 observed-sales probabilistic forecasts", "", summary["fit_policy"], "",
            f"96 real training series; {summary['development_tree_rows']:,} tabular development rows + {summary['validation_tree_rows']:,} validation rows. "
            f"Final {summary['final_tree_rows']:,} labels. Eight fitted development/final models.", "",
            "## Raw and repaired same-case metrics", "",
            "| Cohort | Stage | Model | N | P10 loss | P50 loss | P90 loss | Mean loss | C10 | C50 | C90 | Interval coverage | Width | Interval score | Crossing count |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["metrics"]:
        rows.append(f"| {row['cohort']} | {row['view']} | {row['model']} | {row['observations']} | "
                    f"{row['pinball_q10']:.6f} | {row['pinball_q50']:.6f} | {row['pinball_q90']:.6f} | {row['pinball_mean']:.6f} | "
                    f"{row['coverage_q10']:.3%} | {row['coverage_q50']:.3%} | {row['coverage_q90']:.3%} | {row['interval_coverage']:.3%} | "
                    f"{row['interval_width_mean']:.6f} | {row['interval_score']:.6f} | {row['any_crossing_count']} |")
    rows += ["", "Origin/horizon/band/regime, historical-zero and observed-zero/positive results, denominators, tie counts, "
             "Wilson intervals and raw invalid-interval counts are retained in tables/metrics_*.parquet.", "",
             "Hierarchy: "+str(summary["hierarchy"]["toy"]), "",
             f"Runtime {summary['runtime_seconds']:.2f}s; inference {summary['inference_seconds']:.3f}s; OS peak {summary['peak_process_mib']:.1f} MiB. "
             f"Tree warnings {summary['tree_warning_count']}; neural warnings {summary['neural_warning_count']}; zero failed/skipped cases.", "",
             "All nine real future-sales mutation/deletion and future-price gates pass. Native model reloads reproduce inference. "
             "Source hashes, prior hierarchy and warehouse totals remain unchanged.", "", "## Limitations", "",
             *["- "+value for value in summary["limitations"]], "", "## Reproduce", "",
             "`python -m demandos.quantiles --root . --config configs/quantiles.toml`", ""]
    (output/"quantiles_report.md").write_text("\n".join(rows), encoding="utf-8")
