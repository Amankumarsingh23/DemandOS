"""Measured coverage, sharpness and explicit assumptions for retrospective M5."""


def artifacts(summary, tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output/"figures"; figures.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (cohort, group) in zip(axes, tables["metrics_overall"].groupby("cohort")):
        for row in group.itertuples():
            ax.scatter(row.width_mean, row.coverage, label=row.method)
        ax.axhline(.8, color="black", linestyle="--")
        ax.set(title=cohort, xlabel="Mean interval width", ylabel="Daily empirical coverage")
    axes[1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1, .5))
    fig.tight_layout(); fig.savefig(figures/"coverage_width.png", dpi=120); plt.close(fig)
    fig, axis = plt.subplots(figsize=(12, 5))
    for key, group in tables["aci_feedback"].groupby(["family", "cohort", "horizon_band"]):
        daily = group.groupby("feedback_date").alpha_after.last()
        axis.plot(daily.index, daily, label="/".join(key))
    axis.set(ylabel="Bounded ACI alpha after daily feedback", xlabel="Feedback availability date")
    axis.legend(fontsize=6, ncol=3)
    fig.tight_layout(); fig.savefig(figures/"aci_alpha.png", dpi=120); plt.close(fig)
    rows = ["# Season 17 real M5 conformal calibration", "", "New models use nested September selection and an October 11 fitting cutoff. "
            "The same frozen checkpoints issue all calibration/evaluation forecasts. Season 16 final models remain separate references.", "",
            "| Cohort | Method | N | Coverage | Mean width | Interval score | Zero-target coverage |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["metrics"]:
        rows.append(f"| {row['cohort']} | {row['method']} | {row['observations']} | {row['coverage']:.3%} | "
                    f"{row['width_mean']:.6f} | {row['interval_score']:.6f} | {row['zero_sales_coverage']:.3%} |")
    rows += ["", "Whole-trajectory coverage, origin/horizon/band/regime tables, threshold score IDs, effective support, "
             "maturity timestamps and all ACI updates remain in tables/.", "",
             f"Runtime {summary['runtime_seconds']:.2f}s; peak OS process working set {summary['peak_process_mib']:.1f} MiB; "
             f"training warnings {summary['warnings']}; failed/skipped cases {summary['failed_or_skipped_cases']}.", "",
             "All predictor/calibrator future mutation/deletion/price gates, source hashes, warehouse totals and Season 16 reference hashes pass.",
             "", "## Limitations", "", *["- "+value for value in summary["limitations"]], "", "## Reproduce", "",
             "`python -m demandos.conformal --root . --config configs/conformal.toml`", ""]
    (output/"conformal_report.md").write_text("\n".join(rows), encoding="utf-8")
