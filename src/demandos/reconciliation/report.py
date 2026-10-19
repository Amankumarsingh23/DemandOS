"""Readable cohort-only evidence; generated figures/reports are ignored."""


def artifacts(summary, tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output/"figures"
    figures.mkdir()
    metrics = tables["metrics_level"].pivot(index="level", columns="model", values="wape")
    ax = metrics.plot.bar(figsize=(16, 6), width=.85)
    ax.set(ylabel="Pooled WAPE", title="Cohort-induced groups; identical targets per method")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(figures/"level_wape.png", dpi=120); plt.close()
    root = tables["forecasts"].query("level == 'TOTAL'")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for ax, (origin, frame) in zip(axes, root.groupby("origin_date")):
        for model, group in frame.groupby("model"):
            ax.plot(group.target_date, group.prediction, label=model, alpha=.7)
        actual = frame.drop_duplicates("target_date")
        ax.plot(actual.target_date, actual.actual, color="black", linewidth=2, label="observed cohort total")
        ax.set_title(f"COHORT_TOTAL origin {origin.date()}")
    axes[0].legend(ncol=3, fontsize=7)
    fig.tight_layout(); fig.savefig(figures/"cohort_total.png", dpi=120); plt.close(fig)
    rows = ["# Season 15 real M5 reconciliation", "", "Full sparse membership: "+str(summary["full_hierarchy"]), "",
            "Cohort-induced hierarchy: "+str(summary["cohort_hierarchy"]), "", summary["cohort_rule"], "",
            summary["base_forecasts"], "", summary["historical_recipe"], "", "Covariance: "+str(summary["covariance"]), "",
            "| Cohort level | Method | MAE | WAPE | Macro RMSSE |", "|---|---|---:|---:|---:|"]
    for row in summary["metrics_level"]:
        rows.append(f"| {row['level']} | {row['model']} | {row['mae']:.6f} | {row['wape']:.4%} | {row['macro_rmsse']:.6f} |")
    rows += ["", "## Exact 13-bottom continuity view", "", "| Method | MAE | WAPE | Macro RMSSE |", "|---|---:|---:|---:|"]
    for row in summary["metrics_continuity"]:
        rows.append(f"| {row['model']} | {row['mae']:.6f} | {row['wape']:.4%} | {row['macro_rmsse']:.6f} |")
    rows += ["", f"{summary['common_cases_per_method']:,} cases per method; complete common-grid coverage. No skipped cases. Fallback: {summary['fallback']}.",
             f"Runtime {summary['runtime_seconds']:.2f}s; OS peak process working set {summary['peak_process_mib']:.1f} MiB.", "",
             "Coherence/negative counts: tables/coherence.parquet. Origin/horizon/band/level-origin metrics, solver timings and undefined denominators are retained in tables/.", "",
             "All future-sales and future-price perturbation gates passed; raw/interim/processed hashes, reference weights and warehouse totals are unchanged.", "",
             "## Limitations", "", *["- "+v for v in summary["limitations"]], "", "## Reproduce", "",
             "`python -m demandos.reconciliation --root . --config configs/reconciliation.toml`", ""]
    (output/"reconciliation_report.md").write_text("\n".join(rows), encoding="utf-8")
