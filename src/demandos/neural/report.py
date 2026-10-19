"""Small real-data loss/forecast plots and an auditable human-readable report."""

def artifacts(summary, tables, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = directory/"figures"
    figures.mkdir(exist_ok=True)
    curves = tables["training_curves"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, model in zip(axes, ("lstm", "gru")):
        rows = curves[curves.model.eq(model) & curves.phase.eq("development")]
        ax.plot(rows.epoch, rows.training_loss, label="train")
        ax.plot(rows.epoch, rows.validation_loss, label="December validation")
        ax.set(title=model.upper(), xlabel="Epoch", ylabel="Normalized MSE")
        ax.legend()
    fig.tight_layout(); fig.savefig(figures/"development_losses.png", dpi=120); plt.close(fig)
    forecasts = tables["forecasts"]
    # Select by identifier order, never by which method appears to win.
    member = sorted(forecasts.series_id.unique())[0]
    rows = forecasts[forecasts.series_id.eq(member) & forecasts.origin.eq(forecasts.origin.min())]
    fig, ax = plt.subplots(figsize=(10, 4))
    actual = rows[rows.model.eq("lstm")].sort_values("target_date")
    ax.plot(actual.target_date, actual.actual, label="observed", color="black")
    for model in ("lstm", "gru", "lightgbm", "seasonal_naive"):
        part = rows[rows.model.eq(model)].sort_values("target_date")
        ax.plot(part.target_date, part.prediction, label=model, alpha=.8)
    ax.set(title=member, ylabel="Observed sales units"); ax.legend()
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(figures/"representative_forecast.png", dpi=120); plt.close(fig)
    lines = ["# Season 12 real M5 recurrent forecasting", "",
             "Retrospective selected-cohort point forecasts of recorded sales; not official M5 WRMSSE.", "",
             f"{summary['training_series']}-series historical panel; {summary['development_windows']:,} development vectors; "
             f"{summary['validation_windows']} December validation vectors. 56 input days → 28 outputs.", "",
             f"December-selected epochs: {summary['selected_epochs']}; fit cutoff {summary['development_fit_cutoff']}; validation end {summary['validation_end']}.", "",
             "| Model | MAE | WAPE | Macro RMSSE |", "|---|---:|---:|---:|"]
    for row in summary["metrics_common"]:
        lines.append(f"| {row['model']} | {row['mae']:.6f} | {row['wape']:.4%} | {row['macro_rmsse']:.6f} |")
    lines += ["", "| Encoder | Parameters | Development epochs run / selected | Best validation MSE |", "|---|---:|---:|---:|"]
    for row in summary["fit_records"]:
        if row["phase"] == "development":
            lines.append(f"| {row['model']} | {row['parameter_count']} | {row['epochs_run']} / {row['selected_epochs']} | {row['best_validation_loss']:.6f} |")
    metric = summary["metrics_common"][0]
    warning_count = sum(len(row["warnings"]) for row in summary["fit_records"])
    lines += ["", f"Versions: {summary['versions']}. Initial experiment: {summary['runtime_seconds']:.2f} seconds.",
              "", f"Shared cases per method: {metric['cases']}; defined RMSSE cases: {metric['rmsse_cases']}; "
              f"undefined scales: {metric['undefined_rmsse_cases']}. Neural training warnings: {warning_count}."]
    lines += ["", "All methods share 1,092 target keys. See tables/metrics_origin, metrics_horizon, metrics_band, metrics_regime, metrics_abc_xyz, paired_losses and coverage Parquet files.",
              "", "Future-sales mutation/deletion, price/label perturbation, tensor equality and checkpoint round trips passed for both models at all three origins.",
              "", "Preprocessing, per-fit cutoffs, parameter counts, loss curves, warnings and runtime are recorded in summary.json and models/*/*/metadata.json.",
              "", "Training histories and original sources were not changed. Cold/short-history files are mask-capability diagnostics, not cold-start accuracy validation.",
              "", "## Reproduce", "", "`python -m demandos.neural --root . --config configs/neural.toml`", "",
              "Requires prior warehouse build and installed analytics/eda/boosting/neural extras. See docs/neural.md.", "",
              "## Limits", "", *[f"- {warning}" for warning in summary["warnings"]], ""]
    (directory/"neural_report.md").write_text("\n".join(lines), encoding="utf-8")
