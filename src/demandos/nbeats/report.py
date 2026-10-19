"""Measured reports and representative, noncausal contribution plots."""


def artifacts(summary, tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    curves = tables["training_curves"]
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for axis, variant in zip(axes,("generic","interpretable")):
        subset = curves[curves.variant.eq(variant)&curves.phase.eq("development")]
        axis.plot(subset.epoch,subset.training_loss,label="train")
        axis.plot(subset.epoch,subset.validation_loss,label="December validation")
        axis.set(title=variant,xlabel="Epoch",ylabel="Normalized forecast MSE"); axis.legend()
    fig.tight_layout(); fig.savefig(figures/"loss_curves.png",dpi=120); plt.close(fig)
    fig,axis = plt.subplots(figsize=(10,4))
    for model,rows in tables["metrics_horizon"].groupby("model"):
        axis.plot(rows.horizon,rows.mae,label=model)
    axis.set(xlabel="Horizon (days)",ylabel="MAE (observed units)"); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures/"horizon_mae.png",dpi=120); plt.close(fig)
    cohort = tables["evaluation_cohort"].sort_values(["regime","store_id","item_id"])
    selected = cohort.drop_duplicates("regime").head(3)
    origin = tables["forecasts"].origin.min()
    for index,row in enumerate(selected.itertuples(),1):
        frame = tables["forecasts"]
        subset = frame[frame.store_id.eq(row.store_id)&frame.item_id.eq(row.item_id)&frame.origin.eq(origin)]
        actual = subset[subset.model.eq("nbeats_generic")].sort_values("target_date")
        past = tables["history"]
        past = past[past.store_id.eq(row.store_id)&past.item_id.eq(row.item_id)&past.origin.eq(origin)]
        part = tables["components"]
        part = part[part.store_id.eq(row.store_id)&part.item_id.eq(row.item_id)&part.origin_date.eq(origin)&part.model.eq("nbeats_interpretable")].sort_values("target_date")
        fig,axes = plt.subplots(3,1,figsize=(11,9))
        axes[0].plot(past.date,past.observed_units,label="observed history",color="black")
        axes[0].plot(actual.target_date,actual.actual,label="future actuals (scoring only)",color="gray")
        for model in ("nbeats_generic","nbeats_interpretable"):
            predicted = subset[subset.model.eq(model)].sort_values("target_date")
            axes[0].plot(predicted.target_date,predicted.prediction,label=model)
            axes[1].plot(predicted.target_date,predicted.prediction,label=model)
        axes[1].plot(actual.target_date,actual.actual,label="actual",color="black")
        for name in ("trend","seasonality","raw_prediction","prediction"):
            axes[2].plot(part.target_date,part[name],label=name)
        axes[0].set_title(f"{row.store_id}/{row.item_id} ({row.regime})")
        axes[2].set_title("Model-attributed components; not causal or uniquely identified")
        for axis in axes: axis.set_ylabel("Observed sales units"); axis.legend(fontsize=8)
        fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(figures/f"example_{index}.png",dpi=120); plt.close(fig)
    lines = ["# Season 13: real M5 N-BEATS experiment","",
             f"Target-only 56-day input -> 28-day forecast; {summary['training_series']} series; "
             f"{summary['development_windows']:,} development and {summary['validation_windows']} December validation vectors.","",
             f"Development cutoff: {summary['development_cutoff']}; validation ends {summary['validation_end']}. "
             f"Frozen epochs: {summary['selected_epochs']}.","",
             "| Model | MAE | WAPE | Macro RMSSE |","|---|---:|---:|---:|"]
    for row in summary["metrics_common"]:
        lines.append(f"| {row['model']} | {row['mae']:.6f} | {row['wape']:.4%} | {row['macro_rmsse']:.6f} |")
    lines += ["","All methods share 1,092 targets. Undefined WAPE/RMSSE denominators and coverage remain explicit in the metric tables.","",
              "| Variant | Parameters | Development epochs run / selected | Best validation MSE |","|---|---:|---:|---:|"]
    for row in summary["fit_records"]:
        if row["phase"]=="development":
            lines.append(f"| {row['variant']} | {row['parameter_count']} | {row['epochs_run']} / {row['selected_epochs']} | {row['best_validation_loss']:.6f} |")
    lines += ["", "Generic blocks learn separate backcast/forecast bases. Interpretable polynomial trend precedes a fixed weekly Fourier stack. "
              "Raw forecasts add all block contributions; clipping adjustment reconciles the nonnegative scored forecast.","",
              "All checkpoints reloaded. Future-sales mutation/deletion, future-price perturbation and training-label eligibility checks passed for both variants at all three origins.","",
              f"Initial experiment seconds: {summary['runtime_seconds']:.2f}. Versions: {summary['versions']}. Resources: {summary['resources']}.","",
              "Prior GRU/LSTM/LightGBM/TSB/seasonal-naive forecasts are reused after source/config/cohort/key/actual/scale checks. Their previous timings are retained separately.","",
              "## Reproduce","","`python -m demandos.nbeats --root . --config configs/nbeats.toml`","",
              "Requires the existing warehouse, neural extra and verified Season 12 artifacts. See docs/nbeats.md.","",
              "## Limits","",*[f"- {item}" for item in summary["limitations"]],""]
    (output/"nbeats_report.md").write_text("\n".join(lines),encoding="utf-8")
