"""Measured point-forecast results and explicitly noncausal TFT diagnostics."""


def artifacts(summary,tables,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures=output/"figures"; figures.mkdir(exist_ok=True)
    development=tables["training_curves"].query("phase=='development'")
    fig,axis=plt.subplots(figsize=(8,4))
    axis.plot(development.epoch,development.training_loss,label="train")
    axis.plot(development.epoch,development.validation_loss,label="December validation")
    axis.set(xlabel="Epoch",ylabel="Normalized point MSE"); axis.legend()
    fig.tight_layout(); fig.savefig(figures/"loss_curves.png",dpi=120); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for axis,family in zip(axes,("static","past","future")):
        rows=tables["vsn_summary"].query("family==@family").sort_values("mean_weight")
        axis.barh(rows.variable,rows.mean_weight); axis.set_title(f"{family}: internal VSN weights")
    fig.tight_layout(); fig.savefig(figures/"variable_selection.png",dpi=120); plt.close(fig)
    attention=tables["attention_cases"]
    matrix=attention.groupby(["horizon","key_relative_day"]).weight.mean().unstack("key_relative_day")
    fig,axis=plt.subplots(figsize=(10,4))
    heat=axis.imshow(matrix,origin="lower",aspect="auto",extent=[-55.5,28.5,.5,28.5])
    axis.set(xlabel="Key day relative to forecast origin",ylabel="Forecast horizon",title="Mean head attention; internal diagnostic, not causal")
    fig.colorbar(heat,ax=axis); fig.tight_layout(); fig.savefig(figures/"attention_horizons.png",dpi=120); plt.close(fig)
    cohort=tables["evaluation_cohort"].sort_values(["regime","store_id","item_id"]).drop_duplicates("regime").head(3)
    forecasts=tables["forecasts"]; origin=forecasts.origin.min()
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    for row_axes,row in zip(axes,cohort.itertuples()):
        selected=forecasts[forecasts.store_id.eq(row.store_id)&forecasts.item_id.eq(row.item_id)&forecasts.origin.eq(origin)]
        observed=selected[selected.model.eq("tft")].sort_values("target_date")
        row_axes[0].plot(observed.target_date,observed.actual,label="actual",color="black")
        for model in ("tft","nbeats_interpretable","gru","lightgbm"):
            values=selected[selected.model.eq(model)].sort_values("target_date")
            row_axes[0].plot(values.target_date,values.prediction,label=model)
        weights=attention[attention.store_id.eq(row.store_id)&attention.item_id.eq(row.item_id)&attention.origin.eq(origin)]
        values=weights.groupby(["horizon","key_relative_day"]).weight.mean().unstack("key_relative_day")
        row_axes[1].imshow(values,origin="lower",aspect="auto",extent=[-55.5,28.5,.5,28.5])
        row_axes[0].set_title(f"{row.store_id}/{row.item_id} ({row.regime})"); row_axes[0].legend(fontsize=7)
        row_axes[1].set(xlabel="Key day relative to origin",ylabel="Horizon",title="Internal attention")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(figures/"representative_cases.png",dpi=120); plt.close(fig)
    fig,axis=plt.subplots(figsize=(10,4))
    for model,rows in tables["metrics_horizon"].groupby("model"): axis.plot(rows.horizon,rows.mae,label=model)
    axis.set(xlabel="Forecast horizon",ylabel="MAE in observed sales units"); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures/"horizon_mae.png",dpi=120); plt.close(fig)
    meta=summary["fit_records"][0]
    lines=["# Season 14: real M5 point TFT","",
           f"Past [B,56,10], future [B,28,7], static [B,5], predictions [B,28]. {summary['training_series']} training series; "
           f"{summary['development_windows']} development and {summary['validation_windows']} validation vectors.","",
           f"December fit cutoff {summary['development_cutoff']}; validation ends {summary['validation_end']}. "
           f"Selected epochs {summary['selected_epochs']}; parameters {meta['parameter_count']}; best normalized validation MSE {meta['best_validation_loss']:.6f}.","",
           "| Model | MAE | WAPE | Macro RMSSE |","|---|---:|---:|---:|"]
    for row in summary["metrics_common"]: lines.append(f"| {row['model']} | {row['mae']:.6f} | {row['wape']:.4%} | {row['macro_rmsse']:.6f} |")
    lines += ["","All methods share 1,092 exact target keys. Coverage, undefined denominators, per-origin/horizon/band/segment scores and paired losses are retained in tables/.","",
              "Static/past/future VSNs normalize over variables. Shared-value, averaged-head attention normalizes over permitted keys; all padded historical and future-prohibited keys have zero probability. Diagnostics do not establish causes or price effects.","",
              "All four TFT checkpoints reloaded. At every evaluation origin, future-sales mutation/deletion and future-price/label perturbation preserve tensors, raw/scored predictions and interpretations.","",
              f"Initial experiment: {summary['runtime_seconds']:.2f}s. Versions: {summary['versions']}. Resources: {summary['resources']}.","",
              "Source hashes and warehouse totals remain unchanged. Prior reference predictions are provenance-checked and reused, not retrained.","",
              "## Reproduce","","`python -m demandos.tft --root . --config configs/tft.toml`","",
              "Requires existing neural environment, warehouse and verified Seasons 12/13 artifacts. See docs/tft.md.","",
              "## Limitations","",*[f"- {value}" for value in summary["limitations"]],""]
    (output/"tft_report.md").write_text("\n".join(lines),encoding="utf-8")
