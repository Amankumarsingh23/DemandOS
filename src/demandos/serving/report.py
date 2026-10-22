"""Measured local replay report; operational assumptions remain explicit."""
def write_report(summary,tables,drifts,path):
    lines=["# Season 23: production-oriented LOCAL historical serving","",
           "**Historical M5 observed sales; SIMULATED decisions, canary, approvals and incident exercises. No live service/order dispatch or measured Walmart policy effects.**","",
           f"Run `{summary['run_id']}`: 25 series (13 continuity / 12 stress), three origins, 28 horizons; {summary['forecast_rows']} unique forecast rows, {summary['origin_decisions']} origin decision rows, {summary['canary_actions']} separate simulated canary actions.","",
           "## Registry / contracts / publication","",
           "Native LightGBM ensembles use existing trusted model.txt/metadata.json files only, verified before loading. No pickle/joblib and no historical refit. Immutable manifests plus hash-linked lifecycle history and one atomic alias state. The historical fallback is an explicit existing algorithm recipe, not claimed fitted weights.","",
           "| Model | Fit cutoff | Artifact digest |","|---|---|---|"]
    for e in summary["registry_entries"]:lines.append(f"| {e['model_version']} | {e['training_cutoff']} | `{e['artifact_digest']}` |")
    lines += ["",f"Feature contract `{summary['feature_contract_digest']}` reuses original Season8/16 transformations/native category matrices. Event-date availability and known-future calendar are replay assumptions, not actual M5 ingestion timestamps. Critical delayed history blocks; no price is available without explicit verified knowledge stamps.",
        f"{summary['feature_parity_checks']} saved-reference parity checks, {summary['future_checks']} future sales/deletion/price forecast/action checks, {summary['forecast_retries']} publication retries. Complete batches are immutable, validated in staging and atomically renamed on the same local volume; corruption/conflicting content fails. No distributed exactly-once claim.","",
        "## Mature-label forecast monitoring","",
        "Two-day assumed label lag: origin+10 monitoring includes horizons1–8 (200/700 rows); origin+30 includes all28. Missing/unavailable labels remain unavailable. Full mature results:","",
        "| Model | Mature | MAE | WAPE | Pinball | P10–P90 coverage | Width |","|---|---:|---:|---:|---:|---:|---:|"]
    for model,m in summary["forecast_metrics"].items():
        w="unavailable" if m["wape"] is None else f"{m['wape']:.4%}"
        lines.append(f"| {model} | {m['mature']} | {m['mae']:.6f} | {w} | {m['pinball']:.6f} | {m['coverage']:.4%} | {m['width']:.6f} |")
    lines += ["", "## Fixed-reference drift and injected shifts","",
        "Reference frozen at Jan31; bins from reference only, NaN separate, epsilon1e-6 floor then renormalization. KS is descriptive with no IID p-value claim. History statistics use25 unique series; prediction/width use700 correlated rows. Thresholds/minimum20/window fixed in config before scoring.","",
        "| Origin | Signal | Injection | PSI | KS | Severity |","|---|---|---|---:|---:|---|"]
    for row in drifts:lines.append(f"| {row['origin_date'][:10]} | {row['feature']} | {row.get('injection','none')} | {row['psi']:.6f} | {row['ks']} | {row['severity']} |")
    lines += ["", "## Simulated downstream quality and routing","",
        "Origin requests use ceil(sum of repaired daily P50 over14 days), an explicit point-target heuristic. No sum of marginal quantiles is claimed to be a calibrated lead-time quantile. Shared resource limits are frozen from prior56-day means; costs, initial stock10/pipeline3 and seven-day lead are hypothetical. Full28-day controllers reset independently under matched observed-sales proxy / distinct historical bootstrap worlds.","",
        "| Model | World kind | Pool runs | Mean simulated cost | Aggregate immediate fill | Lost units/run |","|---|---|---:|---:|---:|---:|"]
    for (model,kind),g in tables["simulation_runs"].groupby(["model_version","world_kind"]):
        fill=g.served.sum()/g.demand.sum() if g.demand.sum() else 0
        lines.append(f"| {model} | {kind} | {len(g)} | {g.cost.mean():.2f} | {fill:.4%} | {g.lost.mean():.2f} |")
    lines += ["", "| Model | Mean origin-specific expected cost | Mean origin-specific CVaR(.75) |","|---|---:|---:|"]
    for model,g in tables["simulation_risk"].groupby("model_version"):lines.append(f"| {model} | {g.expected.mean():.2f} | {g.cvar.mean():.2f} |")
    lines += ["", f"Canary routing: `{summary['canary']}`. One whole real resource pool across three dependent origin episodes, not independent item assignments. Canary JSON commits reference original forecast batches without duplicating forecast rows. Shadow publishes zero actions. Candidate local approval uses hard provenance/feasibility gates only; delayed evaluation outcomes never select origin forecasts or routes.",
        "Rollback restores the prior verified approved alias atomically and preserves all published bytes/actions. Isolated missing/tampered targets block rollback. Model failure recovers through a trusted history-only baseline after remaining gates; stale/missing history, infeasibility and incompatible fallbacks block publication for human review. Zero ordering is not claimed universally safe.","",
        "## Incidents and verification","",f"{summary['fault_cases']} predeclared fault cases: {summary['fault_dispositions']}. Incident severity totals: {summary['incident_counts']}; observed drift warnings {summary['observed_drift_warnings']}; injected shift warnings {summary['injected_shift_warnings']}. These are historical/synthetic exercises, not live incident rates.",
        f"{summary['simulation']['pool_runs']} complete pool runs / {summary['simulation']['daily_rows']} daily rows; {summary['simulation']['replays']} replay interventions and {summary['simulation_temporal_checks']} temporal/reset/hidden-demand checks. Solver statuses {summary['solver_statuses']}; max gap {summary['max_mip_gap']}. Four bootstrap worlds per origin imply CVaR(.75)=worst of four; no real risk/coverage guarantee.",
        f"Independent gates: {summary['verification']['independent_gates']}; source/model fingerprints and exactly58,327,370 warehouse rows unchanged. Runtime {summary['runtime_seconds']:.2f}s / peak process {summary['peak_memory_mib']:.2f}MiB. Environment {summary['environment']}.","",
        "## Reproduce / scope","", "`.venv/Scripts/python -m demandos.serving --root . --config configs/serving.toml`", "",
        "`data/serving/m5_h28/latest_verified.json` points to the successful versioned run. Registry, immutable batches, canary references, contracts, incident/retry/fault logs, Parquet forecasts/features/maturity/metrics and simulator ledgers remain ignored. Code/contract changes create another run namespace; completed batches are never overwritten.","",
        *["- "+s for s in summary["ineligible"]],"",
        "See docs/production_serving.md for exact contracts, limits and official references."]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
