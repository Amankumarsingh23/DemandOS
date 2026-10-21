"""Concise, measured report; no actual-business cost claims."""
def write_report(summary,tables,path):
    front=tables["risk_frontier"]
    operating=tables["operating_risk"]
    paired=tables["paired_operating"]
    lines=["# Season 21: finite-scenario risk optimization", "", "**SIMULATED / SCENARIO-ASSUMED.** M5 recorded sales are a demand proxy. No verified uncensored demand, actual supplier costs, disruptions or business savings.", "",
           f"{summary['pool_runs']} committed pool runs; {summary['series_runs']} series runs; {summary['daily_rows']:,} stateful daily rows. 25 series (13 continuity / 12 sparse stress), three origins, 75 unique recorded-sales paths / 2,100 target dates before replication.","",
           "## Objective and uncertainty definitions","",
           "Mean; E[L]+lambda CVaR with lambda .5/2; minimax; identical-feasible-set scenario-oracle minimax regret; finite two-family probability ambiguity p(shock) in [.05,.20]. Static origin diagnostics also solve hard CVaR ceilings 0/10,000/100,000/1,000,000 at alpha .75/.90. Infeasibility is retained, never relaxed.","",
           "L is Cu/Co protection-window mismatch plus separately recorded soft fairness preferences. Operating costs instead use the original procurement/fee/holding/shortage ledger. These objectives are not interchangeable.","",
           "Planning support: four prefix-only normal 14-day and four assumed 1.5x/21-day adverse joint paths; nominal adverse mass .10, shifted-probability sensitivity .20. Independent per-series blocks coupled by common family; no inferred true joint distribution. Coordinate box bounds are diagnostics, not a continuous robust counterpart.","",
           "Evaluation: observed-sales proxy baseline, 1.5x surge, common seven-day 3x burst, lead 14 instead of 7, receiving closure days 8–14, joint surge/closure, changed planning probabilities. Eight independently keyed prefix-bootstrap complete worlds per origin for baseline/joint; no daily or cross-series pseudo-replication. Empirical CVaR at .75 averages two of these eight worlds and has no coverage guarantee.","",
           "Closures have zero receiving capacity during the closed window and unlimited capacity afterwards. Each whole due outstanding order is re-timed to day 15, with its quantity, identity and prepaid commitment intact; no partial delivery/cancellation/refund. Horizon supplier authorizations use the worst assumed multiplier .75 for every method (nominal multiplier 1); this robust hard constraint differs from realized delivery-delay recourse.","",
           "## Origin risk comparisons","", "Costs below are assumed forecast mismatch, averaged across three origins at alpha .75; not operating costs.","",
           "| Method | Expected | CVaR | Worst | Max regret | Ordered units |","|---|---:|---:|---:|---:|---:|"]
    for method,g in front[front.alpha.eq(.75)&front.method_id.ne("ceiling")].groupby("method_id"):
        lines.append(f"| {method} | {g.expected.mean():.2f} | {g.cvar.mean():.2f} | {g.worst.mean():.2f} | {g.max_regret.mean():.2f} | {g.ordered_units.mean():.1f} |")
    lines += ["", "## Stateful simulated operating comparisons","", "Matched complete-world cohort totals averaged across origins; empirical risk uses eight complete bootstrap worlds per origin/family/cohort. Fill is demand-weighted within each such panel.","",
              "| Family | Cohort | Method | Mean cost | Operating CVaR | Fill |","|---|---|---|---:|---:|---:|"]
    for key,g in operating.groupby(["scenario_family","cohort","method_id"]):
        lines.append(f"| {' | '.join(key)} | {g.expected.mean():.2f} | {g.cvar.mean():.2f} | {g.fill.mean():.2%} |")
    lines += ["", "## Matched observed-path differences","", "Differences versus mean under the same origin/world/resources/economics. Positive cost difference costs more; positive fill difference serves more. Pool-origin/cohort is the comparison unit, not individual days.","",
              "| Family | Method | Cost difference | Fill difference |","|---|---|---:|---:|"]
    for key,g in paired[paired.study.eq("main")&paired.cohort.eq("ALL")].groupby(["scenario_family","method_id"]):
        lines.append(f"| {' | '.join(key)} | {g.cost_difference_vs_mean.mean():.2f} | {g.fill_difference_vs_mean.mean():.2%} |")
    lines += ["", "## Verification and limits","",
              f"{summary['verification']['independent_gates']} independent persisted-output gates; {summary['origin_mutations']} forecast/scenario future mutations; {summary['temporal_checks']} full-pool temporal/hidden-demand checks; {summary['verification']['retiming_events']} conserved order re-timings.",
              f"Live statuses {summary['solver_statuses']}; static statuses {summary['static_statuses']}; maximum recorded live MIP gap {summary['max_mip_gap']}. {summary['presolve_recoveries']} presolve errors recovered on the identical model within the original time budget; all attempts are logged. Runtime {summary['runtime_seconds']:.2f}s; peak RAM {summary['peak_memory_mib']:.2f} MiB. Versions {summary['versions']}.","",
              "Hard budget, persistent supplier, procurement flow, store quotas and packs/MOQ pass; positive-request store shares and soft preference slacks are in fairness_slacks.parquet. Procurement flow is not a physical on-hand storage constraint. No actual M5 shelf capacities exist; on-hand inventory is conserved, not claimed to satisfy unknown physical storage limits.",
              "The Pareto table contains nondominated points only among tested allocations. Integer weighted sums can miss unsupported efficient points. DRO is exact only over the two specified probability endpoints on fixed finite support. Worst-case protection does not cover unmodeled distributions or real disruptions. System CVaR does not guarantee group service.","",
              "Reproduce: `.venv/Scripts/python -m demandos.risk`. Source/model hashes and warehouse totals are rechecked before publication. Generated outputs are Git-ignored under data/risk/m5_h28; no earlier models are retrained."]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
