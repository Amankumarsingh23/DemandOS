"""Measured results with explicit model-world identification limits."""
def write_report(summary,tables,path):
    repeated=tables["ope_repeated_summary"]
    values=tables["oracle_policy_values"]
    controlled=tables["controlled_summary"]
    risk=tables["trajectory_operating_risk"]
    contrast=tables["trajectory_intervals"]
    lines=["# Season 22: causal and off-policy evaluation","",
        "**SIMULATED / SYNTHETIC LOGGING / MODEL-WORLD.** M5 provides recorded sales, not Walmart policy assignments, logging propensities, uncensored demand, inventory decisions or causal financial effects.","",
        f"25 series (13 continuity / 12 stress), three retrospective origins. {summary['contexts']} predecision pool contexts, {summary['one_step_context_worlds']} context-world units, {summary['logging_rows']} log rows across 32 assignment replications. The repeated rows are not independent demand worlds.","",
        "## Track A: one origin-issued feasible allocation","",
        "Action 0: mean-mismatch whole-pool vector. Action 1: mean+2CVaR(.75) vector. One decision at origin, 14-day no-further-order rollout, reward = negative purchase/activation/holding/lost-sales cost. Existing pipeline prepaid, new vector procurement charged once. No unchosen reward enters nuisance fitting.",
        "Logger P(A=1|X)=.2+.6*sigmoid(predecision risk score). Jan31 selected rewards mature Feb14 before Feb28/Mar27 held-out evaluation; 96 development / 192 evaluation units per replication. Ridge, bounded forest and deliberately misspecified intercept model; separate true vs estimated propensity results. The source periods were repeatedly studied in earlier seasons, not prospective unseen data.","",
        "Oracle values are exact over the finite context/eight-world simulator table; not actual business causal values.","",
        "| Target | Oracle reward |","|---|---:|"]
    for row in values[values.population.eq("pooled")].itertuples():lines.append(f"| {row.target} | {row.oracle_value:.4f} |")
    lines += ["", "Repeated-assignment means below use the forest outcome model and known logging propensities. Full estimates/errors/CI/sensitivities for all models, propensity sources and targets are in Parquet.","",
              "| Target | IPS | SNIPS | DM | DR |","|---|---:|---:|---:|---:|"]
    selected=repeated[repeated.model.eq("forest")&repeated.propensity_source.eq("known")&repeated.population.eq("pooled")]
    for target,g in selected.groupby("target"):
        m=g.set_index("estimator").estimate_mean
        lines.append(f"| {target} | {m['ips']:.4f} | {m['snips']:.4f} | {m['dm']:.4f} | {m['dr']:.4f} |")
    lines += ["", "Errors below are empirical assignment-replication bias / RMSE versus the same finite oracle; they are not real-world bias estimates.", "",
              "| Target | Estimator | Bias | Std | RMSE |", "|---|---|---:|---:|---:|"]
    for row in selected.itertuples():
        lines.append(f"| {row.target} | {row.estimator} | {row.bias:.4f} | {row.std:.4f} | {row.rmse:.4f} |")
    overlap=tables["overlap_diagnostics"]
    overlap=overlap[overlap.replication_id.eq(0)&overlap.model.eq("forest")&overlap.propensity_source.eq("known")&overlap.population.eq("pooled")]
    lines += ["", "Overlap, first assignment replication / 192 held-out units. ESS is weight ESS, not the number of independent real pools.", "",
              "| Target | Propensity min/max | Assignment counts [0,1] | ESS | Max weight | Support coverage |", "|---|---|---|---:|---:|---:|"]
    for row in overlap.itertuples():
        lines.append(f"| {row.target} | {row.min_logging_probability:.6f}/{row.max_logging_probability:.6f} | {row.assignment_counts} | {row.ess:.4f} | {row.max_weight:.4f} | {row.target_support_coverage:.0%} |")
    lines += ["", "CVaR target: chronological nuisance/propensity sensitivities, means across 32 assignment replications. Estimated propensities do not replace the known logger for identification.", "",
              "| Outcome model | Propensity | DM | DR | DR bias | DR RMSE |", "|---|---|---:|---:|---:|---:|"]
    sensitivities=repeated[repeated.target.eq("cvar")&repeated.population.eq("pooled")]
    for key,g in sensitivities.groupby(["model","propensity_source"]):
        m=g.set_index("estimator")
        lines.append(f"| {' | '.join(key)} | {m.loc['dm','estimate_mean']:.4f} | {m.loc['dr','estimate_mean']:.4f} | {m.loc['dr','bias']:.4f} | {m.loc['dr','rmse']:.4f} |")
    ate=tables["one_step_ate_att"].set_index("population").loc["pooled"]
    lines += ["", f"One-shot finite-world oracle reward ATE {ate.oracle_ate:.4f}; actual-logged-treated oracle ATT {ate.oracle_att_actual_logged_treated:.4f}. First-replication naive difference {ate.naive_difference:.4f} / normalized IPW ATT {ate.normalized_ipw_att:.4f}; different targets and finite assignment noise preclude equating these estimates."]
    lines += ["", "## Controlled structural demonstration","",
              "Y(a)=10X+2a+epsilon, X~Bernoulli(.5), true mu(1|X)=.1+.8X. True population value(always1)=7 and ATE=ATT=2. Correct nuisance functions are analytic structural conditional means/probabilities; wrong functions omit X. These are separate synthetic units, not M5/Walmart outcomes.","",
              "| Correct propensity | Correct outcome | DR bias | Std | DR RMSE | Nominal Wald coverage |","|---|---|---:|---:|---:|---:|"]
    for row in controlled[controlled.estimator.eq("dr")].itertuples():
        lines.append(f"| {row.correct_propensity} | {row.correct_outcome} | {row.bias:.4f} | {row.std:.4f} | {row.rmse:.4f} | {row.coverage:.3f} |")
    c=tables["confounding"]
    lines += ["",f"Confounding: naive treated-control {c.naive_difference.mean():.4f}; randomized benchmark {c.randomized_benchmark.mean():.4f}; known IPW {c.known_ipw.mean():.4f}; correct DR {c.correct_dr.mean():.4f}; omitted-confounder DR {c.omitted_confounder_dr.mean():.4f}; truth 2.",
              "Double robustness is conditional/asymptotic, not a finite-sample guarantee. Nominal interval coverage under misspecification is a diagnostic, not a valid coverage promise. Missing support returns NONIDENTIFIABLE; clipping never repairs it.","",
              "## Track B: complete 28-day controller interventions","",
              "Fresh initial stock/pipeline, budgets, queues and ledgers for each policy; identical exogenous worlds. This is a model-world full-policy contrast, not one-step IPS/DR on daily rows.","",
              "The following costs/tails and aggregate fill rates are averaged over three origin-specific eight-world distributions; CVaR is not pooled across origins. Paired fill below instead averages individual world-level fill differences.", "",
              "| Family | Policy | Mean full operating cost | CVaR(.75) | Fill |","|---|---|---:|---:|---:|"]
    for key,g in risk[risk.cohort.eq("ALL")].groupby(["scenario_family","policy_id"]):
        lines.append(f"| {' | '.join(key)} | {g.expected.mean():.4f} | {g.cvar.mean():.4f} | {g.fill.mean():.3%} |")
    lines += ["", "Paired differences are CVaR policy minus mean policy; negative cost is favorable, positive fill is favorable.","",
              "| Family | Metric | Paired difference | World-block interval |","|---|---|---:|---|"]
    for row in contrast[contrast.world_kind.eq("prefix_bootstrap")&contrast.cohort.eq("ALL")].itertuples():
        lines.append(f"| {row.scenario_family} | {row.metric} | {row.estimate:.6f} | [{row.lower:.6f}, {row.upper:.6f}] |")
    lines += ["", "Observed-sales proxy paths (three dependent historical origin pairs): no real confidence interval is available.", "",
              "| Family | Cost difference | Fill difference | Lost units difference |", "|---|---:|---:|---:|"]
    for family,g in contrast[contrast.world_kind.eq("observed_sales_proxy")&contrast.cohort.eq("ALL")].groupby("scenario_family"):
        m=g.set_index("metric").estimate
        lines.append(f"| {family} | {m['operating_cost']:.4f} | {m['fill']:.6f} | {m['lost_units']:.4f} |")
    lines += ["", "## Uncertainty, sequential weights and limits","",
              "One-step intervals resample eight complete world panels, carrying all context strata/origins together, conditional on fitted nuisances. Paired controller intervals use the same world blocks. Few clusters, overlapping source histories and one real resource pool mean no reliable real-world coverage or significance claim. Controlled IID synthetic experiments report separate nominal Wald coverage. 2,100 daily targets are never counted as independent policy experiments.",
              "Sequential diagnostics use a separate bounded stateful synthetic logger: complete-trajectory ratios computed in log space for 1/7/14/28 steps. A ratio of 2 for 28 steps equals 268,435,456. No sample matches and structural missing support are distinct statuses; one-step DR is not applied to controller daily rows.","",
              "| Target P(1) | Horizon | Matched/2048 | ESS | Weight variance | Status |", "|---|---:|---:|---:|---:|---|"]
    for row in tables["sequential_diagnostics"].itertuples():
        lines.append(f"| {row.target_probability:.2f} | {row.horizon} | {row.matching} | {row.ess:.4f} | {row.weight_variance:.4f} | {row.status} |")
    lines += ["",
              f"Integrity: {summary['verification']['independent_gates']} saved-output gates, {summary['future_source_checks']} source/context mutations, {summary['future_context_checks']} full-vector/propensity perturbations, {summary['temporal_checks']} reset/temporal checks. Source/model fingerprints and 58,327,370 warehouse observations unchanged.",
              f"Runtime {summary['runtime_seconds']:.2f}s; peak {summary['peak_memory_mib']:.2f} MiB. Catalogue statuses {summary['catalog_solver_statuses']}; controller statuses {summary['trajectory_solver_statuses']}; max gap {summary['max_mip_gap']}. Versions {summary['versions']}.","",
              "Reproduce: `.venv/Scripts/python -m demandos.causal`. Ignored artifacts: data/causal/m5_h28. Oracle outcomes live separately under oracle/, estimator inputs under tables/synthetic_logs.parquet. See docs/causal_evaluation.md and docs/causal_experiment_design.md for assumptions and the future real randomized trial design; no trial was deployed."]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
