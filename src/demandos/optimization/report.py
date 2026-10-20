"""Generated SCENARIO-ASSUMED results; operating costs differ from solver objectives."""
def report(summary,metrics,directory):
    chosen=metrics[metrics.study.eq("main")&metrics.cost_scenario.eq("balanced")&metrics.budget_factor.eq(.7)]
    lines=["# Season 20 — SIMULATED shared supply optimization", "",
        f"75 real recorded-sales paths, 2,100 unique targets; {summary['pool_runs']} coordinated pool runs, {summary['series_runs']} series runs, {summary['daily_rows']} daily rows.",
        "One shared SELECTED_COHORT_DC pool; scenario supplier groups follow department metadata. These are not actual Walmart resource pools.",
        "Budget and supplier units persist for 28 days; procurement-flow volume and store receiving caps apply per review epoch. No delivery refunds.",
        "Quantile/historical scenarios are frozen at E, with independent series bootstraps and a shared hypothetical lead-time draw. No known joint distribution is asserted.",
        "Mismatch MILP uses Cu/Co only; deterministic MILP uses a labeled linear benefit. Realized operating cost includes procurement/fees in the original Season 19 timeline.", "",
        "Balanced cost, medium budget, fixed-seven-day lead:", "",
        "| Cohort | Method | Cost/run | Immediate fill | Lost units/run | Orders/run |", "|---|---|---:|---:|---:|---:|"]
    for r in chosen.itertuples():lines.append(f"| {r.cohort} | {r.method} | {r.cost_mean:.2f} | {r.fill:.4f} | {r.lost_mean:.2f} | {r.orders_mean:.2f} |")
    lines += ["", "The unconstrained reference may overspend and is not a feasible constrained competitor. Low cost alone can mean low service.",
        "LP marginals are local continuous-relaxation diagnostics, not MILP shadow prices. Integer budget perturbation and EVPI are separate exact toy experiments.",
        "Fairness sensitivities use explicitly penalized soft shares/disparity; all physical/resource constraints stay hard.", "",
        "Warnings:",*["- "+x for x in summary["warnings"]], "",
        "Reproduce: `python -m demandos.optimization --root . --config configs/optimization.toml`."]
    (directory/"optimization_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
