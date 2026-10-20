"""Concise generated scenario report and cost/service comparison figure."""


def report(summary, tables, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = tables["metrics"]
    subset = metrics[metrics.cost_regime.eq("balanced") & metrics.action.eq("periodic_up_to")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, cohort in zip(axes, ["continuity", "sparse_stress"]):
        values = subset[subset.cohort.eq(cohort)].sort_values("method")
        ax.barh(values.method, values.cost_mean)
        ax.set_title(f"SIMULATED: {cohort}")
        ax.set_xlabel("One-order mismatch cost, hypothetical INR/case")
    (target/"figures").mkdir()
    fig.savefig(target/"figures/balanced_policy_cost.png", dpi=140)
    plt.close(fig)
    rows = ["# Season 18 — SCENARIO-BASED / SIMULATED", "",
        "M5 observed sales are treated as demand only for this stated experiment. No measured Walmart inventory savings.", "",
        f"25 series × 3 origins = 75 series-origins. Per policy/action: {summary['matched_cases_per_method_action']} matched lead/cost cases: {summary['matched_by_cohort']}.",
        f"{summary['table_rows']['decisions']:,} stock-target plans and {summary['table_rows']['evaluations']:,} one-order outcomes; {summary['future_invariance_checks']} future-data checks.", "",
        f"Weekday-aligned {summary['config']['inventory']['block_days']}-day blocks; {summary['config']['inventory']['lookback_days']}-day prefix; {summary['config']['inventory']['samples']} paths × 28 days. Independent-day sensitivity is separate.",
        "Assumed piecewise inverse-CDF and tail anchors combine marginal quantiles with randomized historical ranks. These are not uniquely identified joint forecasts.",
        "Season 17 rolling interval width scales the same October-frozen base paths heuristically. It is not assigned a service probability.", "",
        "At day E close, place one order. Existing on-hand serves days 1..L; on-order plus the new order arrive at day L+1 open. Finish at day L+R close; no reordering/backlog.",
        "Target cost assumes stock available for the whole period; executed one-order cost accounts for pre-receipt lost sales. The newsvendor target is not a claimed dynamic optimum.",
        "CSL counts whole cycles; fill rate pools units. Random lead outcomes are averaged over the configured hypothetical probabilities.", "",
        "Balanced costs, periodic order-up-to; means across identical series-origin/lead cases:", "",
        "| Cohort | Method | Cost | CSL | Unit fill | Order units |", "|---|---|---:|---:|---:|---:|"]
    for r in subset.itertuples():
        fill = f"{r.fill_rate:.4f}" if r.fill_rate is not None else "undefined"
        rows.append(f"| {r.cohort} | {r.method} | {r.cost_mean:.2f} | {r.csl:.4f} | {fill} | {r.order_mean:.2f} |")
    rows += ["", "All costs are hypothetical mismatch costs, not profit, purchasing cost totals, or actual losses.",
             "Own-distribution expected costs/regret use each method's assumed distribution. Common-historical target costs and common future sales permit matched comparisons.",
             "Clairvoyant one-order regret uses future demand and realized lead time only in the benchmark; it respects initial stock and pre-receipt irrecoverable shortages.", "",
             "Detailed cost/lead/regime/origin metrics, CDF supports/probabilities, cost curves, paths and invariance checks are in tables/.", "",
             "Warnings:", *[f"- {warning}" for warning in summary["warnings"]], "",
             "Reproduce: `python -m demandos.inventory --root . --config configs/inventory.toml`."]
    (target/"inventory_report.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
