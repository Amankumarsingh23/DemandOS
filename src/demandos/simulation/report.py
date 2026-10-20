"""Generated comparisons and sampled inventory trajectories, all hypothetical."""
import pandas as pd


def report(summary, tables, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = tables["runs"]
    base = tables["metrics"]
    chosen = base[base.experiment_id.eq("main") & base.cost_scenario_id.eq("balanced") & base.lead_scenario_id.eq("fixed_7")]
    lines = ["# Season 19 — SCENARIO-BASED sequential inventory simulation", "",
        "Observed M5 sales are treated as simulated customer demand. No measured Walmart cost savings, historical stockouts or supplier behavior.", "",
        f"75 recorded-sales paths / 2,100 unique daily targets; {summary['main_runs']:,} main runs and {summary['main_daily_rows']:,} main daily rows.",
        f"{summary['temporal_checks']} pre-k/hidden-demand checks, {summary['origin_checks']} frozen-origin checks, and all daily/queue conservation gates pass.", "",
        "Day OPEN receipts -> FIFO backlog -> current demand -> unmet lost/backlogged -> post-demand IP -> policy -> queue at t+L -> daily costs.",
        "Procurement at ORDER; initial open orders paid previously; terminal pipeline paid with no credit. Default terminal stock value zero.",
        "Immediate fill uses NEW demand only; daily no-shortage fraction is NOT cycle service level. All-zero demand fill is undefined.",
        "Supplier shocks use issue-calendar-day keys shared by policy/cost, never order sequence. Targets remain frozen at the original issue origin.", "",
        "Balanced operating costs, fixed seven-day lead, default lost sales (two supplier replications; fixed-lead duplicates identified):", "",
        "| Cohort | Source | Rule | Cost/run | Immediate fill | Daily no shortage | Orders | End stock | End pipeline |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in chosen.itertuples():
        lines.append(f"| {r.cohort} | {r.source} | {r.rule} | {r.cost_mean:.2f} | {r.demand_weighted_fill:.4f} | {r.daily_no_shortage_fraction:.4f} | {r.orders_mean:.2f} | {r.ending_stock_mean:.2f} | {r.ending_on_order_mean:.2f} |")
    lines += ["", "Nine sensitivity studies are separate from main: backlog, starting stock 0/100, open orders 0, review 3 days, weekend +2-day delay, terminal salvage credit, historical block and independent bootstrap demand.",
        "Warm-up studies use 0/28/56 burn-in days followed by the same last 56 synthetic days, the same supplier calendar keys and review phase. They never initialize from future M5 outcomes.",
        "Paired cost differences are saved daily and per run. Standard errors average supplier replications within series-origin clusters; days are not independent replications.", "",
        "Warnings:", *["- "+w for w in summary["warnings"]], "",
        "Reproduce: `python -m demandos.simulation --root . --config configs/simulation.toml`."]
    (directory/"simulation_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (directory/"figures").mkdir()
    # Bounded example selected by original identifiers, never by performance.
    first = runs[runs.experiment_id.eq("main")].sort_values(["series_id", "origin_date"]).iloc[0]
    ledger = pd.read_parquet(directory/"tables/daily_ledger.parquet", filters=[("series_id", "==", first.series_id),
        ("origin_date", "==", first.origin_date), ("experiment_id", "==", "main"),
        ("cost_scenario_id", "==", "balanced"), ("lead_scenario_id", "==", "fixed_7"), ("replication_id", "==", 0)])
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, layout="constrained")
    for name, group in ledger.groupby("policy_id"):
        group = group.sort_values("day_index")
        axes[0].plot(group.simulation_date, group.ending_stock, label=name.split(":", 1)[1])
        axes[1].plot(group.simulation_date, group.ordered)
        axes[2].plot(group.simulation_date, group.lost_units.cumsum())
    for ax, label in zip(axes, ["Closing stock", "Orders placed", "Cumulative simulated lost units"]): ax.set_ylabel(label)
    axes[0].legend(fontsize=7, ncol=3); axes[0].set_title("SIMULATED: first identifier, frozen-origin policies")
    fig.savefig(directory/"figures/inventory_trajectories.png", dpi=130); plt.close(fig)
