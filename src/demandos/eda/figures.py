"""Export actual-data figures from bounded summaries; no generative imagery."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .statistics import prior_average


def generate_figures(frames, directory, config, window_label):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2, "figure.dpi": 120})
    written = []

    def save(fig, name, title):
        fig.suptitle(title + "\nM5 recorded sales | " + window_label, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = directory / (name + ".png")
        fig.savefig(path, dpi=150, metadata={"Software": "DemandOS Season 4"})
        plt.close(fig)
        written.append(path.name)

    daily, weekly = frames["daily"], frames["weekly"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(daily.date, daily.units_sold, alpha=0.4, lw=0.6, label="Daily units")
    axes[0].plot(daily.date, prior_average(daily.units_sold, config.moving_average_days), label=f"Prior {config.moving_average_days}-day mean")
    axes[0].set_ylabel("Units / day"); axes[0].legend()
    complete = weekly[weekly.observed_days.eq(7)]
    axes[1].plot(complete.week_start, complete.units_sold, lw=1)
    axes[1].set_ylabel("Units / complete week")
    save(fig, "daily_weekly_trend", "Daily and complete-week sales trajectory")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    days = frames["weekday"]
    axes[0].bar(days.weekday_number, days.weekday_index)
    axes[0].set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[0].axhline(1, color="black", lw=0.8); axes[0].set_ylabel("Mean daily units / overall daily mean")
    months = frames["month_of_year"]
    axes[1].bar(months.month_number, months.month_index)
    axes[1].set_xticks(range(1, 13)); axes[1].set_xlabel("Calendar month (pooled years)")
    axes[1].set_ylabel("Monthly mean / overall daily mean")
    axes[1].axhline(1, color="black", lw=0.8)
    for year, values in frames["weekday_by_year"].groupby("year"):
        axes[2].plot(values.weekday_number, values.weekday_index, marker=".", label=str(year))
    axes[2].set_xticks(range(7), ["M", "T", "W", "T", "F", "S", "S"])
    axes[2].set_title("Within-year weekday index"); axes[2].legend(fontsize=8)
    axes[2].set_ylabel("Mean daily units / within-year daily mean")
    save(fig, "seasonal_profiles", "Descriptive calendar patterns; not causal effects")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    stores = frames["store"].sort_values("units_sold", ascending=False)
    axes[0].bar(stores.store_id, stores.units_sold / 1e6)
    axes[0].set_ylabel("Total units (millions)"); axes[0].tick_params(axis="x", rotation=60)
    axes[1].bar(stores.store_id, stores.units_per_observed_series_day)
    axes[1].set_ylabel("Units / observed store-item-day"); axes[1].tick_params(axis="x", rotation=60)
    categories = frames["category"].sort_values("units_sold", ascending=False)
    axes[2].bar(categories.cat_id, categories.units_sold / 1e6)
    axes[2].set_ylabel("Category units (millions)")
    save(fig, "category_store_comparison", "Sales volume and observed-series-day normalization")

    products = frames["product_segments"]
    eligible = products[products.revenue_share.notna()]
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(eligible):
        ax.plot(np.arange(1, len(eligible) + 1) / len(eligible) * 100, eligible.cumulative_revenue_share.astype(float) * 100)
    ax.axhline(config.abc_a * 100, ls="--", color="#b66b00", label="A cutoff")
    ax.axhline(config.abc_b * 100, ls=":", color="#555555", label="B cutoff")
    ax.set(xlabel="Cumulative eligible products (%)", ylabel="Cumulative known-price revenue (%)", xlim=(0, 100), ylim=(0, 102))
    ax.legend(); save(fig, "revenue_pareto", "Revenue concentration; products ordered by revenue then item ID")

    series = frames["store_item"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(series.zero_fraction.dropna(), bins=np.linspace(0, 1, 31), color="#347b96")
    ax.set(xlabel="Zero-sales days / observed days", ylabel="Store-item series")
    save(fig, "zero_sales_distribution", "Zero-sales frequency; zeros do not establish stockouts")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"smooth": "#287d46", "intermittent": "#2874a6", "erratic": "#c87b12", "lumpy": "#a94358"}
    for regime, values in series[series.regime.ne("unclassified")].groupby("regime"):
        ax.scatter(values.adi, values.positive_cv2, s=6, alpha=0.28, label=f"{regime} ({len(values):,})", color=colors[regime], rasterized=True)
    ax.axvline(config.adi_boundary, color="black", ls="--", lw=0.8)
    ax.axhline(config.positive_cv2_boundary, color="black", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylim(bottom=0)
    ax.set(xlabel="ADI: observed days / positive-sales days (log)", ylabel="CV squared of positive sales sizes (symlog)")
    ax.legend(); save(fig, "intermittent_regimes", "Heuristic sales-occurrence regimes at store-item grain")

    matrix = frames["abc_xyz_matrix"]
    counts = matrix.products.to_numpy().reshape(3, 3)
    shares = matrix.revenue_share.to_numpy(dtype=float).reshape(3, 3)
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(counts, cmap="Blues", vmin=0)
    ax.set_xticks(range(3), list("XYZ")); ax.set_yticks(range(3), list("ABC"))
    ax.set(xlabel="Weekly XYZ variability", ylabel="ABC known-price revenue")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{counts[i,j]:,} products\n{shares[i,j]:.1%} revenue", ha="center", va="center", color="white" if counts[i,j] > counts.max() * .6 else "black")
    fig.colorbar(image, ax=ax, label="Product count")
    save(fig, "abc_xyz_heatmap", "Product-level ABC-XYZ; unsupported labels reported separately")

    anomalies = frames["selected_anomalies"]
    groups = list(anomalies.groupby(["store_id", "item_id"], sort=True))
    fig, axes = plt.subplots(max(1, len(groups)), 1, figsize=(12, max(4, 2.4 * len(groups))), squeeze=False)
    for ax, ((store, item), values) in zip(axes.flat, groups):
        ax.plot(values.date, values.units_sold, lw=.55, alpha=.6, label="Recorded units")
        ax.plot(values.date, values.prior_average, lw=.9, label="Prior-day rolling mean")
        for column, marker, color in (("iqr_flag", ".", "#bb5544"), ("robust_flag", "x", "#693b88")):
            flagged = values[values[column]]
            ax.scatter(flagged.date, flagged.units_sold, s=10, marker=marker, color=color, label=column)
        ax.set_title(f"{store} / {item} | {values.selection.iloc[0]}", fontsize=9)
        ax.set_ylabel("Units / day"); ax.legend(fontsize=7, loc="upper right")
    save(fig, "selected_anomalies", "Selected high/middle/low-volume and regime examples; no rows deleted")

    decompositions = list(frames["decomposition"].groupby("series", sort=True))
    if decompositions:
        fig, axes = plt.subplots(len(decompositions), 3, figsize=(15, 2.8 * len(decompositions)), squeeze=False)
        for row_axes, (name, values) in zip(axes, decompositions):
            row_axes[0].plot(values.date, values.units_sold, alpha=.35, lw=.5)
            row_axes[0].plot(values.date, values.trend, lw=1)
            row_axes[0].set_title(f"{name}: observed and trend")
            profile = values.assign(weekday=pd.to_datetime(values.date).dt.dayofweek).groupby("weekday").seasonal.mean()
            row_axes[1].bar(profile.index, profile.values); row_axes[1].set_title("Mean fitted weekly component")
            row_axes[1].set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=45)
            row_axes[2].plot(values.date, values.residual, lw=.5); row_axes[2].set_title("Remainder (not proven errors)")
            for ax in row_axes:
                ax.set_ylabel("Units / day")
        save(fig, "additive_decomposition", "Robust additive STL, period 7; descriptive fit uses the whole selected window")
    return written
