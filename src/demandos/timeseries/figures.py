"""Actual-data diagnostics; intervals are exploratory, never causal proof."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def figures(data, correlations, decompositions, metadata, output, config, split):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2})

    def save(fig, name, title):
        fig.suptitle(title + f"\nM5 training observations only, through {split['train_end'].date()}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .93))
        fig.savefig(output / name, dpi=140, metadata={"Software": "DemandOS Season 5"})
        plt.close(fig)
        paths.append(name)

    plot_ids = metadata.loc[metadata.kind.eq("store_item"), "series_id"].tolist()[:5]
    plot_ids += metadata.loc[metadata.kind.ne("store_item"), "series_id"].tolist()
    for index, series_id in enumerate(plot_ids):
        values = data[data.series_id.eq(series_id)].sort_values("date")
        fig, axes = plt.subplots(3, 2, figsize=(12, 9))
        axes[0,0].plot(values.date, values.units_sold, alpha=.45, lw=.6, label="Observed")
        axes[0,0].plot(values.date, values.rolling_mean, lw=1, label="Trailing mean incl. current")
        axes[0,0].set_ylabel("Recorded units"); axes[0,0].legend()
        axes[0,1].plot(values.date, values.rolling_variance, lw=.7)
        axes[0,1].set_title(f"Trailing {config.rolling_days}-day population variance"); axes[0,1].set_ylabel("Units squared")
        for variant, color in (("first_difference", "#2874a6"), ("seasonal_difference", "#a45229")):
            axes[1,0].plot(values.date, values[variant], lw=.6, alpha=.7, label=variant)
        axes[1,0].set_ylabel("Change in units"); axes[1,0].legend()
        for ax, coefficient in ((axes[1,1], "acf"), (axes[2,0], "pacf")):
            numbers = correlations[(correlations.series_id == series_id) & correlations.variant.eq("original")]
            numbers = numbers[numbers.lag.gt(0)]
            if numbers.empty:
                ax.text(.5, .5, "Skipped: see diagnostic reason table", ha="center", transform=ax.transAxes)
            else:
                ax.vlines(numbers.lag, 0, numbers[coefficient], lw=1)
                # statsmodels intervals center on coefficients; display relative-to-zero bands.
                ax.fill_between(numbers.lag, numbers[coefficient + "_low"] - numbers[coefficient], numbers[coefficient + "_high"] - numbers[coefficient], alpha=.15)
                for lag in (1, 7, 14, 28):
                    ax.axvline(lag, ls=":", lw=.7, color="#b14d3b")
                ax.set_ylim(-1.05, 1.05)
            ax.set(xlabel="Lag (days)", ylabel=coefficient.upper(), title="Original-series " + coefficient.upper())
        comparison = correlations[(correlations.series_id == series_id) & correlations.lag.gt(0)]
        for variant, group in comparison.groupby("variant"):
            axes[2,1].plot(group.lag, group.acf, lw=.8, label=variant)
        axes[2,1].set(xlabel="Lag (days)", ylabel="ACF", title="Before/after differencing (not model selection)")
        if len(comparison):
            axes[2,1].legend(fontsize=7)
        else:
            axes[2,1].text(.5, .5, "Skipped: no supported correlations", ha="center", transform=axes[2,1].transAxes)
        save(fig, f"diagnostic_{index+1:02d}.png", series_id)

    aggregate_ids = metadata.loc[metadata.kind.ne("store_item"), "series_id"].tolist()
    fitted = decompositions[decompositions.series_id.isin(aggregate_ids)] if len(decompositions) else decompositions
    if len(fitted):
        groups = list(fitted.groupby("series_id"))
        fig, axes = plt.subplots(len(groups), 3, figsize=(14, 2.7 * len(groups)), squeeze=False)
        for row, (name, values) in zip(axes, groups):
            for ax, column in zip(row, ("trend", "seasonal", "residual")):
                ax.plot(values.date, values[column], lw=.6)
                ax.set_title(name + ": " + column, fontsize=8); ax.set_ylabel("Units")
        save(fig, "stl_aggregates.png", "Additive STL period 7 — descriptive only, not historical features")

    fig, ax = plt.subplots(figsize=(11, 5))
    for series_id in aggregate_ids:
        frame = data[data.series_id.eq(series_id)]
        profile = frame.groupby(pd.to_datetime(frame.date).dt.dayofweek).units_sold.mean()
        ax.plot(profile.index, profile / frame.units_sold.mean(), marker=".", label=series_id)
    ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_ylabel("Weekday mean / series mean"); ax.legend(fontsize=8)
    save(fig, "weekly_profiles.png", "Selected aggregate weekday associations — not causal effects")

    fig, ax = plt.subplots(figsize=(11, 4))
    origin = split["train_end"]
    ax.axvspan(origin - pd.Timedelta(days=28), origin, color="#76a98b", alpha=.25, label="Available history")
    ax.axvspan(origin, origin + pd.Timedelta(days=28), color="#cfa65f", alpha=.2, label="Future actuals unavailable")
    ax.axvline(origin, color="black", label="Forecast origin")
    for row, horizon in enumerate(config.horizons):
        target = origin + pd.Timedelta(days=horizon)
        ax.scatter(target, row, color="#286584")
        ax.annotate(f"h={horizon}", (target, row), xytext=(4, 4), textcoords="offset points")
    h7_row = list(config.horizons).index(7)
    ax.scatter(origin + pd.Timedelta(days=6), h7_row, marker="x", s=80, color="red", label="h=7 target lag-1: LEAK")
    ax.scatter(origin, h7_row, marker="o", color="green", label="Corrected input: sales at origin")
    ax.set_yticks([]); ax.legend(fontsize=8, loc="upper left")
    save(fig, "origin_boundary.png", "Direct-horizon inputs share one information cutoff; no future recorded prices")
    return paths
