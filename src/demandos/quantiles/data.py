"""Bounded reuse of real origin-safe tabular and history-only sequence contracts."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd

from demandos.boosting.config import load_config as load_boosting
from demandos.boosting.dataset import build
from demandos.features.pipeline import load_keys
from demandos.features.validation import generate
from demandos.nbeats.config import load_config as load_neural
from demandos.neural.sequences import Panel
from demandos.intermittent.diagnostics import statistics
from demandos.reconciliation.hierarchy import build as build_hierarchy, observed_matrix
from demandos.data.pipeline import fingerprint
from .baseline import nonadditivity

KEYS = ["store_id", "item_id"]


def prepare(con, root, config):
    boost = load_boosting(root/"configs/boosting.toml")
    data = build(con, root, boost)
    continuity = data["evaluation_cohort"][KEYS].copy().assign(cohort="continuity")
    existing = set(map(tuple, continuity[KEYS].to_numpy()))
    available = data["training_cohort"].copy()
    available = available[[tuple(v) not in existing for v in available[KEYS].to_numpy()]]
    stress = available.sort_values(["positive_fraction", *KEYS]).head(config["stress_series"])[KEYS].copy().assign(cohort="sparse_stress")
    if len(stress) != config["stress_series"] or len(continuity) != 13:
        raise ValueError("Required continuity/sparse cohorts unavailable")
    selected = pd.concat([continuity, stress], ignore_index=True)
    eval_keys = load_keys(con, data["origins"][0], selected[KEYS])
    con.register("feature_keys", eval_keys)
    inference = generate(con, data["origins"], data["feature_config"], data["mapping"]).features
    metadata = load_keys(con, data["origins"][0], pd.concat([data["training_cohort"][KEYS], selected[KEYS]]).drop_duplicates())
    con.register("quantile_all_keys", metadata[KEYS])
    daily = con.execute("""select f.store_id,f.item_id,f.date,f.units_sold,f.sell_price
        from main.fact_daily_sales f join quantile_all_keys k using(store_id,item_id)
        order by store_id,item_id,date""").fetchdf()
    calendar = con.execute("select * from main.dim_calendar order by date").fetchdf()
    neural = replace(load_neural(root/"configs/nbeats.toml"), epochs=config["epochs"], patience=config["patience"])
    histories = {key: frame.reset_index(drop=True) for key, frame in daily.groupby(KEYS, sort=True)}
    diagnostics = []
    for origin in data["origins"]:
        for row in selected.itertuples():
            diagnostics.append({"store_id": row.store_id, "item_id": row.item_id, "origin_date": origin,
                                "cohort": row.cohort, **statistics(histories[(row.store_id, row.item_id)], origin, window=365)})
    data.update(boost=boost, inference=inference, evaluation_keys=eval_keys, selected=selected, metadata=metadata,
                histories=histories, daily=daily, calendar=calendar, panel=Panel(daily, calendar, metadata),
                neural_config=neural, origin_diagnostics=pd.DataFrame(diagnostics))
    return data


def hierarchy_diagnostic(root, data, sources, baseline_days):
    directory = root/"data/reconciliation/m5_cohort_h28"
    summary = json.loads((directory/"summary.json").read_text())
    if summary["status"] != "passed" or summary["source_fingerprints"] != sources:
        raise ValueError("Season 15 hierarchy source provenance mismatch")
    bottom = pd.read_parquet(directory/"tables/cohort_bottom_map.parquet")
    hierarchy = build_hierarchy(bottom)
    current = build_hierarchy(data["metadata"])
    pd.testing.assert_frame_equal(hierarchy.bottom, current.bottom)
    dates, observed = observed_matrix(data["daily"], hierarchy)
    records = []
    for origin in data["origins"]:
        take = (dates <= origin) & (dates > origin-pd.Timedelta(days=baseline_days))
        history = observed[:, take]
        marginals = np.quantile(history, [.1, .5, .9], axis=1, method="inverted_cdf").sum(axis=1)
        total = np.quantile(np.asarray(hierarchy.S[hierarchy.rows("TOTAL")]@history).ravel(), [.1, .5, .9], method="inverted_cdf")
        for q, naive, actual in zip([.1, .5, .9], marginals, total):
            records.append({"origin_date": origin, "level": "COHORT_TOTAL", "bottom_series": len(bottom),
                            "quantile": q, "sum_historical_marginal_quantiles": naive,
                            "direct_historical_total_quantile": actual,
                            "interpretation": "Temporal empirical diagnostic on past observed cohort sales; not calibrated future aggregate quantiles"})
    return pd.DataFrame(records), {"toy": nonadditivity(), "hierarchy_summary": fingerprint(directory/"summary.json"),
                                   "bottom_map": fingerprint(directory/"tables/cohort_bottom_map.parquet")}
