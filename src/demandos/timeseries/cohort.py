"""Training-only Season 4 definitions; intentionally illustrative cohort selection."""

import pandas as pd

from dataclasses import replace
from demandos.eda.config import load_config
from demandos.eda.analysis import product_segments
from demandos.eda.queries import create_views, fetch_frame
from demandos.eda.statistics import intermittent_class


def training_labels(con, root, start, end):
    create_views(con, start.date(), end.date())
    series = fetch_frame(con, """select store_id, item_id, count(*) as observed_periods,
        sum(units_sold) as units_sold,
        count(*) filter (where units_sold > 0) as positive_periods,
        case when count(*) filter (where units_sold > 0) >= 2 then
            pow(stddev_pop(units_sold) filter (where units_sold > 0)
                / nullif(avg(units_sold) filter (where units_sold > 0),0),2) end as positive_cv2
        from eda_fact group by store_id, item_id order by store_id, item_id""")
    config = replace(load_config(root / "configs/eda.toml"), start_date=str(start.date()), end_date=str(end.date()))
    series["regime"] = [intermittent_class(r.observed_periods, r.positive_periods, r.positive_cv2, config)[0] for r in series.itertuples()]
    series["positive_fraction"] = series.positive_periods / series.observed_periods
    products = fetch_frame(con, """select item_id, sum(units_sold) as units_sold,
        sum(known_revenue) as known_revenue, sum(unknown_revenue_count) as unknown_revenue_count
        from eda_product_daily group by item_id order by item_id""")
    weekly = fetch_frame(con, (root / "sql/eda/weekly_product_statistics.sql").read_text(encoding="utf-8"))
    products = product_segments(products, weekly, config)
    series = series.merge(products[["item_id", "abc_xyz"]], on="item_id", validate="many_to_one")
    series["segment_fit_end"] = end
    return series, products, config


def select_cohort(labels, limit=14):
    ranked = labels.sort_values(["units_sold", "store_id", "item_id"], ascending=[False, True, True])
    selected = {}

    def add(frame, reason):
        if frame.empty:
            return
        row = frame.iloc[0]
        key = (row.store_id, row.item_id)
        if key in selected:
            selected[key].append(reason)
        elif len(selected) < limit:
            selected[key] = [reason]

    for regime in ("smooth", "intermittent", "erratic", "lumpy"):
        add(ranked[ranked.regime.eq(regime)], "regime:" + regime)
    add(ranked.sort_values(["positive_fraction", "store_id", "item_id"]), "sparsity_example")
    for cell in [a + x for a in "ABC" for x in "XYZ"]:
        add(ranked[ranked.abc_xyz.eq(cell)], "ABC_XYZ:" + cell)
    rows = [{"store_id": key[0], "item_id": key[1], "selection_reason": ";".join(reasons)} for key, reasons in selected.items()]
    return pd.DataFrame(rows).merge(labels, on=["store_id", "item_id"], validate="one_to_one")
