"""Real frozen-state future mutation, deletion and retrospective-price gates."""

import numpy as np
import pandas as pd

from demandos.features.validation import generate
from demandos.boosting.dataset import eligible_rows
from demandos.boosting.models import TARGET
from demandos.nbeats.dataset import TargetDataset
from demandos.neural.sequences import examples, Panel
from .neural import predict
from .baseline import historical


def checks(con, data, origin, trees, model, neural, baseline_days, fit_cutoff):
    origin = pd.Timestamp(origin)
    day = origin.date().isoformat()
    before = data["inference"].loc[data["inference"].origin_date.eq(origin)].reset_index(drop=True)
    con.register("feature_keys", data["evaluation_keys"])
    rows = examples(data["evaluation_keys"], [origin], neural)
    original = TargetDataset(data["panel"], rows, neural)
    first_neural = predict(model, original, neural)
    tree_before = [tree.predict(before).raw_prediction.to_numpy() for tree in trees]
    base_before = [historical(data["histories"][(r.store_id, r.item_id)], origin, baseline_days)[0] for r in rows.itertuples()]
    result = []
    for mode in ["mutate_future_sales", "remove_future_sales", "mutate_future_prices"]:
        condition = "where date<=date '"+day+"'" if mode == "remove_future_sales" else ""
        sale = f"case when date>date '{day}' then units_sold+1000000 else units_sold end" if mode == "mutate_future_sales" else "units_sold"
        con.execute(f"create or replace temp view feature_sales as select store_id,item_id,date,{sale} as units_sold from global_daily_panel {condition}")
        if mode == "mutate_future_prices":
            con.execute(f"""create or replace temp view feature_prices as select p.* replace(
                case when wm_yr_wk in (select wm_yr_wk from feature_calendar group by wm_yr_wk having max(date)>date '{day}')
                then sell_price*100+999 else sell_price end as sell_price) from main.dim_price_week p""")
        changed_features = generate(con, [origin], data["feature_config"], data["mapping"]).features.reset_index(drop=True)
        pd.testing.assert_frame_equal(before, changed_features)
        changed = data["daily"].copy()
        future = changed.date.gt(origin)
        if mode == "mutate_future_sales": changed.loc[future, "units_sold"] += 1000000
        elif mode == "remove_future_sales": changed = changed.loc[~future].copy()
        else: changed.loc[future, "sell_price"] = 99999.
        panel = Panel(changed, data["calendar"], data["metadata"])
        probe = TargetDataset(panel, rows, neural)
        for first, second in zip(original.tensors, probe.tensors):
            np.testing.assert_array_equal(first.numpy(), second.numpy())
        np.testing.assert_array_equal(first_neural, predict(model, probe, neural))
        for i, tree in enumerate(trees):
            np.testing.assert_array_equal(tree_before[i], tree.predict(changed_features).raw_prediction.to_numpy())
        for i, row in enumerate(rows.itertuples()):
            history = changed[changed.store_id.eq(row.store_id) & changed.item_id.eq(row.item_id)]
            np.testing.assert_array_equal(base_before[i], historical(history, origin, baseline_days)[0])
        changed_pool = data["pool"].copy()
        beyond = pd.to_datetime(changed_pool.target_date).gt(origin)
        changed_pool.loc[beyond, TARGET] += 1000000
        pd.testing.assert_frame_equal(eligible_rows(data["pool"], fit_cutoff), eligible_rows(changed_pool, fit_cutoff))
        result.append({"origin_date": origin, "perturbation": mode, "feature_rows": len(before),
                       "unchanged_features": True, "unchanged_history_scale_mask": True,
                       "unchanged_neural_quantiles": True, "unchanged_tree_quantiles": True,
                       "unchanged_baseline": True, "unchanged_eligible_fitting_labels": True,
                       "unchanged_intervals": True})
        con.execute("create or replace temp view feature_sales as select * from global_daily_panel")
        con.execute("create or replace temp view feature_prices as select * from main.dim_price_week")
    return result
