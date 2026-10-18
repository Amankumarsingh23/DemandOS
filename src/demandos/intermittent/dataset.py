"""Bounded real-series stress selection and explicitly simulated visibility windows."""
import hashlib
import numpy as np
import pandas as pd
from demandos.features.pipeline import load_keys
from demandos.features.validation import generate
from .diagnostics import quadrant,describe,prefix


COLD_MODES = {"new_product_0":0,"new_store_0":0,"short_3":3,"short_7":7,"short_28":28,"unknown_metadata_0":0}


def stress_cohort(con,data,config):
    origin = data["origins"][0]
    stats = con.execute("""select store_id,item_id,sum(units_sold) as prefix_units,
        count(*) as n,count(*) filter(where units_sold>0) as k,
        var_pop(units_sold) filter(where units_sold>0)/pow(avg(units_sold) filter(where units_sold>0),2) as cv2
        from main.fact_daily_sales where date between ? and ? group by store_id,item_id order by store_id,item_id""",
        [(origin-pd.Timedelta(days=config.history_window-1)).date(),origin.date()]).df()
    stats["selection_regime"] = ["all_zero" if r.k==0 else "one_positive" if r.k==1 else quadrant(r.n/r.k,r.cv2,config) for r in stats.itertuples()]
    stats["stable_hash"] = [hashlib.sha256(f"{config.seed}/{r.store_id}/{r.item_id}".encode()).hexdigest() for r in stats.itertuples()]
    used = set(map(tuple,data["evaluation_cohort"][["store_id","item_id"]].to_numpy()))
    records = []
    for regime in ("all_zero","one_positive","intermittent","lumpy","smooth","erratic"):
        available = stats[stats.selection_regime.eq(regime)].sort_values(
            ["k","stable_hash"] if regime in ("intermittent","lumpy") else ["stable_hash"])
        available = available[[tuple((r.store_id,r.item_id)) not in used for r in available.itertuples()]]
        for row in available.head(config.stress_per_group).itertuples():
            used.add((row.store_id,row.item_id))
            records.append({"store_id":row.store_id,"item_id":row.item_id,"cohort":"sparse_stress",
                            "scenario":"observed","visible_days":-1,"selection_reason":regime,"selection_cutoff":origin})
    excluded = set(data["pool"].item_id) | set(data["evaluation_cohort"].item_id) | {r["item_id"] for r in records}
    cold = stats[~stats.item_id.isin(excluded)].sort_values("stable_hash").drop_duplicates("item_id")
    required = config.cold_per_mode*len(COLD_MODES)
    if len(cold)<required:
        raise ValueError("Insufficient distinct genuinely training-unseen items for bounded cold simulation")
    for index,(mode,days) in enumerate(COLD_MODES.items()):
        for row in cold.iloc[index*config.cold_per_mode:(index+1)*config.cold_per_mode].itertuples():
            records.append({"store_id":row.store_id,"item_id":row.item_id,"cohort":"cold_simulation",
                            "scenario":mode,"visible_days":days,"selection_reason":"stable_hash_unseen_training_item",
                            "selection_cutoff":origin})
    stress = pd.DataFrame(records)
    matched = data["evaluation_cohort"][["store_id","item_id"]].copy().assign(cohort="matched",scenario="observed",visible_days=-1,
                selection_reason="existing_13_series_benchmark",selection_cutoff=origin)
    selected = pd.concat([matched,stress],ignore_index=True)
    selected = load_keys(con,origin,selected).merge(selected,on=["store_id","item_id"],validate="one_to_one")
    selected["series_id"] = "store_item/"+selected.store_id+"/"+selected.item_id
    if selected.duplicated(["store_id","item_id"]).any():
        raise ValueError("Stress and continuity cohorts must be distinct real store-item keys")
    return selected


def observed_frames(con,data,cohort):
    keys = pd.concat([data["training_cohort"][["store_id","item_id"]],cohort[["store_id","item_id"]]]).drop_duplicates()
    con.register("sparse_selected_keys",keys)
    con.execute("""create or replace temp table sparse_daily as select f.store_id,f.item_id,f.date,f.units_sold
        from main.fact_daily_sales f join sparse_selected_keys k using(store_id,item_id) order by store_id,item_id,date""")
    rows = con.execute("select * from sparse_daily order by store_id,item_id,date").df()
    if len(rows)>1000000:
        raise ValueError("Bounded selected history exceeds one million rows")
    return rows,{key:g.reset_index(drop=True) for key,g in rows.groupby(["store_id","item_id"],sort=True)}


def origin_features(con,cohort,origin,data,frames,config,*,sales_view="sparse_daily"):
    """No-history cases are explicitly unsupported by the observed-history ML schema."""
    blocks,diagnostics = [],[]
    for row in cohort.itertuples():
        history = frames[(row.store_id,row.item_id)]
        limit = int(row.visible_days) if row.visible_days>=0 else None
        stats = describe(history,origin,config,visible_days=limit,
                         known_category=row.scenario!="unknown_metadata_0",item_elsewhere=row.scenario=="new_store_0")
        diagnostics.append({"store_id":row.store_id,"item_id":row.item_id,"origin_date":origin,
                            "cohort":row.cohort,"scenario":row.scenario,**stats})
    diagnostics = pd.DataFrame(diagnostics)
    # At most six distinct visibility groups; aggregate once per group, not per series.
    for days,group in cohort.groupby("visible_days",sort=True):
        if days==0:
            continue
        group = group.copy()
        if days>0:
            group["first_observed_date"] = pd.Timestamp(origin)-pd.Timedelta(days=int(days)-1)
        con.register("feature_keys",group[["store_id","item_id","state_id","dept_id","cat_id","first_observed_date"]])
        if days>0:
            start = (pd.Timestamp(origin)-pd.Timedelta(days=int(days)-1)).date().isoformat()
            con.execute(f"create or replace temp view feature_sales as select * from {sales_view} where date>=date '{start}'")
        else:
            con.execute(f"create or replace temp view feature_sales as select * from {sales_view}")
        blocks.append(generate(con,[origin],data["feature_config"],data["mapping"]).features)
    con.execute(f"create or replace temp view feature_sales as select * from {sales_view}")
    return pd.concat(blocks,ignore_index=True).sort_values(["store_id","item_id","horizon"]).reset_index(drop=True),diagnostics
