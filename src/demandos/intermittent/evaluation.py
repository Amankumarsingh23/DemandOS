"""Exact planned coverage and separate observed/cold-simulation scoring panels."""
import numpy as np
import pandas as pd
from demandos.backtesting.metrics import rmsse_scale,summarize,common_cases,paired_losses,horizon_band,CASE
from .diagnostics import prefix


PRIMARY = ("pooled","hurdle","tsb","seasonal_naive","cold_prior","router","ensemble")
BASELINES = ("seasonal_naive","tsb","croston","sba","moving_average")
MODELS = ("pooled","hurdle",*BASELINES,"cold_prior","router","ensemble")
POINT = ["store_id","item_id","origin_date","horizon","target_date"]


def attach_actuals(predictions,cohort,frames,origins):
    if predictions.duplicated(POINT+["model"]).any():
        raise ValueError("Duplicate forecast model keys")
    plan = []
    for origin in origins:
        for row in cohort.itertuples():
            history = frames[(row.store_id,row.item_id)]
            visible = prefix(history,origin,int(row.visible_days) if row.visible_days>=0 else None)
            scale = rmsse_scale(visible.units_sold.to_numpy(float)) if len(visible) else {"scale":None,"scale_status":"no_visible_history"}
            targets = history.set_index("date").units_sold
            for h in range(1,29):
                date = origin+pd.Timedelta(days=h)
                if date not in targets.index:
                    raise ValueError("Actual M5 target missing")
                for model in MODELS:
                    plan.append({"store_id":row.store_id,"item_id":row.item_id,"origin_date":origin,"horizon":h,"target_date":date,
                                 "model":model,"series_id":row.series_id,"cohort":row.cohort,"scenario":row.scenario,
                                 "actual":targets.loc[date],"scale":scale["scale"],"scale_status":scale["scale_status"]})
    planned = pd.DataFrame(plan)
    merged = planned.merge(predictions,on=POINT+["model"],how="outer",indicator=True,validate="one_to_one")
    if merged._merge.ne("both").any():
        raise ValueError("Predictions and planned cases must match exactly, including explicit skips")
    merged = merged.drop(columns="_merge")
    ok = merged.status.eq("ok")
    if merged.status.isna().any() or not np.isfinite(merged.loc[ok,["prediction","raw_prediction","actual"]].to_numpy(float)).all() or merged.loc[~ok,"prediction"].notna().any():
        raise ValueError("Invalid successful predictions or unavailable-case substitution")
    if merged.loc[ok,"prediction"].lt(0).any():
        raise ValueError("Negative observed-sales prediction")
    merged["origin"],merged["window"],merged["kind"] = merged.origin_date,"expanding","store_item"
    merged["y_pred"],merged["y_true"] = merged.prediction,merged.actual
    merged["absolute_error"] = abs(merged.actual-merged.prediction)
    merged["squared_error"] = (merged.actual-merged.prediction)**2
    merged["horizon_band"] = horizon_band(merged.horizon)
    merged["model_id"] = merged.model+"/"+merged.origin_date.dt.strftime("%Y%m%d")
    return merged


def tables(frame):
    common = common_cases(frame,list(PRIMARY))
    if common[common.cohort.eq("matched")].empty:
        raise ValueError("No common continuity benchmark; publication blocked")
    result = {"metrics_common":summarize(common,["cohort","model"]),
              "comparison_membership":common[["cohort",*CASE]].drop_duplicates(),
              "metrics_available":summarize(frame[frame.status.eq("ok")],["cohort","model"])}
    for name,columns in (("origin",["origin"]),("horizon",["horizon"]),("band",["horizon_band"]),
                         ("regime",["regime"]),("cold_status",["cold_status"]),("series_origin",["series_id","origin"])):
        result[f"metrics_{name}"] = summarize(common,["cohort","model",*columns])
    result["paired_losses"] = paired_losses(frame,baseline="pooled")
    result["paired_origin"] = result["paired_losses"].groupby(["cohort","model","origin"],as_index=False).agg(
        targets=("horizon","size"),mean_absolute_difference=("absolute_loss_difference","mean"))
    result["coverage"] = frame.groupby(["cohort","model","origin","status","reason"],dropna=False,as_index=False).agg(
        targets=("horizon","size"),series=("series_id","nunique"))
    hurdle = frame[frame.model.eq("hurdle") & frame.status.eq("ok")].copy()
    hurdle["occurred"] = hurdle.actual.gt(0).astype(int)
    hurdle["brier"] = (hurdle.p_positive-hurdle.occurred)**2
    hurdle["probability_bin"] = pd.cut(hurdle.p_positive,[0,.2,.4,.6,.8,1.],include_lowest=True).astype(str)
    result["occurrence_metrics"] = hurdle.groupby("cohort",as_index=False).agg(rows=("actual","size"),brier_score=("brier","mean"),observed_occurrence=("occurred","mean"))
    result["calibration"] = hurdle.groupby(["cohort","probability_bin"],as_index=False).agg(
        rows=("actual","size"),mean_probability=("p_positive","mean"),positive_fraction=("occurred","mean"))
    positive = hurdle[hurdle.actual.gt(0)].copy()
    positive["size_absolute_error"] = abs(positive.actual-positive.conditional_positive_mean)
    result["positive_size_metrics"] = positive.groupby("cohort",as_index=False).agg(
        actual_positive_rows=("actual","size"),conditional_size_mae=("size_absolute_error","mean"))
    return result
