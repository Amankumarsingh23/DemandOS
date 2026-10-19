"""Bounded interpretation tables with explicit dates, variable axes and caveats."""

import numpy as np
import pandas as pd
from demandos.neural.sequences import STATIC,PAST,FUTURE


def tables(rows,diagnostics,history_mask,lookback=56,horizon=28):
    variable,attention=[],[]
    averaged=diagnostics["attention"].mean(axis=1)
    for index,row in enumerate(rows.itertuples()):
        key={"store_id":row.store_id,"item_id":row.item_id,"origin":row.origin_date}
        for family,names in (("static",STATIC),("past",PAST),("future",FUTURE)):
            weights=diagnostics[f"{family}_weights"][index]
            if family=="static": weights=weights[None]
            for step,values in enumerate(weights):
                valid=bool(history_mask[index,step]) if family=="past" else True
                relative_day=step-lookback+1 if family=="past" else step+1 if family=="future" else 0
                for name,value in zip(names,values):
                    variable.append({**key,"family":family,"step":step,"relative_day":relative_day,
                                     "variable":name,"weight":float(value),"valid_position":valid})
        for h in range(1,horizon+1):
            for position in range(lookback+horizon):
                relative_day=position-lookback+1
                attention.append({**key,"horizon":h,"target_date":row.origin_date+pd.Timedelta(days=h),
                                  "key_relative_day":relative_day,"key_date":row.origin_date+pd.Timedelta(days=relative_day),
                                  "key_family":"history" if position<lookback else "future_known",
                                  "allowed":bool(diagnostics["allowed"][index,h-1,position]),
                                  "weight":float(averaged[index,h-1,position])})
    variables,attention=pd.DataFrame(variable),pd.DataFrame(attention)
    selected=variables[variables.valid_position]
    return {"vsn_cases":variables,"attention_cases":attention,
            "vsn_summary":selected.groupby(["family","variable"],as_index=False).agg(mean_weight=("weight","mean"),sd_weight=("weight","std"),positions=("weight","size")),
            "attention_lags":attention[attention.key_family.eq("history")].groupby("key_relative_day",as_index=False).agg(mean_weight=("weight","mean"),positions=("weight","size")),
            "attention_horizon":attention.groupby(["horizon","key_family"],as_index=False).agg(total_weight=("weight","sum"),cases=("origin","size"))}
