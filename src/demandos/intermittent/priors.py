"""Leave-one-item-out observed-sales priors; no silent global-mean fallback."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import Config


@dataclass
class PriorState:
    cutoff: pd.Timestamp
    cells: pd.DataFrame
    config: Config

    @classmethod
    def from_daily(cls, daily, cutoff, config=Config()):
        cutoff = pd.Timestamp(cutoff)
        rows = daily.loc[pd.to_datetime(daily.date).between(cutoff-pd.Timedelta(days=config.prior_window-1),cutoff)].copy()
        if rows.duplicated(["store_id","item_id","date"]).any() or rows[["store_id","item_id","cat_id","date","units_sold"]].isna().any().any():
            raise ValueError("Unique observed prior rows with metadata required")
        values = rows.units_sold.to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError("Invalid prior observations")
        cells = rows.groupby(["store_id","item_id","cat_id"],as_index=False).agg(prior_units=("units_sold","sum"),prior_rows=("units_sold","size"))
        return cls(cutoff,cells,config)

    @classmethod
    def from_warehouse(cls, con, cutoff, config=Config(), *, relation="main.fact_daily_sales"):
        cutoff = pd.Timestamp(cutoff)
        if relation not in ("main.fact_daily_sales","sparse_prior_source"):
            raise ValueError("Only verified or explicit perturbation prior source allowed")
        cells = con.execute(f"""select store_id,item_id,cat_id,sum(units_sold) as prior_units,count(*) as prior_rows
            from {relation} where date between ? and ? group by store_id,item_id,cat_id
            order by store_id,item_id""",[(cutoff-pd.Timedelta(days=config.prior_window-1)).date(),cutoff.date()]).df()
        return cls(cutoff,cells,config)

    def estimate(self, store_id, item_id, cat_id, observed, origin):
        if pd.Timestamp(origin) != self.cutoff or (not observed.empty and pd.to_datetime(observed.date).gt(self.cutoff).any()):
            raise ValueError("Prior and visible observations must belong to the permitted origin")
        eligible = self.cells.loc[self.cells.item_id.ne(item_id)]
        selected = eligible.iloc[:0]
        source = "unavailable_category"
        if cat_id is not None and not pd.isna(cat_id):
            category = eligible.loc[eligible.cat_id.eq(cat_id)]
            selected = category.loc[category.store_id.eq(store_id)]
            source = "store_category_leave_item_out"
            if selected.prior_rows.sum() < self.config.minimum_prior_rows:
                selected, source = category, "category_leave_item_out"
        count = int(selected.prior_rows.sum())
        prior = float(selected.prior_units.sum()/count) if count >= self.config.minimum_prior_rows else None
        if prior is None:
            source = "unavailable_prior"
        n = len(observed)
        total = float(observed.units_sold.sum())
        if prior is not None:
            value = (total+self.config.kappa*prior)/(n+self.config.kappa)
            status = "ok"
        elif n:
            value, status = total/n, "observed_mean_only_prior_unavailable"
        else:
            value, status = None, "unavailable_no_history_or_prior"
        return {"prior_mean": prior,"prior_rows":count,"prior_items":selected.item_id.nunique(),
                "prior_source":source,"prior_cutoff":self.cutoff,"excluded_item_id":item_id,
                "observed_n":n,"observed_mean":total/n if n else None,"kappa":self.config.kappa,
                "prior_prediction":value,"prior_status":status}
