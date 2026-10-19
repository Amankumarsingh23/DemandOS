"""Warehouse checks and genuine historical/base-model inference."""

from dataclasses import replace
import json
from time import perf_counter

import numpy as np
import pandas as pd

from demandos.features.pipeline import load_keys
from demandos.nbeats.pipeline import prepare
from demandos.nbeats.config import load_config as neural_config
from demandos.nbeats.training import train, predict, save
from demandos.nbeats.dataset import TargetDataset
from demandos.neural.sequences import Panel, examples
from demandos.data.pipeline import fingerprint
from .hierarchy import build, KEYS, LEVELS, observed_matrix


def prepare_data(con, root, config):
    neural = neural_config(root/"configs/nbeats.toml")
    data = prepare(con, root, neural)
    hierarchy = build(data["metadata"])
    if len(hierarchy.bottom) > config["max_bottom"]:
        raise ValueError("Selected cohort exceeds the explicit solver cap")
    full = build(load_keys(con, data["origins"][0]), scope="FULL")
    full.check(full=True)
    con.register("reconciliation_keys", hierarchy.bottom[KEYS])
    daily = con.execute("""select f.store_id,f.item_id,f.date,f.units_sold,f.sell_price
        from main.fact_daily_sales f join reconciliation_keys k using(store_id,item_id)
        order by store_id,item_id,date""").fetchdf()
    dates, bottom = observed_matrix(daily, hierarchy)
    data.update(neural_config=neural, hierarchy=hierarchy, full=full, daily=daily,
                dates=dates, observed=bottom, panel=Panel(daily, data["calendar"], data["metadata"]))
    return data


def verify_aggregates(con, hierarchy, *, cohort=False):
    """Independent SQL ground truth at all levels on three real source dates."""
    days = ["2011-01-29", "2016-02-29", "2016-04-24"]
    select = "join reconciliation_keys k using(store_id,item_id)" if cohort else ""
    bottom = con.execute(f"""select f.store_id,f.item_id,f.date,f.units_sold
        from main.fact_daily_sales f {select} where f.date in (?,?,?)
        order by store_id,item_id,date""", days).fetchdf()
    pivot = bottom.pivot(index=KEYS, columns="date", values="units_sold")
    pivot = pivot.reindex(pd.MultiIndex.from_frame(hierarchy.bottom[KEYS]))
    if pivot.isna().any().any():
        raise ValueError("Full/cohort warehouse coverage is incomplete")
    aggregate = hierarchy.S@pivot.to_numpy(float)
    records = []
    for level, dims in LEVELS.items():
        columns = ",".join("f."+d for d in dims)
        columns = columns+"," if columns else ""
        truth = con.execute(f"""select {columns}f.date,sum(f.units_sold) as units_sold
            from main.fact_daily_sales f {select} where f.date in (?,?,?) group by all""", days).fetchdf()
        rows = hierarchy.nodes[hierarchy.nodes.level.eq(level)]
        check = rows.merge(truth, on=list(dims), validate="one_to_many") if dims else rows.merge(truth, how="cross")
        positions = pd.DatetimeIndex(pivot.columns).get_indexer(pd.to_datetime(check.date))
        predicted = aggregate[check.node_row.to_numpy(), positions]
        if len(check) != len(rows)*3 or not np.array_equal(predicted, check.units_sold.to_numpy(float)):
            raise ValueError(f"Sparse/warehouse aggregate mismatch: {level}")
        records.append({"scope": "COHORT" if cohort else "FULL", "level": level,
                        "nodes": len(rows), "dates": 3, "max_gap": float(np.abs(predicted-check.units_sold).max())})
    return pd.DataFrame(records)


def direct_history(con, hierarchy):
    """Actual cohort histories independently grouped in DuckDB, not forecast sums."""
    con.register("reconciliation_membership", hierarchy.membership[["node_row", "bottom_col"]])
    con.register("reconciliation_bottom_map", hierarchy.bottom[KEYS+["bottom_col"]])
    frame = con.execute("""select m.node_row,f.date,sum(f.units_sold) as units_sold
        from main.fact_daily_sales f join reconciliation_bottom_map b using(store_id,item_id)
        join reconciliation_membership m using(bottom_col) group by m.node_row,f.date
        order by m.node_row,f.date""").fetchdf()
    pivot = frame.pivot(index="node_row", columns="date", values="units_sold").reindex(hierarchy.nodes.node_row)
    if pivot.isna().any().any():
        raise ValueError("Incomplete independent aggregate observations")
    return frame, pivot.to_numpy(float)


def independent_bases(history):
    history = np.asarray(history, float)
    if history.ndim != 2 or history.shape[1] < 7 or not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("At least seven complete observed prefix days required")
    # Direct seasonal naive on each observed node. It can disagree with N-BEATS bottoms.
    return np.tile(history[:, -7:], (1, 4))


def base_matrix(panel, data, origin, model, neural, aggregate, dates):
    rows = examples(data["hierarchy"].bottom, [pd.Timestamp(origin)], neural)
    dataset = TargetDataset(panel, rows, neural)
    raw, bottom, _ = predict(model, dataset, neural)
    # examples sorts keys in the same canonical bottom-column order.
    pd.testing.assert_frame_equal(rows[KEYS].reset_index(drop=True), data["hierarchy"].bottom[KEYS])
    prefix = aggregate[:, dates <= origin]
    result = independent_bases(prefix)
    result[data["hierarchy"].bottom_rows] = bottom
    return result, dataset, raw


def long_matrix(values, hierarchy, origin):
    n, h = values.shape
    if n != len(hierarchy.nodes) or h != 28 or not np.isfinite(values).all():
        raise ValueError("Complete finite 28-day all-node forecast required")
    frame = pd.DataFrame({"node_id": np.repeat(hierarchy.nodes.node_id, 28),
                          "level": np.repeat(hierarchy.nodes.level, 28),
                          "node_row": np.repeat(hierarchy.nodes.node_row, 28),
                          "origin_date": pd.Timestamp(origin), "horizon": np.tile(np.arange(1, 29), n),
                          "prediction": values.ravel()})
    frame["target_date"] = frame.origin_date+pd.to_timedelta(frame.horizon, unit="D")
    return frame


def verify_reference(root, data, sources):
    directory = root/"data/nbeats/global_h28"
    summary = json.loads((directory/"summary.json").read_text())
    if summary["status"] != "passed" or summary["source_fingerprints"] != sources:
        raise ValueError("Existing N-BEATS source provenance mismatch")
    if summary["config"] != vars(data["neural_config"]):
        raise ValueError("Existing N-BEATS configuration mismatch")
    reference = pd.read_parquet(directory/"tables/forecasts.parquet")
    reference = reference[reference.model.eq("nbeats_interpretable")].copy()
    expected = data["evaluation_cohort"][KEYS].drop_duplicates()
    if len(expected) != 13 or len(reference) != 1092 or not reference.status.eq("ok").all():
        raise ValueError("Original 13-series reference coverage differs")
    actual = reference[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected.sort_values(KEYS).reset_index(drop=True))
    return reference, {"summary": fingerprint(directory/"summary.json"),
                       "forecasts": fingerprint(directory/"tables/forecasts.parquet")}


def historical_residuals(data, aggregate, config, directory):
    """Refit at each historical origin: never backcast a later saved checkpoint."""
    origins = pd.date_range(config["calibration_start"], periods=config["calibration_origins"],
                            freq=f"{config['calibration_stride']}D")
    cutoff = pd.Timestamp(config["covariance_cutoff"])
    if config["calibration_stride"] < 28 or origins[-1]+pd.Timedelta(days=28) > cutoff or cutoff >= data["origins"][0]:
        raise ValueError("Calibration blocks must be disjoint and wholly prior to evaluation")
    records, metadata = [], []
    neural = replace(data["neural_config"], epochs=config["calibration_epochs"])
    for origin in origins:
        began = perf_counter()
        rows = examples(data["train_keys"], data["training_origins"], neural, cutoff=origin)
        training = TargetDataset(data["panel"], rows, neural, label_cutoff=origin)
        model, meta = train("interpretable", training, neural, cutoff=origin, epochs=config["calibration_epochs"])
        base, _, _ = base_matrix(data["panel"], data, origin, model, neural, aggregate, data["dates"])
        idx = data["dates"].get_indexer(pd.date_range(origin+pd.Timedelta(days=1), periods=28))
        if (idx < 0).any():
            raise ValueError("Historical residual targets missing")
        residual = long_matrix(base, data["hierarchy"], origin)
        residual["actual"] = aggregate[:, idx].ravel()
        residual["error"] = residual.actual-residual.prediction
        residual["fit_cutoff"], residual["fit_target_max"] = origin, pd.Timestamp(meta["fit_target_max"])
        residual["source_model"] = np.where(residual.level.eq("ITEM_STORE"), "nbeats_interpretable_historical_fixed2", "seasonal_naive_7")
        save(model, meta, directory/f"origin_{origin:%Y%m%d}")
        records.append(residual)
        metadata.append({"origin_date": origin, "seconds_total": perf_counter()-began, **meta})
        print(f"Historical paired errors {origin.date()}: {len(residual):,} rows", flush=True)
    return pd.concat(records, ignore_index=True), metadata
