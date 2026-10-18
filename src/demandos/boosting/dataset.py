"""Historical direct-horizon examples, chronological eligibility and fixed cohorts."""

from dataclasses import replace
import hashlib

import pandas as pd

from demandos.backtesting.pipeline import extract as extract_benchmark
from demandos.backtesting.design import load_config as load_backtest_config, schedules
from demandos.eda.queries import fetch_frame
from demandos.features.config import load_config as load_feature_config
from demandos.features.pipeline import source_views, load_keys, fit_mapping
from demandos.features.validation import generate, supervised_labels
from demandos.features.contracts import KEYS, predictive_columns
from demandos.features.encoding import CATEGORICAL
from demandos.timeseries.cohort import training_labels, select_cohort
from .models import TARGET, validate_rows


def historical_origins(config, last_evaluation):
    origins = pd.date_range(config.training_start, pd.Timestamp(last_evaluation) - pd.Timedelta(days=1), freq=f"{config.training_stride}D")
    if len(origins) > config.max_origins or pd.Timestamp(config.validation_origin) not in origins:
        raise ValueError("Origin cap exceeded or internal validation origin absent from historical schedule")
    return origins


def eligible_rows(pool, cutoff):
    """Filter by TARGET date, not merely the example's forecast origin."""
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff):
        raise ValueError("Explicit label-availability cutoff required")
    result = pool.loc[pd.to_datetime(pool.target_date).le(cutoff)].copy()
    validate_rows(result, labelled=True, cutoff=cutoff)
    return result


def chronological_development(pool, config, first_evaluation):
    origin = pd.Timestamp(config.validation_origin)
    end = origin + pd.Timedelta(days=28)
    if end >= pd.Timestamp(first_evaluation):
        raise ValueError("Internal validation must finish before any evaluation origin")
    train = eligible_rows(pool, origin)
    validation = pool.loc[pd.to_datetime(pool.origin_date).eq(origin)].copy()
    validate_rows(validation, labelled=True, cutoff=end)
    if not pd.to_datetime(validation.target_date).gt(origin).all():
        raise ValueError("Validation targets overlap fitting targets")
    expected = len(validation[["store_id", "item_id"]].drop_duplicates()) * 28
    if len(validation) != expected:
        raise ValueError("Incomplete internal-validation target grid")
    return train, validation, end


def select_training_panel(labels, config):
    """Training-prefix diversity/store leaders plus deterministic ID-hash sampling."""
    selected, reasons = {}, {}
    def add(row, reason):
        key = (row.store_id, row.item_id)
        if len(selected) < config.training_series or key in selected:
            selected[key] = True
            reasons.setdefault(key, []).append(reason)
    for row in select_cohort(labels, min(14, config.training_series)).itertuples():
        add(row, "prefix_regime_ABCXYZ")
    ranked = labels.sort_values(["units_sold", "store_id", "item_id"], ascending=[False, True, True])
    for _, group in ranked.groupby("store_id", sort=True):
        for row in group.head(2).itertuples():
            add(row, "prefix_store_volume")
    hashed = labels.copy()
    hashed["selection_hash"] = [hashlib.sha256(f"{config.seed}/{r.store_id}/{r.item_id}".encode()).hexdigest() for r in labels.itertuples()]
    for row in hashed.sort_values(["selection_hash", "store_id", "item_id"]).itertuples():
        add(row, "stable_ID_hash")
        if len(selected) == config.training_series:
            break
    if len(selected) != config.training_series:
        raise ValueError("Not enough training-prefix series for requested panel")
    choices = pd.DataFrame([{"store_id": key[0], "item_id": key[1], "training_selection_reason": ";".join(reasons[key])} for key in selected])
    return choices.merge(labels, on=["store_id", "item_id"], validate="one_to_one")


def row_digest(frame):
    """Hash logical daily keys/labels independently of Arrow/pandas storage types."""
    columns = KEYS + ["target_date", TARGET]
    canonical = frame[columns].copy()
    for name in ("store_id", "item_id"):
        canonical[name] = canonical[name].astype(str).astype(object)
    for name in ("origin_date", "target_date"):
        canonical[name] = pd.to_datetime(canonical[name]).dt.strftime("%Y-%m-%d").astype(object)
    for name in ("horizon", TARGET):
        canonical[name] = canonical[name].astype("int64")
    canonical = canonical.sort_values(KEYS).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(canonical, index=False).to_numpy().tobytes()).hexdigest()


def build(con, root, config):
    source_views(con)
    backtest = load_backtest_config(root / "configs/backtesting.toml")
    frames, metadata, _, _, _, _ = extract_benchmark(con, root, backtest)
    evaluation = metadata[metadata.kind.eq("store_item")].copy()
    calendar = fetch_frame(con, "select date from main.dim_calendar order by date").date
    origins = list(pd.DatetimeIndex(schedules(calendar, backtest).origin.drop_duplicates()).sort_values())
    # Historical selection is independent of legacy benchmark IDs and evaluation outcomes.
    start = pd.Timestamp(con.execute("select min(date) from main.dim_calendar").fetchone()[0])
    labels, _, _ = training_labels(con, root, start, pd.Timestamp(config.selection_cutoff))
    selected = select_training_panel(labels, config)
    train_keys = load_keys(con, config.selection_cutoff, selected)
    eval_keys = load_keys(con, origins[0], evaluation[["store_id", "item_id"]])
    train_origins = historical_origins(config, origins[-1])
    if len(train_origins) * config.training_series * 28 > config.max_rows:
        raise ValueError("Planned feature expansion exceeds max_rows before materialization")
    feature_config = replace(load_feature_config(root / "configs/features.toml"), mapping_fit_end=config.selection_cutoff,
                             training_origins=(config.training_start,), full_origin_pass=False)
    mapping = fit_mapping(con, train_keys, config.selection_cutoff)
    union = pd.concat([train_keys[["store_id", "item_id"]], eval_keys[["store_id", "item_id"]]]).drop_duplicates()
    con.register("global_panel_keys", union)
    con.execute("""create or replace temp table global_daily_panel as select f.* from feature_base_sales f
        join global_panel_keys k using(store_id,item_id) where date between ? and ? order by date,store_id,item_id""",
        [(train_origins[0] - pd.Timedelta(days=feature_config.lookback)).date(), (origins[-1] + pd.Timedelta(days=28)).date()])
    con.execute("create or replace temp view feature_sales as select * from global_daily_panel")
    con.register("feature_keys", train_keys)
    batch = generate(con, train_origins, feature_config, mapping)
    candidates = batch.features.loc[batch.features.target_date.le(origins[-1])].copy()
    label_rows = supervised_labels(con, candidates, origins[-1])
    pool = candidates.merge(label_rows, on=KEYS + ["target_date"], validate="one_to_one")
    del candidates, batch, label_rows
    if len(pool) > config.max_rows:
        raise ValueError("Labeled training pool exceeds max_rows")
    con.register("feature_keys", eval_keys)
    inference = generate(con, origins, feature_config, mapping)
    features = [name for name in predictive_columns(feature_config) if name not in {f"{key}_code" for key in CATEGORICAL}]
    training, validation, validation_end = chronological_development(pool, config, origins[0])
    return {"pool": pool, "development_train": training, "development_validation": validation,
            "validation_end": validation_end, "inference": inference.features, "feature_columns": features,
            "feature_config": feature_config, "mapping": mapping, "training_cohort": selected,
            "evaluation_cohort": evaluation, "evaluation_keys": eval_keys, "origins": origins,
            "training_origins": train_origins, "frames": {r.series_id: frames[r.series_id] for r in evaluation.itertuples()},
            "cached_daily_rows": con.execute("select count(*) from global_daily_panel").fetchone()[0]}
