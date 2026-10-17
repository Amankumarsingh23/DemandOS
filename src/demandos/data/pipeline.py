"""Validate M5 CSV sources and stream long-form Parquet outputs."""

import csv
import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .validation import columns, integer_values, nonnull, require, unique


FILES = ("sales_train_validation.csv", "calendar.csv", "sell_prices.csv")
IDS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
PRICE_KEY = ["store_id", "item_id", "wm_yr_wk"]
CALENDAR = ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    require(len(header) == len(set(header)), f"{path.name}: duplicate column names")
    return pd.read_csv(path, dtype="string[pyarrow]")


def validate_calendar(frame):
    frame = frame.copy()
    columns(frame, CALENDAR, "calendar")
    required = [name for name in CALENDAR if not name.startswith("event_")]
    nonnull(frame, required, "calendar")
    unique(frame, ["d"], "calendar")
    unique(frame, ["date"], "calendar")
    require(frame.d.str.fullmatch(r"d_[1-9]\d*").all(), "calendar: invalid day identifiers")
    frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="coerce")
    nonnull(frame, ["date"], "calendar")
    for name in ["wm_yr_wk", "wday", "month", "year", "snap_CA", "snap_TX", "snap_WI"]:
        frame[name] = integer_values(frame[name], f"calendar.{name}", minimum=1 if name in {"wm_yr_wk", "wday", "month", "year"} else 0)
    ordered = frame.assign(day_number=frame.d.str[2:].astype(int)).sort_values("day_number")
    require(ordered.day_number.tolist() == list(range(1, len(frame) + 1)), "calendar: days must be contiguous from d_1")
    require(ordered.date.diff().dropna().eq(pd.Timedelta(days=1)).all(), "calendar: day/date mapping is not consecutive")
    require(frame.month.eq(frame.date.dt.month).all() and frame.year.eq(frame.date.dt.year).all(), "calendar: month/year mismatch")
    require(frame.weekday.eq(frame.date.dt.day_name()).all(), "calendar: weekday mismatch")
    require(frame.wday.eq((frame.date.dt.dayofweek + 2) % 7 + 1).all(), "calendar: wday mismatch (Saturday=1)")
    require(frame[["snap_CA", "snap_TX", "snap_WI"]].isin([0, 1]).all().all(), "calendar: invalid SNAP flag")
    for suffix in ("1", "2"):
        require(frame[f"event_name_{suffix}"].isna().equals(frame[f"event_type_{suffix}"].isna()), "calendar: unmatched event name/type nulls")
    weeks = frame.groupby("wm_yr_wk").date.agg(["min", "max", "count"])
    require(((weeks["max"] - weeks["min"]).dt.days + 1).eq(weeks["count"]).all() and weeks["count"].le(7).all(), "calendar: invalid week mapping")
    return frame[CALENDAR]


def validate_prices(frame):
    frame = frame.copy()
    columns(frame, PRICE_KEY + ["sell_price"], "prices")
    nonnull(frame, PRICE_KEY + ["sell_price"], "prices")
    frame["wm_yr_wk"] = integer_values(frame.wm_yr_wk, "prices.wm_yr_wk", 1)
    frame["sell_price"] = pd.to_numeric(frame.sell_price, errors="coerce").astype("float64")
    require(np.isfinite(frame.sell_price).all() and frame.sell_price.gt(0).all(), "prices: price must be finite and positive")
    unique(frame, PRICE_KEY, "prices")
    return frame[PRICE_KEY + ["sell_price"]]


def sales_header(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    require(len(header) == len(set(header)), "sales: duplicate column names")
    days = [name for name in header if name not in IDS]
    require(set(IDS).issubset(header), "sales: missing identifier columns")
    require(bool(days) and all(re.fullmatch(r"d_[1-9]\d*", name) for name in days), "sales: invalid day columns")
    require(days == [f"d_{i}" for i in range(1, len(days) + 1)], "sales: days must be ordered and contiguous from d_1")
    return days


def validate_sales(frame, days):
    columns(frame, IDS + days, "sales")
    nonnull(frame, IDS, "sales")
    unique(frame, ["store_id", "item_id"], "sales")
    unique(frame, ["id"], "sales")
    for day in days:
        frame[day] = integer_values(frame[day], f"sales.{day}")
    return frame


def wide_to_long(frame, days):
    return frame.melt(id_vars=IDS, value_vars=days, var_name="d", value_name="units_sold").astype({"units_sold": "int32"})


def join_daily(long, calendar, prices):
    daily = long.merge(calendar, on="d", how="left", validate="many_to_one", indicator=True)
    require(daily._merge.eq("both").all(), "sales: calendar day not mapped")
    daily = daily.drop(columns="_merge").merge(prices, on=PRICE_KEY, how="left", validate="many_to_one")
    require(len(daily) == len(long), "join: row count changed")
    unique(daily, ["store_id", "item_id", "date"], "daily")
    return daily


def arrow_table(frame):
    # Explicit strings avoid null-only event columns changing schema between chunks.
    table = pa.Table.from_pandas(frame, preserve_index=False)
    for index, name in enumerate(table.column_names):
        if name in IDS + ["d", "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            table = table.set_column(index, name, table.column(index).cast(pa.string()))
    return table


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def run_pipeline(root=Path("data"), chunk_size=128):
    root = Path(root)
    require(chunk_size > 0, "chunk size must be positive")
    raw = root / "raw"
    missing = [str(raw / name) for name in FILES if not (raw / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    sources = {name: fingerprint(raw / name) for name in FILES}
    calendar = validate_calendar(read_csv(raw / FILES[1]))
    prices = validate_prices(read_csv(raw / FILES[2]))
    require(prices.wm_yr_wk.isin(calendar.wm_yr_wk).all(), "prices: week absent from calendar")
    days = sales_header(raw / FILES[0])
    require(set(days).issubset(set(calendar.d)), "sales: calendar days missing")
    interim, processed = root / "interim", root / "processed"
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    seen_pairs, seen_ids, products, stores = set(), set(), set(), set()
    item_hierarchy, store_states = {}, {}
    rows = zeros = missing_prices = positive_missing_prices = wide_rows = 0
    writers = {}
    reader = pd.read_csv(raw / FILES[0], dtype="string[pyarrow]", chunksize=chunk_size)
    with TemporaryDirectory(prefix=".m5-", dir=root) as temporary:
        temporary = Path(temporary)
        try:
            for chunk in reader:
                chunk = validate_sales(chunk, days)
                pairs = set(zip(chunk.store_id, chunk.item_id))
                require(not seen_pairs.intersection(pairs), "sales: duplicate store/item across chunks")
                require(not seen_ids.intersection(chunk.id), "sales: duplicate id across chunks")
                seen_pairs.update(pairs)
                seen_ids.update(chunk.id)
                products.update(chunk.item_id)
                stores.update(chunk.store_id)
                for item, dept, cat, store, state in chunk[["item_id", "dept_id", "cat_id", "store_id", "state_id"]].itertuples(index=False, name=None):
                    require(item_hierarchy.setdefault(item, (dept, cat)) == (dept, cat), "sales: inconsistent item hierarchy")
                    require(store_states.setdefault(store, state) == state, "sales: inconsistent store state")
                long = wide_to_long(chunk, days)
                daily = join_daily(long, calendar, prices)
                wide_rows += len(chunk)
                rows += len(long)
                zeros += int(daily.units_sold.eq(0).sum())
                missing_prices += int(daily.sell_price.isna().sum())
                positive_missing_prices += int((daily.sell_price.isna() & daily.units_sold.gt(0)).sum())
                for name, frame in (("sales_long", long), ("daily_sales", daily)):
                    table = arrow_table(frame)
                    if name not in writers:
                        writers[name] = pq.ParquetWriter(temporary / f"{name}.parquet", table.schema, compression="snappy")
                    writers[name].write_table(table)
                print(f"Processed {wide_rows:,} series / {rows:,} daily rows", flush=True)
            require(wide_rows > 0, "sales: empty source")
            price_pairs = set(zip(prices.store_id, prices.item_id))
            require(price_pairs.issubset(seen_pairs), "prices: unknown store/item keys")
            require(rows == wide_rows * len(days), "reshape: unexpected row count")
        finally:
            reader.close()
            for writer in writers.values():
                writer.close()
        selected = calendar[calendar.d.isin(days)]
        warnings = []
        if missing_prices:
            warnings.append("Missing joined prices retained as null; price absence does not establish stockout or availability.")
        if positive_missing_prices:
            warnings.append("Some positive-sales rows have no price record.")
        if prices.sell_price.gt(1000).any():
            warnings.append("Source prices above 1000 detected; review currency/business context.")
        report = {
            "status": "passed", "sources": sources, "chunk_size": chunk_size,
            "sales_series": wide_rows, "sales_days": len(days), "daily_rows": rows,
            "calendar_rows": len(calendar), "price_rows": len(prices),
            "distinct_products": len(products), "distinct_stores": len(stores),
            "date_min": str(selected.date.min().date()), "date_max": str(selected.date.max().date()),
            "missing_prices": missing_prices, "missing_price_rate": missing_prices / rows,
            "positive_sales_missing_price": positive_missing_prices,
            "zero_sales_rows": zeros, "zero_sales_rate": zeros / rows,
            "source_null_counts": {"calendar": {key: int(value) for key, value in calendar.isna().sum().items()}, "prices": {key: int(value) for key, value in prices.isna().sum().items()}, "sales": 0},
            "duplicate_checks": {"sales_id": 0, "sales_store_item": 0, "calendar_d": 0, "calendar_date": 0, "price_store_item_week": 0, "daily_store_item_date": 0},
            "join_cardinality": "many_to_one; row counts preserved",
            "warnings": warnings,
            "assumptions": ["Sales are observed units, not uncensored demand.", "Source prices must be finite and strictly positive.", "Missing joined prices are retained, never imputed.", "Daily uniqueness follows globally unique store/item series and unique consecutive calendar days."],
        }
        (temporary / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        # Publish only after all validation passes; previous outputs survive failed runs.
        for name, target in (("sales_long.parquet", interim), ("daily_sales.parquet", processed), ("quality_report.json", processed)):
            (temporary / name).replace(target / name)
    return report
