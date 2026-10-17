"""Deterministic synthetic examples, NOT actual M5 observations."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from demandos.data.cli import main
from demandos.data.pipeline import (CALENDAR, IDS, join_daily, run_pipeline,
                                   validate_calendar, validate_prices,
                                   validate_sales, wide_to_long)
from demandos.data.validation import ValidationError


def fixtures():
    sales = pd.DataFrame([
        ["synthetic_a", "item_a", "dept", "cat", "store", "state", "0", "3"],
        ["synthetic_b", "item_b", "dept", "cat", "store", "state", "2", "1"],
    ], columns=IDS + ["d_1", "d_2"], dtype="string")
    calendar = pd.DataFrame([
        ["2011-01-29", "11101", "Saturday", "1", "1", "2011", "d_1", None, None, None, None, "0", "0", "0"],
        ["2011-01-30", "11101", "Sunday", "2", "1", "2011", "d_2", "Synthetic event", "Synthetic type", None, None, "1", "0", "0"],
    ], columns=CALENDAR, dtype="string")
    prices = pd.DataFrame([["store", "item_a", "11101", "2.5"]],
                          columns=["store_id", "item_id", "wm_yr_wk", "sell_price"], dtype="string")
    return sales, calendar, prices


class IngestionTests(unittest.TestCase):
    def test_reshape_and_missing_prices(self):
        sales, calendar, prices = fixtures()
        long = wide_to_long(validate_sales(sales, ["d_1", "d_2"]), ["d_1", "d_2"])
        self.assertEqual(len(long), 4)
        self.assertEqual(long.units_sold.tolist(), [0, 2, 3, 1])
        result = join_daily(long, validate_calendar(calendar), validate_prices(prices))
        self.assertEqual(len(result), 4)
        self.assertEqual(result.sell_price.isna().sum(), 2)
        self.assertEqual(result.units_sold.sum(), 6)

    def test_duplicate_price_join_rejected(self):
        sales, calendar, prices = fixtures()
        duplicate = pd.concat([prices, prices])
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            validate_prices(duplicate)
        long = wide_to_long(validate_sales(sales, ["d_1", "d_2"]), ["d_1", "d_2"])
        with self.assertRaises(pd.errors.MergeError):
            join_daily(long, validate_calendar(calendar), pd.concat([validate_prices(prices)] * 2))

    def test_invalid_sales(self):
        for value in ("-1", "1.5", "bad", None, "inf"):
            with self.subTest(value=value):
                sales, _, _ = fixtures()
                sales.loc[0, "d_1"] = value
                with self.assertRaises(ValidationError):
                    validate_sales(sales, ["d_1", "d_2"])

    def test_source_keys_and_schema(self):
        sales, calendar, prices = fixtures()
        for frame, validator in ((pd.concat([sales, sales]), lambda x: validate_sales(x, ["d_1", "d_2"])),
                                 (pd.concat([calendar, calendar]), validate_calendar),
                                 (calendar.drop(columns="date"), validate_calendar),
                                 (prices.drop(columns="sell_price"), validate_prices)):
            with self.assertRaises(ValidationError):
                validator(frame)

    def test_invalid_prices_and_calendar(self):
        for value in ("0", "-1", "inf", None, "bad"):
            _, _, prices = fixtures()
            prices.loc[0, "sell_price"] = value
            with self.assertRaises(ValidationError):
                validate_prices(prices)
        _, calendar, _ = fixtures()
        calendar.loc[1, "date"] = "2011-02-01"
        with self.assertRaises(ValidationError):
            validate_calendar(calendar)

    def test_missing_calendar_mapping(self):
        sales, calendar, prices = fixtures()
        long = wide_to_long(validate_sales(sales, ["d_1", "d_2"]), ["d_1", "d_2"])
        with self.assertRaisesRegex(ValidationError, "not mapped"):
            join_daily(long, validate_calendar(calendar.iloc[:1]), validate_prices(prices))

    def test_missing_files_cli(self):
        with TemporaryDirectory() as tmp, redirect_stderr(StringIO()) as errors:
            self.assertEqual(main(["--data-dir", tmp]), 1)
            for name in ("sales_train_validation.csv", "calendar.csv", "sell_prices.csv"):
                self.assertIn(name, errors.getvalue())
            self.assertFalse((Path(tmp) / "processed").exists())

    def test_pipeline_repeatable_across_chunks_and_preserves_raw(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            (root / "raw").mkdir()
            for frame, name in zip(fixtures(), ("sales_train_validation.csv", "calendar.csv", "sell_prices.csv")):
                frame.to_csv(root / "raw" / name, index=False)
            before = {p.name: p.read_bytes() for p in (root / "raw").iterdir()}
            report = run_pipeline(root, 1)
            first = pd.read_parquet(root / "processed/daily_sales.parquet")
            self.assertEqual(report["daily_rows"], 4)
            self.assertEqual(report["missing_prices"], 2)
            run_pipeline(root, 2)
            second = pd.read_parquet(root / "processed/daily_sales.parquet")
            keys = ["store_id", "item_id", "d"]
            pd.testing.assert_frame_equal(first.sort_values(keys).reset_index(drop=True), second.sort_values(keys).reset_index(drop=True))
            self.assertEqual(before, {p.name: p.read_bytes() for p in (root / "raw").iterdir()})
            sales, _, _ = fixtures()
            pd.concat([sales, sales.iloc[:1]]).to_csv(root / "raw/sales_train_validation.csv", index=False)
            with self.assertRaisesRegex(ValidationError, "across chunks"):
                run_pipeline(root, 1)
            pd.testing.assert_frame_equal(second, pd.read_parquet(root / "processed/daily_sales.parquet"))
