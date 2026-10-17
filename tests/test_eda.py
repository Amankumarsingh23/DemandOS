"""Deterministic synthetic EDA fixtures, never presented as M5 observations."""

from decimal import Decimal
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from demandos.eda.analysis import abc_xyz_matrix, series_segments
from demandos.eda.config import EDAConfig
from demandos.eda.statistics import (abc_classification, combine_segments, intermittent_class,
                                     outlier_flags, prior_average, sales_statistics, xyz_class)


class EDAStatisticsTests(unittest.TestCase):
    def test_population_statistics(self):
        a, b = sales_statistics([8, 9, 10, 11, 12]), sales_statistics([0, 0, 0, 0, 50])
        self.assertEqual(a["mean"], b["mean"])
        self.assertAlmostEqual(a["variance"], 2)
        self.assertAlmostEqual(b["std"], 20)
        self.assertAlmostEqual(a["cv"], np.sqrt(2) / 10)

    def test_missing_and_zero_mean(self):
        values = sales_statistics([0, None, 2])
        self.assertEqual((values["observed_periods"], values["missing_periods"], values["mean"], values["zero_fraction"]), (2, 1, 1, .5))
        self.assertIsNone(sales_statistics([0, 0])["cv"])
        self.assertIsNone(sales_statistics([None, None])["adi"])
        for values in ([-1, 2], [float("inf"), 0]):
            with self.assertRaises(ValueError):
                sales_statistics(values)

    def test_adi_and_positive_size_cv2(self):
        stats = sales_statistics([0] * 24 + [2, 2, 3, 3, 2, 3])
        self.assertEqual(stats["adi"], 5)
        self.assertAlmostEqual(stats["positive_cv2"], .04)
        self.assertEqual(intermittent_class(30, 6, stats["positive_cv2"])[0], "intermittent")
        self.assertNotAlmostEqual(stats["positive_cv2"], stats["cv"] ** 2)

    def test_regime_edges_and_unsupported(self):
        self.assertEqual(intermittent_class(132, 100, .49)[0], "lumpy")
        self.assertEqual(intermittent_class(100, 100, .49)[0], "erratic")
        self.assertEqual(intermittent_class(100, 100, .48)[0], "smooth")
        self.assertEqual(intermittent_class(30, 0, None), ("unclassified", "all_zero"))
        self.assertEqual(intermittent_class(30, 1, None)[1], "insufficient_positive_history")

    def test_xyz_thresholds_and_guards(self):
        self.assertEqual(xyz_class(20, 20, 10, 5)[0], "X")
        self.assertEqual(xyz_class(20, 20, 10, 10)[0], "Y")
        self.assertEqual(xyz_class(20, 20, 10, 11)[0], "Z")
        self.assertEqual(xyz_class(20, 0, 0, 0)[1], "all_zero")
        self.assertEqual(xyz_class(3, 3, 10, 1)[1], "insufficient_weeks")
        self.assertEqual(xyz_class(20, 1, 1, 5)[1], "insufficient_positive_weeks")
        self.assertEqual(xyz_class(20, 10, .1, .1)[1], "low_mean")
        self.assertEqual(xyz_class(0, 0, None, None)[1], "no_complete_weeks")

    @staticmethod
    def products(values):
        return pd.DataFrame({"item_id": list(values), "known_revenue": [Decimal(str(x)) for x in values.values()],
                             "unknown_revenue_count": 0, "units_sold": 1})

    def test_abc_crossing_and_exact_boundaries(self):
        result = abc_classification(self.products({"a": 50, "b": 30, "c": 10, "d": 7, "e": 3}))
        self.assertEqual(result.abc_class.tolist(), list("AABBC"))
        result = abc_classification(self.products({"a": 90, "b": 6, "c": 4, "zero": 0}))
        self.assertEqual(result.abc_class.tolist(), list("ABCC"))
        self.assertAlmostEqual(result.revenue_share.sum(), 1)
        self.assertEqual(result.cumulative_revenue_share.iloc[-1], 1)

    def test_abc_ties_incomplete_and_all_zero(self):
        products = self.products({"c": 1, "a": 1, "b": 1})
        first = abc_classification(products)
        second = abc_classification(products.sample(frac=1, random_state=7))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.item_id.tolist(), ["a", "b", "c"])
        products.loc[0, "unknown_revenue_count"] = 1
        result = abc_classification(products)
        self.assertEqual(result.loc[result.item_id.eq("c"), "abc_class"].iloc[0], "unclassified")
        self.assertTrue(abc_classification(self.products({"a": 0, "b": 0})).abc_class.eq("unclassified").all())

    def test_join_grain_and_matrix_reconciliation(self):
        abc = abc_classification(self.products({"a": 80, "b": 20}))
        xyz = pd.DataFrame({"item_id": ["a", "b"], "xyz_class": ["X", "Z"]})
        combined = combine_segments(abc, xyz)
        matrix, unsupported = abc_xyz_matrix(combined, 2, Decimal(100))
        self.assertEqual(len(matrix), 9)
        self.assertEqual(matrix.products.sum(), 2)
        self.assertAlmostEqual(matrix.volume_share.sum(), 1)
        self.assertAlmostEqual(matrix.revenue_share.sum(), 1)
        self.assertEqual(len(unsupported), 0)
        with self.assertRaises(ValueError):
            combine_segments(abc, pd.concat([xyz, xyz.iloc[:1]]))
        with self.assertRaises(ValueError):
            combine_segments(abc, xyz.iloc[:1])

    def test_prior_average_no_current_or_future_leakage(self):
        before = prior_average([1, 2, 3, 100], 2)
        after = prior_average([1, 2, 3, 999, 1000], 2)
        pd.testing.assert_series_equal(before.iloc[:4], after.iloc[:4])
        self.assertEqual(before.iloc[3], 2.5)

    def test_outliers_do_not_mutate_and_history_is_prior_only(self):
        frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=70), "units_sold": [10] * 69 + [1000]})
        saved = frame.copy(deep=True)
        flags = outlier_flags(frame)
        self.assertTrue(flags.iqr_flag.iloc[-1])
        self.assertEqual(len(flags), len(frame))
        pd.testing.assert_frame_equal(frame, saved)
        self.assertEqual(flags.prior_average.iloc[-1], 10)
        self.assertFalse(flags.robust_evaluable.iloc[-1])  # zero MAD is undefined, not a forced z-score

    def test_config_rejects_invalid_choices(self):
        for kwargs in ({"abc_a": .96}, {"xyz_y": .1}, {"min_positive_days": 1}, {"start_date": "bad"}):
            with self.assertRaises(ValueError):
                EDAConfig(**kwargs)


@unittest.skipUnless(importlib.util.find_spec("duckdb"), "Install .[eda] for SQL fixture checks")
class EDASQLTests(unittest.TestCase):
    def test_weekly_grain_partial_weeks_and_training_cutoff(self):
        import duckdb
        from demandos.eda.queries import create_views, fetch_frame
        root = Path(__file__).resolve().parents[1]
        dates = pd.date_range("2020-01-04", periods=16)  # Saturday; two full weeks plus two days
        product = pd.DataFrame({"item_id": "synthetic", "date": dates, "units_sold": [1] * 14 + [999, 999]})
        calendar = pd.DataFrame({"date": dates, "wm_yr_wk": [1] * 7 + [2] * 7 + [3] * 2})
        with duckdb.connect() as con:
            con.register("eda_product_daily", product)
            con.register("eda_calendar", calendar)
            sql = (root / "sql/eda/weekly_product_statistics.sql").read_text()
            result = fetch_frame(con, sql).iloc[0]
            self.assertEqual(result.weekly_periods, 2)
            self.assertEqual(result.mean_weekly_units, 7)
            self.assertEqual(result.std_weekly_units, 0)

    def test_store_item_sql_and_reconciliation(self):
        import duckdb
        from demandos.eda.queries import expand_quantiles, fetch_frame
        root = Path(__file__).resolve().parents[1]
        data = pd.DataFrame({"store_id": "store", "item_id": "item", "date": pd.date_range("2020-01-01", periods=4),
                             "units_sold": [0, 0, 2, 4], "sell_price": [None, 2.0, 2.0, 2.0],
                             "observed_revenue": [0, 0, 4, 8]})
        with duckdb.connect() as con:
            con.register("eda_fact", data)
            frame = expand_quantiles(fetch_frame(con, (root / "sql/eda/store_item_statistics.sql").read_text(), ["store"]))
        row = frame.iloc[0]
        self.assertEqual(row.units_sold, 6)
        self.assertEqual(row.missing_price_count, 1)
        self.assertEqual(row.zero_fraction, .5)
        self.assertEqual(row.adi, 2)
        self.assertAlmostEqual(row.positive_cv2, 1 / 9)
        self.assertEqual(row.leading_zero_days, 2)
        self.assertEqual(series_segments(frame, EDAConfig()).regime_after_first_positive.iloc[0], "smooth")

    @unittest.skipUnless(importlib.util.find_spec("statsmodels") and importlib.util.find_spec("matplotlib") and importlib.util.find_spec("jinja2"), "Install .[analytics,eda] for full analysis fixtures")
    def test_full_synthetic_aggregation_and_training_window(self):
        import duckdb
        from jinja2 import Environment
        from demandos.eda.pipeline import collect

        root = Path(__file__).resolve().parents[1]
        dates = pd.date_range("2020-01-04", periods=16)
        rows = []
        for day, when in enumerate(dates):
            for store in ("store_a", "store_b"):
                for item in ("a", "b"):
                    units = 2 if item == "a" else day % 2
                    rows.append({"store_id": store, "state_id": store, "item_id": item,
                                 "dept_id": "dept", "cat_id": "cat", "date": when,
                                 "d": f"d_{day + 1}", "wm_yr_wk": day // 7 + 1,
                                 "units_sold": units, "sell_price": 1.5 if item == "a" else None,
                                 "observed_revenue": units * 1.5 if item == "a" else 0 if units == 0 else None})
        data = pd.DataFrame(rows)
        calendar = pd.DataFrame({"date": dates, "wm_yr_wk": [i // 7 + 1 for i in range(16)],
                                 "event_name_1": None, "event_name_2": None})
        with duckdb.connect() as con:
            con.register("fixture", data)
            con.register("calendar_fixture", calendar)
            con.execute("""create table fact_daily_sales as select * exclude(sell_price, observed_revenue),
                cast(sell_price as decimal(12,4)) as sell_price, sell_price is not null as price_is_known,
                cast(observed_revenue as decimal(30,4)) as observed_revenue from fixture""")
            con.execute("create table dim_product as select distinct item_id, dept_id, cat_id from fixture")
            con.execute("create table dim_store as select distinct store_id, state_id from fixture")
            con.execute("create table dim_calendar as select * from calendar_fixture")
            macro = (root / "dbt/macros/sales_metrics.sql").read_text()
            environment = Environment()
            for name in ("store", "category", "product"):
                sql = (root / f"dbt/models/marts/mart_{name}_daily.sql").read_text()
                rendered = environment.from_string(macro + sql).render(ref=lambda name: name)
                con.execute(f"create table mart_{name}_daily as " + rendered)
            config = EDAConfig(end_date="2020-01-17", xyz_min_weeks=2)
            frames, baseline, _, _, full = collect(con, root, config)
            self.assertFalse(full)
            self.assertEqual((baseline["daily_rows"], baseline["units_sold"], baseline["known_revenue"]), (56, 70, Decimal(84)))
            self.assertEqual(frames["product_segments"].set_index("item_id").abc_xyz.to_dict(), {"a": "AX", "b": "unclassified"})
            fitted = frames["decomposition"]
            np.testing.assert_allclose(fitted.trend + fitted.seasonal + fitted.residual, fitted.units_sold, atol=1e-8)
            self.assertEqual(frames["selected_anomalies"].date.max(), pd.Timestamp("2020-01-17"))
            self.assertEqual(frames["product_segments"].weekly_periods.unique().tolist(), [2])
