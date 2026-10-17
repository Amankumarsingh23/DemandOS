"""Deterministic synthetic time-series fixtures; not actual M5 findings."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from demandos.timeseries.temporal import (FeatureEvidence, validate_feature, daily_panel, descriptive_lags,
                                        difference, fit_location_scale, origin_features, chronological_split, origin_schedule)


def panel(n=140):
    return pd.DataFrame([{"store_id": store, "item_id": "synthetic", "date": when, "units_sold": i + offset}
                         for store, offset in (("A", 0), ("B", 1000))
                         for i, when in enumerate(pd.date_range("2020-01-01", periods=n))])


class TemporalTests(unittest.TestCase):
    def test_canonical_order_duplicates_and_missing_dates(self):
        frame = panel(40)
        pd.testing.assert_frame_equal(daily_panel(frame), daily_panel(frame.sample(frac=1, random_state=8)))
        for invalid in (frame.drop(index=5), pd.concat([frame, frame.iloc[:1]]), frame.assign(units_sold=np.nan)):
            with self.assertRaises(ValueError):
                daily_panel(invalid)

    def test_lags_1_7_14_28_and_no_cross_series_mixing(self):
        lags = descriptive_lags(panel(40))
        for _, group in lags.groupby(["store_id", "item_id"]):
            for lag in (1, 7, 14, 28):
                self.assertTrue(group[f"target_lag_{lag}"].iloc[:lag].isna().all())
                np.testing.assert_equal(group[f"target_lag_{lag}"].iloc[lag:].to_numpy(), group.units_sold.iloc[:-lag].to_numpy())
            self.assertTrue(np.isnan(group.prior_mean.iloc[0]))
            self.assertEqual(group.prior_mean.iloc[1], group.units_sold.iloc[0])

    def test_first_second_and_seasonal_difference_alignment(self):
        values = pd.Series(np.arange(35) * 10 + 100)
        self.assertTrue(difference(values).iloc[1:].eq(10).all())
        self.assertTrue(difference(values, order=2).iloc[2:].eq(0).all())
        self.assertTrue(difference(values, lag=7).iloc[7:].eq(70).all())
        self.assertTrue(difference(values, lag=7).iloc[:7].isna().all())
        with self.assertRaises(ValueError):
            difference(values, lag=0)

    def test_horizon_features_future_invariance_and_fit_cutoff(self):
        frame = panel()
        origin = pd.Timestamp("2020-04-09")  # day index 99
        fitted = fit_location_scale(frame, origin)
        first = origin_features(frame, origin, fitted=fitted)
        changed = frame.copy()
        changed.loc[changed.date.gt(origin), "units_sold"] = 1000000
        second = origin_features(changed, origin, fitted=fit_location_scale(changed, origin))
        pd.testing.assert_frame_equal(first, second)
        a = first[first.store_id.eq("A")]
        self.assertEqual(a.horizon.tolist(), [1, 7, 14, 28])
        self.assertTrue(a.sales_at_origin.eq(99).all())
        self.assertTrue(a.origin_minus_1.eq(98).all())
        self.assertTrue(a.origin_minus_28.eq(71).all())
        self.assertTrue(a.target_date.eq(a.origin + pd.to_timedelta(a.horizon, unit="D")).all())
        with self.assertRaises(ValueError):
            origin_features(frame, origin, fitted=fit_location_scale(frame, frame.date.max()))

    def test_evidence_rejects_future_centered_descriptive_and_unverified_prices(self):
        safe = FeatureEvidence("past", 1, "2020-01-10", "2020-01-10")
        self.assertTrue(validate_feature(safe, "2020-01-10"))
        cases = [replace(safe, source_date="2020-01-16"), replace(safe, available_at="2020-01-11"),
                 replace(safe, fitted_through="2020-01-11"), replace(safe, centered=True),
                 replace(safe, descriptive_only=True), replace(safe, kind="price"),
                 replace(safe, source_date=None)]
        for evidence in cases:
            with self.subTest(evidence=evidence):
                with self.assertRaises(ValueError):
                    validate_feature(evidence, "2020-01-10")
        self.assertTrue(validate_feature(replace(safe, kind="calendar", source_date="2020-02-01"), "2020-01-10"))

    def test_h7_naive_target_shift_is_rejected(self):
        origin = pd.Timestamp("2020-01-10")
        target = origin + pd.Timedelta(days=7)
        with self.assertRaises(ValueError):
            validate_feature(FeatureEvidence("naive_lag1", 99, target-pd.Timedelta(days=1), target-pd.Timedelta(days=1)), origin)

    def test_split_and_expanding_rolling_target_windows(self):
        dates = pd.date_range("2020-01-01", periods=140)
        split = chronological_split(dates, validation_days=28, test_days=28)
        self.assertEqual(split["train_end"] + pd.Timedelta(days=1), split["validation_start"])
        self.assertEqual(split["validation_end"] + pd.Timedelta(days=1), split["test_start"])
        schedule = origin_schedule(dates, 84, 28, 28)
        self.assertEqual(len(schedule), 2)
        self.assertTrue(schedule.target_end.le(dates[-1]).all())
        self.assertLess(schedule.target_end.iloc[0], schedule.target_start.iloc[1])
        rolling = origin_schedule(dates, 84, 28, 28, training_window=30)
        self.assertTrue((rolling.origin - rolling.history_start).dt.days.eq(29).all())
        with self.assertRaises(ValueError):
            chronological_split(dates.delete(2))


@unittest.skipUnless(importlib.util.find_spec("statsmodels"), "Install .[eda] for statistical diagnostics")
class DiagnosticTests(unittest.TestCase):
    def test_skip_cases_without_fabricated_estimates(self):
        from demandos.timeseries.diagnostics import correlation_diagnostics, stationarity_diagnostics
        for values in (np.zeros(140), np.ones(140), np.arange(8), [0]*139 + [1], [1, np.nan]*70):
            self.assertEqual(correlation_diagnostics(values)["status"], "skipped")
            result = stationarity_diagnostics(values)
            self.assertEqual(result["status"], "skipped")
            self.assertIsNone(result["adf"])

    def test_acf_estimator_and_pacf_sample_bound(self):
        from demandos.timeseries.diagnostics import correlation_diagnostics, DiagnosticsConfig
        values = np.random.default_rng(42).normal(size=200)
        result = correlation_diagnostics(values, DiagnosticsConfig(max_lags=200))
        self.assertEqual(result["status"], "ok")
        centered = values - values.mean()
        expected = np.dot(centered[1:], centered[:-1]) / np.dot(centered, centered)
        self.assertAlmostEqual(result["coefficients"][1]["acf"], expected)
        self.assertLess(result["max_lags"], len(values)/2)

    def test_stationary_like_and_trending_without_brittle_significance(self):
        from demandos.timeseries.diagnostics import stationarity_diagnostics
        noise = np.random.default_rng(22).normal(size=256)
        for values in (noise, noise + np.arange(256)/10):
            result = stationarity_diagnostics(values)
            self.assertEqual(result["status"], "ok")
            self.assertTrue(np.isfinite(result["adf"]["pvalue"]))
            self.assertIn(result["interpretation"], ("compatible_with_stationarity", "compatible_with_nonstationarity", "conflicting", "inconclusive"))

    def test_numerical_test_failure_is_explicit(self):
        from demandos.timeseries.diagnostics import stationarity_diagnostics, correlation_diagnostics
        values = np.random.default_rng(9).normal(size=140)
        with patch("demandos.timeseries.diagnostics.adfuller", side_effect=ValueError("synthetic failure")):
            result = stationarity_diagnostics(values)
            self.assertEqual(result["status"], "partial_or_failed")
            self.assertEqual(result["interpretation"], "inconclusive")
            self.assertIsNone(result["adf"]["pvalue"])
        with patch("demandos.timeseries.diagnostics.pacf", side_effect=np.linalg.LinAlgError("synthetic singularity")):
            self.assertEqual(correlation_diagnostics(values)["status"], "failed")

    def test_stl_alignment_origin_prefix_and_usage_contract(self):
        from demandos.timeseries.diagnostics import additive_stl
        frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=140),
                              "units_sold": 20 + np.sin(np.arange(140)*2*np.pi/7)})
        fitted, reason = additive_stl(frame, origin="2020-04-09")
        self.assertIsNone(reason)
        self.assertEqual(len(fitted), 100)
        np.testing.assert_allclose(fitted.trend + fitted.seasonal + fitted.residual, fitted.units_sold)
        self.assertTrue(fitted.usage.eq("descriptive_only").all())
        with self.assertRaises(ValueError):
            additive_stl(frame, usage="forecast_feature")
        with self.assertRaises(ValueError):
            additive_stl(frame.drop(index=4))


@unittest.skipUnless(importlib.util.find_spec("duckdb") and importlib.util.find_spec("statsmodels"), "Install analytics/eda extras")
class TrainingSelectionTests(unittest.TestCase):
    def test_training_segments_and_cohort_ignore_changed_future_sales(self):
        import duckdb
        from demandos.timeseries.cohort import training_labels, select_cohort
        root = Path(__file__).resolve().parents[1]
        with duckdb.connect() as con:
            con.execute("""create table fact_daily_sales as select 'S' as store_id, item_id,
                date '2020-01-04' + cast(i as integer) as date,
                1 + cast(floor(i/7) as integer) as wm_yr_wk,
                case when item_id='A' then 2 else i%3 end as units_sold,
                cast((case when item_id='A' then 2 else i%3 end) * 0.5 as decimal(20,4)) as observed_revenue
                from range(140) x(i) cross join (values ('A'),('B')) p(item_id)""")
            con.execute("create table dim_calendar as select distinct date, wm_yr_wk from fact_daily_sales")
            con.execute("create table dim_store as select 'S' as store_id, 'state' as state_id")
            def marts():
                for name, key in (("product", "item_id"), ("store", "store_id")):
                    con.execute(f"""create or replace table mart_{name}_daily as select {key}, date,
                        count(*) as observation_count, sum(units_sold) as units_sold,
                        sum(observed_revenue) as known_revenue,
                        count(*) filter (where units_sold=0) as zero_sales_count,
                        0 as missing_price_count, 0 as unknown_revenue_count
                        from fact_daily_sales group by {key}, date""")
                con.execute("create or replace view mart_category_daily as select * from mart_store_daily")
            marts()
            start, end = pd.Timestamp("2020-01-04"), pd.Timestamp("2020-04-24")
            first, products, _ = training_labels(con, root, start, end)
            con.execute("update fact_daily_sales set units_sold=100000, observed_revenue=50000 where date > date '2020-04-24'")
            marts()
            second, other_products, _ = training_labels(con, root, start, end)
            pd.testing.assert_frame_equal(first, second)
            pd.testing.assert_frame_equal(products, other_products)
            pd.testing.assert_frame_equal(select_cohort(first), select_cohort(first.sample(frac=1, random_state=3)))
