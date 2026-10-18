"""Deterministic synthetic backtest fixtures, not measured M5 data."""

from dataclasses import replace
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from demandos.data.pipeline import fingerprint


AVAILABLE = importlib.util.find_spec("statsmodels") is not None and importlib.util.find_spec("duckdb") is not None


def fixture():
    from demandos.backtesting.design import BacktestConfig, schedules
    config = BacktestConfig(first_origin="2020-04-09", last_origin="2020-05-07", sliding_days=84,
                            models=("naive", "seasonal_naive", "moving_average", "croston", "sba", "tsb"))
    dates = pd.date_range("2020-01-01", periods=160)
    frames = {key: pd.DataFrame({"date": dates, "units_sold": (np.arange(len(dates)) % 7 + 1) * multiplier})
              for key, multiplier in (("A", 1), ("B", 10))}
    cohort = pd.DataFrame([{"series_id": key, "kind": "store_item", "regime": regime, "abc_xyz": "AX",
                            "segment_fit_end": pd.Timestamp(config.first_origin)} for key, regime in (("A", "smooth"), ("B", "intermittent"))])
    return config, dates, frames, cohort, schedules(dates, config)


@unittest.skipUnless(AVAILABLE, "Install analytics/eda extras")
class BacktestTests(unittest.TestCase):
    def test_primary_schedule_leap_day_and_final_target(self):
        from demandos.backtesting.design import BacktestConfig, schedules
        dates = pd.date_range("2011-01-29", "2016-04-24")
        schedule = schedules(dates, BacktestConfig())
        expanding = schedule[schedule.window.eq("expanding")]
        self.assertEqual(expanding.origin.dt.strftime("%Y-%m-%d").tolist(), ["2016-01-31", "2016-02-28", "2016-03-27"])
        self.assertEqual(expanding.iloc[1].target_start, pd.Timestamp("2016-02-29"))
        self.assertEqual(expanding.iloc[-1].target_end, pd.Timestamp("2016-04-24"))
        self.assertEqual(expanding.training_days.tolist(), [1829, 1857, 1885])
        self.assertTrue(schedule[schedule.window.eq("sliding")].training_days.eq(730).all())
        self.assertTrue(expanding.history_start.eq(dates[0]).all())
        with self.assertRaises(ValueError):
            schedules(dates, replace(BacktestConfig(), last_origin="2016-03-28"))

    def test_overlapping_schedules_and_invalid_calendar(self):
        from demandos.backtesting.design import schedules
        config, dates, _, _, _ = fixture()
        overlapped = schedules(dates, replace(config, stride=7, windows=("expanding",)))
        self.assertEqual(len(overlapped), 5)
        targets = [date for row in overlapped.itertuples() for date in pd.date_range(row.target_start, row.target_end)]
        self.assertLess(len(set(targets)), len(targets))
        regular = schedules(dates, replace(config, windows=("expanding",)))
        targets = [date for row in regular.itertuples() for date in pd.date_range(row.target_start, row.target_end)]
        self.assertEqual(len(set(targets)), len(targets))
        for bad in (dates.delete(5), dates.append(dates[:1]), dates.append(pd.DatetimeIndex([pd.NaT]))):
            with self.assertRaises(ValueError):
                schedules(bad, config)

    def test_configuration_and_training_limits(self):
        from demandos.backtesting.design import BacktestConfig, schedules
        for kwargs in ({"horizon": 29}, {"stride": 0}, {"windows": ["random"]}, {"models": ["fake"]},
                       {"arima_series": 17}, {"sarima_series": 9}, {"first_origin": "2017-01-01"}, {"max_store_items": 31}):
            with self.assertRaises(ValueError):
                BacktestConfig(**kwargs)
        config, dates, _, _, _ = fixture()
        with self.assertRaises(ValueError):
            schedules(dates, replace(config, sliding_days=120))

    def test_numeric_metrics_and_first_positive_scale(self):
        from demandos.backtesting.metrics import rmsse_scale, point_metrics
        scale = rmsse_scale([0, 0, 10, 12, 11, 15])
        self.assertEqual(scale["scale"], 7.)
        self.assertEqual(scale["scale_pairs"], 3)
        self.assertEqual(scale["first_positive_index"], 2)
        measured = point_metrics([14, 18], [12, 15], scale["scale"])
        self.assertEqual(measured["mae"], 2.5)
        self.assertEqual(measured["wape"], 5 / 32)
        self.assertAlmostEqual(measured["rmsse"], np.sqrt(6.5 / 7))
        self.assertEqual(rmsse_scale([0, 5, 0])["scale"], 25.)

    def test_zero_constant_one_event_and_invalid_scales(self):
        from demandos.backtesting.metrics import rmsse_scale, point_metrics
        self.assertEqual(rmsse_scale([0, 0])["scale_status"], "no_positive_history")
        self.assertEqual(rmsse_scale([0, 3])["scale_status"], "insufficient_after_first_positive")
        self.assertEqual(rmsse_scale([0, 3, 3])["scale_status"], "zero_scale")
        for values in ([], [np.nan], [-1], [[1, 2]]):
            with self.assertRaises(ValueError):
                rmsse_scale(values)
        result = point_metrics([0, 0], [1, 2], 0)
        self.assertEqual(result["mae"], 1.5)
        self.assertIsNone(result["wape"])
        self.assertIsNone(result["rmsse"])
        self.assertEqual(result["wape_status"], "zero_actual_total")

    def test_micro_macro_and_undefined_coverage(self):
        from demandos.backtesting.metrics import score
        frame = pd.DataFrame({"window": "expanding", "origin": pd.Timestamp("2020-01-01"),
                              "series_id": ["A", "B", "C"], "actual": [1., 100., 0.],
                              "prediction": [0., 90., 0.], "scale": [1., 100., 0.]})
        measured = score(frame)
        self.assertAlmostEqual(measured["wape"], 11 / 101)
        self.assertAlmostEqual(measured["macro_wape"], .55)
        self.assertEqual(measured["macro_wape_cases"], 2)
        self.assertEqual(measured["macro_rmsse"], 1.)
        self.assertEqual(measured["rmsse_cases"], 2)
        self.assertEqual(measured["undefined_rmsse_cases"], 1)
        self.assertEqual(measured["rmsse_observations"], 2)
        self.assertEqual(measured["rmsse_status"], "partial_undefined_scales")
        frame.loc[1, "prediction"] = 60.
        measured = score(frame)
        self.assertEqual(measured["macro_rmsse"], 2.5)
        self.assertAlmostEqual(measured["micro_scaled_rmse"], np.sqrt(8.5))
        uneven = pd.concat([frame.iloc[:2], frame.iloc[[1, 1]]])
        self.assertNotEqual(score(uneven)["mae"], score(uneven)["macro_mae"])

    def test_horizon_and_bands(self):
        from demandos.backtesting.metrics import horizon_band
        np.testing.assert_equal(horizon_band([1, 7, 8, 14, 15, 28]), ["01-07", "01-07", "08-14", "08-14", "15-28", "15-28"])
        for bad in ([0], [29], [1.5]):
            with self.assertRaises(ValueError):
                horizon_band(bad)

    def test_origin_invariance_scale_and_no_cross_series(self):
        from demandos.backtesting.pipeline import execute
        config, dates, frames, cohort, schedule = fixture()
        schedule = schedule[schedule.origin.eq(pd.Timestamp(config.first_origin))]
        original, _ = execute(frames, cohort, schedule, config)
        changed = {key: frame.copy() for key, frame in frames.items()}
        for frame in changed.values():
            frame.loc[frame.date.gt(pd.Timestamp(config.first_origin)), "units_sold"] += 100000
        later, _ = execute(changed, cohort, schedule, config)
        for column in ("prediction", "raw_prediction", "scale"):
            pd.testing.assert_series_equal(original["forecasts"][column], later["forecasts"][column])
        self.assertFalse(original["forecasts"].actual.equals(later["forecasts"].actual))
        forecasts = original["forecasts"].query("model == 'naive' and window == 'expanding'")
        self.assertAlmostEqual(forecasts[forecasts.series_id.eq("B")].prediction.iloc[0], forecasts[forecasts.series_id.eq("A")].prediction.iloc[0] * 10)
        for h in (1, 7, 14, 28):
            self.assertTrue(original["forecasts"].horizon.eq(h).any())
        self.assertTrue(original["training"].date.le(original["training"].origin).all())

    def test_future_segment_and_bad_targets_rejected(self):
        from demandos.backtesting.pipeline import execute
        config, _, frames, cohort, schedule = fixture()
        with self.assertRaisesRegex(ValueError, "future"):
            execute(frames, cohort.assign(segment_fit_end=pd.Timestamp("2025-01-01")), schedule, config)
        changed = dict(frames)
        changed["A"] = frames["A"].drop(index=103)
        with self.assertRaisesRegex(ValueError, "planned targets"):
            execute(changed, cohort, schedule, config)

    def test_failures_caps_planned_coverage_and_exact_pairs(self):
        from demandos.backtesting.pipeline import execute, analyze
        from demandos.backtesting.metrics import common_cases, paired_losses
        from demandos.forecasting.models import ForecastResult, forecast
        config, _, frames, cohort, schedule = fixture()
        config = replace(config, models=("naive", "seasonal_naive", "arima"), arima_series=1)
        def selective(history, horizon, model, **kwargs):
            return ForecastResult(model, "failed", "synthetic_failure") if model == "arima" else forecast(history, horizon, model, **kwargs)
        with patch("demandos.backtesting.pipeline.forecast", side_effect=selective):
            outputs, _ = execute(frames, cohort, schedule, config)
        planned = len(cohort) * len(schedule) * len(config.models)
        self.assertEqual(len(outputs["fit_status"]), planned)
        self.assertEqual(len(outputs["forecasts"]), planned * 28)
        statuses = outputs["fit_status"]
        self.assertEqual(set(statuses[statuses.model.eq("arima")].status), {"failed", "skipped"})
        self.assertTrue(common_cases(outputs["forecasts"], ["naive", "arima"]).empty)
        paired = paired_losses(outputs["forecasts"])
        self.assertEqual(set(paired.model), {"naive"})
        tables = analyze(outputs, config)
        self.assertFalse(tables["coverage"].empty)
        self.assertNotIn("arima_capped", set(tables["metrics_common"].comparison))
        with self.assertRaises(ValueError):
            common_cases(pd.concat([outputs["forecasts"], outputs["forecasts"].query("status == 'ok'").iloc[:1]]), ["naive", "seasonal_naive"])

    def test_paired_numeric_sign_and_origin_keys(self):
        from demandos.backtesting.metrics import paired_losses
        frame = pd.DataFrame([{"window": "expanding", "series_id": "A", "origin": origin,
                               "target_date": origin + pd.Timedelta(days=1), "horizon": 1, "model": model,
                               "status": "ok", "kind": "store_item", "actual": 10., "prediction": prediction}
                              for origin in pd.to_datetime(["2020-01-01", "2020-01-02"])
                              for model, prediction in (("seasonal_naive", 8.), ("naive", 9.))])
        pair = paired_losses(frame)
        self.assertEqual(len(pair), 2)
        np.testing.assert_equal(pair.absolute_loss_difference.to_numpy(), [-1., -1.])
        np.testing.assert_equal(pair.squared_loss_difference.to_numpy(), [-3., -3.])
        changed = frame.copy()
        changed.loc[0, "actual"] = 100.
        with self.assertRaises(ValueError):
            paired_losses(changed)

    def test_repeatability_and_source_immutability(self):
        from demandos.backtesting.pipeline import execute, analyze
        config, _, frames, cohort, schedule = fixture()
        config = replace(config, models=("naive", "seasonal_naive", "moving_average"))
        before = {key: frame.copy(deep=True) for key, frame in frames.items()}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            frames["A"].to_csv(path, index=False)
            digest = fingerprint(path)
            first, _ = execute(frames, cohort, schedule, config)
            second, _ = execute(frames, cohort, schedule, config)
            cols = [c for c in first["forecasts"] if c != "runtime_seconds"]
            pd.testing.assert_frame_equal(first["forecasts"][cols], second["forecasts"][cols])
            self.assertEqual(digest, fingerprint(path))
        for key in frames:
            pd.testing.assert_frame_equal(frames[key], before[key])
        tables = analyze(first, config)
        self.assertEqual(tables["metrics_common_horizon"].horizon.nunique(), 28)
        self.assertEqual(set(tables["metrics_common_band"].horizon_band), {"01-07", "08-14", "15-28"})
        # Last-observation, last-cycle and fixed trailing means are window invariant.
        sensitivity = tables["window_sensitivity"]
        for _, group in sensitivity.groupby("model"):
            self.assertEqual(group.mae.nunique(), 1)

    def test_hierarchy_reconciles_all_twelve_levels(self):
        from demandos.backtesting.hierarchy import aggregate_hierarchy, LEVELS
        bottom = pd.DataFrame([{"store_id": store, "state_id": state, "item_id": item, "dept_id": "D", "cat_id": "C", "date": date, "units_sold": value}
                               for store, state, value in (("S1", "CA", 2), ("S2", "TX", 3))
                               for item in ("I1", "I2") for date in pd.date_range("2020-01-01", periods=2)])
        result = aggregate_hierarchy(bottom)
        self.assertEqual(len(LEVELS), 12)
        self.assertEqual(result.level.nunique(), 12)
        self.assertTrue(result.groupby(["level", "date"]).units_sold.sum().eq(10).all())
        with self.assertRaises(ValueError):
            aggregate_hierarchy(pd.concat([bottom, bottom.iloc[:1]]))

    def test_historical_prices_availability_and_future_invariance(self):
        from demandos.backtesting.hierarchy import historical_revenue_weights, cohort_weighted_rmsse
        origin = pd.Timestamp("2020-01-02")
        data = pd.DataFrame([{"series_id": key, "date": date, "units_sold": units, "sell_price": price, "price_available_at": date}
                             for key, units, price in (("A", 1., 2.), ("B", 2., 3.))
                             for date in pd.date_range("2020-01-01", periods=3)])
        weights = historical_revenue_weights(data, origin, days=2)
        np.testing.assert_allclose(weights.weight, [.25, .75])
        changed = data.copy()
        changed.loc[changed.date.gt(origin), ["units_sold", "sell_price"]] = 1e9
        pd.testing.assert_frame_equal(weights, historical_revenue_weights(changed, origin, days=2))
        invalid = data.copy()
        invalid.loc[0, "price_available_at"] = origin + pd.Timedelta(days=1)
        with self.assertRaises(ValueError):
            historical_revenue_weights(invalid, origin, days=2)
        invalid.loc[0, "price_available_at"] = pd.NaT
        with self.assertRaises(ValueError):
            historical_revenue_weights(invalid, origin, days=2)
        scores = pd.DataFrame({"series_id": ["A", "B"], "rmsse": [1., 2.]})
        self.assertEqual(cohort_weighted_rmsse(scores, weights)["cohort_weighted_rmsse"], 1.75)
        scores.loc[0, "rmsse"] = np.nan
        self.assertIsNone(cohort_weighted_rmsse(scores, weights)["cohort_weighted_rmsse"])
        with self.assertRaises(ValueError):
            cohort_weighted_rmsse(scores.iloc[:1], weights)
        zero = historical_revenue_weights(data.assign(units_sold=0., sell_price=np.nan, price_available_at=pd.NaT), origin, days=2)
        self.assertTrue(zero.weight.isna().all())
        self.assertTrue(zero.weight_status.eq("zero_revenue_total").all())
        with self.assertRaises(ValueError):
            historical_revenue_weights(data.drop(index=0), origin, days=2)

    def test_manifest_clean_rerun_and_path_protection(self):
        import json
        from demandos.backtesting.pipeline import publish
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "new", root / "output"
            source.mkdir()
            (source / "old.txt").write_text("generated")
            publish(source, target)
            (target / "notes.txt").write_text("user notes")
            (source / "current.txt").write_text("regenerated")
            publish(source, target)
            self.assertFalse((target / "old.txt").exists())
            self.assertEqual((target / "current.txt").read_text(), "regenerated")
            self.assertEqual((target / "notes.txt").read_text(), "user notes")
            (target / "manifest.json").write_text(json.dumps(["../outside.txt"]))
            with self.assertRaises(ValueError):
                publish(source, target)

    def test_all_failed_experiment_reports_no_winner(self):
        from demandos.backtesting.pipeline import execute, analyze
        from demandos.backtesting.report import write_report, figures
        from demandos.forecasting.models import ForecastResult
        config, _, frames, cohort, schedule = fixture()
        config = replace(config, models=("arima",), windows=("expanding",))
        schedule = schedule[schedule.window.eq("expanding")]
        with patch("demandos.backtesting.pipeline.forecast", return_value=ForecastResult("arima", "failed", "synthetic")):
            outputs, _ = execute(frames, cohort, schedule, config)
        outputs.update(analyze(outputs, config), cohort=cohort, origin_calendar=schedule)
        self.assertTrue(outputs["metrics_common"].empty)
        self.assertTrue(outputs["forecasts"].prediction.isna().all())
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = figures(outputs, directory / "figures")
            summary = {"runtime_seconds": 0., "cohort": cohort.to_dict("records"), "fit_counts": {"failed": 4}, "warnings": 0,
                       "resources": {}, "figures": paths}
            write_report(summary, outputs, directory / "report.md")
            self.assertIn("No shared successful", (directory / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
