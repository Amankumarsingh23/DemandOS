"""Deterministic synthetic fixtures; these are not real M5 observations."""

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


def history(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values)))


@unittest.skipUnless(AVAILABLE, "Install the analytics/eda extras")
class ForecastTests(unittest.TestCase):
    def setUp(self):
        from demandos.forecasting.models import forecast
        self.forecast = forecast

    def test_exact_simple_forecasts_all_horizons_and_cycling(self):
        source = history(np.arange(1, 15))
        for h in (1, 7, 14, 28):
            with self.subTest(h=h):
                np.testing.assert_equal(self.forecast(source, h, "naive").predictions, np.repeat(14, h))
                np.testing.assert_equal(self.forecast(source, h, "seasonal_naive").predictions, np.resize(np.arange(8, 15), h))
                np.testing.assert_equal(self.forecast(source, h, "moving_average", params={"window": 3}).predictions, np.repeat(13, h))
        self.assertEqual(self.forecast(source.iloc[:6], 28, "seasonal_naive").status, "skipped")
        self.assertEqual(self.forecast(source, 28, "moving_average", params={"window": 15}).status, "skipped")
        self.assertTrue(self.forecast(history([9, 0, 0]), 7, "moving_average", params={"window": 2}).predictions.sum() == 0)

    def test_ses_known_initial_level_and_alpha(self):
        result = self.forecast(history([120, 130]), 7, "ses", params={"initial_level": 100, "smoothing_level": .3, "optimized": False})
        # Known initial state is before the first observation: 106, then 113.2.
        # Three points are the generic optimization guard; known states need only one.
        self.assertEqual(result.status, "ok", result.reason)
        np.testing.assert_allclose(result.predictions, np.repeat(113.2, 7))

    def test_holt_exact_linear_and_additive_weekly(self):
        linear = history(np.arange(100., 240., 5))
        fitted = self.forecast(linear, 7, "holt")
        self.assertEqual(fitted.status, "ok", fitted.reason)
        np.testing.assert_allclose(fitted.predictions, linear.iloc[-1] + 5 * np.arange(1, 8), atol=.03)
        seasonal = history(np.tile([10., 12., 14., 16., 18., 20., 22.], 12))
        fitted = self.forecast(seasonal, 28, "holt_winters", params={"initial_level": 16., "initial_trend": 0.,
            "initial_seasonal": [-6., -4., -2., 0., 2., 4., 6.], "optimized": False,
            "smoothing_level": .2, "smoothing_trend": .1, "smoothing_seasonal": .1})
        self.assertEqual(fitted.status, "ok", fitted.reason)
        np.testing.assert_allclose(fitted.predictions, np.tile(seasonal.iloc[-7:].to_numpy(), 4), atol=.05)

    def test_damped_trend_analytic_states(self):
        source = history(np.arange(105., 245., 5))
        result = self.forecast(source, 7, "holt_damped", params={"initial_level": 100., "initial_trend": 5.,
            "smoothing_level": .3, "smoothing_trend": .2, "damping_trend": .9, "optimized": False})
        self.assertEqual(result.status, "ok", result.reason)
        # Derive the final level/trend independently using the additive damped recursions.
        level, trend = 100., 5.
        for value in source:
            previous = level
            level = .3 * value + .7 * (level + .9 * trend)
            trend = .2 * (level - previous) + .8 * .9 * trend
        expected = level + trend * np.cumsum(.9 ** np.arange(1, 8))
        np.testing.assert_allclose(result.predictions, expected)

    def test_croston_intervals_sizes_sba_and_tsb(self):
        source = history([0., 4., 0., 0., 8., 0., 0.])
        # First event t=2: z=4,p=2. Second event gap=3: z=6,p=2.5 (alpha=.5).
        croston = self.forecast(source, 28, "croston", params={"alpha": .5})
        np.testing.assert_allclose(croston.predictions, 2.4)
        self.assertEqual(croston.metadata["interval"], 2.5)
        self.assertEqual(croston.metadata["size"], 6.)
        np.testing.assert_allclose(self.forecast(source, 28, "sba", params={"alpha": .5}).predictions, 1.8)
        tsb = self.forecast(source, 28, "tsb", params={"alpha": .5, "beta": .5})
        # p=.5, then .25,.125,.5625,.28125,.140625; z=6.
        np.testing.assert_allclose(tsb.predictions, .84375)
        extended = history(source.tolist() + [0.] * 4)
        np.testing.assert_allclose(self.forecast(extended, 28, "tsb", params={"alpha": .5, "beta": .5}).predictions, tsb.predictions / 16)
        np.testing.assert_allclose(self.forecast(extended, 28, "croston", params={"alpha": .5}).predictions, croston.predictions)

    def test_zero_constant_short_and_one_event_statuses(self):
        from demandos.forecasting.models import MODELS
        for model in MODELS:
            zero = self.forecast(history(np.zeros(50)), 28, model)
            if model in ("naive", "seasonal_naive", "moving_average", "croston", "sba", "tsb"):
                self.assertEqual(zero.status, "ok")
                np.testing.assert_equal(zero.predictions, np.zeros(28))
            else:
                self.assertEqual(zero.status, "skipped")
                self.assertIsNone(zero.predictions)
            constant = self.forecast(history(np.ones(50)), 28, model)
            self.assertIn(constant.status, ("ok", "skipped"))
        for model in ("croston", "sba", "tsb"):
            self.assertEqual(self.forecast(history([0, 4, 0]), 7, model).reason, "fewer_than_two_positive_events")
        self.assertEqual(self.forecast(history([2]), 7, "holt").status, "skipped")

    def test_arima_sarima_output_and_bounded_failures(self):
        rng = np.random.default_rng(17)
        source = history(20 + np.tile(np.arange(7), 20) + rng.normal(size=140))
        for model in ("arima", "sarima"):
            result = self.forecast(source, 28, model, params={"maxiter": 100})
            self.assertEqual(result.status, "ok", result.reason)
            self.assertEqual(result.predictions.shape, (28,))
            self.assertTrue(np.isfinite(result.predictions).all())
            self.assertEqual(result.metadata["maxiter"], 100)
            with patch("demandos.forecasting.models.ARIMA", side_effect=RuntimeError("synthetic fit failure")):
                failed = self.forecast(source, 28, model)
                self.assertEqual(failed.status, "failed")
                self.assertIn("synthetic fit failure", failed.reason)
                self.assertIsNone(failed.predictions)
        with patch("demandos.forecasting.models._library_fit") as fit:
            fit.return_value = (type("Fit", (), {"mle_retvals": {"converged": False}})(), {})
            failed = self.forecast(source, 7, "arima", params={"maxiter": 1})
            self.assertEqual(failed.reason, "optimizer_did_not_converge")
        self.assertEqual(self.forecast(source, 28, "arima", params={"order": (4, 0, 0)}).status, "failed")

    def test_invalid_input_nonfinite_output_and_clipping(self):
        source = history([1., 2., 3., 4.])
        invalid = [history([]), history([1, np.nan]), history([1, -1]), source.iloc[::-1],
                   source.drop(source.index[2]), pd.concat([source, source.iloc[:1]])]
        for bad in invalid:
            self.assertEqual(self.forecast(bad, 7, "naive").status, "failed")
        for kwargs in ({"horizon": 0}, {"horizon": True}, {"seasonal_period": 1}):
            self.assertEqual(self.forecast(source, kwargs.get("horizon", 7), "naive", seasonal_period=kwargs.get("seasonal_period", 7)).status, "failed")
        for raw in (np.array([-3., 2.]), np.array([np.nan, 2.])):
            dummy = type("Fit", (), {"forecast": lambda self, h: raw})()
            with patch("demandos.forecasting.models._library_fit", return_value=(dummy, {})):
                result = self.forecast(source, 2, "ses")
            if np.isfinite(raw).all():
                np.testing.assert_equal(result.raw_predictions, raw)
                np.testing.assert_equal(result.predictions, [0, 2])
                self.assertEqual(result.metadata["clipped_count"], 1)
            else:
                self.assertEqual(result.status, "failed")

    def test_origin_future_invariance_and_no_mixing(self):
        from demandos.forecasting.models import MODELS, at_origin
        source = history(np.arange(100.) % 7 + 1)
        frame = pd.DataFrame({"date": source.index, "units_sold": source.to_numpy(), "store_id": "A", "item_id": "synthetic"})
        original = frame.copy(deep=True)
        origin = source.index[69]
        changed = frame.copy()
        changed.loc[changed.date.gt(origin), "units_sold"] = 1e9
        for model in MODELS:
            first = self.forecast(at_origin(frame, origin), 28, model)
            second = self.forecast(at_origin(changed, origin), 28, model)
            self.assertEqual(first.status, second.status)
            if first.status == "ok":
                np.testing.assert_equal(first.predictions, second.predictions)
            self.assertLessEqual(first.metadata["fit_end"], origin)
        pd.testing.assert_frame_equal(frame, original)
        mixed = frame.copy()
        mixed.loc[0, "store_id"] = "B"
        with self.assertRaises(ValueError):
            at_origin(mixed, origin)
        with self.assertRaises(ValueError):
            at_origin(frame.drop(index=69), origin)

    def test_metrics_and_common_cohort(self):
        from demandos.forecasting.evaluation import metrics, comparisons
        measured = metrics([0, 2, 8], [1, 4, 5])
        self.assertEqual(measured["mae"], 2.)
        self.assertEqual(measured["wape"], .6)
        self.assertIsNone(metrics([0, 0], [1, 2])["wape"])
        for a, p in (([], []), ([1], [1, 2]), ([np.nan], [0]), ([-1], [0])):
            with self.assertRaises(ValueError):
                metrics(a, p)
        frame = pd.DataFrame([{"series_id": key, "target_date": pd.Timestamp("2020-01-01"), "horizon": 1, "model": model}
                              for key, model in (("A", "naive"), ("A", "ses"), ("B", "naive"))])
        common = comparisons(frame, ["naive", "ses"])
        self.assertEqual(set(common.series_id), {"A"})
        self.assertEqual(len(common), 2)
        with self.assertRaises(ValueError):
            comparisons(pd.concat([frame, frame.iloc[:1]]), ["naive", "ses"])

    def test_config_and_no_future_labels(self):
        from demandos.forecasting.pipeline import ForecastConfig, fit_cohort, sarima_cohort
        config = ForecastConfig()
        for kwargs in ({"maxiter": 501}, {"sarima_series": 9}, {"arima_order": [4, 0, 0]}, {"lookback": 2}):
            with self.assertRaises(ValueError):
                replace(config, **kwargs)
        origin = pd.Timestamp("2020-02-01")
        cohort = pd.DataFrame([{"series_id": "A", "regime": "smooth", "kind": "store_item", "segment_fit_end": origin + pd.Timedelta(days=1)}])
        self.assertEqual(sarima_cohort(cohort, 1), ["A"])
        with self.assertRaisesRegex(ValueError, "Segment label"):
            fit_cohort({}, cohort, origin, config)

    def test_holdout_cardinality_immutability_and_artifacts(self):
        from demandos.forecasting.pipeline import ForecastConfig, fit_cohort, attach_actuals, evaluation_tables, validate_outputs
        source = history(10 + np.arange(130.) % 7 + np.random.default_rng(42).uniform(0, 1, 130))
        origin = source.index[99]
        frame = pd.DataFrame({"date": source.index, "units_sold": source.to_numpy()})
        cohort = pd.DataFrame([{"series_id": "synthetic/A", "kind": "store_item", "regime": "smooth", "abc_xyz": "AX", "segment_fit_end": origin}])
        with TemporaryDirectory() as directory:
            raw, processed = Path(directory) / "source.csv", Path(directory) / "processed.parquet"
            frame.to_csv(raw, index=False)
            frame.to_parquet(processed, index=False)
            before = [fingerprint(raw), fingerprint(processed)]
            predictions, status, training, _ = fit_cohort({"synthetic/A": frame}, cohort, origin, ForecastConfig())
            joined = attach_actuals(predictions, {"synthetic/A": frame}, origin, 28)
            self.assertEqual(len(joined), len(predictions))
            self.assertEqual(joined.horizon.max(), 28)
            self.assertTrue(training.date.le(origin).all())
            self.assertTrue(status.fit_end.eq(origin).all())
            validate_outputs(joined, status, cohort, origin, 28)
            with self.assertRaises(ValueError):
                validate_outputs(joined.iloc[1:], status, cohort, origin, 28)
            with self.assertRaises(ValueError):
                validate_outputs(joined, pd.concat([status, status.iloc[:1]]), cohort, origin, 28)
            tables = evaluation_tables(joined, status)
            self.assertFalse(tables["metrics_common"].empty)
            self.assertEqual(before, [fingerprint(raw), fingerprint(processed)])
            with self.assertRaises(ValueError):
                attach_actuals(predictions, {"synthetic/A": frame.drop(index=101)}, origin, 28)

    def test_warnings_retained_and_unknown_parameters_rejected(self):
        import warnings
        source = history([1., 3., 2., 4.])
        def fit(*args):
            warnings.warn("synthetic optimizer warning", RuntimeWarning)
            return type("Fit", (), {"forecast": lambda self, h: np.ones(h)})(), {}
        with patch("demandos.forecasting.models._library_fit", side_effect=fit):
            result = self.forecast(source, 7, "ses")
        self.assertEqual(result.status, "ok")
        self.assertIn("synthetic optimizer warning", result.warnings[0])
        self.assertEqual(self.forecast(source, 7, "naive", params={"future_price": 99}).status, "failed")


if __name__ == "__main__":
    unittest.main()
