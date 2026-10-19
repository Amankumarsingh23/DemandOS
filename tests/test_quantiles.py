"""Controlled synthetic quantile arithmetic and temporal contracts, not M5 metrics."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_pinball_loss

from demandos.quantiles.contracts import LEVELS, Q, levels, crossing, repair, forecast_rows, validate, matched
from demandos.quantiles.metrics import pinball, interval_score, wilson, summarize
from demandos.quantiles.baseline import historical, nonadditivity, inverse_cdf
from demandos.quantiles.neural import QuantileNBeats, joint_loss, train, save, load, predict
from demandos.quantiles.tree import QuantileLightGBM
from demandos.nbeats.config import NBeatsConfig
from demandos.nbeats.dataset import TargetDataset
from demandos.neural.sequences import Panel, examples


def sequence_fixture():
    from test_neural import fixture
    daily, calendar, metadata, _, panel, rows, _ = fixture()
    config = replace(NBeatsConfig(), hidden_size=8, layers=2, generic_blocks=2,
                     generic_theta=4, epochs=2, patience=1, threads=1, batch_size=2)
    return daily, calendar, metadata, config, panel, rows


def wide():
    rows = pd.DataFrame([{"store_id": "CA_1", "item_id": "toy", "origin_date": pd.Timestamp("2016-02-28")}])
    q = np.tile([0., 1., 3.], (1, 28, 1))
    return forecast_rows(rows, q, "toy_model", "2016-01-31", "2016-01-31").assign(actual=np.arange(28)%5)


class QuantileMathTests(unittest.TestCase):
    def test_levels_strict_bounds_order_and_uniqueness(self):
        np.testing.assert_array_equal(levels(), [.1, .5, .9])
        for bad in [[], [0, .5, .9], [.1, .5, 1], [.1, .1, .9], [.9, .5, .1], [np.nan]]:
            with self.assertRaises(ValueError): levels(bad)

    def test_pinball_exact_under_over_ties_and_zero(self):
        self.assertAlmostEqual(float(pinball(30, 20, .9)), 9)
        self.assertAlmostEqual(float(pinball(10, 20, .9)), 1)
        self.assertAlmostEqual(float(pinball(30, 20, .1)), 1)
        self.assertAlmostEqual(float(pinball(10, 20, .1)), 9)
        np.testing.assert_array_equal(pinball([0, 3], [0, 3], .5), [0, 0])
        np.testing.assert_allclose(pinball([0, 10], [2, 7], .5), .5*np.abs(np.array([0, 10])-np.array([2, 7])))

    def test_pinball_matches_sklearn(self):
        actual, prediction = [0, 1, 30, 10, 0], [0, 4, 20, 20, -2]
        for tau in LEVELS:
            self.assertAlmostEqual(float(pinball(actual, prediction, tau).mean()), mean_pinball_loss(actual, prediction, alpha=tau))

    def test_nonfinite_and_invalid_target_rejected(self):
        for y, q, tau in [(np.nan, 0, .5), (0, np.inf, .5), (-1, 0, .5), (1, 0, 1)]:
            with self.assertRaises(ValueError): pinball(y, q, tau)

    def test_interval_score_examples_ties_and_invalid(self):
        self.assertEqual(float(interval_score(20, 10, 30)), 20)
        self.assertEqual(float(interval_score(35, 10, 30)), 70)
        self.assertEqual(float(interval_score(5, 10, 30)), 70)
        self.assertEqual(float(interval_score(0, 0, 0)), 0)
        self.assertEqual(float(interval_score(10, 10, 30)), 20)
        with self.assertRaises(ValueError): interval_score(20, 30, 10)
        with self.assertRaises(ValueError): interval_score(20, 10, 30, alpha=0)

    def test_crossing_repair_preserves_raw_and_needs_no_target(self):
        raw = np.array([[4., -2., 1.], [0., 0., 0.]])
        original = raw.copy()
        np.testing.assert_array_equal(crossing(raw)[0], [True, False, True])
        np.testing.assert_array_equal(repair(raw), [[0, 1, 4], [0, 0, 0]])
        np.testing.assert_array_equal(raw, original)
        with self.assertRaises(ValueError): repair(np.array([[0., np.nan, 1.]]))

    def test_exact_keys_horizons_leapday_and_provenance(self):
        frame = wide()
        self.assertEqual(frame.target_date.min(), pd.Timestamp("2016-02-29"))
        self.assertEqual(frame.target_date.max(), pd.Timestamp("2016-03-27"))
        for bad in [frame.iloc[1:], pd.concat([frame, frame.iloc[:1]]),
                    frame.assign(target_date=pd.Timestamp("2020-01-01")), frame.assign(fit_cutoff=pd.Timestamp("2017-01-01")),
                    frame.assign(fit_target_max=pd.Timestamp("2016-02-01")), frame.assign(q10=-1), frame.assign(q50=99)]:
            with self.assertRaises(ValueError): validate(bad)

    def test_matched_models_and_labels_not_silent_intersection(self):
        frame = wide()
        all_rows = pd.concat([frame, frame.assign(model="second")], ignore_index=True)
        self.assertEqual(len(matched(all_rows, ["toy_model", "second"])), 56)
        with self.assertRaises(ValueError): matched(all_rows.iloc[1:], ["toy_model", "second"])
        with self.assertRaises(ValueError): matched(pd.concat([frame, frame.assign(model="second", actual=999)]), ["toy_model", "second"])

    def test_inclusive_zero_ties_and_interval_coverage(self):
        frame = wide().assign(actual=0., q10=0., q50=0., q90=0.)
        metric = summarize(frame, ["model"]).iloc[0]
        self.assertEqual(metric.coverage_q10, 1.)
        self.assertEqual(metric.q10_tie_count, 28)
        self.assertEqual(metric.interval_coverage, 1.)
        self.assertEqual(metric.zero_width_count, 28)
        self.assertEqual(metric.multiple_zero_quantiles_count, 28)
        self.assertEqual(metric.interval_score, 0.)
        self.assertTrue(pd.isna(metric.median_wape))

    def test_raw_invalid_intervals_explicit_exclusion(self):
        frame = wide(); frame.loc[:2, Q] = [5., 2., 0.]
        metric = summarize(frame, ["model"]).iloc[0]
        self.assertEqual(metric.observations, 28)
        self.assertEqual(metric.any_crossing_count, 3)
        self.assertEqual(metric.invalid_interval_rows, 3)
        self.assertEqual(metric.interval_rows, 25)
        self.assertTrue(np.isfinite(metric.pinball_mean))

    def test_pooled_horizon_losses_reconcile_and_p50_half_mae(self):
        frame = wide()
        pooled = summarize(frame, ["model"]).iloc[0]
        horizons = summarize(frame, ["model", "horizon"])
        self.assertAlmostEqual(pooled.pinball_mean, np.average(horizons.pinball_mean, weights=horizons.observations))
        self.assertAlmostEqual(pooled.pinball_q50, pooled.median_absolute_error/2)

    def test_wilson_counts_and_correlation_caveat_input(self):
        lower, upper = wilson(80, 100)
        self.assertLess(lower, .8); self.assertGreater(upper, .8)
        self.assertGreaterEqual(wilson(0, 10)[0], 0)
        self.assertLessEqual(wilson(10, 10)[1], 1)
        with self.assertRaises(ValueError): wilson(1, 0)

    def test_empirical_includes_zero_and_ignores_future(self):
        history = pd.DataFrame({"date": pd.date_range("2016-01-01", periods=20), "units_sold": [0]*9+[10]+[999]*10})
        q, audit = historical(history, "2016-01-10", 10)
        np.testing.assert_array_equal(q[0], [0, 0, 0])
        self.assertEqual(audit["history_days"], 10)
        changed = history.copy(); changed.loc[changed.date.gt("2016-01-10"), "units_sold"] = 999999
        np.testing.assert_array_equal(q, historical(changed, "2016-01-10", 10)[0])

    def test_baseline_allzero_short_missing_and_crossseries(self):
        history = pd.DataFrame({"date": pd.date_range("2016-01-01", periods=3), "units_sold": [0, 0, 0]})
        result, audit = historical(history, "2016-01-03")
        self.assertEqual(audit["status"], "all_zero")
        self.assertTrue((result == 0).all())
        self.assertEqual(historical(history.assign(units_sold=1), "2016-01-03")[1]["status"], "short_history")
        with self.assertRaises(ValueError): historical(history, "2015-01-01")
        with self.assertRaises(ValueError): historical(history.iloc[[0, 2]], "2016-01-03")
        with self.assertRaises(ValueError): historical(history.assign(item_id=["a", "b", "a"]), "2016-01-03")

    def test_hierarchy_inversecdf_nonadditivity(self):
        result = nonadditivity()
        self.assertEqual(result["sum_marginal_q90"], 0)
        self.assertEqual(result["q90_sum"], 10)
        self.assertEqual(inverse_cdf([0, 10], [.9, .1], .9), 0)
        with self.assertRaises(ValueError): inverse_cdf([10, 0], [.9, .1], .9)


class JointNeuralTests(unittest.TestCase):
    def test_joint_loss_equals_numpy(self):
        y = torch.tensor([[30., 10.]])
        q = torch.tensor([[[20., 20., 20.], [20., 20., 20.]]], requires_grad=True)
        value = joint_loss(y, q)
        expected = pinball(y.numpy()[..., None], q.detach().numpy(), np.array(LEVELS)).mean()
        self.assertAlmostEqual(float(value.detach()), expected)
        value.backward(); self.assertTrue(torch.isfinite(q.grad).all())
        self.assertEqual(float(joint_loss(torch.zeros((1, 28)), torch.zeros((1, 28, 3)))), 0.)

    def test_shape_monotonicity_backcast_residuals_and_latent_sum(self):
        _, _, _, config, _, _ = sequence_fixture()
        model = QuantileNBeats(config)
        x, mask = torch.rand(4, 56), torch.ones(4, 56); mask[:, :5] = 0
        out = model(x, mask, diagnostics=True)
        self.assertEqual(tuple(out["quantiles"].shape), (4, 28, 3))
        self.assertTrue((out["quantiles"] >= 0).all())
        self.assertTrue((torch.diff(out["quantiles"], dim=-1) >= 0).all())
        torch.testing.assert_close(out["latent"], out["contributions"].sum(1))
        for i in range(config.generic_blocks):
            torch.testing.assert_close(out["residuals"][:, i+1], (out["residuals"][:, i]-out["backcasts"][:, i])*mask)

    def test_softplus_near_zero_not_exact_zero_parameterization(self):
        value = torch.nn.functional.softplus(torch.tensor(-20.))
        self.assertGreater(float(value), 0)
        self.assertLess(float(value), 3e-9)
        _, _, _, config, _, _ = sequence_fixture()
        model = QuantileNBeats(config)
        zero = model(torch.zeros((1, 56)), torch.zeros((1, 56)))
        self.assertTrue(torch.isfinite(zero).all())
        self.assertTrue((zero > 0).all())
        # Even an arbitrarily small positive P10 excludes exact observed zeros.
        frame = wide().assign(actual=0.)
        frame[Q] = zero.detach().numpy().reshape(28, 3)
        metric = summarize(frame, ["model"]).iloc[0]
        self.assertEqual(metric.coverage_q10, 1.)
        self.assertEqual(metric.interval_coverage, 0.)

    def test_gradients_and_invalid_masks_nonfinite_fail(self):
        _, _, _, config, _, _ = sequence_fixture()
        model = QuantileNBeats(config)
        loss = joint_loss(torch.ones((2, 28)), model(torch.ones((2, 56)), torch.ones((2, 56))))
        loss.backward()
        for name, parameter in model.named_parameters():
            if name not in model.expected_unused_parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        with self.assertRaises(ValueError): model(torch.ones((2, 56)), torch.full((2, 56), .5))
        with self.assertRaises(ValueError): model(torch.full((2, 56), np.nan), torch.ones((2, 56)))

    def test_train_cutoffs_selection_serialization_and_seed(self):
        _, _, metadata, config, panel, rows = sequence_fixture()
        training = TargetDataset(panel, rows, config, label_cutoff="2016-01-31")
        validation_rows = examples(metadata, [pd.Timestamp("2016-01-31")], config)
        validation = TargetDataset(panel, validation_rows, config, label_cutoff="2016-02-28")
        model, meta = train(training, config, cutoff="2016-01-31", validation=validation, validation_cutoff="2016-02-28")
        probe = TargetDataset(panel, validation_rows, config)
        expected = predict(model, probe, config)
        with TemporaryDirectory() as directory:
            save(model, meta, directory)
            restored, loaded = load(directory)
            np.testing.assert_array_equal(expected, predict(restored, probe, config))
            self.assertEqual(restored.fit_target_max, pd.Timestamp(meta["fit_target_max"]))
            payload = json.loads((Path(directory)/"metadata.json").read_text()); payload["fit_target_max"] = "2020-01-01"
            (Path(directory)/"metadata.json").write_text(json.dumps(payload))
            with self.assertRaises(ValueError): load(directory)
        with self.assertRaises(ValueError): train(training, config, cutoff="2016-01-30")
        with self.assertRaises(ValueError): predict(model, training, config)
        with self.assertRaises(ValueError): predict(model, TargetDataset(panel, rows, config), config)

    def test_frozen_future_mutation_deletion_prices_inputs_and_scaling(self):
        daily, calendar, metadata, config, panel, _ = sequence_fixture()
        origin = pd.Timestamp("2016-02-28")
        rows = examples(metadata, [origin], config)
        expected = TargetDataset(panel, rows, config)
        model = QuantileNBeats(config); model.fit_cutoff = pd.Timestamp("2016-01-31")
        first = predict(model, expected, config)
        future = daily.date.gt(origin)
        changed = daily.copy(); changed.loc[future, "units_sold"] += 1000000; changed.loc[future, "sell_price"] = 999999.
        for frame in [changed, daily.loc[~future].copy()]:
            probe = TargetDataset(Panel(frame, calendar, metadata), rows, config)
            for a, b in zip(expected.tensors, probe.tensors): torch.testing.assert_close(a, b, rtol=0, atol=0)
            np.testing.assert_array_equal(first, predict(model, probe, config))
        self.assertEqual(expected.audits[0]["history_start"], pd.Timestamp("2016-01-04"))
        self.assertEqual(expected.audits[0]["target_end"], pd.Timestamp("2016-03-27"))


class QuantileTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_boosting import BoostingTests
        BoostingTests.setUpClass()
        for name in ["training", "validation", "inference", "features", "feature_config", "config"]:
            setattr(cls, name, getattr(BoostingTests, name))
        cls.config = replace(cls.config, iterations=6)

    def test_native_three_alphas_chronology_save_load_and_unknown_missing(self):
        changed = self.training.copy(); changed.loc[changed.index[::5], "relative_level_change_7"] = np.nan
        for alpha in LEVELS:
            model = QuantileLightGBM(alpha, self.features, self.config, self.feature_config)
            model.fit(changed, allowed_cutoff="2016-03-27", validation=self.validation, validation_cutoff="2016-04-24")
            self.assertEqual(model.native_parameters["objective"], "quantile")
            self.assertEqual(model.native_parameters["alpha"], alpha)
            self.assertLessEqual(pd.Timestamp(model.metadata["training_target_max"]), pd.Timestamp("2016-03-27"))
            unknown = self.inference.copy(); unknown.loc[unknown.index[:1], "item_id"] = "UNKNOWN_ITEM"
            self.assertNotIn("UNKNOWN_ITEM", model.vocabulary["item_id"])
            expected = model.predict(unknown)
            self.assertTrue(np.isfinite(expected.raw_prediction).all())
            with TemporaryDirectory() as directory:
                model.save(directory)
                restored = QuantileLightGBM.load(directory)
                np.testing.assert_array_equal(expected.raw_prediction, restored.predict(unknown).raw_prediction)

    def test_tree_future_labels_audits_and_prediction_labels_rejected(self):
        for features in [self.features+["target_units_sold"], self.features+["future_price"]]:
            with self.assertRaises(ValueError): QuantileLightGBM(.5, features, self.config, self.feature_config)
        model = QuantileLightGBM(.5, self.features, self.config, self.feature_config)
        with self.assertRaises(ValueError): model.fit(self.training, allowed_cutoff="2016-03-01")
        with self.assertRaises(ValueError): model.fit(self.training, allowed_cutoff="2016-03-27", validation=self.training, validation_cutoff="2016-04-24")
        model.fit(self.training, allowed_cutoff="2016-03-27")
        with self.assertRaises(ValueError): model.predict(self.inference.assign(actual=999))
        with self.assertRaises(ValueError): model.predict(self.training.drop(columns="target_units_sold"))


if __name__ == "__main__":
    unittest.main()
