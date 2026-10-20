"""Deterministic SYNTHETIC fixtures only; real M5 verification is a separate CLI."""
from itertools import product
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from demandos.inventory.core import (Economics, Distribution, service, lead_moments,
    covariance_variance, normal_rop, inventory_position, order_units, policy, cycle)
from demandos.inventory.scenarios import bootstrap, quantile_paths, period_distribution, targets, seed_for
from demandos.inventory.experiment import forecast_bundle, plan, evaluate, matched, summarize, STATUS
from demandos.inventory.pipeline import load_config, controlled
from demandos.data.pipeline import fingerprint


def fixture():
    origin = pd.Timestamp("2016-02-28")
    history = pd.DataFrame({"date": pd.date_range(origin-pd.Timedelta(days=399), periods=428),
                            "units_sold": np.tile([0, 0, 1, 0, 3, 0, 10], 62)[:428], "sell_price": 2.})
    rows = pd.DataFrame({"target_date": pd.date_range(origin+pd.Timedelta(days=1), periods=28), "horizon": np.arange(1, 29),
                         "q10": 0., "q50": 2., "q90": 8., "prediction": 3., "actual": 999,
                         "fit_cutoff": origin, "fit_target_max": origin, "history_cutoff": origin, "model_fit_cutoff": origin,
                         "selection_cutoff": origin, "model_id": "SYNTHETIC_FIXTURE", "base_lower": 0.,
                         "base_upper": 8., "lower": 0., "upper": 12.})
    config = load_config(Path(__file__).resolve().parents[1]/"configs/inventory.toml")
    config["inventory"]["samples"] = 16
    return origin, history, rows, config


class EconomicsTests(unittest.TestCase):
    def test_required_hand_costs(self):
        e = Economics(100, 60, 40)
        self.assertEqual((e.cu, e.co), (40, 20))
        self.assertAlmostEqual(e.fractile, 2/3)
        self.assertEqual(e.cost(25, 30), 200)
        self.assertEqual(e.cost(35, 30), 100)

    def test_invalid_economics(self):
        for args in [(1, 2, 0), (10, 2, 3), (0, 0, 0), (np.nan, 1, 0), (2, -1, 0)]:
            with self.assertRaises(ValueError): Economics(*args)

    def test_zero_inflated_optimum_and_direct_enumeration(self):
        d = Distribution([0, 100], [.8, .2])
        for cu, co, expected in [(10, 10, 0), (90, 10, 100)]:
            e = Economics(cu, 0, 0, overage_penalty=co)
            q = d.quantile(e.fractile)
            costs = [.8*float(e.cost(stock, 0))+.2*float(e.cost(stock, 100)) for stock in range(101)]
            self.assertEqual(q, expected)
            self.assertEqual(q, int(np.argmin(costs)))
            self.assertAlmostEqual(d.metrics(q, e)["expected_cost"], min(costs))
        self.assertGreater(d.mean, 0)

    def test_cdf_quantile_ties_and_boundaries(self):
        d = Distribution([0, 0, 5, 100, 1000], [.2, .3, .3, .2, 0])
        self.assertEqual(d.quantile(.5), 0)
        self.assertEqual(d.quantile(.8), 5)
        self.assertEqual(d.quantile(.8001), 100)
        self.assertEqual(d.quantile(1), 100)
        self.assertEqual(d.quantile(0), 0)
        self.assertEqual(d.cdf(-1), 0)
        self.assertEqual(d.cdf(5), .8)
        self.assertEqual(Distribution([5]).quantile(0), 0)

    def test_probability_and_support_validation(self):
        for y, p in [([1, 2], [.2, .2]), ([1, 2], [-1, 2]), ([1], [np.nan]), ([1, 2], [1]), ([-1], [1]), ([.5], [1]), ([], [])]:
            with self.assertRaises(ValueError): Distribution(y, p)
        with self.assertRaises(ValueError): Distribution([0]).quantile(1.1)

    def test_cost_curve_exhaustive_integer_minimum(self):
        d = Distribution([0, 2, 2, 5, 8])
        for e in [Economics(3, 1, 0), Economics(1, 1, 0), Economics(2, 1, 1)]:
            q, costs = d.curve(e)
            all_costs = [d.metrics(i, e)["expected_cost"] for i in range(12)]
            self.assertAlmostEqual(costs.min(), min(all_costs))
            self.assertEqual(d.quantile(e.fractile), int(np.argmin(all_costs)))

    def test_all_zero_fill_rate_is_undefined(self):
        m = Distribution([0, 0]).metrics(0, Economics(10, 5, 0))
        self.assertEqual(m["csl"], 1)
        self.assertIsNone(m["fill_rate"])
        self.assertIsNone(service([0], [0])["fill_rate"])

    def test_required_cycle_service_vs_unit_fill(self):
        m = service([100]*10, [100]*9+[90])
        self.assertAlmostEqual(m["csl"], .9)
        self.assertAlmostEqual(m["fill_rate"], .99)
        with self.assertRaises(ValueError): service([10], [11])


class LeadPolicyTests(unittest.TestCase):
    def test_exact_iid_and_covariance(self):
        paths = np.array(list(product([0, 2], repeat=3)))
        d = period_distribution(paths, [3], [1.])
        self.assertEqual((d.mean, d.variance), (3, 3))
        self.assertEqual(lead_moments(1, 1, [3], [1]), (3, 3))
        covariance = np.cov(paths, rowvar=False, ddof=0)
        self.assertEqual(covariance_variance(covariance), d.variance)
        shared = np.array([[0]*3, [2]*3])
        self.assertEqual(covariance_variance(np.cov(shared, rowvar=False, ddof=0)), 9)
        with self.assertRaises(ValueError): covariance_variance([[1, 2], [2, 1]])

    def test_exact_random_lead_iid_formula(self):
        paths = np.array(list(product([0, 2], repeat=3)))
        mixture = period_distribution(paths, [1, 3], [.5, .5])
        self.assertEqual(lead_moments(1, 1, [1, 3], [.5, .5]), (mixture.mean, mixture.variance))
        self.assertEqual((mixture.mean, mixture.variance), (2, 3))

    def test_protection_and_no_truncation(self):
        paths = np.array([[1, 2, 3, 4]])
        self.assertEqual(period_distribution(paths, [2], [1], 2).mean, 10)
        with self.assertRaises(ValueError): period_distribution(paths, [3], [1], 2)

    def test_nonadditive_daily_quantiles(self):
        y = np.array([[0, 10], [10, 0]])
        self.assertEqual(sum(Distribution(y[:, h]).quantile(.9) for h in range(2)), 20)
        self.assertEqual(period_distribution(y, [2], [1]).quantile(.9), 10)

    def test_normal_hand_example_and_signed_offset(self):
        continuous, rounded = normal_rop(20, 5, 4, .95)
        self.assertAlmostEqual(continuous, 96.4485362695)
        self.assertEqual(rounded, 97)
        d = Distribution([0, 100], [.8, .2])
        offset = d.quantile(.5)-d.mean
        self.assertEqual(offset, -20)
        self.assertEqual(max(0, offset), 0)

    def test_position_trigger_quantity_and_rounding(self):
        self.assertEqual(inventory_position(10, 4, 3), 11)
        a = policy(11, 11, 30, 7, multiple=5)
        self.assertTrue(a["trigger"])
        self.assertEqual((a["periodic_up_to"], a["continuous_fixed"]), (20, 10))
        a = policy(12, 11, 30, 7)
        self.assertFalse(a["trigger"])
        self.assertEqual((a["periodic_up_to"], a["continuous_up_to"]), (18, 0))
        self.assertEqual(policy(40, 50, 30, 7)["continuous_up_to"], 0)
        self.assertEqual(order_units(1.1, multiple=4, minimum=5), 8)
        self.assertEqual(order_units(-10), 0)
        with self.assertRaises(ValueError): order_units(2, multiple=0)
        with self.assertRaises(ValueError): inventory_position(-1, 0)

    def test_one_order_timing_hand_trajectory(self):
        # E-close order 5; day1 consumes2, day2 loses1, day3 receives2+5, serves4.
        result = cycle([[2, 2, 4, 1]], 2, 2, on_hand=3, on_order=2, order=5, trace=True)
        expected = dict(demand=9, served=8, shortage=1, leftover=2, stock_unit_days=6)
        for name, value in expected.items(): self.assertEqual(result[name][0], value)
        self.assertEqual([r["receipt"] for r in result["daily"]], [0, 0, 7, 0])
        self.assertEqual([r["lost_sales"] for r in result["daily"]], [0, 1, 0, 0])
        with self.assertRaises(ValueError): cycle([[1, 2]], 2, 2, 0, 0, 0)


class ScenarioContractTests(unittest.TestCase):
    def test_bootstrap_seed_prefix_dates_and_zero_mass(self):
        origin, history, _, _ = fixture()
        a, u, provenance = bootstrap(history, origin, samples=16, seed=42)
        b, v, _ = bootstrap(history, origin, samples=16, seed=42)
        np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(u, v)
        self.assertEqual(provenance["history_end"], origin)
        self.assertEqual(provenance["history_rows"], 365)
        self.assertTrue(set(np.unique(a)).issubset(set(history.units_sold)))
        self.assertTrue(np.all((u >= 0) & (u <= 1)))
        self.assertEqual(seed_for(42, "a"), seed_for(42, "a"))
        self.assertNotEqual(seed_for(42, "a"), seed_for(42, "b"))

    def test_historical_blocks_preserve_weekday_sequence(self):
        origin = pd.Timestamp("2016-02-28")
        history = pd.DataFrame({"date": pd.date_range("2015-01-01", origin)})
        history["units_sold"] = history.date.dt.dayofweek
        paths, _, _ = bootstrap(history, origin, samples=16)
        np.testing.assert_array_equal(paths, np.tile(np.arange(28) % 7, (16, 1)))

    def test_path_rounding_tails_and_zero_only(self):
        q = np.array([[0., 0., 0.], [1., 2., 3.]])
        # Deliberately assumed positive tail beyond zero P90; not uniquely identified.
        paths = quantile_paths(q, [[.5, .5], [0., 1.]])
        np.testing.assert_array_equal(paths, [[0, 2], [0, 4]])
        with self.assertRaises(ValueError): quantile_paths([[3, 2, 1]], [[.5]])
        with self.assertRaises(ValueError): quantile_paths(q, [[.5, 2.]])

    def test_leap_day_and_complete_windows(self):
        origin, _, rows, _ = fixture()
        self.assertEqual(targets(rows, origin).target_date.iloc[0], pd.Timestamp("2016-02-29"))
        for frame in [rows.iloc[:-1], pd.concat([rows.iloc[:-1], rows.iloc[:1]])]:
            with self.assertRaises(ValueError): targets(frame, origin)

    def test_future_fitted_and_wrong_base_forecasts_rejected(self):
        origin, _, rows, _ = fixture()
        changed = rows.copy(); changed["fit_target_max"] = origin+pd.Timedelta(days=1)
        with self.assertRaises(ValueError): forecast_bundle(changed, rows, rows, rows, origin)
        changed = rows.copy(); changed["history_cutoff"] = origin+pd.Timedelta(days=1)
        with self.assertRaises(ValueError): forecast_bundle(changed, rows, rows, rows, origin)
        changed = rows.copy(); changed["model_id"] = "OTHER_CHECKPOINT"
        with self.assertRaises(ValueError): forecast_bundle(rows, rows, rows, changed, origin)

    def test_future_outcomes_and_prices_cannot_change_plan(self):
        origin, history, rows, config = fixture()
        bundle = forecast_bundle(rows, rows, rows, rows, origin)
        baseline, decisions, audit = plan(history, origin, bundle, config, ("SYNTHETIC", "ITEM"))
        for change in ["sales", "prices", "remove"]:
            altered = history.copy(); mask = altered.date.gt(origin)
            if change == "sales": altered.loc[mask, "units_sold"] = 1000000
            elif change == "prices": altered.loc[mask, "sell_price"] = 1000000.
            else: altered = altered.loc[~mask]
            actual, new, new_audit = plan(altered, origin, bundle, config, ("SYNTHETIC", "ITEM"))
            pd.testing.assert_frame_equal(decisions, new)
            self.assertEqual(audit, new_audit)
            for name in baseline: np.testing.assert_array_equal(baseline[name], actual[name])
        changed = rows.copy(); changed["actual"] = -999999
        new_bundle = forecast_bundle(changed, changed, changed, changed, origin)
        for name in ["point", "s16", "s17", "width"]: np.testing.assert_array_equal(bundle[name], new_bundle[name])

    def test_bounded_integration_matched_status_and_source_immutability(self):
        origin, history, rows, config = fixture()
        snapshot = history.copy(deep=True)
        bundle = forecast_bundle(rows, rows, rows, rows, origin)
        paths, decisions, _ = plan(history, origin, bundle, config, ("SYNTHETIC", "ITEM"))
        outcomes = evaluate(decisions, paths, np.arange(28) % 4, config)
        self.assertEqual(len(decisions), 63)
        self.assertEqual(len(outcomes), 189)
        self.assertEqual(matched(outcomes), 9)
        self.assertEqual(set(outcomes.status), {STATUS})
        with self.assertRaises(ValueError): evaluate(decisions, paths, np.full(28, .5), config)
        self.assertTrue(outcomes.clairvoyant_regret.ge(0).all())
        self.assertTrue(outcomes.quantity.ge(0).all())
        self.assertEqual(len(summarize(outcomes, ["method", "action"])), 21)
        with self.assertRaises(ValueError): matched(outcomes.iloc[1:])
        pd.testing.assert_frame_equal(history, snapshot)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"SYNTHETIC.csv"; history.to_csv(source, index=False)
            before = fingerprint(source)
            bootstrap(pd.read_csv(source), origin, samples=16)
            self.assertEqual(before, fingerprint(source))

    def test_controlled_dependence_families(self):
        frame = controlled()
        self.assertEqual(len(frame), 18)
        self.assertEqual(set(frame.status), {"EXACT_SYNTHETIC_TOY"})
        a = frame[frame.family.eq("independent_daily")].iloc[0]
        b = frame[frame.family.eq("positive_shared_shocks")].iloc[0]
        self.assertEqual(a["mean"], b["mean"])
        self.assertGreater(b.variance, a.variance)

    def test_configuration_rejects_unknown_or_unsafe_scenarios(self):
        _, _, _, config = fixture()
        # TOML case is deliberately incomplete: no guessed costs or fallback config.
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/"bad.toml"; file.write_text("[inventory]\nmode='backorders'\n")
            with self.assertRaises(ValueError): load_config(file)


if __name__ == "__main__":
    unittest.main()
