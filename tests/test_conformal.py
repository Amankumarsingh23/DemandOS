"""Deterministic synthetic score mathematics and delayed-feedback contracts."""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from demandos.conformal.core import scores, corrected, weighted, intervals, update_alpha
from demandos.conformal.ledger import validate, scored, eligible, recent_pool, split_pool, complete_paths, GROUP
from demandos.conformal.engine import static_intervals, aci_intervals
from demandos.conformal.metrics import summarize, trajectory_coverage
from demandos.conformal.pipeline import load_config
from demandos.backtesting.metrics import horizon_band


def fixture():
    rows = []
    for issue in pd.date_range("2020-01-01", periods=10, freq="7D"):
        for number in range(4):
            for horizon in range(1, 29):
                target = issue+pd.Timedelta(days=horizon)
                rows.append({"store_id": f"toy_{number}", "item_id": "synthetic", "origin_date": issue,
                    "target_date": target, "horizon": horizon, "family": "toy", "model_id": "toy_frozen",
                    "model_fit_cutoff": pd.Timestamp("2019-12-01"), "fit_target_max": pd.Timestamp("2019-12-01"),
                    "selection_cutoff": pd.Timestamp("2019-12-29"), "observation_available_date": target,
                    "cohort": "continuity" if number < 2 else "sparse_stress", "q10": 5., "q50": 10., "q90": 15.,
                    "actual": int((number*3+target.day+horizon)%19), "score_id": len(rows)})
    ledger = pd.DataFrame(rows)
    ledger["horizon_band"] = horizon_band(ledger.horizon)
    config = {"alpha": .2, "split_origins": ["2020-01-01", "2020-01-29"], "split_end": "2020-02-26",
              "rolling_scores": 64, "half_life_days": 14, "path_min_origins": 2, "path_min_trajectories": 3,
              "aci_start": "2020-01-29", "aci_gamma": .001, "aci_alpha_min": .02, "aci_alpha_max": .5}
    return scored(ledger), config


class ConformalTests(unittest.TestCase):
    def test_corrected_nine_scores_rank_eight_and_point_interval(self):
        c, info = corrected([1,2,3,4,5,6,7,8,10], .2)
        self.assertEqual((c, info["n"], info["rank"]), (8., 9, 8))
        lower, upper, _ = intervals([20.], [20.], c)
        np.testing.assert_array_equal(lower, [12]); np.testing.assert_array_equal(upper, [28])

    def test_extended_infinity_instead_of_interpolation(self):
        c, info = corrected([1,2,3,4], .1)
        self.assertTrue(np.isposinf(c)); self.assertEqual(info["rank"], 5)
        self.assertEqual(info["status"], "extended_infinite")
        lo, hi, _ = intervals([20.], [20.], c)
        self.assertEqual(lo[0], 0); self.assertTrue(np.isposinf(hi[0]))
        self.assertTrue(np.isposinf(corrected([], .2)[0]))
        metric = summarize(pd.DataFrame({"method": ["toy"], "actual": [100.], "lower": lo, "upper": hi}), ["method"]).iloc[0]
        self.assertEqual(metric.coverage, 1); self.assertEqual(metric.infinite_intervals, 1)
        self.assertTrue(np.isposinf(metric.interval_score))

    def test_ties_and_negative_corrected_rank(self):
        self.assertEqual(corrected([2]*9, .2)[0], 2)
        self.assertEqual(corrected([-10]*9, .2)[0], -10)
        self.assertEqual(corrected([1,2,3,4], .2)[1]["rank"], 4)
        for bad, alpha in [([np.nan], .2), ([np.inf], .2), ([1], 0), ([1], 1)]:
            with self.assertRaises(ValueError): corrected(bad, alpha)

    def test_signed_and_nonnegative_cqr_arithmetic(self):
        result = scores([35.,20.], [10.,10.], [30.,30.], [20.,20.])
        np.testing.assert_array_equal(result["cqr_signed"], [5.,-10.])
        np.testing.assert_array_equal(result["cqr_nonnegative"], [5.,0.])
        np.testing.assert_array_equal(result["absolute"], [15.,0.])
        with self.assertRaises(ValueError): scores([20], [30], [10])

    def test_nonnegative_support_clip_preserves_valid_inclusions(self):
        lo, hi, _ = intervals([1.], [5.], 4.)
        np.testing.assert_array_equal(lo, [0.]); np.testing.assert_array_equal(hi, [9.])
        y = np.arange(15)
        before = (y >= -3)&(y <= 9)
        after = (y >= lo[0])&(y <= hi[0])
        self.assertTrue((~before|after).all())

    def test_signed_empty_set_enlargement_is_target_free_and_ordered(self):
        lo, hi, c = intervals([10., .13], [30., .8], [-100., -100.])
        np.testing.assert_array_equal(lo, hi)
        self.assertEqual(lo[0], 20.); self.assertEqual(c[0], -10.)
        lo, hi, _ = intervals([10.], [30.], -5.)
        np.testing.assert_array_equal(lo, [15]); np.testing.assert_array_equal(hi, [25])
        with self.assertRaises(ValueError): intervals([1.], [2.], np.nan)
        with self.assertRaises(ValueError): intervals([1.], [2.], -np.inf)

    def test_weighted_stable_normalization_and_hand_case(self):
        value, info = weighted([1,2,3], [1,2,1], .25)
        self.assertEqual(value, 2)
        self.assertTrue(info["heuristic"]); self.assertFalse(info["finite_sample_rank_correction"])
        self.assertAlmostEqual(info["kish_effective_n"], 16/6)
        self.assertEqual(weighted([3,2,1], [1,2,1], .25)[0], 2)
        self.assertEqual(weighted([2,2,2], [1,2,1], .2)[0], 2)
        with self.assertRaises(ValueError): weighted([1,2], [0,0])
        with self.assertRaises(ValueError): weighted([1,2], [1,-1])

    def test_aci_formula_direction_and_bounds(self):
        self.assertAlmostEqual(update_alpha(.2, 0, .2, .1)[0], .22)
        self.assertAlmostEqual(update_alpha(.2, 1, .2, .1)[0], .12)
        self.assertEqual(update_alpha(.02, 1)[0], .02)
        self.assertTrue(update_alpha(.02, 1)[1])
        with self.assertRaises(ValueError): update_alpha(.2, .5)

    def test_heldout_provenance_in_sample_and_selection_targets_rejected(self):
        ledger, _ = fixture()
        validate(ledger)
        for bad in [ledger.assign(model_fit_cutoff=pd.Timestamp("2020-04-01")),
                    ledger.assign(fit_target_max=pd.Timestamp("2020-04-01")),
                    ledger.assign(selection_cutoff=pd.Timestamp("2020-04-01")),
                    pd.concat([ledger, ledger.iloc[:1]]), ledger.assign(q10=99)]:
            with self.assertRaises(ValueError): validate(bad)

    def test_maturity_uses_target_not_only_issue(self):
        ledger, _ = fixture()
        result = eligible(ledger, "2020-01-10")
        self.assertTrue(result.target_date.le("2020-01-10").all())
        self.assertTrue(result.observation_available_date.le("2020-01-10").all())
        self.assertTrue(ledger[ledger.origin_date.eq("2020-01-01")].target_date.gt("2020-01-10").any())
        self.assertLess(len(result), len(ledger[ledger.origin_date.le("2020-01-10")]))

    def test_observation_delay_is_honored(self):
        ledger, config = fixture()
        changed = ledger.copy()
        first = changed.score_id.eq(0)
        changed.loc[first, "observation_available_date"] = pd.Timestamp("2020-01-20")
        self.assertNotIn(0, eligible(changed, "2020-01-10").score_id.tolist())
        self.assertIn(0, eligible(changed, "2020-01-20").score_id.tolist())
        delayed = ledger.copy()
        ids = delayed.loc[delayed.origin_date.eq("2020-01-29") & delayed.horizon.eq(1), "score_id"].tolist()
        delayed.loc[delayed.score_id.isin(ids), "observation_available_date"] = pd.Timestamp("2020-02-03")
        _, _, feedback = aci_intervals(delayed, "2020-02-26", config)
        self.assertTrue(feedback.loc[feedback.score_id.isin(ids), "feedback_date"].eq("2020-02-03").all())

    def test_future_candidate_mutation_or_removal_cannot_change_past_scores(self):
        ledger, _ = fixture()
        expected = eligible(ledger, "2020-02-01")
        changed = ledger.copy(); changed.loc[changed.target_date.gt("2020-02-01"), "actual"] = np.nan
        pd.testing.assert_frame_equal(expected, eligible(changed, "2020-02-01"), check_dtype=False)
        changed.loc[changed.target_date.gt("2020-02-01"), "actual"] = 999999
        pd.testing.assert_frame_equal(expected, eligible(changed, "2020-02-01"), check_dtype=False)

    def test_rolling_fixed_size_and_target_cutoff(self):
        ledger, _ = fixture()
        pool = recent_pool(ledger, "2020-02-26", 64)
        self.assertTrue(pool.groupby(GROUP).size().eq(64).all())
        self.assertTrue(pool.target_date.le("2020-02-26").all())
        self.assertTrue(pool.observation_available_date.le("2020-02-26").all())

    def test_split_pool_freezes_dates_and_origins(self):
        ledger, config = fixture()
        before = split_pool(ledger, "2020-02-26", config)
        after = split_pool(ledger, "2020-03-04", config)
        pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
        self.assertEqual(set(before.origin_date), set(pd.to_datetime(config["split_origins"])))

    def test_complete_trajectory_needs_day_28_and_separate_support(self):
        ledger, _ = fixture()
        paths, statuses = complete_paths(eligible(ledger, "2020-01-28"), 1, 2)
        self.assertTrue(paths.empty)
        paths, statuses = complete_paths(eligible(ledger, "2020-01-29"), 1, 2)
        self.assertEqual(len(paths), 4)
        self.assertTrue(statuses.status.eq("available").all())
        _, statuses = complete_paths(eligible(ledger, "2020-01-29"), 3, 10)
        self.assertTrue(statuses.status.eq("insufficient_complete_trajectories").all())

    def test_static_outputs_unique_ordered_and_audited(self):
        ledger, config = fixture()
        output, thresholds, availability = static_intervals(ledger, ["2020-02-26"], config)
        self.assertFalse(output.duplicated(["store_id","item_id","origin_date","horizon","method"]).any())
        self.assertTrue(output.lower.ge(0).all()); self.assertTrue(output.lower.le(output.upper).all())
        self.assertTrue(thresholds.dropna(subset=["availability_max"]).availability_max.le("2020-02-26").all())
        self.assertTrue(availability.status.eq("available").all())
        weighted_rows = thresholds[thresholds.method.str.contains("weighted")]
        self.assertTrue(weighted_rows.heuristic.all())

    def test_aci_mature_feedback_exact_update_and_nonfuture_state(self):
        ledger, config = fixture()
        output, thresholds, events = aci_intervals(ledger, "2020-02-26", config)
        self.assertTrue(events.target_date.le(events.feedback_date).all())
        self.assertTrue(events.feedback_date.le("2020-02-26").all())
        self.assertTrue(events.origin_date.lt(events.feedback_date).all())
        expected = np.clip(events.alpha_before+config["aci_gamma"]*(config["alpha"]-events.miss), .02, .5)
        np.testing.assert_array_equal(events.alpha_after, expected)
        self.assertTrue(output.lower.le(output.upper).all())
        self.assertTrue(thresholds.alpha.between(.02, .5).all())

    def test_aci_future_mutation_preserves_issued_intervals_and_updates(self):
        ledger, config = fixture()
        first = aci_intervals(ledger, "2020-02-26", config)
        changed = ledger.copy(); changed.loc[changed.target_date.gt("2020-02-26"), "actual"] = np.nan
        second = aci_intervals(changed, "2020-02-26", config)
        for a, b in zip(first, second): pd.testing.assert_frame_equal(a, b)
        longer = aci_intervals(ledger, "2020-03-04", config)
        for index, date_column in [(0, "origin_date"), (1, "origin_date"), (2, "feedback_date")]:
            prefix = longer[index][longer[index][date_column].le("2020-02-26")].reset_index(drop=True)
            pd.testing.assert_frame_equal(first[index].reset_index(drop=True), prefix)

    def test_static_future_mutation_preserves_thresholds_and_intervals(self):
        ledger, config = fixture()
        first = static_intervals(ledger, ["2020-02-26"], config)
        changed = ledger.copy(); changed.loc[changed.target_date.gt("2020-02-26"), "actual"] += 1000000
        second = static_intervals(changed, ["2020-02-26"], config)
        for a, b in zip(first, second): pd.testing.assert_frame_equal(a, b)

    def test_aci_deterministic_feedback_order_with_overlapping_horizons(self):
        ledger, config = fixture()
        first = aci_intervals(ledger, "2020-02-26", config)
        shuffled = ledger.sample(frac=1, random_state=42)
        second = aci_intervals(shuffled, "2020-02-26", config)
        keys = ["method","origin_date","store_id","item_id","horizon"]
        pd.testing.assert_frame_equal(first[0].sort_values(keys).reset_index(drop=True), second[0].sort_values(keys).reset_index(drop=True))
        pd.testing.assert_frame_equal(first[2], second[2])

    def test_interval_metrics_math_and_zero_inclusion(self):
        frame = pd.DataFrame({"method": ["toy"]*2, "actual": [20.,35.], "lower": [10.,10.], "upper": [30.,30.]})
        metric = summarize(frame, ["method"]).iloc[0]
        self.assertEqual(metric.coverage, .5); self.assertEqual(metric.width_mean, 20)
        self.assertEqual(metric.interval_score, 45); self.assertEqual(metric.upper_miss_rate, .5)
        zeros = summarize(pd.DataFrame({"method": ["toy"], "actual": [0.], "lower": [0.], "upper": [0.]}), ["method"]).iloc[0]
        self.assertEqual(zeros.zero_sales_coverage, 1); self.assertEqual(zeros.interval_score, 0)

    def test_daily_coverage_does_not_imply_simultaneous_coverage(self):
        frame = pd.DataFrame({"cohort": ["toy"]*28, "method": ["toy"]*28, "store_id": ["one"]*28,
                              "item_id": ["one"]*28, "origin_date": [pd.Timestamp("2020-01-01")]*28,
                              "horizon": np.arange(1,29), "actual": [0.]*27+[2.], "lower": [0.]*28, "upper": [1.]*28})
        _, joint = trajectory_coverage(frame)
        self.assertEqual(joint.simultaneous_coverage.iloc[0], 0)
        self.assertAlmostEqual(summarize(frame, ["method"]).coverage.iloc[0], 27/28)
        with self.assertRaises(ValueError): trajectory_coverage(frame.iloc[:-1])

    def test_checked_predeclared_real_config(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root/"configs/conformal.toml")
        self.assertEqual(config["rolling_scores"], 512)
        self.assertEqual(pd.Timestamp(config["validation_origin"])+pd.Timedelta(days=28), pd.Timestamp(config["fit_cutoff"]))
        self.assertEqual(len(config["split_origins"]), 4)


if __name__ == "__main__":
    unittest.main()
