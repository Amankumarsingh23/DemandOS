"""Small deterministic synthetic cases; the optional raw-M5 membership test is real."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from demandos.reconciliation.hierarchy import build, dense_cohort, observed_matrix, LEVELS
from demandos.reconciliation.methods import projection, shares, reconcile
from demandos.reconciliation.covariance import estimate, eligible
from demandos.reconciliation.experiment import independent_bases, long_matrix
from demandos.reconciliation.pipeline import load_config
from demandos.backtesting.metrics import rmsse_scale, summarize, common_cases


def tiny():
    return pd.DataFrame([
        {"store_id": store, "state_id": state, "item_id": item, "dept_id": dept, "cat_id": cat}
        for store, state in [("CA_1", "CA"), ("TX_1", "TX")]
        for item, dept, cat in [("a", "food_1", "food"), ("b", "food_2", "food"), ("c", "home_1", "home")]])


def errors(nodes, blocks=8):
    rng = np.random.default_rng(42)
    records = []
    for origin in pd.date_range("2014-01-01", periods=blocks, freq="28D"):
        for horizon in range(1, 29):
            for i, node in enumerate(nodes):
                records.append({"node_id": node, "origin_date": origin,
                                "target_date": origin+pd.Timedelta(days=horizon), "horizon": horizon,
                                "error": (i+1)*rng.normal()+rng.normal(),
                                "fit_cutoff": origin, "fit_target_max": origin})
    return pd.DataFrame(records)


class ReconciliationTests(unittest.TestCase):
    def test_twelve_exact_dimensions_and_sparse_identity(self):
        h = build(tiny())
        self.assertEqual(len(LEVELS), 12)
        self.assertEqual(LEVELS["ITEM_STATE"], ("item_id", "state_id"))
        self.assertEqual(LEVELS["STORE_DEPARTMENT"], ("store_id", "dept_id"))
        self.assertIsInstance(h.S, sparse.csr_matrix)
        self.assertEqual(h.S.nnz, 12*6)
        np.testing.assert_array_equal(h.S[h.bottom_rows].toarray(), np.eye(6))
        self.assertFalse(h.nodes.node_id.duplicated().any())
        self.assertTrue(h.nodes.node_id.str.startswith("COHORT_").all())
        self.assertEqual(h.check()["level_counts"]["ITEM"], 3)

    def test_deterministic_maps_independent_of_input_order(self):
        a, b = build(tiny()), build(tiny().sample(frac=1, random_state=2))
        pd.testing.assert_frame_equal(a.nodes, b.nodes)
        pd.testing.assert_frame_equal(a.bottom, b.bottom)
        self.assertEqual((a.S != b.S).nnz, 0)

    def test_duplicate_and_inconsistent_dimensions_rejected(self):
        for frame in [pd.concat([tiny(), tiny().iloc[:1]]), tiny().assign(state_id=None)]:
            with self.assertRaises(ValueError): build(frame)
        frame = tiny(); frame.loc[1, "state_id"] = "WI"
        with self.assertRaises(ValueError): build(frame)
        frame = tiny(); frame.loc[3, "cat_id"] = "bad"
        with self.assertRaises(ValueError): build(frame)

    def test_crossed_levels_each_sum_total_once(self):
        h = build(tiny())
        bottom = np.arange(1, 7)[:, None]
        all_nodes = h.S@bottom
        for level in LEVELS:
            self.assertEqual(all_nodes[h.rows(level)].sum(), bottom.sum())
        self.assertEqual(all_nodes.sum(), 12*bottom.sum())

    def test_full_matrix_densification_blocked_before_conversion(self):
        S = sparse.csr_matrix((42840, 30490))
        with self.assertRaisesRegex(ValueError, "bounded cohort"):
            dense_cohort(S)
        with self.assertRaises(ValueError): projection(S)

    def test_observed_keys_gap_and_values(self):
        h = build(tiny())
        rows = [{**key, "date": date, "units_sold": i+1}
                for i, key in enumerate(h.bottom[["store_id", "item_id"]].to_dict("records"))
                for date in pd.date_range("2016-02-28", periods=3)]
        daily = pd.DataFrame(rows)
        dates, values = observed_matrix(daily, h)
        self.assertIn(pd.Timestamp("2016-02-29"), dates)
        np.testing.assert_array_equal(values[:, 0], np.arange(1, 7))
        for bad in [daily.iloc[1:], pd.concat([daily, daily.iloc[:1]]), daily.assign(units_sold=-1), daily.assign(units_sold=.5),
                    daily.assign(date=daily.date+pd.Timedelta(hours=1)), daily.iloc[:0]]:
            with self.assertRaises(ValueError): observed_matrix(bad, h)

    def test_ols_hand_calculation(self):
        S = sparse.csr_matrix([[1., 1.], [1., 0.], [0., 1.]])
        result, info = projection(S)(np.array([[350.], [100.], [200.]]))
        np.testing.assert_allclose(result[:, 0], [333.3333333333, 116.6666666667, 216.6666666667])
        self.assertAlmostEqual(result[0, 0], result[1:, 0].sum())
        self.assertLess(info["design_condition"], 10)

    def test_ols_already_coherent_is_unchanged(self):
        h = build(tiny())
        target = h.S@np.arange(1, 7)[:, None]
        result, _ = projection(h.S)(target)
        np.testing.assert_allclose(result, target, atol=1e-12)

    def test_weighted_solution_matches_normal_equations(self):
        S = sparse.csr_matrix([[1., 1.], [1., 0.], [0., 1.]])
        y = np.array([[350.], [100.], [200.]])
        for W in [np.array([4., 1., 2.]), np.array([[4., .1, .2], [.1, 1., .1], [.2, .1, 2.]])]:
            matrix = np.diag(W) if W.ndim == 1 else W
            dense = S.toarray()
            expected = S@np.linalg.solve(dense.T@np.linalg.solve(matrix, dense), dense.T@np.linalg.solve(matrix, y))
            np.testing.assert_allclose(projection(S, W)(y)[0], expected)

    def test_invalid_variance_and_forecast_shape(self):
        S = sparse.eye(2, format="csr")
        for W in [np.array([0., 1.]), np.array([np.nan, 1.]), np.array([-1., 2.]), np.eye(3)]:
            with self.assertRaises(ValueError): projection(S, W)
        with self.assertRaises(ValueError): projection(S)(np.ones((3, 28)))
        with self.assertRaises(ValueError): projection(S)(np.full((2, 28), np.inf))

    def test_nonnegative_is_solved_at_bottom_then_aggregated(self):
        S = sparse.csr_matrix([[1., 1.], [1., 0.], [0., 1.]])
        y = np.array([[1.], [-20.], [20.]])
        signed = projection(S)(y)[0]
        constrained = projection(S, nonnegative=True)(y)[0]
        self.assertLess(signed.min(), 0)
        self.assertGreaterEqual(constrained.min(), 0)
        self.assertAlmostEqual(constrained[0, 0], constrained[1:, 0].sum())
        self.assertGreater(abs(np.maximum(signed, 0)[0, 0]-np.maximum(signed, 0)[1:, 0].sum()), 0)

    def test_top_down_and_middle_partition_and_zero_fallback(self):
        h = build(tiny())
        history = np.arange(1, 7)[:, None]*np.ones((6, 40))
        for level in ["TOTAL", "STORE_DEPARTMENT"]:
            P = dense_cohort(h.S[h.rows(level)])
            allocation, status = shares(history, P)
            np.testing.assert_allclose(allocation.sum(axis=0), 1)
            self.assertTrue((allocation >= 0).all())
            zero, statuses = shares(history*0, P)
            np.testing.assert_allclose(zero.sum(axis=0), 1)
            self.assertTrue(statuses.status.eq("equal_zero_volume").all())
        with self.assertRaises(ValueError): shares(history, np.ones((2, 6)))
        with self.assertRaises(ValueError): shares(history, np.zeros((1, 6)))
        with self.assertRaises(ValueError): shares(history, np.ones((1, 6)), supported=np.zeros(6, bool))
        with self.assertRaises(ValueError): shares(history*np.nan, np.ones((1, 6)))
        with self.assertRaises(ValueError): shares(history[:, :0], np.ones((1, 6)))

    def test_unsupported_new_product_zero_share(self):
        allocation, status = shares(np.ones((3, 10)), np.ones((1, 3)), supported=[True, True, False])
        np.testing.assert_array_equal(allocation[:, 0], [.5, .5, 0.])
        self.assertEqual(status.unsupported_descendants.iloc[0], 1)

    def test_all_methods_coherent_and_independent_bases_not_presummed(self):
        h = build(tiny())
        history = np.arange(1, 7)[:, None]*np.ones((6, 40))
        base = independent_bases(h.S@history)
        base[h.bottom_rows] += 3
        self.assertGreater(np.abs(base-h.S@base[h.bottom_rows]).max(), 0)
        cov = {"W": np.ones(len(h.nodes)), "diagonal": np.ones(len(h.nodes)), "method": "mint_shrinkage"}
        outputs, _, allocation = reconcile(h, base, history, cov)
        self.assertEqual(len(outputs), 8)
        for name, result in outputs.items():
            np.testing.assert_allclose(result, h.S@result[h.bottom_rows], atol=1e-10)
        np.testing.assert_array_equal(outputs["bottom_up"][h.bottom_rows], base[h.bottom_rows])
        np.testing.assert_allclose(outputs["top_down"][h.rows("TOTAL")], base[h.rows("TOTAL")])
        structural = np.asarray(h.S.sum(axis=1)).ravel()
        fallback = {"W": structural, "diagonal": structural, "method": "structural_fallback"}
        alternatives, _, _ = reconcile(h, base, history, fallback)
        self.assertNotIn("wls_variance", alternatives)
        self.assertIn("nonnegative_wls_structural", alternatives)

    def test_paired_covariance_alignment_shrinkage_and_finite_condition(self):
        nodes = ["root", "a", "b"]
        residuals = errors(nodes)
        cov = estimate(residuals, nodes, "2015-01-01", np.array([2., 1., 1.]))
        reordered = estimate(residuals.sample(frac=1, random_state=2), nodes, "2015-01-01", np.array([2., 1., 1.]))
        np.testing.assert_array_equal(cov["W"], reordered["W"])
        self.assertEqual(cov["method"], "mint_shrinkage")
        self.assertEqual(cov["diagnostics"]["origin_blocks"], 8)
        self.assertGreater(np.linalg.eigvalsh(cov["W"]).min(), 0)
        self.assertFalse(cov["diagnostics"]["sample_covariance_full_rank_claim"])

    def test_covariance_explicit_support_and_condition_fallbacks(self):
        nodes = ["root", "a", "b"]
        residuals = errors(nodes, blocks=1)
        cov = estimate(residuals, nodes, "2015-01-01", [2, 1, 1])
        self.assertEqual(cov["method"], "variance_fallback")
        self.assertIn("insufficient", cov["diagnostics"]["fallback"])
        cov = estimate(errors(nodes), nodes, "2015-01-01", [2, 1, 1], maximum_condition=1)
        self.assertEqual(cov["method"], "variance_fallback")
        empty = estimate(residuals, nodes, "2013-01-01", [2, 1, 1])
        self.assertEqual(empty["method"], "structural_fallback")

    def test_future_target_excluded_even_when_origin_precedes_cutoff(self):
        nodes = ["root", "a", "b"]
        residuals = errors(nodes)
        cutoff = pd.Timestamp("2014-01-14")
        _, matrix = eligible(residuals, nodes, cutoff)
        changed = residuals.copy()
        changed.loc[changed.target_date.gt(cutoff), "error"] += 999999
        _, perturbed = eligible(changed, nodes, cutoff)
        pd.testing.assert_frame_equal(matrix, perturbed)
        self.assertEqual(len(matrix), 13)

    def test_invalid_residual_provenance_alignment_and_duplicates(self):
        nodes = ["root", "a", "b"]
        source = errors(nodes)
        variants = [source.iloc[1:], pd.concat([source, source.iloc[:1]]),
                    source.assign(fit_target_max=pd.Timestamp("2020-01-01")),
                    source.assign(fit_cutoff=pd.Timestamp("2020-01-01")),
                    source.assign(target_date=pd.Timestamp("2020-01-01")), source.assign(error=np.nan)]
        for residuals in variants:
            with self.assertRaises(ValueError): eligible(residuals, nodes, "2015-01-01")

    def test_zero_variance_floor(self):
        nodes = ["root", "a", "b"]
        source = errors(nodes).assign(error=0.)
        cov = estimate(source, nodes, "2015-01-01", [2, 1, 1])
        self.assertTrue((cov["diagonal"] > 0).all())
        self.assertEqual(cov["diagnostics"]["floored_nodes"], 3)
        self.assertEqual(cov["method"], "variance_fallback")

    def test_rank_deficient_covariance_is_not_mislabeled_mint(self):
        source = errors(["root", "a", "b"])
        first = source[source.node_id.eq("a")].set_index(["origin_date", "horizon"]).error
        source.loc[source.node_id.eq("root"), "error"] = first.to_numpy()
        result = estimate(source, ["root", "a", "b"], "2015-01-01", [2, 1, 1])
        self.assertEqual(result["method"], "variance_fallback")
        self.assertIn("underdetermined", result["diagnostics"]["fallback"])
        self.assertEqual(result["W"].ndim, 1)

    def test_origin_prefix_shares_and_base_forecasts_ignore_future(self):
        history = np.arange(1, 61).reshape(3, 20).astype(float)
        changed = history.copy(); changed[:, 10:] += 999999
        for values in [history, changed]:
            allocation, _ = shares(values[:, :10], np.ones((1, 3)))
            np.testing.assert_allclose(allocation[:, 0], history[:, :10].sum(axis=1)/history[:, :10].sum())
        np.testing.assert_array_equal(independent_bases(history[:, :10]), independent_bases(changed[:, :10]))

    def test_exact_keys_and_leap_day(self):
        h = build(tiny())
        frame = long_matrix(np.ones((len(h.nodes), 28)), h, pd.Timestamp("2016-02-28"))
        self.assertEqual(frame.target_date.min(), pd.Timestamp("2016-02-29"))
        self.assertEqual(frame.target_date.max(), pd.Timestamp("2016-03-27"))
        self.assertFalse(frame.duplicated(["node_id", "origin_date", "horizon", "target_date"]).any())
        with self.assertRaises(ValueError): long_matrix(np.ones((len(h.nodes), 27)), h, pd.Timestamp("2016-02-28"))

    def test_signed_metric_undefined_denominators_and_common_cases(self):
        frame = pd.DataFrame({"model": ["ols"]*2, "window": ["expanding"]*2,
                              "series_id": ["zero"]*2, "origin": [pd.Timestamp("2016-01-31")]*2,
                              "target_date": pd.date_range("2016-02-01", periods=2), "horizon": [1, 2],
                              "actual": [0., 0.], "prediction": [-1., 2.], "scale": [None, None], "status": ["ok"]*2})
        metric = summarize(frame, ["model"]).iloc[0]
        self.assertEqual(metric.mae, 1.5)
        self.assertTrue(pd.isna(metric.wape))
        self.assertEqual(metric.undefined_rmsse_cases, 1)
        self.assertEqual(rmsse_scale(np.zeros(5))["scale_status"], "no_positive_history")
        both = pd.concat([frame, frame.assign(model="bottom_up")], ignore_index=True)
        self.assertEqual(len(common_cases(both, ["ols", "bottom_up"])), 4)
        with self.assertRaises(ValueError): common_cases(pd.concat([both, frame]), ["ols", "bottom_up"])

    def test_config_bounded_defaults(self):
        config = load_config(Path(__file__).resolve().parents[1]/"configs/reconciliation.toml")
        self.assertLessEqual(config["max_bottom"], 256)
        self.assertLess(pd.Timestamp(config["covariance_cutoff"]), pd.Timestamp("2016-01-31"))

    def test_per_level_metrics_match_explicit_losses(self):
        h = build(tiny())
        values = long_matrix(h.S@np.arange(1, 7)[:, None]*np.ones((1, 28)), h, pd.Timestamp("2016-01-31"))
        values = values.assign(actual=values.prediction+1, scale=2., model="test", window="expanding",
                               origin=values.origin_date, series_id=values.node_id)
        scored = summarize(values, ["level", "model"])
        self.assertEqual(len(scored), 12)
        self.assertTrue(scored.mae.eq(1).all())
        np.testing.assert_allclose(scored.macro_rmsse, np.sqrt(.5))
        for record in scored.itertuples():
            level = values[values.level.eq(record.level)]
            self.assertAlmostEqual(record.wape, len(level)/level.actual.sum())

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/"data/raw/sales_train_validation.csv").exists(), "Real M5 source not available")
    def test_real_m5_full_membership_without_dense_matrix(self):
        path = Path(__file__).resolve().parents[1]/"data/raw/sales_train_validation.csv"
        before = path.stat()
        metadata = pd.read_csv(path, usecols=["store_id", "item_id", "state_id", "cat_id", "dept_id"])
        h = build(metadata, scope="FULL")
        stats = h.check(full=True)
        self.assertEqual(stats["shape"], [42840, 30490])
        self.assertEqual(stats["nnz"], 365880)
        self.assertLess(stats["sparse_bytes"], 6_000_000)
        self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(path.stat().st_size, before.st_size)


if __name__ == "__main__":
    unittest.main()
