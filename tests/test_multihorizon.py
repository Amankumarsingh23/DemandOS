"""Deterministic synthetic architecture tests; not real M5 observations."""

from dataclasses import replace
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd


AVAILABLE = all(importlib.util.find_spec(name) for name in ("duckdb", "lightgbm"))


class FeedbackModel:
    """Transparent synthetic recurrence; independent of any real-data fit."""
    features = ["sales_at_origin", "rolling_mean_7_at_origin", "origin_lag_1"]

    def matrix(self, rows):
        return rows[self.features].astype(float)

    def predict(self, rows):
        from demandos.features.contracts import KEYS
        result = rows[KEYS+["target_date"]].reset_index(drop=True).copy()
        result["raw_prediction"] = .8*rows.sales_at_origin.to_numpy(float)+2
        result["prediction"] = np.maximum(result.raw_prediction, 0)
        result["model"] = "synthetic_feedback"
        return result


@unittest.skipUnless(AVAILABLE, "Install analytics/eda/boosting extras")
class MultiHorizonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_features import fixture
        from demandos.features.validation import generate, supervised_labels
        from demandos.features.contracts import KEYS, predictive_columns
        from demandos.features.encoding import CATEGORICAL
        from demandos.boosting.config import BoostConfig
        con, fc, cls.sales, _, _, _, mapping = fixture()
        with con:
            rows = generate(con, ["2016-02-28", "2016-03-06", "2016-03-13"], fc, mapping).features
            rows = rows[rows.target_date.le(pd.Timestamp("2016-03-27"))]
            cls.training = rows.merge(supervised_labels(con, rows, "2016-03-27"), on=KEYS+["target_date"], validate="one_to_one")
            val = generate(con, ["2016-03-27"], fc, mapping, inference_only=False, label_cutoff="2016-04-24")
            cls.validation = val.features.merge(val.labels, on=KEYS+["target_date"], validate="one_to_one")
            cls.inference = val.features
            cls.leap = generate(con, ["2016-02-28"], fc, mapping).features
        cls.fc = fc
        cls.config = replace(BoostConfig(), iterations=4, patience=2, threads=1, min_child_samples=2)
        cls.features = [name for name in predictive_columns(fc) if name not in {f"{c}_code" for c in CATEGORICAL}]

    def prefix(self, frame=None, sales=None):
        from demandos.multihorizon.recursive import history_prefix
        frame = self.inference if frame is None else frame
        return history_prefix(self.sales if sales is None else sales, frame, frame.origin_date.iloc[0], self.fc.lookback)

    def test_recursive_first_step_equals_original_features_and_leap_day(self):
        from demandos.multihorizon.recursive import recursive_predict
        from demandos.boosting.models import GlobalModel
        model = GlobalModel("lightgbm", self.features, self.config, self.fc)
        model.fit(self.training[self.training.horizon.eq(1)], allowed_cutoff="2016-03-27")
        first = model.predict(self.inference[self.inference.horizon.eq(1)]).sort_values(["store_id", "item_id"])
        predicted, trace = recursive_predict(model, self.inference, self.prefix(), self.fc)
        np.testing.assert_allclose(predicted[predicted.horizon.eq(1)].prediction, first.prediction, rtol=0, atol=1e-12)
        leap, _ = recursive_predict(FeedbackModel(), self.leap, self.prefix(self.leap), self.fc)
        self.assertTrue(leap[leap.horizon.eq(1)].target_date.eq(pd.Timestamp("2016-02-29")).all())
        self.assertTrue(trace.actual_source_max.eq(pd.Timestamp("2016-03-27")).all())

    def test_recursive_feedback_recomputes_fractional_lags_and_rolls(self):
        from demandos.multihorizon.recursive import recursive_predict
        frame = self.inference[self.inference.store_id.eq("CA_1")].copy()
        prefix = self.prefix(frame).assign(units_sold=20)
        # History-derived columns in the original template must agree at step one.
        for name in frame:
            if name.startswith("rolling_") and not name.startswith("rolling_observations_"):
                if "zero_fraction" in name or "std" in name:
                    frame[name] = 0.
                elif "positive_count" in name:
                    frame[name] = float(name.split("_")[3])
                elif "sum" in name:
                    frame[name] = 20.*int(name.split("_")[2])
                else:
                    frame[name] = 20.
            if name.startswith("origin_lag_") and not name.endswith("_source_date"):
                frame[name] = 20.
        frame["sales_at_origin"] = 20.
        predicted, trace = recursive_predict(FeedbackModel(), frame, prefix, self.fc)
        np.testing.assert_allclose(predicted.prediction.iloc[:3], [18., 16.4, 15.12])
        self.assertAlmostEqual(trace.feedback_last_value.iloc[2], 16.4)
        self.assertAlmostEqual(trace.rolling_sum_7.iloc[2], 5*20+18+16.4)
        self.assertAlmostEqual(trace.origin_lag_1.iloc[2], 18.)
        self.assertEqual(trace.prior_predictions.tolist(), list(range(28)))

    def test_recursive_future_actuals_rejected_prefix_and_output_invariant(self):
        from demandos.multihorizon.recursive import recursive_predict
        changed = self.sales.copy()
        changed.loc[changed.date.gt(pd.Timestamp("2016-03-27")), "units_sold"] += 1000000
        a = self.prefix()
        b = self.prefix(sales=changed)
        pd.testing.assert_frame_equal(a, b)
        left, _ = recursive_predict(FeedbackModel(), self.inference, a, self.fc)
        right, _ = recursive_predict(FeedbackModel(), self.inference, b, self.fc)
        pd.testing.assert_frame_equal(left, right)
        with self.assertRaises(ValueError):
            recursive_predict(FeedbackModel(), self.inference, self.sales, self.fc)

    def test_no_cross_series_contamination(self):
        from demandos.multihorizon.recursive import recursive_predict
        first, _ = recursive_predict(FeedbackModel(), self.inference, self.prefix(), self.fc)
        changed = self.prefix()
        changed.loc[changed.store_id.eq("TX_1"), "units_sold"] += 100
        # Use an isolated CA call to prove its state does not depend on TX.
        ca = self.inference[self.inference.store_id.eq("CA_1")]
        second, _ = recursive_predict(FeedbackModel(), ca, changed[changed.store_id.eq("CA_1")], self.fc)
        pd.testing.assert_frame_equal(first[first.store_id.eq("CA_1")].reset_index(drop=True), second)

    def test_every_recursive_history_statistic_and_short_missingness(self):
        from demandos.multihorizon.recursive import update_history_features
        row = self.inference.iloc[:1].reset_index(drop=True).copy()
        origin = pd.Timestamp(row.origin_date.iloc[0])
        key = (row.store_id.iloc[0], row.item_id.iloc[0])
        values = np.arange(57, dtype=float)
        values[-2:] = [31.5, 0.]
        updated = update_history_features(row, {key: values}, {key: pd.date_range(origin-pd.Timedelta(days=56), origin)}, origin, self.fc).iloc[0]
        for w in self.fc.windows:
            tail = values[-w:]
            expected = {"mean": np.mean(tail), "sum": sum(tail), "std": np.std(tail), "max": max(tail),
                        "zero_fraction": np.mean(tail == 0), "positive_count": np.sum(tail > 0)}
            for name, result in expected.items():
                self.assertAlmostEqual(updated[f"rolling_{name}_{w}_at_origin"], result)
        self.assertAlmostEqual(updated.mean_previous_7_at_origin, np.mean(values[-14:-7]))
        self.assertAlmostEqual(updated.level_change_7, np.mean(values[-7:])-np.mean(values[-14:-7]))
        self.assertAlmostEqual(updated.relative_level_change_7, updated.level_change_7/updated.mean_previous_7_at_origin)
        self.assertEqual(updated.origin_lag_1, 31.5)
        row["first_observed_date"] = origin-pd.Timedelta(days=2)
        short = update_history_features(row, {key: [0., 0., 1.5]}, {key: pd.date_range(origin-pd.Timedelta(days=2), origin)}, origin, self.fc).iloc[0]
        self.assertTrue(pd.isna(short.rolling_mean_7_at_origin))
        self.assertEqual(short.rolling_observations_7, 3)
        self.assertTrue(pd.isna(short.origin_lag_7))

    def test_recursive_missing_observation_and_nonfinite_feedback_fail(self):
        from demandos.multihorizon.recursive import recursive_predict
        bad = self.prefix().drop(index=self.prefix().index[0])
        with self.assertRaises(ValueError):
            recursive_predict(FeedbackModel(), self.inference, bad, self.fc)
        class Nonfinite(FeedbackModel):
            def predict(self, rows):
                self.calls = getattr(self, "calls", 0)+1
                result = super().predict(rows)
                return result.assign(prediction=np.nan) if self.calls == 28 else result
        with self.assertRaises(ValueError):
            recursive_predict(Nonfinite(), self.inference, self.prefix(), self.fc)

    def test_vector_cutoff_entire_target_and_partial_samples(self):
        from demandos.multihorizon.vector import complete_rows, target_vectors, VectorLayout, attach_vectors
        from demandos.boosting.models import TARGET
        complete = complete_rows(self.training, "2016-03-27")
        self.assertEqual(len(complete), 3*28)
        self.assertTrue(complete.origin_date.eq(pd.Timestamp("2016-02-28")).all())
        layout = VectorLayout.from_long(complete.drop(columns=TARGET))
        X, Y = layout.pack(complete.drop(columns=TARGET)), target_vectors(complete)
        with self.assertRaises(ValueError):
            attach_vectors(layout, X, Y, "2016-03-26")
        for bad in (Y.drop(columns="y_28"), Y.assign(y_28=np.nan), pd.concat([Y, Y.iloc[:1]])):
            with self.assertRaises(ValueError):
                attach_vectors(layout, X, bad, "2016-03-27")
        reconstructed = attach_vectors(layout, X, Y, "2016-03-27")
        self.assertEqual(len(reconstructed), len(complete))

    def test_vector_schema_roundtrip_forbids_targets_and_fanout(self):
        from demandos.multihorizon.vector import VectorLayout
        from demandos.features.contracts import KEYS
        layout = VectorLayout.from_long(self.inference)
        wide = layout.pack(self.inference)
        restored = layout.unpack(wide)
        columns = sorted(self.inference.columns)
        pd.testing.assert_frame_equal(restored[columns], self.inference.sort_values(KEYS).reset_index(drop=True)[columns], check_dtype=False)
        with self.assertRaises(ValueError):
            layout.unpack(wide.assign(target_units_sold=1))
        with self.assertRaises(ValueError):
            layout.pack(self.inference.iloc[1:])
        with self.assertRaises(ValueError):
            layout.pack(pd.concat([self.inference, self.inference.iloc[:1]]))

    def test_direct_and_vector_fit_28_models_save_load_unknown_missing(self):
        from demandos.multihorizon.models import HorizonModels, IndependentMultiOutput
        from demandos.multihorizon.vector import complete_rows, VectorLayout, target_vectors
        from demandos.boosting.models import TARGET
        records = []
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            direct = HorizonModels.fit(self.training, self.features, self.config, self.fc, "2016-03-27", directory/"direct", records,
                phase="development", validation=self.validation, validation_end="2016-04-24")
            complete = complete_rows(self.training, "2016-03-27")
            layout = VectorLayout.from_long(complete.drop(columns=TARGET))
            vector = IndependentMultiOutput.fit(layout.pack(complete.drop(columns=TARGET)), target_vectors(complete), layout,
                self.features, self.config, self.fc, "2016-03-27", directory/"vector", records, phase="development",
                validation=(layout.pack(self.inference), target_vectors(self.validation)), validation_end="2016-04-24")
            predicted = direct.predict(self.inference)
            np.testing.assert_allclose(predicted.prediction, HorizonModels.load(directory/"direct").predict(self.inference).prediction)
            raw, scored = vector.predict_vector(layout.pack(self.inference))
            self.assertEqual(raw.shape, (3, 28))
            self.assertEqual(scored.shape, (3, 28))
            restored = IndependentMultiOutput.load(directory/"vector")
            np.testing.assert_allclose(scored, restored.predict_vector(layout.pack(self.inference))[1])
            unknown = self.inference.copy()
            unknown["item_id"] = "new_item"
            unknown["relative_level_change_7"] = np.nan
            self.assertTrue(np.isfinite(restored.predict_vector(layout.pack(unknown))[1]).all())
        self.assertEqual(len(records), 56)
        self.assertTrue(all(r["target_max"] <= pd.Timestamp("2016-03-27") for r in records))
        self.assertTrue(all(r["save_load_equal"] for r in records))

    def test_future_vector_labels_do_not_enter_eligible_targets(self):
        from demandos.multihorizon.vector import complete_rows, target_vectors
        from demandos.boosting.models import TARGET
        pool = pd.concat([self.training, self.validation], ignore_index=True)
        changed = pool.copy()
        changed.loc[changed.target_date.gt(pd.Timestamp("2016-03-27")), TARGET] += 100000
        pd.testing.assert_frame_equal(target_vectors(complete_rows(pool, "2016-03-27")), target_vectors(complete_rows(changed, "2016-03-27")))

    def test_ensemble_validation_selection_alignment_and_final_cutoff(self):
        from demandos.multihorizon.ensemble import select_weight, blend
        from demandos.features.contracts import KEYS
        from demandos.boosting.models import TARGET
        actual = self.validation[KEYS+["target_date", TARGET]]
        pooled = actual.drop(columns=TARGET).assign(prediction=10., raw_prediction=10.)
        seasonal = pooled.assign(prediction=20., raw_prediction=20.)
        labels = actual.assign(target_units_sold=15)
        weight, table = select_weight(pooled, seasonal, labels, (0., .5, 1.), validation_end="2016-04-24", first_evaluation="2016-05-01")
        self.assertEqual(weight, .5)
        self.assertEqual(table.iloc[0].mae, 0.)
        with self.assertRaises(ValueError):
            select_weight(pooled, seasonal, labels, (0., .5, 1.), validation_end="2016-04-24", first_evaluation="2016-04-01")
        for bad in (seasonal.iloc[1:], pd.concat([seasonal, seasonal.iloc[:1]])):
            with self.assertRaises(ValueError):
                blend(pooled, bad, .5)

    def test_deterministic_budget_and_complete_model_count(self):
        from demandos.multihorizon.config import StrategyConfig
        from demandos.multihorizon.models import HorizonModels
        self.assertEqual(StrategyConfig().max_native_models, 232)
        for kwargs in ({"max_native_models": 231}, {"ensemble_weights": [0., .2, 1.]}, {"boosting_config": "../elsewhere.toml"}):
            with self.assertRaises(ValueError):
                StrategyConfig(**kwargs)
        with self.assertRaises(ValueError):
            HorizonModels({})


if __name__ == "__main__":
    unittest.main()
