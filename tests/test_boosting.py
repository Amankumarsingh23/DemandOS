"""Deterministic synthetic global-model contracts; not actual M5 performance."""

from dataclasses import replace
from io import BytesIO
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("duckdb", "lightgbm", "xgboost", "catboost"))


@unittest.skipUnless(AVAILABLE, "Install analytics/eda/boosting extras")
class BoostingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_features import fixture
        from demandos.features.validation import generate, supervised_labels
        from demandos.features.contracts import KEYS, predictive_columns
        from demandos.features.encoding import CATEGORICAL
        from demandos.boosting.config import BoostConfig
        con, feature_config, sales, _, _, _, mapping = fixture()
        with con:
            batch = generate(con, ["2016-02-28", "2016-03-06", "2016-03-13"], feature_config, mapping)
            rows = batch.features[batch.features.target_date.le(pd.Timestamp("2016-03-27"))]
            cls.training = rows.merge(supervised_labels(con, rows, "2016-03-27"), on=KEYS + ["target_date"], validate="one_to_one")
            validation = generate(con, ["2016-03-27"], feature_config, mapping, inference_only=False, label_cutoff="2016-04-24")
            cls.validation = validation.features.merge(validation.labels, on=KEYS + ["target_date"], validate="one_to_one")
            cls.inference = generate(con, ["2016-04-24"], feature_config, mapping).features
        cls.features = [name for name in predictive_columns(feature_config) if name not in {f"{k}_code" for k in CATEGORICAL}]
        cls.feature_config, cls.sales = feature_config, sales
        cls.config = replace(BoostConfig(), iterations=10, patience=3, threads=1, min_child_samples=2)

    def model(self, library="lightgbm", features=None, config=None):
        from demandos.boosting.models import GlobalModel
        return GlobalModel(library, features or self.features, config or self.config, self.feature_config)

    def test_labels_audits_and_integer_id_codes_forbidden(self):
        from demandos.boosting.models import GlobalModel
        for forbidden in ("target_units_sold", "actual", "target_date", "max_sales_source_date", "item_id_code", "future_price"):
            with self.assertRaises(ValueError):
                GlobalModel("lightgbm", self.features + [forbidden], self.config, self.feature_config)
        with self.assertRaises(ValueError):
            GlobalModel("lightgbm", [f for f in self.features if f != "horizon"], self.config, self.feature_config)

    def test_training_target_cutoff_and_chronological_early_stopping(self):
        from demandos.boosting.models import validate_rows
        with self.assertRaises(ValueError):
            self.model().fit(self.training, allowed_cutoff="2016-03-01")
        with self.assertRaises(ValueError):
            self.model().fit(self.training, allowed_cutoff="2016-03-27", validation=self.validation, validation_cutoff="2016-03-31")
        with self.assertRaises(ValueError):
            self.model().fit(self.training, allowed_cutoff="2016-03-27", validation=self.training, validation_cutoff="2016-04-24")
        bad = self.training.assign(max_sales_source_date=pd.Timestamp("2017-01-01"))
        with self.assertRaises(ValueError):
            validate_rows(bad, labelled=True, cutoff="2016-03-27")
        bad = self.training.copy()
        bad["origin_lag_7_source_date"] = bad.target_date - pd.Timedelta(days=7)
        with self.assertRaises(ValueError):
            validate_rows(bad, labelled=True, cutoff="2016-03-27")

    def test_global_all_libraries_native_missing_unknown_and_save_reload(self):
        from demandos.boosting.models import GlobalModel
        changed = self.training.copy()
        changed.loc[changed.index[::5], "relative_level_change_7"] = np.nan
        changed.loc[changed.index[::7], "dept_id"] = pd.NA
        for library in ("lightgbm", "xgboost", "catboost"):
            with self.subTest(library=library), TemporaryDirectory() as directory:
                model = self.model(library).fit(changed, allowed_cutoff="2016-03-27", validation=self.validation, validation_cutoff="2016-04-24")
                predicted = model.predict(self.inference)
                self.assertEqual(len(predicted), 3 * 28)
                self.assertFalse(predicted.duplicated(["store_id", "item_id", "origin_date", "horizon"]).any())
                self.assertTrue(np.isfinite(predicted.prediction).all())
                self.assertEqual(model.metadata["training_series"], 3)
                self.assertGreater(model.metadata["training_origins"], 1)
                self.assertLessEqual(model.rounds, self.config.iterations)
                model.save(directory)
                restored = GlobalModel.load(directory)
                np.testing.assert_allclose(predicted.prediction, restored.predict(self.inference).prediction, rtol=1e-10, atol=1e-10)
                unknown = self.inference.iloc[:1].copy()
                unknown["item_id"] = "new_item"
                unknown["dept_id"] = pd.NA
                unknown["relative_level_change_7"] = np.nan
                self.assertTrue(np.isfinite(restored.predict(unknown).prediction).all())
                self.assertEqual(set(model.importance().feature), set(self.features))

    def test_vocabulary_fit_only_on_eligible_training(self):
        validation = self.validation.copy()
        validation["item_id"] = validation.item_id.astype(str) + "_validation_only"
        model = self.model().fit(self.training, allowed_cutoff="2016-03-27", validation=validation, validation_cutoff="2016-04-24")
        self.assertNotIn("I_validation_only", model.vocabulary["item_id"])
        self.assertEqual(model.metadata["validation_target_max"], "2016-04-24")

    def test_schema_order_output_alignment_and_label_free_prediction(self):
        model = self.model().fit(self.training, allowed_cutoff="2016-03-27")
        shuffled = self.inference.sample(frac=1, random_state=10)
        original = model.predict(self.inference)
        other = model.predict(shuffled[list(reversed(shuffled.columns))])
        keys = ["store_id", "item_id", "origin_date", "horizon"]
        pd.testing.assert_frame_equal(original.sort_values(keys).reset_index(drop=True), other.sort_values(keys).reset_index(drop=True))
        with self.assertRaises(ValueError):
            model.predict(self.inference.drop(columns=self.features[-1]))
        with self.assertRaises(ValueError):
            model.predict(self.inference.assign(target_units_sold=1))
        with self.assertRaises(ValueError):
            model.predict(pd.concat([self.inference, self.inference.iloc[:1]]))
        with self.assertRaises(ValueError):
            model.predict(self.training.drop(columns="target_units_sold"))

    def test_negative_clipping_nonfinite_hard_failure(self):
        model = self.model().fit(self.training, allowed_cutoff="2016-03-27")
        with patch.object(model.booster, "predict", return_value=np.repeat(-3., len(self.inference))):
            output = model.predict(self.inference)
            self.assertTrue(output.prediction.eq(0).all())
            self.assertTrue(output.raw_prediction.eq(-3).all())
        with patch.object(model.booster, "predict", return_value=np.repeat(np.nan, len(self.inference))):
            with self.assertRaises(ValueError):
                model.predict(self.inference)

    def test_future_labels_excluded_and_forecasts_invariant(self):
        from demandos.boosting.dataset import eligible_rows
        future = self.validation.copy()
        pool = pd.concat([self.training, future], ignore_index=True)
        changed = pool.copy()
        changed.loc[changed.target_date.gt(pd.Timestamp("2016-03-27")), "target_units_sold"] += 100000
        first, second = eligible_rows(pool, "2016-03-27"), eligible_rows(changed, "2016-03-27")
        pd.testing.assert_frame_equal(first, second)
        for library in ("lightgbm", "xgboost", "catboost"):
            left = self.model(library).fit(first, allowed_cutoff="2016-03-27")
            right = self.model(library).fit(second, allowed_cutoff="2016-03-27")
            np.testing.assert_allclose(left.predict(self.inference).prediction, right.predict(self.inference).prediction, rtol=1e-10, atol=1e-10)

    def test_fit_digest_survives_parquet_storage_and_row_order(self):
        from demandos.boosting.dataset import row_digest
        from demandos.boosting.models import TARGET
        buffer = BytesIO()
        self.training.to_parquet(buffer, index=False)
        buffer.seek(0)
        reloaded = pd.read_parquet(buffer).sample(frac=1, random_state=42)
        reloaded["target_date"] = reloaded.target_date.astype("datetime64[ms]")
        reloaded["origin_date"] = reloaded.origin_date.astype("datetime64[ns]")
        reloaded[TARGET] = reloaded[TARGET].astype("Int64")
        self.assertEqual(row_digest(self.training), row_digest(reloaded))
        changed = reloaded.copy()
        changed.loc[changed.index[0], TARGET] += 1
        self.assertNotEqual(row_digest(reloaded), row_digest(changed))

    def test_resource_caps_and_origin_generator(self):
        from demandos.boosting.config import BoostConfig
        from demandos.boosting.dataset import historical_origins
        for kwargs in ({"threads": 8}, {"iterations": 1000}, {"max_depth": 20}, {"training_series": 1000}, {"seed": -1}):
            with self.assertRaises(ValueError):
                BoostConfig(**kwargs)
        origins = historical_origins(BoostConfig(), "2016-03-27")
        self.assertEqual(len(origins), 58)
        self.assertEqual(origins[0], pd.Timestamp("2014-01-05"))
        self.assertTrue(origins.max() < pd.Timestamp("2016-03-27"))
        with self.assertRaises(ValueError):
            historical_origins(replace(BoostConfig(), max_origins=2), "2016-03-27")
        with self.assertRaises(ValueError):
            self.model(config=replace(self.config, max_rows=100)).fit(self.training, allowed_cutoff="2016-03-27")
        with self.assertRaises(ValueError):
            self.model(config=replace(self.config, training_series=2)).fit(self.training, allowed_cutoff="2016-03-27")
        with self.assertRaises(ValueError):
            self.model(config=replace(self.config, max_origins=2)).fit(self.training, allowed_cutoff="2016-03-27")

    def test_exact_grid_common_case_metrics_and_fanout(self):
        from demandos.boosting.evaluation import attach_targets, comparison_tables
        cohort = pd.DataFrame([{"series_id": f"store_item/{store}/I", "store_id": store, "item_id": "I", "regime": "smooth", "abc_xyz": "AX"}
                               for store in self.sales.store_id.unique()])
        origin = pd.Timestamp("2016-04-24")
        data = {"origins": [origin], "evaluation_cohort": cohort,
                "frames": {r.series_id: self.sales[self.sales.store_id.eq(r.store_id)][["date", "units_sold"]].copy() for r in cohort.itertuples()}}
        model = self.model().fit(self.training, allowed_cutoff="2016-03-27")
        output = model.predict(self.inference)
        baseline = output.assign(model="seasonal_naive")
        coverage = [{"fit_id": name, "model": name, "origin": origin, "series_id": r.series_id, "status": "ok", "reason": None}
                    for name in ("lightgbm", "seasonal_naive") for r in cohort.itertuples()]
        forecasts, _ = attach_targets(pd.concat([output, baseline]), coverage, data, ["lightgbm", "seasonal_naive"])
        tables = comparison_tables(forecasts, ["lightgbm", "seasonal_naive"])
        self.assertEqual(len(forecasts), 168)
        self.assertEqual(tables["metrics_common"].observations.tolist(), [84, 84])
        self.assertTrue(tables["paired_origin"].mean_absolute_difference.eq(0).all())
        with self.assertRaises(ValueError):
            attach_targets(pd.concat([output, output.iloc[:1], baseline]), coverage, data, ["lightgbm", "seasonal_naive"])
        with self.assertRaises(ValueError):
            attach_targets(pd.concat([output, baseline, output.iloc[:1].assign(item_id="unplanned")]), coverage, data, ["lightgbm", "seasonal_naive"])

    def test_internal_validation_cannot_move_into_final_targets(self):
        from demandos.boosting.dataset import chronological_development
        config = replace(self.config, validation_origin="2016-03-27")
        pool = pd.concat([self.training, self.validation])
        with self.assertRaises(ValueError):
            chronological_development(pool, config, "2016-04-01")


if __name__ == "__main__":
    unittest.main()
