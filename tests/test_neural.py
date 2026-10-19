"""Deterministic synthetic fixtures; these are not real M5 results."""

from dataclasses import replace
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


AVAILABLE = importlib.util.find_spec("torch") is not None


def fixture():
    from demandos.neural.config import NeuralConfig
    from demandos.neural.sequences import Panel, Preprocessor, examples
    dates = pd.date_range("2015-10-01", "2016-05-31")
    calendar = pd.DataFrame({"date": dates, "event_name_1": [None]*len(dates), "event_name_2": [None]*len(dates),
                             "snap_CA": (dates.day <= 10).astype(int), "snap_TX": (dates.day > 10).astype(int), "snap_WI": 0})
    metadata = pd.DataFrame([
        {"store_id": "CA_1", "state_id": "CA", "item_id": "A", "cat_id": "FOOD", "dept_id": "D1"},
        {"store_id": "TX_1", "state_id": "TX", "item_id": "B", "cat_id": "HOME", "dept_id": "D2"}])
    daily = pd.concat([pd.DataFrame({"store_id": row.store_id, "item_id": row.item_id, "date": dates,
                                    "units_sold": (np.arange(len(dates)) % 7)*(i+1), "sell_price": 3.})
                       for i, row in enumerate(metadata.itertuples())], ignore_index=True)
    config = replace(NeuralConfig(), epochs=2, patience=1, threads=1, batch_size=2)
    panel = Panel(daily, calendar, metadata)
    rows = examples(metadata, [pd.Timestamp("2016-01-03")], config, "2016-01-31")
    prep = Preprocessor.fit(panel, rows, config, "2016-01-31")
    return daily, calendar, metadata, config, panel, rows, prep


@unittest.skipUnless(AVAILABLE, "Install neural extra")
class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.daily, self.calendar, self.metadata, self.config, self.panel, self.rows, self.prep = fixture()

    def test_exact_56_28_dates_and_leap_day(self):
        from demandos.neural.sequences import SequenceDataset, examples
        rows = examples(self.metadata, [pd.Timestamp("2016-02-28")], self.config)
        dataset = SequenceDataset(self.panel, rows, self.prep, self.config)
        audit = dataset.audits[0]
        self.assertEqual(audit["history_start"], pd.Timestamp("2016-01-04"))
        self.assertEqual(audit["history_end"], pd.Timestamp("2016-02-28"))
        self.assertEqual(audit["target_start"], pd.Timestamp("2016-02-29"))
        self.assertEqual(audit["target_end"], pd.Timestamp("2016-03-27"))
        self.assertEqual(tuple(dataset.tensors[0].shape), (2, 56, 10))
        self.assertEqual(tuple(dataset.tensors[1].shape), (2, 28, 7))
        self.assertEqual(len(dataset.tensors), 4)

    def test_full_vector_cutoff_not_just_origin(self):
        from demandos.neural.sequences import examples, Preprocessor, SequenceDataset
        rows = examples(self.metadata, pd.to_datetime(["2016-01-03", "2016-01-17"]), self.config, "2016-01-31")
        pd.testing.assert_frame_equal(rows, self.rows)
        for callback in [lambda: self.panel.labels("CA_1", "A", "2016-01-03", self.config, "2016-01-30"),
                         lambda: Preprocessor.fit(self.panel, self.rows, self.config, "2016-01-30"),
                         lambda: SequenceDataset(self.panel, self.rows, self.prep, self.config, label_cutoff="2016-01-30")]:
            with self.assertRaises(ValueError): callback()

    def test_bad_daily_keys_and_invalid_units_are_hard_failures(self):
        from demandos.neural.sequences import Panel
        for daily in (self.daily.iloc[::-1], pd.concat([self.daily, self.daily.iloc[:1]]),
                      self.daily.drop(index=80), self.daily.assign(units_sold=-1), self.daily.assign(units_sold=.5),
                      self.daily.assign(units_sold=np.nan)):
            with self.assertRaises(ValueError): Panel(daily, self.calendar, self.metadata)
        for calendar in (self.calendar.iloc[::-1], self.calendar.drop(index=12), self.calendar.assign(snap_CA=2)):
            with self.assertRaises(ValueError): Panel(self.daily, calendar, self.metadata)

    def test_no_cross_series_and_correct_state_snap(self):
        left, _, _, _ = self.panel.inputs("CA_1", "A", "2016-02-28", self.config)
        right, _, _, _ = self.panel.inputs("TX_1", "B", "2016-02-28", self.config)
        # Original units differ by a factor two but both normalize to the same pattern.
        np.testing.assert_allclose(left[:, 0], right[:, 0])
        np.testing.assert_array_equal(left[:, -1]+right[:, -1], np.ones(56))
        a = self.panel.labels("CA_1", "A", "2016-01-03", self.config, "2016-01-31")
        b = self.panel.labels("TX_1", "B", "2016-01-03", self.config, "2016-01-31")
        np.testing.assert_array_equal(2*a, b)

    def test_padding_observed_zero_and_missing_source_distinct(self):
        from demandos.neural.sequences import Panel
        origin = pd.Timestamp("2016-02-28")
        for n, zero in ((0, False), (3, False), (56, True)):
            chosen = self.daily[self.daily.date.between(origin-pd.Timedelta(days=n-1), origin)] if n else self.daily.iloc[:0]
            if zero: chosen = chosen.assign(units_sold=0)
            p = Panel(chosen, self.calendar, self.metadata)
            past, _, _, audit = p.inputs("CA_1", "A", origin, self.config)
            self.assertEqual(int(past[:, 1].sum()), n)
            self.assertTrue(np.isfinite(past).all()); self.assertGreaterEqual(audit["scale"], 1)
            self.assertEqual(audit["history_status"], "all_zero" if zero else "short_history" if n else "no_history")
            if zero: self.assertTrue((past[:, 1] == 1).all())
        p = Panel(self.daily[self.daily.date < origin], self.calendar, self.metadata)
        with self.assertRaisesRegex(ValueError, "Missing-source"):
            p.inputs("CA_1", "A", origin, self.config)
        supported = self.metadata.assign(first_observed_date=pd.Timestamp("2015-10-01"))
        p = Panel(self.daily[self.daily.date > pd.Timestamp("2016-01-04")], self.calendar, supported)
        with self.assertRaisesRegex(ValueError, "Missing-source"):
            p.inputs("CA_1", "A", origin, self.config)

    def test_train_only_normalizer_vocabulary_and_unknown(self):
        from demandos.neural.sequences import Panel, Preprocessor
        changed = self.daily.copy()
        changed.loc[changed.date > pd.Timestamp("2016-01-31"), "units_sold"] = 1000000
        extra = self.metadata.iloc[:1].assign(item_id="UNSEEN", cat_id="NEW", dept_id="NEW")
        meta = pd.concat([self.metadata, extra], ignore_index=True)
        panel = Panel(changed, self.calendar, meta)
        fresh = Preprocessor.fit(panel, self.rows, self.config, "2016-01-31")
        self.assertEqual(fresh.state(), self.prep.state())
        past, _, static, _ = panel.inputs("CA_1", "UNSEEN", "2016-02-28", self.config)
        values, ids = fresh.transform(past, static)
        self.assertEqual(ids[-1], 0); self.assertEqual(ids[2], 0)
        self.assertTrue(np.isfinite(values).all())

    def test_prices_target_columns_and_all_future_actuals_excluded(self):
        from demandos.neural.sequences import SequenceDataset, examples, Preprocessor
        from demandos.neural.pipeline import future_panels, assert_same_tensors
        origin = pd.Timestamp("2016-02-28")
        rows = examples(self.metadata, [origin], self.config)
        original = SequenceDataset(self.panel, rows, self.prep, self.config)
        for panel in future_panels(self.daily, self.calendar, self.metadata, origin):
            fresh = SequenceDataset(panel, rows, self.prep, self.config)
            assert_same_tensors(original, fresh)
            # Future outcomes cannot change training statistics or vocab either.
            self.assertEqual(self.prep.state(), Preprocessor.fit(panel, self.rows, self.config, "2016-01-31").state())
        self.assertTrue((original.tensors[0][:, :, 2:4].numpy() == 0).all())
        with self.assertRaises(ValueError):
            SequenceDataset(self.panel, rows.assign(actual=1), self.prep, self.config)

    def test_all_zero_scaler_and_source_immutability(self):
        from demandos.neural.sequences import Panel, Preprocessor, SequenceDataset
        before = self.daily.copy(deep=True)
        zero = Panel(self.daily.assign(units_sold=0), self.calendar, self.metadata)
        processor = Preprocessor.fit(zero, self.rows, self.config, "2016-01-31")
        data = SequenceDataset(zero, self.rows, processor, self.config, label_cutoff="2016-01-31")
        self.assertTrue(all(np.isfinite(t.numpy()).all() for t in data.tensors))
        pd.testing.assert_frame_equal(before, self.daily)

    def test_bounds_unique_keys_and_incomplete_calendar(self):
        from demandos.neural.config import NeuralConfig
        from demandos.neural.sequences import examples, SequenceDataset
        for options in ({"clip_norm": 0}, {"dropout": float("nan")}, {"epochs": 100}, {"threads": 10}, {"lookback": 55}):
            with self.assertRaises(ValueError): NeuralConfig(**options)
        with self.assertRaises(ValueError): examples(self.metadata, [pd.Timestamp("2016-01-03")]*2, self.config)
        with self.assertRaises(ValueError): examples(self.metadata, [pd.Timestamp("2016-01-03")], replace(self.config, max_windows=1))
        with self.assertRaises(ValueError): SequenceDataset(self.panel, pd.concat([self.rows]*2), self.prep, self.config)
        with self.assertRaises(ValueError): self.panel.inputs("CA_1", "A", "2016-05-31", self.config)

    def test_inference_cannot_use_later_fitted_state(self):
        from demandos.neural.sequences import SequenceDataset
        with self.assertRaisesRegex(ValueError, "fitting cutoff"):
            SequenceDataset(self.panel, self.rows, self.prep, self.config)

    def test_model_consumes_observed_history_mask(self):
        import torch
        from demandos.neural.models import RecurrentForecaster, seed_run
        from demandos.neural.sequences import SequenceDataset, examples
        seed_run(self.config)
        rows = examples(self.metadata, [pd.Timestamp("2016-02-28")], self.config)
        dataset = SequenceDataset(self.panel, rows, self.prep, self.config)
        for arch in ("lstm", "gru"):
            model = RecurrentForecaster(arch, self.prep, self.config).eval()
            past = dataset.tensors[0].clone().requires_grad_(True)
            model(past, dataset.tensors[1], dataset.tensors[2]).sum().backward()
            self.assertGreater(float(past.grad[:, :, 1].abs().sum()), 0.)


@unittest.skipUnless(AVAILABLE, "Install neural extra")
class NeuralModelTests(unittest.TestCase):
    def test_both_encoders_gradients_dropout_clipping_save_load_and_future_invariance(self):
        import torch
        from demandos.neural.models import train, predict, save, load, clip_gradients, RecurrentForecaster
        from demandos.neural.sequences import SequenceDataset, examples
        from demandos.neural.pipeline import future_panels
        daily, calendar, metadata, config, panel, rows, prep = fixture()
        fitting = SequenceDataset(panel, rows, prep, config, label_cutoff="2016-01-31")
        val_rows = examples(metadata, [pd.Timestamp("2016-02-28")], config)
        validation = SequenceDataset(panel, val_rows, prep, config, label_cutoff="2016-03-27")
        inference = SequenceDataset(panel, val_rows, prep, config)
        for arch in ("lstm", "gru"):
            with self.subTest(architecture=arch), TemporaryDirectory() as tmp:
                model, meta = train(arch, prep, fitting, config, cutoff="2016-01-31", validation=validation)
                self.assertEqual(model.encoder.dropout, 0.)
                self.assertEqual(model.encoder.num_layers, 1)
                self.assertEqual(model.head[2].p, config.dropout)
                self.assertEqual(meta["fit_target_max"], "2016-01-31")
                result = predict(model, inference, config)
                self.assertEqual(result.shape, (2, 28)); self.assertTrue(np.isfinite(result).all())
                for parameter in model.parameters(): parameter.grad = torch.full_like(parameter, 100.)
                norm = clip_gradients(model, config.clip_norm)
                self.assertGreater(norm, config.clip_norm)
                post = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters()))
                self.assertLessEqual(float(post), config.clip_norm+1e-5)
                save(model, prep, meta, Path(tmp))
                loaded, processor, _ = load(tmp)
                self.assertEqual(prep.state(), processor.state())
                np.testing.assert_allclose(result, predict(loaded, inference, config), atol=1e-6, rtol=1e-6)
                for candidate in future_panels(daily, calendar, metadata, pd.Timestamp("2016-02-28")):
                    changed = SequenceDataset(candidate, val_rows, processor, config)
                    np.testing.assert_allclose(result, predict(loaded, changed, config), atol=1e-6, rtol=1e-6)
                with self.assertRaises(ValueError): predict(model, validation, config)
                with self.assertRaises(ValueError): model(inference.tensors[0][:, :-1], *inference.tensors[1:3])
                model.train()
                with torch.no_grad():
                    self.assertFalse(torch.equal(model(*inference.tensors[:3]), model(*inference.tensors[:3])))

    def test_nonfinite_gradients_and_invalid_training_are_hard_errors(self):
        import torch
        from demandos.neural.models import train, RecurrentForecaster, clip_gradients
        from demandos.neural.sequences import SequenceDataset, examples
        _, _, _, config, panel, rows, prep = fixture()
        model = RecurrentForecaster("gru", prep, config)
        for parameter in model.parameters(): parameter.grad = torch.full_like(parameter, float("nan"))
        with self.assertRaises(RuntimeError): clip_gradients(model, 1.)
        inference = SequenceDataset(panel, examples(panel.metadata.reset_index(drop=True), [pd.Timestamp("2016-02-28")], config), prep, config)
        with self.assertRaises(ValueError): train("gru", prep, inference, config, cutoff="2016-01-31")

    def test_seeded_training_equivalence(self):
        from demandos.neural.models import train, predict
        from demandos.neural.sequences import SequenceDataset, examples
        _, _, _, config, panel, rows, prep = fixture()
        fitting = SequenceDataset(panel, rows, prep, config, label_cutoff="2016-01-31")
        inference = SequenceDataset(panel, examples(panel.metadata.reset_index(drop=True), [pd.Timestamp("2016-02-28")], config), prep, config)
        a, _ = train("gru", prep, fitting, config, cutoff="2016-01-31", epochs=1)
        b, _ = train("gru", prep, fitting, config, cutoff="2016-01-31", epochs=1)
        np.testing.assert_array_equal(predict(a, inference, config), predict(b, inference, config))

    def test_early_stopping_selects_best_earlier_epoch(self):
        from demandos.neural.models import train
        from demandos.neural.sequences import SequenceDataset, examples
        _, _, metadata, config, panel, rows, prep = fixture()
        fitting = SequenceDataset(panel, rows, prep, config, label_cutoff="2016-01-31")
        validation = SequenceDataset(panel, examples(metadata, [pd.Timestamp("2016-02-28")], config), prep, config, label_cutoff="2016-03-27")
        with patch("demandos.neural.models.loss_epoch", side_effect=[(1., 0.), (.2, 0.), (.9, 0.), (.3, 0.)]):
            _, meta = train("gru", prep, fitting, replace(config, epochs=3), cutoff="2016-01-31", validation=validation)
        self.assertEqual(meta["selected_epochs"], 1)
        self.assertEqual(meta["epochs_run"], 2)
        self.assertEqual(meta["best_validation_loss"], .2)

    def test_prediction_key_alignment_and_undefined_metrics_skip_coverage(self):
        from demandos.neural.pipeline import prediction_rows
        from demandos.backtesting.metrics import point_metrics, common_cases
        _, _, _, _, _, rows, _ = fixture()
        values = np.stack([np.arange(28), np.arange(28)+100])
        frame = prediction_rows(rows.iloc[::-1], values, "gru")
        self.assertEqual(frame.iloc[0].store_id, "TX_1")
        self.assertEqual(frame.iloc[28].prediction, 100)
        with self.assertRaises(ValueError): prediction_rows(rows, values[:, :-1], "gru")
        score = point_metrics([0, 0], [0, 0], None)
        self.assertIsNone(score["wape"]); self.assertIsNone(score["rmsse"])
        known = point_metrics([1, 3], [0, 5], 2.)
        self.assertEqual(known["mae"], 1.5); self.assertEqual(known["wape"], .75)
        self.assertAlmostEqual(known["rmsse"], np.sqrt(1.25))
        # Failed/skipped cases cannot enter a successful common-key ranking.
        source = frame.rename(columns={"origin_date": "origin"}).assign(window="expanding", series_id=frame.store_id+"/"+frame.item_id, actual=0, status="ok")
        skipped = source.assign(model="lstm", status="skipped", prediction=np.nan)
        self.assertTrue(common_cases(pd.concat([source, skipped]), ["gru", "lstm"]).empty)


if __name__ == "__main__":
    unittest.main()
