"""Small deterministic synthetic N-BEATS tests; no invented M5 performance."""

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
    from test_neural import fixture as neural_fixture
    from demandos.nbeats.config import NBeatsConfig
    daily,calendar,metadata,_,panel,rows,_ = neural_fixture()
    config = replace(NBeatsConfig(),hidden_size=16,layers=2,batch_size=2,epochs=2,threads=1)
    return daily,calendar,metadata,config,panel,rows


@unittest.skipUnless(AVAILABLE,"Install neural extra")
class NBeatsTests(unittest.TestCase):
    def test_calendar_lookback_leap_labels_and_no_cross_series(self):
        from demandos.neural.sequences import examples
        from demandos.nbeats.dataset import TargetDataset
        _,_,metadata,config,panel,_ = fixture()
        rows = examples(metadata,[pd.Timestamp("2016-02-28")],config)
        dataset = TargetDataset(panel,rows,config,label_cutoff="2016-03-27")
        self.assertEqual(tuple(dataset.tensors[0].shape),(2,56))
        self.assertEqual(tuple(dataset.tensors[3].shape),(2,28))
        self.assertEqual(dataset.audits[0]["history_start"],pd.Timestamp("2016-01-04"))
        self.assertEqual(dataset.audits[0]["target_end"],pd.Timestamp("2016-03-27"))
        y = dataset.tensors[3].numpy()*dataset.tensors[2].numpy()[:,None]
        np.testing.assert_allclose(2*y[0],y[1],atol=1e-6)
        self.assertEqual(rows.origin_date.iloc[0]+pd.Timedelta(days=1),pd.Timestamp("2016-02-29"))

    def test_strict_daily_grid_and_invalid_sales(self):
        from demandos.neural.sequences import Panel
        daily,calendar,metadata,_,_,_ = fixture()
        for bad in (daily.iloc[::-1],daily.drop(index=80),pd.concat([daily,daily.iloc[:1]]),daily.assign(units_sold=-1)):
            with self.assertRaises(ValueError): Panel(bad,calendar,metadata)

    def test_mask_padding_observed_zero_and_finite_origin_scale(self):
        from demandos.neural.sequences import Panel,examples
        from demandos.nbeats.dataset import TargetDataset
        daily,calendar,metadata,config,_,_ = fixture()
        origin = pd.Timestamp("2016-02-28")
        rows = examples(metadata,[origin],config)
        for n in (0,3,56):
            restricted = daily[daily.date.between(origin-pd.Timedelta(days=n-1),origin)].assign(units_sold=0) if n else daily.iloc[:0]
            dataset = TargetDataset(Panel(restricted,calendar,metadata),rows,config)
            self.assertEqual(float(dataset.tensors[1][0].sum()),n)
            self.assertEqual(float(dataset.tensors[2][0]),1.)
            self.assertTrue(np.isfinite(dataset.tensors[0].numpy()).all())
            self.assertEqual(dataset.audits[0]["history_status"],"all_zero" if n else "no_history")

    def test_full_vector_eligibility_and_label_free_api(self):
        from demandos.nbeats.dataset import TargetDataset
        from demandos.neural.sequences import examples
        _,_,metadata,config,panel,rows = fixture()
        with self.assertRaises(ValueError): TargetDataset(panel,rows,config,label_cutoff="2016-01-30")
        eligible = examples(metadata,pd.to_datetime(["2016-01-03","2016-01-17"]),config,"2016-01-31")
        pd.testing.assert_frame_equal(eligible,rows)
        inference = TargetDataset(panel,rows,config)
        self.assertEqual(len(inference.tensors),3)
        with self.assertRaises(ValueError): TargetDataset(panel,rows.assign(actual=3),config)

    def test_polynomial_values_shapes_and_origin_grid(self):
        import torch
        from demandos.nbeats.models import trend_basis
        past,future = trend_basis(56,28,2)
        self.assertEqual(tuple(past.shape),(3,56)); self.assertEqual(tuple(future.shape),(3,28))
        torch.testing.assert_close(past[:, -1],torch.tensor([1.,0.,0.]))
        torch.testing.assert_close(past[:,0],torch.tensor([1.,-55/56,(55/56)**2]))
        torch.testing.assert_close(future[:,-1],torch.tensor([1.,.5,.25]))

    def test_weekly_fourier_values_harmonics_and_periodicity(self):
        import torch
        from demandos.nbeats.models import fourier_basis
        past,future = fourier_basis(56,28,3,7)
        self.assertEqual(tuple(past.shape),(7,56)); self.assertEqual(tuple(future.shape),(7,28))
        torch.testing.assert_close(past[:,-1],torch.tensor([1.,1.,0.,1.,0.,1.,0.]),atol=1e-6,rtol=0)
        torch.testing.assert_close(future[:,0],past[:,-7],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(future[:,:7],future[:,7:14],atol=4e-6,rtol=1e-6)

    def test_blocks_shapes_sharing_and_fixed_basis_buffers(self):
        import torch
        from demandos.nbeats.models import NBeats,Block
        *_,config,_,_ = fixture()
        for kind in ("generic","trend","seasonality"):
            b,f = Block(kind,config)(torch.ones(2,56))
            self.assertEqual(tuple(b.shape),(2,56)); self.assertEqual(tuple(f.shape),(2,28))
        model = NBeats("interpretable",config)
        self.assertIs(model.blocks[0],model.blocks[1]); self.assertIs(model.blocks[2],model.blocks[3])
        self.assertIsNot(model.blocks[0],model.blocks[2])
        self.assertTrue(dict(model.named_buffers())); self.assertFalse(any("basis" in name for name,_ in model.named_parameters()))
        generic = NBeats("generic",config)
        self.assertIsNot(generic.blocks[0],generic.blocks[1])
        shared = NBeats("generic",replace(config,generic_share=True))
        self.assertIs(shared.blocks[0],shared.blocks[2])

    def test_zero_initialization_residual_subtraction_accumulation_and_mask(self):
        import torch
        from torch import nn
        from demandos.nbeats.models import NBeats
        *_,config,_,_ = fixture()
        class Constant(nn.Module):
            def forward(self,x): return torch.full_like(x,2.),x.new_full((len(x),28),3.)
        model = NBeats("generic",config)
        model.blocks = nn.ModuleList([Constant(),Constant(),Constant()])
        x,mask = torch.ones(2,56),torch.ones(2,56)
        mask[:, :4] = 0
        result = model(x,mask,diagnostics=True)
        torch.testing.assert_close(result["total"],torch.full((2,28),9.))
        torch.testing.assert_close(result["residuals"][:,0],x*mask)
        for i in range(3):
            torch.testing.assert_close(result["residuals"][:,i+1],(result["residuals"][:,i]-2)*mask)
        self.assertTrue((result["residuals"][:,:,:4]==0).all())
        # Empty-stack diagnostic isolates the explicitly zero initial forecast.
        model.blocks = nn.ModuleList([])
        torch.testing.assert_close(model(x,mask),torch.zeros(2,28))

    def test_multiblock_gradients_and_component_reconciliation(self):
        import torch
        from demandos.nbeats.models import NBeats
        from demandos.neural.models import seed_run
        *_,config,_,_ = fixture()
        seed_run(config)
        for variant in ("generic","interpretable"):
            model = NBeats(variant,config)
            result = model(torch.ones(2,56),torch.ones(2,56),diagnostics=True)
            torch.testing.assert_close(result["total"],result["contributions"].sum(dim=1))
            if variant=="interpretable":
                torch.testing.assert_close(result["total"],result["contributions"][:,:2].sum(1)+result["contributions"][:,2:].sum(1))
            result["total"].square().mean().backward()
            self.assertGreater(float(model.blocks[0].theta_backcast.weight.grad.abs().sum()),0.)
            for name,p in model.named_parameters():
                if name in model.expected_unused_parameters(): self.assertIsNone(p.grad)
                else: self.assertIsNotNone(p.grad); self.assertTrue(torch.isfinite(p.grad).all())

    def test_future_actual_price_and_calendar_columns_cannot_change_inputs(self):
        from demandos.neural.pipeline import future_panels,assert_same_tensors
        from demandos.neural.sequences import examples,Panel
        from demandos.nbeats.dataset import TargetDataset
        daily,calendar,metadata,config,panel,_ = fixture()
        origin = pd.Timestamp("2016-02-28")
        rows = examples(metadata,[origin],config)
        before = TargetDataset(panel,rows,config)
        for candidate in future_panels(daily,calendar,metadata,origin):
            assert_same_tensors(before,TargetDataset(candidate,rows,config))
        changed = calendar.assign(snap_CA=1-calendar.snap_CA,event_name_1="changed",event_name_2="changed")
        assert_same_tensors(before,TargetDataset(Panel(daily,changed,metadata),rows,config))

    def test_seeded_training_reload_cutoffs_finite_and_raw_clipping(self):
        from demandos.neural.sequences import examples
        from demandos.nbeats.dataset import TargetDataset
        from demandos.nbeats.training import train,predict,save,load
        _,_,metadata,config,panel,rows = fixture()
        fitting = TargetDataset(panel,rows,config,label_cutoff="2016-01-31")
        infer_rows = examples(metadata,[pd.Timestamp("2016-02-28")],config)
        inference = TargetDataset(panel,infer_rows,config)
        for variant in ("generic","interpretable"):
            with self.subTest(variant=variant),TemporaryDirectory() as tmp:
                model,meta = train(variant,fitting,config,cutoff="2016-01-31",epochs=1)
                raw,scored,parts = predict(model,inference,config)
                self.assertEqual(raw.shape,(2,28)); self.assertTrue(np.isfinite(raw).all())
                np.testing.assert_array_equal(scored,np.maximum(raw,0))
                np.testing.assert_allclose(raw,parts.sum(1),atol=1e-4,rtol=1e-6)
                save(model,meta,tmp); restored,metadata_saved = load(tmp)
                self.assertEqual(metadata_saved["config"]["seed"],42)
                for a,b in zip((raw,scored,parts),predict(restored,inference,config)):
                    np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-6)
                second,_ = train(variant,fitting,config,cutoff="2016-01-31",epochs=1)
                np.testing.assert_array_equal(raw,predict(second,inference,config)[0])
                with self.assertRaises(ValueError): predict(model,fitting,config)
                with self.assertRaises(ValueError): predict(model,TargetDataset(panel,rows,config),config)
                with self.assertRaises(ValueError): train(variant,fitting,config,cutoff="2016-01-30")

    def test_chronological_validation_and_best_checkpoint_selection(self):
        from demandos.nbeats.dataset import TargetDataset
        from demandos.nbeats.training import train
        from demandos.neural.sequences import examples
        _,_,metadata,config,panel,rows = fixture()
        fitting = TargetDataset(panel,rows,config,label_cutoff="2016-01-31")
        validation = TargetDataset(panel,examples(metadata,[pd.Timestamp("2016-02-28")],config),config,label_cutoff="2016-03-27")
        with self.assertRaises(ValueError): train("generic",fitting,config,cutoff="2016-01-31",validation=validation,validation_cutoff="2016-03-01")
        with patch("demandos.nbeats.training.epoch_loss",side_effect=[(1.,0.),(.2,0.),(.9,0.),(.3,0.)]):
            _,meta = train("generic",fitting,replace(config,patience=1),cutoff="2016-01-31",validation=validation,validation_cutoff="2016-03-27")
        self.assertEqual(meta["selected_epochs"],1); self.assertEqual(meta["epochs_run"],2)

    def test_invalid_inputs_masks_bounds_and_nonfinite_gradients_fail(self):
        import torch
        from demandos.nbeats.config import NBeatsConfig
        from demandos.nbeats.models import NBeats
        from demandos.neural.models import clip_gradients
        *_,config,_,_ = fixture()
        model = NBeats("generic",config)
        for x,mask in [(torch.ones(2,55),torch.ones(2,55)),(torch.ones(2,56),torch.full((2,56),.5)),(torch.full((2,56),float("nan")),torch.ones(2,56))]:
            with self.assertRaises(ValueError): model(x,mask)
        for bad in ({"period":0},{"harmonics":4},{"generic_blocks":100},{"clip_norm":0},{"generic_share":1}):
            with self.assertRaises(ValueError): NBeatsConfig(**bad)
        for p in model.parameters(): p.grad=torch.full_like(p,float("nan"))
        with self.assertRaises(RuntimeError): clip_gradients(model,1.)

    def test_forecast_keys_components_metrics_and_undefined_coverage(self):
        from demandos.nbeats.pipeline import forecast_tables
        from demandos.backtesting.metrics import point_metrics,common_cases
        *_,config,_,rows = fixture()
        raw = np.stack([np.arange(28)-5,np.arange(28)+5]).astype(float)
        scored = np.maximum(raw,0)
        parts = np.stack([raw/2,raw/2],axis=1)
        values,blocks = forecast_tables(rows,raw,scored,parts,"generic")
        self.assertFalse(values.duplicated(["store_id","item_id","origin_date","horizon"]).any())
        np.testing.assert_array_equal(values.raw_prediction,raw.ravel())
        self.assertEqual(len(blocks),112)
        self.assertIsNone(point_metrics([0,0],[0,0],None)["rmsse"])
        self.assertIsNone(point_metrics([0,0],[0,0],None)["wape"])
        score=point_metrics([1,3],[0,5],2)
        self.assertEqual(score["mae"],1.5); self.assertEqual(score["wape"],.75)
        frame=values.rename(columns={"origin_date":"origin"}).assign(series_id=values.store_id+"/"+values.item_id,window="expanding",actual=0,status="ok")
        skipped=frame.assign(model="other",status="skipped",prediction=np.nan)
        self.assertTrue(common_cases(pd.concat([frame,skipped]),["nbeats_generic","other"]).empty)

    def test_reference_flat_and_nested_cutoffs_must_be_explicit(self):
        from demandos.nbeats.pipeline import validate_reference_cutoff
        origin=pd.Timestamp("2016-01-31")
        validate_reference_cutoff({"fit_cutoff":"2016-01-31","fit_target_max":"2016-01-31"},origin)
        validate_reference_cutoff({"metadata":{"allowed_cutoff":"2016-01-31","training_target_max":"2016-01-31"}},origin)
        for payload in ({}, {"metadata":{}}, {"fit_cutoff":"2016-01-31","fit_target_max":"2016-02-01"}):
            with self.assertRaises(ValueError): validate_reference_cutoff(payload,origin)

    def test_supported_source_gaps_and_invalid_checkpoint_metadata_fail(self):
        import json
        from demandos.neural.sequences import Panel,examples
        from demandos.nbeats.dataset import TargetDataset
        from demandos.nbeats.training import train,save,load
        daily,calendar,metadata,config,panel,rows=fixture()
        origin=pd.Timestamp("2016-02-28")
        requested=examples(metadata,[origin],config)
        supported=metadata.assign(first_observed_date=pd.Timestamp("2015-10-01"))
        for daily_bad in (daily.iloc[:0],daily[daily.date<origin],daily[daily.date>pd.Timestamp("2016-01-04")]):
            with self.assertRaises(ValueError): TargetDataset(Panel(daily_bad,calendar,supported),requested,config)
        model,meta=train("generic",TargetDataset(panel,rows,config,label_cutoff="2016-01-31"),config,cutoff="2016-01-31",epochs=1)
        with TemporaryDirectory() as tmp:
            save(model,meta,tmp)
            for changes in ({"fit_cutoff":None},{"fit_target_max":"2016-02-01"}):
                (Path(tmp)/"metadata.json").write_text(json.dumps({**meta,**changes}),encoding="utf-8")
                with self.assertRaises(ValueError): load(tmp)


if __name__=="__main__": unittest.main()
