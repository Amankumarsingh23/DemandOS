"""Deterministic synthetic TFT mechanism/temporal contracts, not real M5 scores."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


AVAILABLE=importlib.util.find_spec("torch") is not None


def fixture():
    from test_neural import fixture as sequence_fixture
    from demandos.tft.config import TFTConfig
    daily,calendar,metadata,_,panel,rows,processor=sequence_fixture()
    config=replace(TFTConfig(),hidden_size=8,heads=2,epochs=2,threads=1,batch_size=2)
    return daily,calendar,metadata,config,panel,rows,processor


@unittest.skipUnless(AVAILABLE,"Install neural extra")
class TFTTests(unittest.TestCase):
    def test_glu_gate_suppression_and_pass_through(self):
        import torch
        from demandos.tft.model import GLU
        layer=GLU(3,3)
        with torch.no_grad():
            layer.value.weight.copy_(torch.eye(3)); layer.value.bias.zero_()
            layer.gate.weight.zero_(); layer.gate.bias.fill_(-100)
        x=torch.ones(2,3)
        torch.testing.assert_close(layer(x),torch.zeros_like(x),atol=1e-6,rtol=0)
        with torch.no_grad(): layer.gate.bias.fill_(100)
        torch.testing.assert_close(layer(x),x)

    def test_grn_projection_context_residual_normalization_and_gradients(self):
        import torch
        from demandos.tft.model import GRN
        layer=GRN(3,8,5,context_size=4)
        x,context=torch.randn(2,7,3,requires_grad=True),torch.randn(2,4,requires_grad=True)
        y=layer(x,context)
        self.assertEqual(tuple(y.shape),(2,7,5)); self.assertTrue(torch.isfinite(y).all())
        y.square().sum().backward()
        self.assertGreater(float(context.grad.abs().sum()),0.)
        self.assertGreater(float(x.grad.abs().sum()),0.)
        self.assertFalse(torch.equal(y,layer(x,context+torch.randn_like(context))))
        with torch.no_grad():
            layer.glu.gate.weight.zero_(); layer.glu.gate.bias.fill_(-100)
        torch.testing.assert_close(layer(x,context),layer.norm(layer.skip(x)),atol=1e-6,rtol=1e-6)
        with self.assertRaises(ValueError): layer(x)

    def test_vsn_variable_axis_shapes_and_separate_parameters(self):
        import torch
        from demandos.tft.model import VariableSelection
        static=VariableSelection(5,8)
        past=VariableSelection(10,8,context=True)
        future=VariableSelection(7,8,context=True)
        for layer,shape,context in [(static,(2,5,8),None),(past,(2,56,10,8),torch.randn(2,8)),(future,(2,28,7,8),torch.randn(2,8))]:
            x=torch.randn(*shape)
            selected,weights=layer(x,context)
            self.assertEqual(tuple(weights.shape),shape[:-1])
            self.assertEqual(tuple(selected.shape),shape[:-2]+(8,))
            torch.testing.assert_close(weights.sum(-1),torch.ones_like(weights.sum(-1)))
            self.assertTrue((weights>=0).all())
        self.assertIsNot(past.weights.input.weight,future.weights.input.weight)

    def test_vsn_static_context_changes_selection(self):
        import torch
        from demandos.tft.model import VariableSelection
        layer=VariableSelection(3,8,context=True).eval()
        x=torch.randn(2,4,3,8)
        self.assertFalse(torch.equal(layer(x,torch.zeros(2,8))[1],layer(x,torch.randn(2,8))[1]))

    def test_attention_exact_dot_product_softmax_and_shared_values(self):
        import torch
        from demandos.tft.model import InterpretableAttention
        layer=InterpretableAttention(4,2,0.)
        with torch.no_grad():
            layer.query.weight.copy_(torch.eye(4)); layer.key.weight.copy_(torch.eye(4))
            layer.value.weight.copy_(torch.eye(4)[:2]); layer.output.weight.copy_(torch.eye(4)[:,:2])
        q=torch.tensor([[[1.,0.,0.,1.]]]); k=torch.tensor([[[1.,0.,1.,0.],[0.,1.,0.,1.],[1.,1.,1.,1.]]])
        allowed=torch.tensor([[[True,False,True]]])
        output,weights=layer(q,k,allowed)
        query=q.view(1,1,2,2).transpose(1,2); keys=k.view(1,3,2,2).transpose(1,2)
        expected=torch.softmax(((query@keys.transpose(-2,-1))/np.sqrt(2)).masked_fill(~allowed[:,None],float("-inf")),dim=-1)
        torch.testing.assert_close(weights,expected)
        combined=(expected@k[:,:,:2][:,None]).mean(1)
        torch.testing.assert_close(output,combined@layer.output.weight.T)
        self.assertEqual(tuple(layer.value.weight.shape),(2,4))
        self.assertTrue((weights[:,:,:,1]==0).all())
        with self.assertRaises(ValueError): layer(q,k,torch.zeros_like(allowed))

    def test_attention_padding_and_causal_horizon_masks(self):
        import torch
        from demandos.tft.model import attention_mask
        valid=torch.ones(2,56,dtype=torch.bool); valid[0,:53]=False; valid[1]=False
        allowed=attention_mask(valid,28)
        self.assertEqual(tuple(allowed.shape),(2,28,84))
        self.assertFalse(allowed[0,:,:53].any()); self.assertFalse(allowed[1,:,:56].any())
        for h in range(28):
            self.assertTrue(allowed[:,h,56+h].all()); self.assertFalse(allowed[:,h,57+h:].any())

    def test_full_model_shapes_states_probabilities_and_finite_gradients(self):
        import torch
        from demandos.neural.sequences import SequenceDataset
        from demandos.tft.model import TemporalFusionTransformer
        from demandos.tft.training import validate_diagnostics
        *_,config,panel,rows,processor=fixture()
        data=SequenceDataset(panel,rows,processor,config,label_cutoff="2016-01-31")
        model=TemporalFusionTransformer(processor,config).eval()
        result=model(*data.tensors[:3],diagnostics=True)
        self.assertEqual(tuple(result["prediction"].shape),(2,28))
        self.assertEqual(tuple(result["encoder_state_h"].shape),(1,2,8))
        self.assertEqual(tuple(result["encoder_state_c"].shape),(1,2,8))
        self.assertEqual(tuple(result["attention"].shape),(2,2,28,84))
        validate_diagnostics(result)
        result["prediction"].square().mean().backward()
        for name,p in model.named_parameters():
            self.assertIsNotNone(p.grad,name); self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertEqual(model.encoder.dropout,0.); self.assertEqual(model.decoder.dropout,0.)

    def test_future_known_changes_cannot_affect_earlier_horizons(self):
        import torch
        from demandos.neural.sequences import SequenceDataset
        from demandos.tft.model import TemporalFusionTransformer
        *_,config,panel,rows,processor=fixture()
        data=SequenceDataset(panel,rows,processor,config,label_cutoff="2016-01-31")
        model=TemporalFusionTransformer(processor,config).eval()
        past,future,ids=data.tensors[:3]
        changed=future.clone(); changed[:,7:]+=1000
        with torch.no_grad():
            first=model(past,future,ids); second=model(past,changed,ids)
        torch.testing.assert_close(first[:,:7],second[:,:7],atol=1e-6,rtol=1e-6)

    def test_short_no_history_masks_skip_padding_in_recurrence_and_attention(self):
        import torch
        from demandos.neural.sequences import Panel,SequenceDataset,examples
        from demandos.tft.model import TemporalFusionTransformer
        from demandos.tft.training import validate_diagnostics
        daily,calendar,metadata,config,_,_,processor=fixture()
        origin=pd.Timestamp("2016-02-28"); rows=examples(metadata,[origin],config)
        model=TemporalFusionTransformer(processor,config).eval()
        for n in (0,3,56):
            restricted=daily[daily.date.between(origin-pd.Timedelta(days=n-1),origin)].assign(units_sold=0) if n else daily.iloc[:0]
            dataset=SequenceDataset(Panel(restricted,calendar,metadata),rows,processor,config)
            past,future,ids=dataset.tensors[:3]
            first=model(past,future,ids,diagnostics=True); validate_diagnostics(first)
            self.assertTrue(torch.isfinite(first["prediction"]).all())
            self.assertEqual(float(past[0,:,1].sum()),n)
            if n<56:
                changed=past.clone(); changed[:,:56-n,0]=100000
                torch.testing.assert_close(first["prediction"],model(changed,future,ids),atol=1e-6,rtol=1e-6)
                self.assertTrue((first["attention"][:,:,:,:56-n]==0).all())

    def test_shared_sequence_dates_leap_cutoff_features_and_unknown_vocab(self):
        from demandos.neural.sequences import Panel,Preprocessor,SequenceDataset,examples
        daily,calendar,metadata,config,panel,rows,processor=fixture()
        inference_rows=examples(metadata,[pd.Timestamp("2016-02-28")],config)
        dataset=SequenceDataset(panel,inference_rows,processor,config)
        self.assertEqual(dataset.audits[0]["history_start"],pd.Timestamp("2016-01-04"))
        self.assertEqual(dataset.audits[0]["target_start"],pd.Timestamp("2016-02-29"))
        self.assertEqual(dataset.audits[0]["target_end"],pd.Timestamp("2016-03-27"))
        with self.assertRaises(ValueError): SequenceDataset(panel,rows,processor,config,label_cutoff="2016-01-30")
        changed=daily.copy(); changed.loc[changed.date>pd.Timestamp("2016-01-31"),"units_sold"]=1000000
        extra=metadata.iloc[:1].assign(item_id="UNSEEN")
        combined=pd.concat([metadata,extra],ignore_index=True)
        fresh=Panel(changed,calendar,combined)
        self.assertEqual(processor.state(),Preprocessor.fit(fresh,rows,config,"2016-01-31").state())
        unknown=SequenceDataset(fresh,examples(extra,[pd.Timestamp("2016-02-28")],config),processor,config)
        self.assertEqual(int(unknown.tensors[2][0,-1]),0)
        self.assertTrue(np.isfinite(unknown.tensors[0].numpy()).all())

    def test_future_sales_prices_labels_and_inputs_are_independent(self):
        import torch
        from demandos.neural.sequences import SequenceDataset,examples
        from demandos.neural.pipeline import future_panels,assert_same_tensors
        from demandos.tft.model import TemporalFusionTransformer
        daily,calendar,metadata,config,panel,_,processor=fixture()
        origin=pd.Timestamp("2016-02-28"); rows=examples(metadata,[origin],config)
        original=SequenceDataset(panel,rows,processor,config)
        model=TemporalFusionTransformer(processor,config).eval()
        for candidate in future_panels(daily,calendar,metadata,origin):
            changed=SequenceDataset(candidate,rows,processor,config); assert_same_tensors(original,changed)
            torch.testing.assert_close(model(*original.tensors[:3]),model(*changed.tensors[:3]),atol=1e-6,rtol=1e-6)
        with self.assertRaises(ValueError): SequenceDataset(panel,rows.assign(actual=1),processor,config)

    def test_training_checkpoint_determinism_and_temporal_guards(self):
        from demandos.neural.sequences import SequenceDataset,examples
        from demandos.tft.training import train,predict,save,load
        *_,config,panel,rows,processor=fixture()
        fitting=SequenceDataset(panel,rows,processor,config,label_cutoff="2016-01-31")
        inference=SequenceDataset(panel,examples(panel.metadata.reset_index(drop=True),[pd.Timestamp("2016-02-28")],config),processor,config)
        model,meta=train(fitting,processor,config,cutoff="2016-01-31",epochs=1)
        raw,scored,diagnostics=predict(model,inference,config,diagnostics=True)
        np.testing.assert_array_equal(scored,np.maximum(raw,0))
        with TemporaryDirectory() as temporary:
            save(model,processor,meta,temporary); loaded,saved,metadata=load(temporary)
            self.assertEqual(saved.state(),processor.state()); self.assertEqual(metadata["config"]["seed"],42)
            again=predict(loaded,inference,config,diagnostics=True)
            np.testing.assert_allclose(raw,again[0],atol=1e-6,rtol=1e-6)
            for name in diagnostics: np.testing.assert_allclose(diagnostics[name],again[2][name],atol=1e-6,rtol=1e-6)
            broken={**meta,"fit_target_max":"2020-01-01"}
            (Path(temporary)/"metadata.json").write_text(json.dumps(broken),encoding="utf-8")
            with self.assertRaises(ValueError): load(temporary)
        second,_=train(fitting,processor,config,cutoff="2016-01-31",epochs=1)
        np.testing.assert_array_equal(raw,predict(second,inference,config)[0])
        with self.assertRaises(ValueError): predict(model,fitting,config)
        with self.assertRaises(ValueError): train(fitting,processor,config,cutoff="2016-01-30")

    def test_chronological_early_stopping_restores_best_epoch(self):
        from demandos.neural.sequences import SequenceDataset,examples
        from demandos.tft.training import train
        *_,config,panel,rows,processor=fixture()
        fitting=SequenceDataset(panel,rows,processor,config,label_cutoff="2016-01-31")
        validation=SequenceDataset(panel,examples(panel.metadata.reset_index(drop=True),[pd.Timestamp("2016-02-28")],config),processor,config,label_cutoff="2016-03-27")
        with self.assertRaises(ValueError): train(fitting,processor,config,cutoff="2016-01-31",validation=validation,validation_cutoff="2016-03-01")
        with patch("demandos.tft.training.epoch_loss",side_effect=[(1.,0.),(.2,0.),(.9,0.),(.3,0.)]):
            _,meta=train(fitting,processor,replace(config,patience=1),cutoff="2016-01-31",validation=validation,validation_cutoff="2016-03-27")
        self.assertEqual(meta["selected_epochs"],1); self.assertEqual(meta["epochs_run"],2)

    def test_bad_shapes_mask_numeric_bounds_and_diagnostics_fail(self):
        import torch
        from demandos.tft.model import TemporalFusionTransformer
        from demandos.tft.config import TFTConfig
        from demandos.tft.training import validate_diagnostics
        from demandos.neural.sequences import SequenceDataset
        *_,config,panel,rows,processor=fixture()
        data=SequenceDataset(panel,rows,processor,config,label_cutoff="2016-01-31")
        past,future,ids=data.tensors[:3]; model=TemporalFusionTransformer(processor,config).eval()
        with self.assertRaises(ValueError): model(past[:,:55],future,ids)
        bad=past.clone(); bad[:,5,1]=0
        with self.assertRaises(ValueError): model(bad,future,ids)
        with self.assertRaises(ValueError): model(past,future*float("nan"),ids)
        for options in ({"heads":3},{"heads":0},{"hidden_size":128},{"clip_norm":0}):
            with self.assertRaises(ValueError): TFTConfig(**options)
        result=model(past,future,ids,diagnostics=True)
        result["attention"]=torch.ones_like(result["attention"])
        with self.assertRaises(AssertionError): validate_diagnostics(result)

    def test_forecast_key_grain_metrics_and_undefined_cases(self):
        from demandos.neural.pipeline import prediction_rows
        from demandos.backtesting.metrics import point_metrics,common_cases
        *_,rows,_=fixture()
        frame=prediction_rows(rows,np.ones((2,28)),"tft")
        self.assertEqual(len(frame),56); self.assertFalse(frame.duplicated(["store_id","item_id","origin_date","horizon"]).any())
        self.assertTrue((pd.to_datetime(frame.target_date)==pd.to_datetime(frame.origin_date)+pd.to_timedelta(frame.horizon,unit="D")).all())
        known=point_metrics([1,3],[0,5],2.)
        self.assertEqual(known["mae"],1.5); self.assertEqual(known["wape"],.75)
        zero=point_metrics([0,0],[0,0],None); self.assertIsNone(zero["wape"]); self.assertIsNone(zero["rmsse"])
        source=frame.rename(columns={"origin_date":"origin"}).assign(window="expanding",series_id=frame.store_id+"/"+frame.item_id,actual=0,status="ok")
        skipped=source.assign(model="other",status="skipped",prediction=np.nan)
        self.assertTrue(common_cases(pd.concat([source,skipped]),["tft","other"]).empty)

    def test_interpretation_axis_dates_and_normalized_summaries(self):
        from demandos.neural.sequences import SequenceDataset,examples
        from demandos.tft.model import TemporalFusionTransformer
        from demandos.tft.interpretation import tables
        *_,config,panel,_,processor=fixture()
        rows=examples(panel.metadata.reset_index(drop=True),[pd.Timestamp("2016-02-28")],config)
        data=SequenceDataset(panel,rows,processor,config)
        result=TemporalFusionTransformer(processor,config).eval()(*data.tensors[:3],diagnostics=True)
        diagnostics={name:result[name].detach().numpy() for name in ("static_weights","past_weights","future_weights","attention","allowed")}
        output=tables(rows,diagnostics,data.tensors[0][:,:,1].numpy())
        self.assertEqual(len(output["attention_cases"]),2*28*84)
        points=output["attention_cases"]
        self.assertTrue(points.loc[points.allowed,"key_date"].le(points.loc[points.allowed,"target_date"]).all())
        self.assertTrue((points.loc[~points.allowed,"weight"]==0).all())
        np.testing.assert_allclose(output["vsn_summary"].groupby("family").mean_weight.sum(),1,atol=1e-6)

    def test_shared_daily_schema_no_series_mixing_and_own_origin_scale(self):
        from demandos.neural.sequences import Panel,SequenceDataset,examples
        daily,calendar,metadata,config,panel,rows,processor=fixture()
        for broken in (daily.iloc[::-1],daily.drop(index=80),pd.concat([daily,daily.iloc[:1]])):
            with self.assertRaises(ValueError): Panel(broken,calendar,metadata)
        data=SequenceDataset(panel,rows,processor,config,label_cutoff="2016-01-31")
        np.testing.assert_allclose(data.tensors[0][0,:,0],data.tensors[0][1,:,0])
        self.assertEqual(float(data.tensors[3][1]),2*float(data.tensors[3][0]))
        with self.assertRaises(ValueError): examples(metadata,[pd.Timestamp("2016-01-03")]*2,config)


if __name__=="__main__": unittest.main()
