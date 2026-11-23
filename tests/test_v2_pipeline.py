import random
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from dataloader.custom_dataset import augment_pair
from dataloader.dataloader import load_manifest, make_loader
from dataloader.preprocessing import prepare_rgb, prepare_depth, restore_prediction, spatial_geometry
from model.canonical_evaluation import evaluate_canonical, PROTOCOL
from model.evaluation import DepthMetrics, evaluate
from model.model import Model, transfer_checkpoint
from model.staged_training import train_staged
from script.train_v2 import promote_if_better


class GeometryTests(unittest.TestCase):
    def test_portrait_letterbox_and_padding_mask(self):
        image=Image.new('RGB',(40,80),(255,0,0))
        tensor,geo=prepare_rgb(image,(64,96),'letterbox')
        self.assertEqual(geo['resized_hw'],[64,32])
        self.assertEqual(geo['left'],32)
        self.assertTrue(torch.equal(tensor[:,:,:32],torch.zeros(3,64,32)))
        depth,mask=prepare_depth(np.full((80,40),2.,dtype=np.float32),geo,.001,8)
        self.assertEqual(mask.sum().item(),64*32)
        self.assertTrue((depth[mask]==2).all())
        self.assertFalse(mask[:,:,:32].any())
        metrics=DepthMetrics()
        metrics.update(torch.full_like(depth,2.),depth,mask)
        self.assertEqual(metrics.compute()['gradient_loss'],0)

    def test_padding_is_removed_before_restoring(self):
        geo=spatial_geometry((80,40),(64,96),'letterbox')
        prediction=torch.full((1,1,64,96),100.)
        prediction[:,:,:,32:64]=2.
        restored=restore_prediction(prediction,geo)
        self.assertEqual(tuple(restored.shape),(1,1,80,40))
        torch.testing.assert_close(restored,torch.full_like(restored,2.),rtol=0,atol=1e-6)

    def test_paired_flip_and_rgb_only_color(self):
        values=np.arange(12,dtype=np.uint8).reshape(3,4)*10
        image=Image.fromarray(np.stack([values,values,values],-1))
        depth=values.astype(np.float32)
        rgb,flipped=augment_pair(image,depth,{'horizontal_flip':1},random.Random(1))
        np.testing.assert_array_equal(np.array(rgb)[...,0],np.fliplr(values))
        np.testing.assert_array_equal(flipped,np.fliplr(depth))
        colored,unchanged=augment_pair(image,depth,{'brightness':.2},random.Random(2))
        np.testing.assert_array_equal(unchanged,depth)
        self.assertFalse(np.array_equal(np.array(colored),np.array(image)))

    def test_augmentation_is_per_record_epoch_and_training_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            Image.new('RGB',(20,40),(140,100,70)).save(root/'x.jpg')
            Image.fromarray(np.tile(np.arange(1,21,dtype=np.uint16)*100,(40,1))).save(root/'x.png')
            record={'id':'x','rgb':str(root/'x.jpg'),'depth':str(root/'x.png')}
            config={'size':[64,96],'batch_size':1,'seed':174,'depth_encoding':'millimeters',
                    'min_depth':.001,'max_depth':8,'resize_mode':'letterbox',
                    'augmentation':{'horizontal_flip':1,'brightness':.2}}
            a=next(iter(make_loader([record],config,augment=True,epoch=2)))
            torch.randn(100)
            b=next(iter(make_loader([record],config,augment=True,epoch=2)))
            for x,y in zip(a,b): self.assertTrue(torch.equal(x,y))
            validation=next(iter(make_loader([record],config)))
            self.assertFalse(torch.equal(a[1],validation[1]))
            self.assertFalse(torch.equal(a[0],validation[0]))


class ArchitectureTests(unittest.TestCase):
    def test_v2_transfer_forward_and_late_block_gradients(self):
        source=Model(pretrained=False)
        model=Model(pretrained=False,architecture='v2')
        info=transfer_checkpoint(model,{'config':{},'model':source.state_dict()})
        self.assertEqual(info['mode'],'v1_to_v2')
        self.assertTrue(torch.equal(model.layer4[0].conv1.weight,source.layer4[0].conv1.weight))
        self.assertTrue(torch.equal(model.up1.refine[0].weight,source.conv1_d.weight))
        self.assertEqual(model.full_resolution_refine[2].weight.abs().sum().item(),0)
        model.configure_encoder('layer3_4')
        model.train()
        before=model.layer3[0].bn1.running_mean.clone()
        prediction=model(torch.randn(1,3,64,96))
        self.assertEqual(tuple(prediction.shape),(1,1,64,96))
        prediction.square().mean().backward()
        self.assertIsNone(model.layer2[0].conv1.weight.grad)
        self.assertIsNotNone(model.layer3[0].conv1.weight.grad)
        self.assertIsNotNone(model.layer4[0].conv1.weight.grad)
        self.assertTrue(torch.equal(before,model.layer3[0].bn1.running_mean))

    def test_legacy_checkpoint_factory_is_exact(self):
        model=Model(pretrained=False).eval()
        loaded=Model.from_config({}).eval()
        loaded.load_state_dict(model.state_dict())
        image=torch.randn(1,3,64,96)
        with torch.inference_mode(): self.assertTrue(torch.equal(model(image),loaded(image)))

    def test_edge_metrics_detect_smoothing(self):
        target=torch.tensor([[[[1.,1.,3.,3.]]]])
        valid=torch.ones_like(target,dtype=torch.bool)
        perfect=DepthMetrics()
        perfect.update(target,target,valid)
        self.assertEqual(perfect.compute()['depth_edge_f1'],1)
        blurred=DepthMetrics()
        blurred.update(torch.tensor([[[[1.,1.8,2.2,3.]]]]),target,valid)
        self.assertAlmostEqual(blurred.compute()['depth_edge_f1'],.5)
        self.assertAlmostEqual(blurred.compute()['boundary_gradient_mae_m'],1.6,places=5)


class StagedTrainingTests(unittest.TestCase):
    def fixture(self,root):
        for split,ids in [('train',['a','b','c','d']),('test',['e'])]:
            for kind,suffix in [('rgb','.jpg'),('depth_bfx','.png')]:
                directory=root/'data'/kind/split
                directory.mkdir(parents=True)
                for i,name in enumerate(ids):
                    image=Image.new('RGB',(24,32),(i*40,100,70)) if kind=='rgb' else Image.fromarray(np.full((32,24),(1000+200*i)*8,dtype=np.uint16))
                    image.save(directory/(name+suffix))
        manifest=load_manifest(root/'data',root/'splits.json')
        config={'size':[32,64],'batch_size':1,'architecture':'v2','resize_mode':'letterbox',
            'depth_encoding':'sunrgbd','min_depth':.001,'max_depth':8.,'seed':174,
            'warmup_epochs':1,'layer4_epochs':1,'warmup_learning_rate':3e-4,'learning_rate':1e-4,
            'encoder_learning_rate':1e-5,'gradient_weight':.1,'patience':5,'lr_patience':2,
            'augmentation':{'horizontal_flip':.5,'brightness':.1,'contrast':.1,'saturation':.1},
            'smoke_test':True,'limit':None}
        return manifest,config

    def test_canonical_protocol_reproduces_legacy_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest,config=self.fixture(Path(folder))
            config.update(architecture='v1',resize_mode='stretch',size=[320,384])
            model=Model(pretrained=False).eval()
            old=evaluate(model,make_loader(manifest['validation'],config),'cpu',config)
            new=evaluate_canonical(model,manifest['validation'],config,'cpu')
            for key in ['mae_m','rmse_m','abs_rel','delta_1','smooth_l1','gradient_loss','valid_pixels']:
                self.assertAlmostEqual(old[key],new[key],places=10)

    def test_resume_across_unfreezing_keeps_weights_and_augmentation(self):
        import model.staged_training as training
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            manifest,config=self.fixture(root)
            def fresh_model():
                torch.manual_seed(174)
                return Model.from_config(config)
            def run(name,resume=False):
                return train_staged(fresh_model(),manifest,config,root/name,3,time.monotonic()+180,
                    resume=resume,checkpoint_every=1,examples=0)
            run('full')
            real_save=training.atomic_save
            def interrupt(payload,path):
                real_save(payload,path)
                if Path(path).name=='recovery.pt' and payload['epoch']==2 and not payload['epoch_complete']:
                    raise RuntimeError('interruption after encoder unfreeze')
            with patch.object(training,'atomic_save',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'after encoder unfreeze'):run('resumed')
            run('resumed',True)
            full=torch.load(root/'full/latest.pt',weights_only=True)
            resumed=torch.load(root/'resumed/latest.pt',weights_only=True)
            self.assertEqual([h['encoder_stage'] for h in full['history']],['frozen','layer4','layer3_4'])
            for a,b in zip(full['history'],resumed['history']):
                self.assertEqual({k:v for k,v in a.items() if k!='elapsed_seconds'},
                                 {k:v for k,v in b.items() if k!='elapsed_seconds'})
            self.assertEqual(full['scheduler'],resumed['scheduler'])
            for key,weights in full['model'].items():self.assertTrue(torch.equal(weights,resumed['model'][key]),key)
            self.assertEqual(full['optimizer']['param_groups'][0]['lr'],1e-4)

    def test_no_warmup_uses_fine_tuning_rate_immediately(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            manifest,config=self.fixture(root)
            config.update(warmup_epochs=0,layer4_epochs=0)
            train_staged(Model.from_config(config),manifest,config,root/'immediate',epochs=1,
                deadline=time.monotonic()+60,examples=0)
            state=torch.load(root/'immediate/latest.pt',weights_only=True)
            self.assertEqual(state['history'][0]['learning_rates']['decoder'],config['learning_rate'])
            self.assertEqual(state['encoder_stage'],'layer3_4')

    def test_time_budget_retains_a_resumable_initial_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            manifest,config=self.fixture(root)
            result=train_staged(Model.from_config(config),manifest,config,root/'timed',epochs=30,
                deadline=time.monotonic()-1,examples=0)
            self.assertEqual(result['completed_epochs'],0)
            self.assertEqual(result['stop_reason'],'time_budget')
            state=torch.load(root/'timed/recovery.pt',weights_only=True)
            self.assertEqual(state['epoch'],0)
            self.assertTrue(state['epoch_complete'])
            self.assertTrue((root/'timed/best.pt').is_file())

    def test_registry_only_promotes_comparable_full_validation(self):
        import script.predict as predict
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            checkpoint=root/'model.pt'
            checkpoint.write_bytes(b'checkpoint fingerprint')
            reference={'evaluation_protocol':PROTOCOL,'manifest_hash':'same','images':800,'valid_pixels':100,
                       'min_depth':.001,'max_depth':8.,'smooth_l1':.2}
            candidate={**reference,'smooth_l1':.19,'smoke_test':False}
            registry=root/'results/selected_model.json'
            self.assertTrue(promote_if_better(candidate,reference,checkpoint,registry))
            with patch.object(predict,'ROOT',root):
                self.assertEqual(predict.default_checkpoint(),checkpoint.resolve())
                checkpoint.write_bytes(b'changed since validation')
                self.assertEqual(predict.default_checkpoint(),root/'results/baseline/best.pt')
            self.assertFalse(promote_if_better({**candidate,'smoke_test':True},reference,checkpoint,registry))
            self.assertFalse(promote_if_better({**candidate,'smooth_l1':.21},reference,checkpoint,registry))
            with self.assertRaisesRegex(ValueError,'images differs'):
                promote_if_better({**candidate,'images':2},reference,checkpoint,registry)
