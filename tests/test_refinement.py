import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from dataloader.dataloader import load_manifest, make_loader
from model.evaluation import evaluate
from model.model import Model
from model.refinement import refine_stage
from script.CNN import manifest_hash


class RefinementTests(unittest.TestCase):
    def fixture(self, root):
        for split, ids in [('train',['a','b','c','d']),('test',['e'])]:
            for kind, extension in [('rgb','.jpg'),('depth_bfx','.png')]:
                directory = root/'data'/kind/split
                directory.mkdir(parents=True)
                for i, name in enumerate(ids):
                    image = Image.new('RGB',(32,32),(40*i,100,50)) if kind=='rgb' else Image.fromarray(np.full((32,32),(1000+200*i)*8,dtype=np.uint16))
                    image.save(directory/(name+extension))
        manifest = load_manifest(root/'data',root/'splits.json')
        config = {'size':[32,32], 'batch_size':1, 'depth_encoding':'sunrgbd',
                  'min_depth':.001, 'max_depth':8., 'seed':174, 'learning_rate':.001}
        model = Model(pretrained=False).eval()
        initial = evaluate(model,make_loader(manifest['validation'],config),'cpu',config)
        source = {'model':model.state_dict(),'config':config,'manifest_hash':manifest_hash(manifest),'smoke_test':True,'epoch':4}
        settings = {'encoder_training':'layer4','gradient_weight':.1,'learning_rate':3e-4,'encoder_learning_rate':3e-5}
        return source, manifest, settings, initial

    def test_partial_tuning_gradients_and_batchnorm(self):
        model = Model(pretrained=False)
        model.configure_encoder('layer4')
        model.train()
        before = model.layer4[0].bn1.running_mean.clone()
        loss = model(torch.randn(2,3,32,32)).square().mean()
        loss.backward()
        self.assertIsNone(model.layer3[0].conv1.weight.grad)
        self.assertIsNotNone(model.layer4[0].conv1.weight.grad)
        self.assertGreater(model.layer4[0].conv1.weight.grad.abs().sum().item(),0)
        self.assertTrue(torch.equal(before,model.layer4[0].bn1.running_mean))
        groups = model.training_parameter_groups(3e-4,3e-5)
        self.assertEqual([g['lr'] for g in groups],[3e-4,3e-5])
        ids = [id(p) for group in groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})
        model.configure_encoder('frozen')
        self.assertFalse(model.layer4[0].conv1.weight.requires_grad)

    def test_resume_matches_with_encoder_and_scheduler(self):
        import model.refinement as refinement
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, manifest, settings, initial = self.fixture(root)
            common = dict(source=source,manifest=manifest,settings=settings,epochs=2,
                          deadline=time.monotonic()+120,initial_validation=initial,
                          initialization={'epoch':4},checkpoint_every=1,examples=0)
            refine_stage(output=root/'full',**common)
            real_save = refinement.atomic_save
            def interrupt(payload,path):
                real_save(payload,path)
                if Path(path).name=='recovery.pt':
                    raise RuntimeError('simulated interruption')
            with patch.object(refinement,'atomic_save',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    refine_stage(output=root/'resumed',**common)
            refine_stage(output=root/'resumed',resume=True,**common)
            full = torch.load(root/'full/latest.pt',weights_only=True)
            resumed = torch.load(root/'resumed/latest.pt',weights_only=True)
            self.assertEqual(full['history'],resumed['history'])
            self.assertEqual(full['scheduler'],resumed['scheduler'])
            for name, weights in full['model'].items():
                self.assertTrue(torch.equal(weights,resumed['model'][name]),name)

    def test_time_budget_retains_initial_model_and_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source, manifest, settings, initial = self.fixture(root)
            result=refine_stage(source,manifest,settings,root/'limited',epochs=3,
                deadline=time.monotonic()-1,initial_validation=initial,initialization={'epoch':4},examples=0)
            self.assertEqual(result['completed_epochs'],0)
            self.assertEqual(result['checkpoint_epoch'],0)
            self.assertEqual(result['stop_reason'],'time_budget')
            best=torch.load(root/'limited/best.pt',weights_only=True)
            for name, weights in source['model'].items():
                self.assertTrue(torch.equal(weights,best['model'][name]))
            self.assertTrue((root/'limited/recovery.pt').exists())
