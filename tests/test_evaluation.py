import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataloader.custom_dataset import CustomDataset, decode_depth
from dataloader.dataloader import load_manifest, audit_manifest
from model.evaluation import DepthMetrics
from model.model import Model


class EvaluationTests(unittest.TestCase):
    def test_global_metrics_and_invalid_targets(self):
        # Errors are [0, 2, 1] across unequal batches: MAE=1, RMSE=sqrt(5/3).
        metrics = DepthMetrics(max_depth=10)
        metrics.update(torch.tensor([1.]), torch.tensor([1.]), torch.tensor([True]))
        metrics.update(torch.tensor([4., 3., float('nan'), 9.]),
                       torch.tensor([2., 4., float('nan'), 0.]), torch.ones(4, dtype=torch.bool))
        result = metrics.compute()
        self.assertEqual(result['valid_pixels'], 3)
        self.assertAlmostEqual(result['mae_m'], 1)
        self.assertAlmostEqual(result['rmse_m'], (5/3)**0.5)
        self.assertAlmostEqual(result['abs_rel'], 1.25/3)
        self.assertAlmostEqual(result['delta_1'], 1/3)
        self.assertAlmostEqual(result['delta_2'], 2/3)
        self.assertAlmostEqual(result['delta_3'], 2/3)

    def test_clipping_nonfinite_and_empty(self):
        metrics = DepthMetrics(min_depth=1, max_depth=8)
        metrics.update(torch.tensor([-2., 20.]), torch.tensor([1., 8.]), torch.tensor([True, True]))
        self.assertEqual(metrics.compute()['mae_m'], 0)
        self.assertEqual(metrics.compute()['clipped_prediction_fraction'], 1)
        with self.assertRaises(ValueError):
            metrics.update(torch.tensor([float('nan')]), torch.tensor([2.]), torch.tensor([True]))
        with self.assertRaises(ValueError):
            DepthMetrics().compute()

    def test_sunrgbd_bit_rotation(self):
        mm = np.array([[0, 1000, 2000, 8000, 10000, 65535]], dtype=np.uint32)
        encoded = (((mm << 3) | (mm >> 13)) & 65535).astype(np.uint16)
        np.testing.assert_allclose(decode_depth(encoded, 'sunrgbd'), mm.astype(np.float32)/1000)

    def test_loader_masks_invalid_depth_and_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            rgb, depth = Path(folder)/'rgb.jpg', Path(folder)/'depth.png'
            Image.new('RGB', (2, 2)).save(rgb)
            Image.fromarray(np.array([[0, 2000], [9000, 1000]], dtype=np.uint16)).save(depth)
            dataset = CustomDataset([{'id':'sample', 'rgb':str(rgb), 'depth':str(depth)}],
                                    size=(2, 2), encoding='millimeters')
            _, target, mask = dataset[0]
            self.assertEqual(mask.sum().item(), 2)
            self.assertEqual(target.sum().item(), 3)

    def test_manifest_reuse_and_dataset_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split, ids in [('train', ['a','b','c']), ('test', ['d'])]:
                for kind, suffix in [('rgb','.jpg'), ('depth_bfx','.png')]:
                    target = root/kind/split
                    target.mkdir(parents=True)
                    for name in ids:
                        (target/(name+suffix)).touch()
            path = root/'splits.json'
            first = load_manifest(root, path)
            self.assertEqual(first, load_manifest(root, path, seed=999))
            self.assertFalse({r['id'] for r in first['train']} & {r['id'] for r in first['validation']})
            (root/'rgb/test/d.jpg').unlink()
            with self.assertRaises(ValueError):
                load_manifest(root, path)

    def test_forward_and_frozen_batchnorm(self):
        model = Model(pretrained=False).train()
        before = model.bn1.running_mean.clone()
        prediction = model(torch.randn(1, 3, 64, 96))
        self.assertEqual(tuple(prediction.shape), (1, 1, 64, 96))
        prediction.mean().backward()
        self.assertTrue(torch.equal(before, model.bn1.running_mean))
        self.assertIsNone(model.conv1.weight.grad)
        self.assertIsNotNone(model.output.weight.grad)

    def test_audit_reports_corrupt_path_and_explicit_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split, ids in [('train', ['a', 'b', 'd', 'bad']), ('test', ['c'])]:
                for kind, suffix in [('rgb', '.jpg'), ('depth_bfx', '.png')]:
                    directory = root / kind / split
                    directory.mkdir(parents=True)
                    for name in ids:
                        image = Image.new('RGB', (32, 32)) if kind == 'rgb' else Image.fromarray(np.full((32,32), 8000, dtype=np.uint16))
                        image.save(directory / (name + suffix))
            (root / 'depth_bfx/train/bad.png').write_bytes(b'broken PNG')
            original = load_manifest(root, root/'original.json', val_fraction=0.5)
            with self.assertRaisesRegex(ValueError, 'bad'):
                audit_manifest(original, 'sunrgbd', root/'audit.json')
            self.assertIn('bad.png', (root/'audit.json').read_text())
            clean = load_manifest(root, root/'clean.json', val_fraction=0.5, exclude_ids=['bad'])
            self.assertEqual(clean['excluded'][0]['record']['id'], 'bad')
            self.assertEqual(clean, load_manifest(root, root/'clean.json'))
            audit_manifest(clean, 'sunrgbd', root/'clean-audit.json')
            self.assertTrue((root/'depth_bfx/train/bad.png').exists())

    def test_mid_epoch_resume_matches_uninterrupted_training(self):
        import sys
        from unittest.mock import patch
        from script import CNN
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split, ids in [('train', ['a', 'b', 'c', 'd']), ('test', ['e'])]:
                for kind, suffix in [('rgb', '.jpg'), ('depth_bfx', '.png')]:
                    directory = root / 'data' / kind / split
                    directory.mkdir(parents=True)
                    for i, name in enumerate(ids):
                        image = Image.new('RGB', (32, 32), (i*40, 100, 50)) if kind == 'rgb' else Image.fromarray(np.full((32,32), (1000+i*200)*8, dtype=np.uint16))
                        image.save(directory / (name + suffix))
            common = ['train', '--data', str(root/'data'), '--manifest', str(root/'splits.json'),
                      '--device', 'cpu', '--size', '32', '32', '--batch-size', '1',
                      '--no-pretrained', '--examples', '0', '--checkpoint-every', '1', '--gradient-weight', '0.2']
            def run(extra):
                with patch.object(sys, 'argv', ['CNN'] + common + extra):
                    CNN.main()
            run(['--output', str(root/'full')])
            real_save = CNN.atomic_save
            def interrupt_after_save(payload, path):
                real_save(payload, path)
                if Path(path).name == 'recovery.pt':
                    raise RuntimeError('simulated interruption')
            with patch.object(CNN, 'atomic_save', side_effect=interrupt_after_save):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    run(['--output', str(root/'resumed')])
            recovery = torch.load(root/'resumed/recovery.pt', weights_only=True)
            self.assertEqual(recovery['next_batch'], 1)
            run(['--output', str(root/'resumed'), '--resume', str(root/'resumed/recovery.pt')])
            full = torch.load(root/'full/latest.pt', weights_only=True)
            resumed = torch.load(root/'resumed/latest.pt', weights_only=True)
            self.assertEqual(full['history'], resumed['history'])
            for name, weights in full['model'].items():
                self.assertTrue(torch.equal(weights, resumed['model'][name]), name)


if __name__ == '__main__':
    unittest.main()
