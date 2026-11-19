import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from model.inference import load_predictor, predict_depth, read_rgb, save_prediction
from model.model import Model


class ConstantModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, rgb):
        return torch.full((1, 1, *rgb.shape[-2:]), self.value)


class InferenceTests(unittest.TestCase):
    config = {'size':[32, 64], 'min_depth':.001, 'max_depth':8.}

    def test_original_size_values_clipping_and_nonfinite(self):
        image = Image.new('RGB', (71, 49))
        raw, depth = predict_depth(ConstantModel(2.), image, self.config, 'cpu')
        self.assertEqual(depth.shape, (49, 71))
        self.assertEqual(depth.dtype, np.float32)
        np.testing.assert_allclose(depth, 2.)
        raw, depth = predict_depth(ConstantModel(-1.), image, self.config, 'cpu')
        np.testing.assert_allclose(raw, -1.)
        np.testing.assert_allclose(depth, .001)
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            predict_depth(ConstantModel(float('nan')), image, self.config, 'cpu')

    def test_orientation_and_transparency(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'photo.jpg'
            exif = Image.Exif()
            exif[274] = 6
            Image.new('RGB', (20, 10)).save(path, exif=exif)
            self.assertEqual(read_rgb(path).size, (10, 20))
            path = Path(folder)/'transparent.png'
            Image.new('RGBA', (2, 2), (0, 0, 0, 0)).save(path)
            np.testing.assert_array_equal(np.array(read_rgb(path)), np.full((2,2,3), 255))

    def test_saved_arrays_previews_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Image.new('RGB', (71, 49))
            raw, depth = predict_depth(ConstantModel(2.), image, self.config, 'cpu')
            files = save_prediction(image, raw, depth, self.config, folder, {})
            np.testing.assert_array_equal(np.load(files['depth_m'], allow_pickle=False), depth)
            np.testing.assert_array_equal(np.load(files['raw_depth_m'], allow_pickle=False), raw)
            with Image.open(files['depth_color']) as preview:
                self.assertEqual(preview.size, image.size)
            info = json.loads(files['metadata'].read_text())
            self.assertEqual(info['clipped_fraction'], 0)
            self.assertEqual(info['alignment'], 'none; no ground truth used')
            with self.assertRaises(FileExistsError):
                save_prediction(image, raw, depth, self.config, folder, {})

    def test_checkpoint_load_needs_no_download_or_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Model(pretrained=False).eval()
            path = Path(folder)/'checkpoint.pt'
            torch.save({'model':model.state_dict(), 'config':self.config, 'epoch':3, 'smoke_test':True}, path)
            with patch('torch.hub.load', side_effect=AssertionError('Unexpected download')):
                loaded, config, epoch, smoke = load_predictor(path, 'cpu')
                image = Image.new('RGB', (71,49))
                raw, depth = predict_depth(loaded, image, config, 'cpu')
            self.assertEqual(epoch, 3)
            self.assertTrue(smoke)
            self.assertEqual(raw.shape, (49, 71))
            self.assertTrue(np.isfinite(depth).all())
            self.assertTrue((depth >= .001).all() and (depth <= 8).all())

    def test_default_checkpoint_prefers_completed_selection(self):
        import script.predict as cli
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(cli, 'ROOT', root):
                self.assertEqual(cli.default_checkpoint(), root/'results/baseline/best.pt')
                selected=root/'results/improvement_pass/best.pt'
                selected.parent.mkdir(parents=True)
                selected.touch()
                self.assertEqual(cli.default_checkpoint(), selected)

    def test_heic_round_trip_and_orientation(self):
        from pillow_heif import register_heif_opener
        register_heif_opener(thumbnails=False)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'phone.HEIC'
            image=Image.new('RGB',(20,10),'green')
            image.paste('red',(0,0,10,10))
            exif=Image.Exif()
            exif[274]=6
            image.save(path,format='HEIF',quality=-1,exif=exif.tobytes())
            decoded=read_rgb(path)
            self.assertEqual(decoded.mode,'RGB')
            self.assertEqual(decoded.size,(10,20))
            top,bottom=decoded.getpixel((5,3)),decoded.getpixel((5,16))
            self.assertGreater(top[0],top[1])
            self.assertGreater(bottom[1],bottom[0])
