"""Paired RGB/depth loading with explicit depth decoding in meters."""
import hashlib
import random

import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from dataloader.preprocessing import prepare_rgb, prepare_depth


def decode_depth(raw, encoding):
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise ValueError(f"Expected single-channel depth, got {raw.shape}")
    if encoding == 'sunrgbd':
        if raw.dtype != np.uint16:
            raise ValueError('SUN RGB-D encoding requires uint16 PNG targets')
        values = raw.astype(np.uint32)
        raw = ((values >> 3) | (values << 13)) & 0xffff
        return raw.astype(np.float32) / 1000.0
    if encoding == 'millimeters':
        return raw.astype(np.float32) / 1000.0
    if encoding == 'meters':
        return raw.astype(np.float32)
    raise ValueError(f'Unknown depth encoding: {encoding}')


def load_pair(record, encoding):
    try:
        with Image.open(record['rgb']) as image:
            rgb = image.convert('RGB')
        with Image.open(record['depth']) as image:
            depth = decode_depth(np.array(image), encoding)
        if depth.shape != (rgb.height, rgb.width):
            raise ValueError('RGB/depth dimensions differ')
        return rgb, depth
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid pair {record['id']}: RGB={record['rgb']}, "
                         f"depth={record['depth']}: {error}") from error



class CustomDataset(Dataset):
    def __init__(self, records, size=(320, 384), encoding='sunrgbd', min_depth=0.001, max_depth=8.0,
                 resize_mode='stretch', augmentation=None, seed=174, epoch=0):
        self.records = records
        self.size = tuple(size)
        self.encoding = encoding
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.resize_mode = resize_mode
        self.augmentation = augmentation or {}
        self.seed, self.epoch = seed, epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        rgb, depth = load_pair(record, self.encoding)
        if self.augmentation:
            # Stable per-record randomness makes an interrupted epoch reproducible.
            key = f'{self.seed}:{self.epoch}:{record["id"]}'.encode()
            rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], 'big'))
            rgb, depth = augment_pair(rgb, depth, self.augmentation, rng)
        rgb_tensor, geometry = prepare_rgb(rgb, self.size, self.resize_mode)
        depth_tensor, mask = prepare_depth(depth, geometry, self.min_depth, self.max_depth)
        return rgb_tensor, depth_tensor, mask


def augment_pair(rgb, depth, settings, rng):
    probability = settings.get('horizontal_flip', 0.)
    if not 0 <= probability <= 1:
        raise ValueError('Horizontal flip probability must be in [0,1]')
    if rng.random() < probability:
        rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        depth = np.fliplr(depth).copy()
    for key, enhancer in [('brightness', ImageEnhance.Brightness), ('contrast', ImageEnhance.Contrast), ('saturation', ImageEnhance.Color)]:
        amount = settings.get(key, 0.)
        if not 0 <= amount < 1:
            raise ValueError(f'{key} jitter must be in [0,1)')
        if amount:
            rgb = enhancer(rgb).enhance(rng.uniform(1-amount, 1+amount))
    return rgb, depth
