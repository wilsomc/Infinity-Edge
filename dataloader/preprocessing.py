"""Versioned image geometry shared by paired training and standalone prediction."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def spatial_geometry(source_hw, target_hw, mode='stretch'):
    h, w = map(int, source_hw)
    th, tw = map(int, target_hw)
    if min(h, w, th, tw) < 1:
        raise ValueError('Image dimensions must be positive')
    if mode == 'stretch':
        rh, rw = th, tw
    elif mode == 'letterbox':
        scale = min(th/h, tw/w)
        rh, rw = min(th, max(1, round(h*scale))), min(tw, max(1, round(w*scale)))
    else:
        raise ValueError(f'Unknown resize mode: {mode}')
    top, left = (th-rh)//2, (tw-rw)//2
    return {'source_hw': [h, w], 'canvas_hw': [th, tw], 'resized_hw': [rh, rw], 'top': top, 'left': left, 'mode': mode}


def build_rgb_transform(size):
    # Keep legacy PIL resizing/normalization exactly unchanged for old checkpoints.
    return transforms.Compose([
        transforms.Resize(tuple(size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def prepare_rgb(image, size, mode='stretch'):
    geometry = spatial_geometry((image.height, image.width), size, mode)
    tensor = build_rgb_transform(geometry['resized_hw'])(image)
    h, w = geometry['resized_hw']
    th, tw = geometry['canvas_hw']
    top, left = geometry['top'], geometry['left']
    # Zero after normalization represents the ImageNet mean, not black padding.
    tensor = F.pad(tensor, (left, tw-w-left, top, th-h-top), value=0.)
    return tensor, geometry


def prepare_depth(depth, geometry, min_depth, max_depth):
    depth = np.asarray(depth, dtype=np.float32)
    if list(depth.shape) != geometry['source_hw']:
        raise ValueError('Depth geometry does not match the RGB image')
    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    rh, rw = geometry['resized_hw']
    # Nearest-neighbor and the identical geometry keep validity and depth aligned.
    clean = np.where(valid, depth, 0).astype(np.float32)
    resized = np.array(Image.fromarray(clean).resize((rw, rh), Image.Resampling.NEAREST), dtype=np.float32)
    resized_mask = np.array(Image.fromarray(valid.astype(np.uint8)).resize((rw, rh), Image.Resampling.NEAREST), dtype=bool)
    canvas = np.zeros(geometry['canvas_hw'], dtype=np.float32)
    mask = np.zeros(geometry['canvas_hw'], dtype=bool)
    top, left = geometry['top'], geometry['left']
    canvas[top:top+rh, left:left+rw] = resized
    mask[top:top+rh, left:left+rw] = resized_mask
    return torch.from_numpy(canvas).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)


def restore_prediction(prediction, geometry, output_hw=None):
    """Remove letterbox padding BEFORE resizing back to a photo/evaluation grid."""
    rh, rw = geometry['resized_hw']
    top, left = geometry['top'], geometry['left']
    content = prediction[..., top:top+rh, left:left+rw]
    if list(prediction.shape[-2:]) != geometry['canvas_hw'] or content.shape[-2:] != (rh, rw):
        raise ValueError('Prediction shape does not match preprocessing geometry')
    return F.interpolate(content, size=output_hw or geometry['source_hw'], mode='bilinear', align_corners=False)
