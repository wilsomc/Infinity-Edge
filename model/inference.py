"""Standalone inference with checkpoint preprocessing and no ground-truth data."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from dataloader.preprocessing import prepare_rgb, restore_prediction
from model.model import Model


def read_rgb(path):
    try:
        from pillow_heif import register_heif_opener
    except ImportError as error:
        if Path(path).suffix.lower() in {'.heic', '.heif', '.heics', '.heifs', '.hif'}:
            raise ValueError('HEIC/HEIF photos require pillow-heif; install it with .venv/bin/python -m pip install pillow-heif') from error
    else:
        register_heif_opener(thumbnails=False)
    with Image.open(path) as image:
        oriented = ImageOps.exif_transpose(image).convert('RGBA')
        # Give transparent input pixels an explicit, reproducible background.
        background = Image.new('RGBA', oriented.size, 'white')
        return Image.alpha_composite(background, oriented).convert('RGB')


def load_predictor(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = dict(checkpoint['config'])
    config['inference_encoder_stage'] = checkpoint.get('encoder_stage', config.get('encoder_training', 'frozen'))
    if len(config['size']) != 2 or min(config['size']) < 32:
        raise ValueError('Invalid checkpoint input size')
    if not 0 < config['min_depth'] < config['max_depth'] < float('inf'):
        raise ValueError('Invalid checkpoint depth range')
    model = Model.from_config(config)
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), config, checkpoint.get('epoch'), checkpoint.get('smoke_test', False)


def predict_depth(model, image, config, device):
    rgb, geometry = prepare_rgb(image, config['size'], config.get('resize_mode', 'stretch'))
    rgb = rgb.unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        prediction = model(rgb)
        if not torch.isfinite(prediction).all():
            raise ValueError('Model produced non-finite depth values')
        # Restore spatial dimensions only; this cannot add learned detail.
        restored = restore_prediction(prediction.cpu(), geometry)
    raw = restored[0, 0].numpy().astype(np.float32)
    return raw, np.clip(raw, config['min_depth'], config['max_depth'])


def save_prediction(image, raw, depth, config, output, metadata, overwrite=False):
    output = Path(output)
    files = {name: output / filename for name, filename in {
        'depth_m': 'depth_m.npy', 'raw_depth_m': 'raw_depth_m.npy',
        'depth_color': 'depth_color.png', 'preview': 'preview.png', 'metadata': 'metadata.json'}.items()}
    if not overwrite and any(path.exists() for path in files.values()):
        raise FileExistsError(f'Prediction output already exists in {output}; choose another --output or use --overwrite')
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    low, high = config['min_depth'], config['max_depth']
    np.save(files['depth_m'], depth, allow_pickle=False)
    np.save(files['raw_depth_m'], raw, allow_pickle=False)
    normalized = (depth - low) / (high - low)
    color = (matplotlib.colormaps['viridis'](normalized)[..., :3] * 255).round().astype(np.uint8)
    Image.fromarray(color).save(files['depth_color'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title('Input photo')
    im = axes[1].imshow(depth, cmap='viridis', vmin=low, vmax=high)
    axes[1].set_title('Predicted depth (m)')
    fig.colorbar(im, ax=axes[1], label='Estimated distance (m)', shrink=.8)
    for ax in axes:
        ax.axis('off')
    fig.savefig(files['preview'], dpi=140)
    plt.close(fig)
    metadata.update(units='meters', original_size_hw=[image.height, image.width],
        network_size_hw=list(config['size']), raw_depth_range_m=[float(raw.min()), float(raw.max())],
        output_range_m=[low, high], clipped_fraction=float(((raw < low) | (raw > high)).mean()),
        pixel_orientation='EXIF orientation applied; transparent pixels composited on white',
        output_policy='Remove padding, bilinearly resize raw prediction to original dimensions, then clip to checkpoint range; raw array also saved',
        architecture=config.get('architecture', 'v1'), resize_mode=config.get('resize_mode', 'stretch'),
        alignment='none; no ground truth used',
        files={name: str(path.resolve()) for name, path in files.items()},
        limitations='Learned indoor depth estimates; original-size output is interpolated, not additional model detail. Accuracy on arbitrary photos is unmeasured.')
    files['metadata'].write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return files
