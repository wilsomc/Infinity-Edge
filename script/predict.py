"""Predict depth for one photo using a saved Infinity Edge checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from model.inference import file_hash, load_predictor, predict_depth, read_rgb, save_prediction

ROOT = Path(__file__).resolve().parent.parent


def default_checkpoint():
    registry = ROOT/'results/selected_model.json'
    if registry.is_file():
        try:
            entry = json.loads(registry.read_text())
            checkpoint = Path(entry['checkpoint'])
            if (entry.get('selected_on_validation') and not entry.get('smoke_test', True)
                    and checkpoint.is_file() and file_hash(checkpoint) == entry.get('sha256')):
                return checkpoint
        except (OSError, ValueError, KeyError, TypeError):
            pass
    selected = ROOT/'results/improvement_pass/best.pt'
    return selected if selected.is_file() else ROOT/'results/baseline/best.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, help='Path to a photo (no depth map or dataset required)')
    parser.add_argument('--checkpoint', type=Path, default=default_checkpoint())
    parser.add_argument('--output', type=Path, help='Output directory; default results/predictions/<photo stem>')
    parser.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    parser.add_argument('--overwrite', action='store_true', help='Replace this prediction output')
    args = parser.parse_args()
    for path in [args.image, args.checkpoint]:
        if not path.is_file():
            parser.error(f'File does not exist: {path}')
    output = args.output or ROOT/'results/predictions'/args.image.stem
    destinations = [output/name for name in ['depth_m.npy', 'raw_depth_m.npy', 'depth_color.png', 'preview.png', 'metadata.json']]
    if any(path.resolve() in {args.image.resolve(), args.checkpoint.resolve()} for path in destinations):
        parser.error('Output would overwrite the input photo or model checkpoint; choose another --output')
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    try:
        image = read_rgb(args.image)
        model, config, epoch, smoke = load_predictor(args.checkpoint, device)
        raw, depth = predict_depth(model, image, config, device)
        files = save_prediction(image, raw, depth, config, output, {
            'input': str(args.image.resolve()), 'input_sha256': file_hash(args.image),
            'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': file_hash(args.checkpoint),
            'checkpoint_epoch': epoch, 'smoke_test': smoke, 'device': device,
            'initialization': config.get('initialization'),
            'encoder_training': config.get('inference_encoder_stage', config.get('encoder_training', 'frozen')),
            'gradient_weight': config.get('gradient_weight', 0.0),
        }, overwrite=args.overwrite)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Prediction failed: {error}\n')
    print(f'Saved depth at {image.width}×{image.height} pixels using {device} (checkpoint epoch {epoch}).')
    for name, path in files.items():
        print(f'{name}: {path.resolve()}')


if __name__ == '__main__':
    main()
