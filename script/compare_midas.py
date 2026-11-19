"""Oracle-aligned relative-depth comparison; never a raw metric-depth score for MiDaS."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataloader.custom_dataset import CustomDataset
from dataloader.preprocessing import prepare_rgb, restore_prediction
from model.canonical_evaluation import EVALUATION_SIZE, PROTOCOL
from dataloader.dataloader import audit_manifest, load_manifest
from model.evaluation import DepthMetrics, save_panel
from model.model import Model
from script.CNN import ROOT, configure_https_certificates, manifest_hash, write_json

EFFICIENTNET_REVISION = '771ce082b2ce6d033f55b3d47c1f77389ad3c180'
MIDAS_REVISION = '1645b7e1675301fdfac03640738fe5a6531e17d6'  # official v3_1


def align_inverse_depth(inverse, target, mask, min_depth, max_depth):
    """Per-image least squares: a*inverse+b ~= 1/target, with a >= 0.

    Uses each evaluation image's ground truth: an oracle relative-depth protocol.
    Degenerate/negative-correlation inputs use constant mean inverse depth.
    """
    inverse, target = np.asarray(inverse, dtype=np.float64), np.asarray(target, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(target) & (target >= min_depth) & (target <= max_depth)
    if not valid.any():
        raise ValueError('No valid pixels for alignment')
    if not np.isfinite(inverse).all():
        raise ValueError('Non-finite inverse-depth prediction')
    x, y = inverse[valid], 1.0 / target[valid]
    centered = x - x.mean()
    variance = np.dot(centered, centered)
    scale = max(0., float(np.dot(centered, y - y.mean()) / variance)) if variance > 1e-12 else 0.
    shift = float(y.mean() - scale * x.mean())
    fitted = np.clip(scale * inverse + shift, 1.0/max_depth, 1.0/min_depth)
    return (1.0 / fitted).astype(np.float32), {'scale': scale, 'shift': shift, 'constant_fit': scale == 0.}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data')
    parser.add_argument('--manifest', type=Path, default=ROOT/'results/baseline/splits.json')
    parser.add_argument('--checkpoints', type=Path, nargs='+', default=[ROOT/'results/baseline/best.pt', ROOT/'results/gradient_loss/best.pt'])
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--output', type=Path, default=ROOT/'results/midas_comparison')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--limit', type=int, help='Smoke test: evaluate only the first N images')
    parser.add_argument('--examples', type=int, default=4)
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or args.examples < 0:
        parser.error('Require positive --limit and nonnegative --examples')
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = [torch.load(p, map_location='cpu', weights_only=True) for p in args.checkpoints]
    input_configs = [checkpoint['config'] for checkpoint in checkpoints]
    config = dict(input_configs[0])
    canonical = any(c.get('resize_mode', 'stretch') != 'stretch' or c.get('evaluation_protocol') == PROTOCOL
                    or c['size'] != config['size'] for c in input_configs)
    if canonical:
        config.update(size=list(EVALUATION_SIZE), resize_mode='stretch')
    manifest = load_manifest(args.data, args.manifest)
    fingerprint = manifest_hash(manifest)
    for checkpoint in checkpoints:
        if checkpoint['manifest_hash'] != fingerprint:
            raise ValueError('All checkpoints must use the same manifest')
        for key in ['depth_encoding', 'min_depth', 'max_depth']:
            if checkpoint['config'][key] != config[key]:
                raise ValueError(f'Checkpoint preprocessing mismatch: {key}')
    audit_manifest(manifest, config['depth_encoding'], args.output/'integrity.json', (args.split,))
    records = manifest[args.split][:args.limit] if args.limit else manifest[args.split]
    dataset = CustomDataset(records, config['size'], config['depth_encoding'], config['min_depth'], config['max_depth'])
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('Device:', device, flush=True)
    configure_https_certificates()
    # Keep downloaded source and weights local and out of Git.
    torch.hub.set_dir(str(ROOT/'results/midas-cache'))
    repository = f'isl-org/MiDaS:{MIDAS_REVISION}'
    hub_load = torch.hub.load
    def pinned_load(repo, *arguments, **kwargs):
        if repo == 'rwightman/gen-efficientnet-pytorch':
            repo = f'{repo}:{EFFICIENTNET_REVISION}'
            kwargs.update(trust_repo=True, skip_validation=True)
        return hub_load(repo, *arguments, **kwargs)
    with patch('torch.hub.load', side_effect=pinned_load):
        midas = torch.hub.load(repository, 'MiDaS_small', trust_repo=True, skip_validation=True).to(device).eval()
    transform = torch.hub.load(repository, 'transforms', trust_repo=True, skip_validation=True).small_transform
    models = [Model.from_config(c).to(device).eval() for c in input_configs]
    names = [f'{i+1}_{p.parent.name}' for i, p in enumerate(args.checkpoints)]
    for model, checkpoint in zip(models, checkpoints):
        model.load_state_dict(checkpoint['model'])
    meters = {name: DepthMetrics(config['min_depth'], config['max_depth']) for name in names}
    aligned = {name: DepthMetrics(config['min_depth'], config['max_depth']) for name in names + ['MiDaS_small']}
    fits = []
    with torch.inference_mode():
        for index, (rgb, target, mask) in enumerate(dataset):
            record = records[index]
            with Image.open(record['rgb']) as original:
                original_image = original.convert('RGB')
            # Legacy runs preserve their original protocol. Mixed/new models use
            # original-image inputs with each model's documented preprocessing.
            image = np.array(original_image if canonical else original_image.resize(
                (config['size'][1], config['size'][0]), Image.Resampling.BILINEAR))
            output = midas(transform(image).to(device))
            inverse = F.interpolate(output.unsqueeze(1), size=config['size'], mode='bicubic', align_corners=False)[0].cpu().numpy()
            inverses = {'MiDaS_small': inverse}
            for name, model, input_config in zip(names, models, input_configs):
                if canonical:
                    prepared, geometry = prepare_rgb(original_image, input_config['size'], input_config.get('resize_mode','stretch'))
                    output = model(prepared.unsqueeze(0).to(device)).cpu()
                    prediction = restore_prediction(output, geometry, EVALUATION_SIZE)[0]
                else:
                    prediction = model(rgb.unsqueeze(0).to(device)).cpu()[0]
                if not torch.isfinite(prediction).all():
                    raise ValueError(f'Non-finite prediction from {name}')
                meters[name].update(prediction, target, mask)
                inverses[name] = 1.0 / prediction.clamp_min(config['min_depth']).numpy()
            image_fits = {'id': record['id']}
            for name, values in inverses.items():
                depth, fit = align_inverse_depth(values, target.numpy(), mask.numpy(), config['min_depth'], config['max_depth'])
                aligned[name].update(torch.from_numpy(depth), target, mask)
                image_fits[name] = fit
                if index < args.examples:
                    save_panel(rgb, target, torch.from_numpy(depth), mask,
                               args.output/'oracle_aligned'/name/f"{record['id']}.png", config['min_depth'], config['max_depth'])
            fits.append(image_fits)
            if index == 0 or (index+1) % 50 == 0:
                print(f'Compared {index+1}/{len(dataset)} images', flush=True)
    result = {'split': args.split, 'images': len(dataset), 'smoke_test': args.limit is not None or any(c['smoke_test'] for c in checkpoints),
        'manifest_hash': fingerprint, 'image_ids': [r['id'] for r in records], 'resolution': config['size'],
        'comparison_protocol': 'canonical_original_inputs_v2' if canonical else 'legacy_shared_resize_v1',
        'model_inputs': {name:{'size':c['size'],'resize_mode':c.get('resize_mode','stretch'),'architecture':c.get('architecture','v1')} for name,c in zip(names,input_configs)},
        'min_depth': config['min_depth'], 'max_depth': config['max_depth'], 'depth_encoding': config['depth_encoding'],
        'device': device, 'midas': {'model': 'MiDaS_small', 'source_revision': MIDAS_REVISION, 'efficientnet_revision': EFFICIENTNET_REVISION,
            'preprocessing': 'official small_transform on original RGB; bicubic prediction resize to canonical target grid' if canonical else 'official small_transform after shared RGB resize; bicubic prediction resize',
            'weights': [{'file': p.name, 'sha256': sha256(p)} for p in (ROOT/'results/midas-cache/checkpoints').glob('midas*')]},
        'checkpoints': {name: {'path': str(path.resolve()), 'sha256': sha256(path), 'epoch': c['epoch']} for name,path,c in zip(names,args.checkpoints,checkpoints)},
        'metric_depth_unaligned': {name: m.compute() for name,m in meters.items()},
        'oracle_aligned_relative_depth': {name: m.compute() for name,m in aligned.items()},
        'protocol': 'Per-image ground-truth least-squares scale/shift in inverse depth; nonnegative scale; inverse depth clamped to configured range before inversion. Identical protocol for all methods. Global valid-pixel aggregation.',
        'limitations': ['Oracle alignment uses evaluation targets and does not demonstrate meter-depth inference without ground truth.',
            'MiDaS pretraining overlap with SUN RGB-D source datasets has not been ruled out; no independently verified zero-shot claim.',
            'Single-seed custom-model comparison; image split is not verified scene-independent.']}
    write_json(args.output/f'{args.split}_comparison.json', result)
    write_json(args.output/f'{args.split}_alignment_fits.json', fits)
    print(json.dumps({k:result[k] for k in ['metric_depth_unaligned','oracle_aligned_relative_depth']}, indent=2))


if __name__ == '__main__':
    main()
