"""Discover paired images and persist a reproducible train/validation/test manifest."""
import json
import random
from pathlib import Path

import torch

from dataloader.custom_dataset import CustomDataset, load_pair


def discover(root, split):
    root = Path(root).resolve()
    # Support both the actual local layout and the original proposed layout.
    rgb_dir, depth_dir = root / 'rgb' / split, root / 'depth_bfx' / split
    if not rgb_dir.is_dir():
        rgb_dir, depth_dir = root / split / 'rgb', root / split / 'depth_bfx'
    images = sorted(p for p in rgb_dir.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    depths = {p.stem: p for p in depth_dir.glob('*.png')}
    if not images:
        raise ValueError(f'No RGB images found in {rgb_dir}')
    if len({p.stem for p in images}) != len(images):
        raise ValueError(f'Duplicate RGB identifiers in {rgb_dir}')
    if {p.stem for p in images} != set(depths):
        raise ValueError(f'RGB/depth identifiers do not match in {split}')
    return [{'id': p.stem, 'rgb': str(p), 'depth': str(depths[p.stem])} for p in images]


def load_manifest(root, path, seed=174, val_fraction=0.1, exclude_ids=None):
    path = Path(path)
    available = {'train': discover(root, 'train'), 'test': discover(root, 'test')}
    if {r['id'] for r in available['train']} & {r['id'] for r in available['test']}:
        raise ValueError('Training/test identifiers overlap')
    if path.exists():
        manifest = json.loads(path.read_text())
        if exclude_ids and set(exclude_ids) != {r['record']['id'] for r in manifest.get('excluded', [])}:
            raise ValueError('Exclusions differ from saved manifest; use a new manifest path')
        for source, splits in [('train', ('train', 'validation')), ('test', ('test',))]:
            saved = [r for split in splits for r in manifest[split]]
            saved += [entry['record'] for entry in manifest.get('excluded', []) if entry['split'] in splits]
            if len(saved) != len({r['id'] for r in saved}):
                raise ValueError('Duplicate or overlapping identifiers in manifest')
            if sorted(saved, key=lambda r: r['id']) != available[source]:
                raise ValueError('Manifest differs from current dataset; use a new manifest path')
        if not all(manifest[s] for s in ('train', 'validation', 'test')):
            raise ValueError('Manifest contains an empty split')
        return manifest
    if not 0 < val_fraction < 1 or len(available['train']) < 2:
        raise ValueError('Need at least two training images and 0 < val_fraction < 1')
    records = available['train'].copy()
    random.Random(seed).shuffle(records)
    n_val = min(len(records) - 1, max(1, round(len(records) * val_fraction)))
    manifest = {'seed': seed, 'val_fraction': val_fraction,
                'split_policy': 'seeded image split; scene identities unavailable',
                'train': sorted(records[n_val:], key=lambda r: r['id']),
                'validation': sorted(records[:n_val], key=lambda r: r['id']), 'test': available['test']}
    excluded = set(exclude_ids or [])
    known = {r['id'] for split in ('train', 'validation', 'test') for r in manifest[split]}
    if excluded - known:
        raise ValueError(f'Unknown excluded identifiers: {sorted(excluded - known)}')
    if excluded:
        manifest['excluded'] = []
        for split in ('train', 'validation', 'test'):
            manifest['excluded'].extend({'split': split, 'record': r, 'reason': 'explicit exclusion of damaged pair'}
                                        for r in manifest[split] if r['id'] in excluded)
            manifest[split] = [r for r in manifest[split] if r['id'] not in excluded]
        if not all(manifest[split] for split in ('train', 'validation', 'test')):
            raise ValueError('Exclusions leave an empty split')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def make_loader(records, config, shuffle=False, limit=None, augment=False, epoch=0):
    if limit is not None:
        records = records[:limit]
    dataset = CustomDataset(records, size=config['size'], encoding=config['depth_encoding'],
                            min_depth=config['min_depth'], max_depth=config['max_depth'],
                            resize_mode=config.get('resize_mode', 'stretch'),
                            augmentation=config.get('augmentation') if augment else None,
                            seed=config.get('seed', 174), epoch=epoch)
    generator = torch.Generator().manual_seed(config.get('seed', 174))
    return torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle,
                                       num_workers=0, generator=generator)


def audit_manifest(manifest, encoding, report_path, splits=('train', 'validation', 'test')):
    """Decode every selected original pair before starting expensive computation."""
    report = {'pairs_checked': 0, 'errors': [], 'splits': list(splits), 'depth_encoding': encoding}
    for split in splits:
        for record in manifest[split]:
            try:
                load_pair(record, encoding)
            except (ValueError, OSError) as error:
                report['errors'].append({'split': split, 'id': record['id'], 'error': str(error)})
            report['pairs_checked'] += 1
            if report['pairs_checked'] % 1000 == 0:
                print(f"Preflight: checked {report['pairs_checked']} pairs", flush=True)
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    if report['errors']:
        ids = ', '.join(e['id'] for e in report['errors'])
        raise ValueError(f"Preflight found invalid pairs: {ids}. See {path}. "
                         'Restore the source files or create a new manifest with --exclude-ids ID ... .')
    print(f"Preflight passed: {report['pairs_checked']} pairs", flush=True)
    return report
