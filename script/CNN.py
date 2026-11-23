import argparse
import hashlib
import json
import os
import ssl
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model.runtime import select_device
from dataloader.custom_dataset import decode_depth
from dataloader.dataloader import load_manifest, make_loader, audit_manifest
from model.evaluation import evaluate
from model.canonical_evaluation import evaluate_canonical, PROTOCOL
from model.model import Model
from model.losses import depth_loss

ROOT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def epoch_records(records, seed, epoch):
    order = torch.randperm(len(records), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    return [records[i] for i in order]


def configure_https_certificates():
    paths = ssl.get_default_verify_paths()
    if (paths.cafile is None and paths.capath is None
            and 'SSL_CERT_FILE' not in os.environ and 'SSL_CERT_DIR' not in os.environ):
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'train', 'evaluate'])
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'results/splits.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/baseline')
    parser.add_argument('--depth-encoding', choices=['sunrgbd', 'millimeters', 'meters'], default='sunrgbd')
    parser.add_argument('--min-depth', type=float, default=0.001)
    parser.add_argument('--max-depth', type=float, default=8.0)
    parser.add_argument('--size', type=int, nargs=2, default=[320, 384], metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--batch-size', type=int, help='Defaults to 4 for training or the saved batch size for evaluation')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--gradient-weight', type=float, help='Blend weight in [0,1]; new runs default to 0 (Smooth L1 only)')
    parser.add_argument('--seed', type=int, default=174)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', type=Path, help='Resume a training checkpoint in the same output directory')
    parser.add_argument('--checkpoint-every', type=int, default=100, help='Save recovery.pt every N batches')
    parser.add_argument('--exclude-ids', nargs='+', help='Explicit pair exclusions when creating a new manifest')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--limit', type=int, help='Smoke test only: restrict images in each processed split')
    parser.add_argument('--examples', type=int, default=4)
    parser.add_argument('--no-pretrained', action='store_true', help='Random frozen encoder; offline smoke tests only')
    args = parser.parse_args()
    if args.gradient_weight is not None and not 0 <= args.gradient_weight <= 1:
        parser.error('--gradient-weight must be in [0,1]')
    if args.gradient_weight is not None and args.command != 'train':
        parser.error('--gradient-weight applies to train; evaluate uses checkpoint settings')
    if args.checkpoint_every < 1:
        parser.error('--checkpoint-every must be positive')
    if args.resume is not None and args.command != 'train':
        parser.error('--resume is only supported for train')
    if (args.batch_size is not None and args.batch_size < 1) or args.epochs < 1 or min(args.size) < 32 or (args.limit is not None and args.limit < 1):
        parser.error('Batch size, epochs and limit must be positive; dimensions must be >=32')
    if not 0 < args.min_depth < args.max_depth or args.learning_rate <= 0 or args.examples < 0:
        parser.error('Require 0 < min-depth < max-depth, positive learning rate and nonnegative examples')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = load_manifest(args.data, args.manifest, args.seed, args.val_fraction, args.exclude_ids)
    print('Split sizes:', {s: len(manifest[s]) for s in ('train', 'validation', 'test')}, flush=True)
    config = {'size': args.size, 'batch_size': args.batch_size or 4, 'depth_encoding': args.depth_encoding,
              'min_depth': args.min_depth, 'max_depth': args.max_depth, 'seed': args.seed}
    if args.command == 'inspect':
        audit_manifest(manifest, config['depth_encoding'], args.output / 'integrity.json')
        report = {'counts': {s: len(manifest[s]) for s in ('train', 'validation', 'test')},
                  'config': config, 'split_policy': manifest['split_policy'], 'samples': []}
        for split in ('train', 'validation', 'test'):
            for record in manifest[split][:3]:
                with Image.open(record['depth']) as image:
                    raw = np.array(image)
                depth = decode_depth(raw, args.depth_encoding)
                valid = np.isfinite(depth) & (depth >= args.min_depth) & (depth <= args.max_depth)
                report['samples'].append({'split': split, 'id': record['id'], 'dtype': str(raw.dtype),
                    'raw_range': [int(raw.min()), int(raw.max())], 'depth_percentiles_m': np.percentile(depth, [0, 50, 99, 100]).tolist(),
                    'valid_fraction': float(valid.mean())})
        write_json(args.output / 'data_inspection.json', report)
        print(json.dumps(report, indent=2))
        return
    device = select_device(args.device)
    device = torch.device(device)
    print('Device:', device, flush=True)
    if args.command == 'evaluate':
        if args.checkpoint is None:
            parser.error('evaluate requires --checkpoint')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if checkpoint['manifest_hash'] != manifest_hash(manifest):
            raise ValueError('Checkpoint split manifest differs; supply its original --manifest')
        config = checkpoint['config'] | {'batch_size': args.batch_size or checkpoint['config']['batch_size']}
        audit_manifest(manifest, config['depth_encoding'], args.output / 'integrity.json', (args.split,))
        model = Model.from_config(config).to(device)
        model.load_state_dict(checkpoint['model'])
        if config.get('evaluation_protocol') == PROTOCOL:
            records = manifest[args.split][:args.limit] if args.limit else manifest[args.split]
            result = evaluate_canonical(model, records, config, device, args.output / args.split, args.examples)
        else:
            loader = make_loader(manifest[args.split], config, limit=args.limit)
            result = evaluate(model, loader, device, config, args.output / args.split, args.examples)
        result.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_epoch=checkpoint['epoch'],
                      manifest_hash=checkpoint['manifest_hash'], split=args.split, smoke_test=checkpoint['smoke_test'] or args.limit is not None)
        write_json(args.output / f'{args.split}_metrics.json', result)
        print(json.dumps(result, indent=2))
        return
    if args.checkpoint is not None:
        parser.error('Use --resume for training; --checkpoint is for evaluate')
    resume = None
    if args.resume is not None:
        if args.resume.resolve().parent != args.output.resolve():
            parser.error('--resume must belong to the --output directory')
        resume = torch.load(args.resume, map_location='cpu', weights_only=True)
        if resume['manifest_hash'] != manifest_hash(manifest):
            raise ValueError('Resume manifest differs from checkpoint; use its saved splits.json')
        config = resume['config']
        if resume.get('training_workflow') in {'refine', 'v2'}:
            parser.error('Resume this checkpoint with its original refinement/v2 training workflow')
        if args.gradient_weight is not None and args.gradient_weight != config.get('gradient_weight', 0.0):
            parser.error('--resume preserves the checkpoint loss; start a new output for a different loss')
        args.limit = resume.get('limit')
        if resume['smoke_test'] and args.limit is None and 'limit' not in resume:
            raise ValueError('Legacy smoke checkpoint lacks subset metadata and cannot safely resume')
    elif any((args.output / name).exists() for name in ('latest.pt', 'best.pt', 'recovery.pt')):
        raise ValueError('Output contains checkpoints; use --resume or a new --output directory')
    else:
        config.update(pretrained=not args.no_pretrained, learning_rate=args.learning_rate,
                      gradient_weight=args.gradient_weight if args.gradient_weight is not None else 0.0)
    audit_manifest(manifest, config['depth_encoding'], args.output / 'integrity.json', ('train', 'validation'))
    if not resume and config['pretrained']:
        configure_https_certificates()
    model = Model(pretrained=config['pretrained'] and resume is None).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config['learning_rate'])
    records = manifest['train'][:args.limit] if args.limit is not None else manifest['train']
    valloader = make_loader(manifest['validation'], config, limit=args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'config.json', config)
    write_json(args.output / 'splits.json', manifest)
    history, best, start_epoch, start_batch = [], float('inf'), 1, 0
    total, count = 0.0, 0
    gradient_total, edge_count = 0.0, 0
    if resume:
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        history = resume.get('history', [])
        best = resume.get('best_loss', (resume.get('validation') or {}).get('smooth_l1', float('inf')))
        complete = resume.get('epoch_complete', True)
        start_epoch = resume['epoch'] + int(complete)
        if not complete:
            start_batch = resume['next_batch']
            total, count = resume['train_loss_sum'], resume['train_valid_pixels']
            gradient_total, edge_count = resume.get('train_gradient_sum', 0.0), resume.get('train_valid_edges', 0)
        if 'torch_rng_state' in resume:
            torch.set_rng_state(resume['torch_rng_state'])
        print(f'Resuming epoch {start_epoch}, next batch {start_batch + 1}; using saved configuration', flush=True)
    if start_epoch > args.epochs:
        raise ValueError('--epochs is the total target epoch count; increase it to continue this checkpoint')
    smoke_test = resume['smoke_test'] if resume else args.limit is not None or args.no_pretrained
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        order = epoch_records(records, config['seed'], epoch)
        batches = (len(order) + config['batch_size'] - 1) // config['batch_size']
        trainloader = make_loader(order[start_batch * config['batch_size']:], config)
        def snapshot(next_batch, complete=False, validation=None):
            return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                    'epoch_complete': complete, 'next_batch': next_batch, 'train_loss_sum': total,
                    'train_valid_pixels': count, 'train_gradient_sum': gradient_total,
                    'train_valid_edges': edge_count, 'config': config, 'manifest_hash': manifest_hash(manifest),
                    'smoke_test': smoke_test, 'limit': args.limit, 'history': history, 'best_loss': best,
                    'validation': validation, 'torch_rng_state': torch.get_rng_state()}
        for step, (rgb, target, mask) in enumerate(trainloader, start_batch + 1):
            rgb, target, mask = rgb.to(device), target.to(device), mask.to(device)
            n = int(mask.sum().item())
            if n:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(rgb)
                loss, smooth_l1, gradient_sum, edges = depth_loss(prediction, target, mask, config.get('gradient_weight', 0.0))
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')
                loss.backward()
                optimizer.step()
                total += smooth_l1.item() * n
                count += n
                gradient_total += gradient_sum.item()
                edge_count += edges
            if step == 1 or step % 50 == 0:
                mean_l1 = total / count if count else 0.0
                mean_grad = gradient_total / edge_count if edge_count else 0.0
                weight = config.get('gradient_weight', 0.0)
                mean_loss = (1 - weight) * mean_l1 + weight * mean_grad
                print(f'Epoch {epoch} batch {step}/{batches} loss {mean_loss:.5f} smooth_l1 {mean_l1:.5f} gradient {mean_grad:.5f}', flush=True)
            if step % args.checkpoint_every == 0:
                atomic_save(snapshot(step), args.output / 'recovery.pt')
        if not count:
            raise ValueError('Training epoch contains no valid target pixels')
        atomic_save(snapshot(batches), args.output / 'recovery.pt')
        validation = evaluate(model, valloader, device, config)
        mean_grad = gradient_total / edge_count if edge_count else 0.0
        weight = config.get('gradient_weight', 0.0)
        history.append({'epoch': epoch, 'train_smooth_l1': total / count, 'train_gradient_loss': mean_grad,
                        'train_total_loss': (1 - weight) * total / count + weight * mean_grad,
                        'train_valid_edges': edge_count, 'validation': validation})
        improved = validation['smooth_l1'] < best
        if improved:
            best = validation['smooth_l1']
            atomic_save(snapshot(batches, True, validation), args.output / 'best.pt')
        atomic_save(snapshot(batches, True, validation), args.output / 'latest.pt')
        write_json(args.output / 'history.json', history)
        print(f'Epoch {epoch}: train={total/count:.5f}, validation={validation["smooth_l1"]:.5f}', flush=True)
        start_batch, total, count = 0, 0.0, 0
        gradient_total, edge_count = 0.0, 0
    payload = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(payload['model'])
    result = evaluate(model, valloader, device, config, args.output / 'validation', args.examples)
    result.update(checkpoint=str((args.output / 'best.pt').resolve()), checkpoint_epoch=payload['epoch'],
                  manifest_hash=payload['manifest_hash'], split='validation', smoke_test=payload['smoke_test'])
    write_json(args.output / 'validation_metrics.json', result)
    print('Saved best.pt, latest.pt, history.json, validation_metrics.json and example panels.', flush=True)


if __name__ == '__main__':
    main()
