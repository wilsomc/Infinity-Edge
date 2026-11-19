"""Bounded validation-selected improvement pass, starting each candidate from one checkpoint."""
import argparse
import copy
import shutil
import time
from pathlib import Path

import torch

from dataloader.dataloader import audit_manifest, load_manifest, make_loader
from model.evaluation import evaluate
from model.inference import file_hash
from model.model import Model
from model.refinement import refine_stage
from script.CNN import ROOT, manifest_hash, write_json

EXPERIMENTS = [
    {'name':'longer_decoder', 'encoder_training':'frozen', 'gradient_weight':0.0},
    {'name':'finetune_layer4', 'encoder_training':'layer4', 'gradient_weight':0.0},
    {'name':'finetune_gradient_01', 'encoder_training':'layer4', 'gradient_weight':0.1},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'results/baseline/best.pt')
    parser.add_argument('--manifest', type=Path, default=ROOT/'results/baseline/splits.json')
    parser.add_argument('--data', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path, default=ROOT/'results/improvement_pass')
    parser.add_argument('--minutes', type=float, default=55, help='Training-pass wall-time budget; final test evaluation may add time')
    parser.add_argument('--epochs', type=int, default=3, help='Maximum additional epochs per candidate')
    parser.add_argument('--decoder-lr', type=float, default=3e-4)
    parser.add_argument('--encoder-lr', type=float, default=3e-5)
    parser.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    parser.add_argument('--resume', action='store_true', help='Continue incomplete experiments in this output directory')
    parser.add_argument('--evaluate-test', action='store_true', help='Evaluate only the validation-selected model on test')
    args = parser.parse_args()
    if args.minutes <= 0 or args.epochs < 1 or args.decoder_lr <= 0 or args.encoder_lr <= 0:
        parser.error('Budgets, epoch count, and learning rates must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'plan.json').exists() and not args.resume:
        parser.error('Improvement pass already exists; choose a new --output or use --resume')
    start = time.monotonic()
    deadline = start + args.minutes*60
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if source['smoke_test']:
        parser.error('Use a full-data trained checkpoint, not a smoke-test checkpoint')
    manifest = load_manifest(args.data, args.manifest)
    if source['manifest_hash'] != manifest_hash(manifest):
        parser.error('Source checkpoint and manifest differ')
    initialization = {'checkpoint':str(args.checkpoint.resolve()), 'sha256':file_hash(args.checkpoint), 'epoch':source['epoch']}
    plan = {'initialization':initialization, 'manifest_hash':source['manifest_hash'],
        'epochs_per_candidate':args.epochs, 'decoder_lr':args.decoder_lr, 'encoder_lr':args.encoder_lr,
        'experiments':EXPERIMENTS, 'selection':'lowest validation raw Smooth L1, including original source checkpoint',
        'early_stopping':'three non-improving epochs', 'scheduler':'ReduceLROnPlateau, factor 0.5, patience 1',
        'normalization':'encoder BatchNorm running statistics fixed during fine-tuning'}
    if args.resume:
        import json
        if not (args.output/'plan.json').exists() or json.loads((args.output/'plan.json').read_text()) != plan:
            parser.error('Saved plan differs; resume with the original settings')
    write_json(args.output/'plan.json', plan)
    write_json(args.output/'splits.json', manifest)
    audit_manifest(manifest, source['config']['depth_encoding'], args.output/'integrity.json', ('train','validation'))
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('Device:',device,flush=True)
    reference = Model.from_config(source['config']).to(device).eval()
    reference.load_state_dict(source['model'])
    initial = evaluate(reference, make_loader(manifest['validation'], source['config']), device, source['config'])
    del reference
    if device == 'mps':
        torch.mps.empty_cache()
    original = copy.deepcopy(initial)
    original.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_epoch=source['epoch'], name='original_baseline')
    candidates = [original]
    write_json(args.output/'initial_validation.json', original)
    statuses = []
    for index, experiment in enumerate(EXPERIMENTS):
        remaining = deadline-time.monotonic()
        if remaining < 90:
            statuses.append({'name':experiment['name'], 'status':'skipped_time_budget'})
            continue
        output = args.output/experiment['name']
        settings = {k:v for k,v in experiment.items() if k != 'name'}
        settings.update(learning_rate=args.decoder_lr, encoder_learning_rate=args.encoder_lr)
        # Reserve equal remaining wall time for each remaining candidate.
        stage_deadline = min(deadline, time.monotonic()+remaining/(len(EXPERIMENTS)-index))
        print(f'Starting {experiment["name"]}; up to {args.epochs} additional epochs',flush=True)
        result = refine_stage(source, manifest, settings, output, args.epochs, stage_deadline,
                              initial, initialization, resume=args.resume and (output/'recovery.pt').exists(), device=device)
        result['name'] = experiment['name']
        candidates.append(result)
        statuses.append({'name':experiment['name'], 'status':result['stop_reason'], 'completed_epochs':result['completed_epochs']})
        write_json(args.output/'progress.json', {'candidates':candidates,'statuses':statuses,'elapsed_minutes':(time.monotonic()-start)/60})
        if device == 'mps':
            torch.mps.empty_cache()
    selected = min(candidates, key=lambda c:c['smooth_l1'])
    # A separate inference checkpoint keeps the old experiment directories intact.
    shutil.copy2(selected['checkpoint'], args.output/'best.pt')
    report = {'plan':plan,'candidates':candidates,'statuses':statuses,'selected':selected,
              'selected_checkpoint':str((args.output/'best.pt').resolve()), 'device':device,
              'elapsed_minutes':(time.monotonic()-start)/60,
              'limitations':'Same image-level split and one seed; test results were previously observed but are not used for selection.'}
    write_json(args.output/'summary.json', report)
    print('Validation-selected model:',selected['name'], 'MAE:',selected['mae_m'],flush=True)
    if args.evaluate_test:
        chosen = torch.load(args.output/'best.pt', map_location='cpu', weights_only=True)
        config = chosen['config']
        audit_manifest(manifest, config['depth_encoding'], args.output/'test_integrity.json', ('test',))
        model = Model.from_config(config).to(device).eval()
        model.load_state_dict(chosen['model'])
        metrics = evaluate(model, make_loader(manifest['test'], config), device, config, args.output/'test', 4)
        metrics.update(checkpoint=str((args.output/'best.pt').resolve()),checkpoint_epoch=chosen['epoch'],
                       selected_experiment=selected['name'], split='test', manifest_hash=source['manifest_hash'], smoke_test=False)
        write_json(args.output/'test_metrics.json', metrics)
        print('Selected model test results:',metrics,flush=True)
    report['elapsed_minutes'] = (time.monotonic()-start)/60
    write_json(args.output/'summary.json',report)


if __name__ == '__main__':
    main()
