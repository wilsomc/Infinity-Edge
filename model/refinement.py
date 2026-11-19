"""Warm-start training with partial encoder tuning, scheduling, and recovery."""
import copy
import time
from pathlib import Path

import torch

from dataloader.dataloader import make_loader
from model.evaluation import evaluate
from model.losses import depth_loss
from model.model import Model
from script.CNN import atomic_save, epoch_records, manifest_hash, write_json


def refine_stage(source, manifest, settings, output, epochs, deadline, initial_validation,
                 initialization, resume=False, checkpoint_every=100, device='cpu', examples=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(source['config'])
    config.update(settings)
    config['initialization'] = initialization
    fingerprint = manifest_hash(manifest)
    if source['manifest_hash'] != fingerprint:
        raise ValueError('Initialization checkpoint and manifest differ')
    model = Model.from_config(config)
    model.load_state_dict(source['model'])
    model.configure_encoder(config['encoder_training'])
    model.to(device)
    optimizer = torch.optim.Adam(model.training_parameter_groups(config['learning_rate'], config['encoder_learning_rate']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=1, min_lr=1e-6)
    valloader = make_loader(manifest['validation'], config)
    history, epoch, next_batch = [], 1, 0
    total, pixels, gradient_sum, edges = 0., 0, 0., 0
    best = initial_validation['smooth_l1']
    stale = 0
    initial = copy.deepcopy(initial_validation)
    weight = config['gradient_weight']
    initial.update(gradient_weight=weight, total_loss=(1-weight)*initial['smooth_l1']+weight*initial['gradient_loss'])

    def snapshot(epoch_number, batch, complete, validation=None):
        return {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(), 'config':config, 'epoch':epoch_number,
            'epoch_complete':complete, 'next_batch':batch, 'train_loss_sum':total,
            'train_valid_pixels':pixels, 'train_gradient_sum':gradient_sum, 'train_valid_edges':edges,
            'history':history, 'best_loss':best, 'stale_epochs':stale,
            'manifest_hash':fingerprint, 'validation':validation, 'smoke_test':source['smoke_test'],
            'limit':source.get('limit'), 'torch_rng_state':torch.get_rng_state(), 'training_workflow':'refine'}

    if resume:
        recovery = output/'recovery.pt'
        if not recovery.exists():
            recovery = output/'latest.pt'
        saved = torch.load(recovery, map_location='cpu', weights_only=True)
        if saved['config'] != config or saved['manifest_hash'] != fingerprint:
            raise ValueError('Recovery settings/manifest do not match the experiment plan')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history, best, stale = saved['history'], saved['best_loss'], saved['stale_epochs']
        epoch = saved['epoch'] + int(saved['epoch_complete'])
        if not saved['epoch_complete']:
            next_batch = saved['next_batch']
            total, pixels = saved['train_loss_sum'], saved['train_valid_pixels']
            gradient_sum, edges = saved['train_gradient_sum'], saved['train_valid_edges']
        torch.set_rng_state(saved['torch_rng_state'])
    else:
        if any((output/f).exists() for f in ['best.pt','latest.pt','recovery.pt']):
            raise ValueError(f'Experiment exists: {output}; use --resume')
        write_json(output/'config.json', config)
        write_json(output/'splits.json', manifest)
        atomic_save(snapshot(0, 0, True, initial), output/'best.pt')

    stop_reason = 'epoch_budget'
    while epoch <= epochs:
        model.train()
        ordered = epoch_records(manifest['train'], config['seed'], epoch)
        batches = (len(ordered)+config['batch_size']-1)//config['batch_size']
        loader = make_loader(ordered[next_batch*config['batch_size']:], config)
        timed_out = False
        for step, (rgb, target, mask) in enumerate(loader, next_batch+1):
            if time.monotonic() >= deadline:
                timed_out = True
                break
            rgb, target, mask = rgb.to(device), target.to(device), mask.to(device)
            n = int(mask.sum().item())
            if n:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(rgb)
                loss, smooth, grad_sum, valid_edges = depth_loss(prediction, target, mask, weight)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite fine-tuning loss')
                loss.backward()
                optimizer.step()
                total += smooth.item()*n
                pixels += n
                gradient_sum += grad_sum.item()
                edges += valid_edges
            next_batch = step
            if step == 1 or step % 100 == 0:
                print(f'{output.name}: epoch {epoch}/{epochs} batch {step}/{batches} Smooth L1 {total/pixels if pixels else 0:.5f}', flush=True)
            if step % checkpoint_every == 0:
                atomic_save(snapshot(epoch, next_batch, False), output/'recovery.pt')
        if not pixels and not timed_out:
            raise ValueError('No valid training pixels')
        atomic_save(snapshot(epoch, next_batch, False), output/'recovery.pt')
        if timed_out:
            # Keep incomplete weights for resumption; do not rank partially evaluated candidates.
            stop_reason = 'time_budget'
            break
        validation = evaluate(model, valloader, device, config)
        improved = validation['smooth_l1'] < best
        if improved:
            best, stale = validation['smooth_l1'], 0
        else:
            stale += 1
        rates = {g['name']:g['lr'] for g in optimizer.param_groups}
        scheduler.step(validation['smooth_l1'])
        history.append({'epoch':epoch, 'train_smooth_l1':total/pixels,
            'train_gradient_loss':gradient_sum/edges if edges else 0.,
            'learning_rates':rates, 'validation':validation})
        if improved:
            atomic_save(snapshot(epoch, batches, True, validation), output/'best.pt')
        payload = snapshot(epoch, batches, True, validation)
        atomic_save(payload, output/'latest.pt')
        atomic_save(payload, output/'recovery.pt')
        write_json(output/'history.json', history)
        print(f'{output.name}: epoch {epoch} validation Smooth L1={validation["smooth_l1"]:.5f}, MAE={validation["mae_m"]:.4f} m, best={best:.5f}', flush=True)
        epoch += 1
        next_batch, total, pixels, gradient_sum, edges = 0, 0., 0, 0., 0
        if stale >= 3:
            stop_reason = 'early_stopping'
            break
    selected = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
    # The selected best already has a full validation pass. Avoid repeating it unless saving panels.
    result = copy.deepcopy(selected['validation'])
    if examples and time.monotonic() < deadline:
        model.load_state_dict(selected['model'])
        panel_records = manifest['validation'][:examples]
        evaluate(model, make_loader(panel_records, config), device, config, output/'validation', examples)
    result.update(checkpoint=str((output/'best.pt').resolve()), checkpoint_epoch=selected['epoch'],
        manifest_hash=fingerprint, split='validation', smoke_test=source['smoke_test'],
        completed_epochs=len(history), stop_reason=stop_reason, initialization=initialization)
    write_json(output/'validation_metrics.json', result)
    return result
