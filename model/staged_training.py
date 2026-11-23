"""Staged training of the versioned high-resolution model with exact CPU recovery."""
import copy
import time
from pathlib import Path

import torch

from dataloader.dataloader import make_loader
from model.canonical_evaluation import evaluate_canonical, PROTOCOL
from model.losses import depth_loss
from script.CNN import atomic_save, epoch_records, manifest_hash, write_json


def encoder_stage(epoch, config):
    if epoch <= config['warmup_epochs']:
        return 'frozen'
    if epoch <= config['warmup_epochs']+config['layer4_epochs']:
        return 'layer4'
    return 'layer3_4'


def staged_parameter_groups(model, config):
    # Include both future encoder blocks from the start: Adam retains decoder
    # moments when blocks are unfrozen, and resume always has the same groups.
    decoder, encoder = [], []
    for name, parameter in model.named_parameters():
        if name.startswith(('layer3.', 'layer4.')):
            encoder.append(parameter)
        elif not name.startswith(('conv1.', 'bn1.', 'layer1.', 'layer2.')):
            decoder.append(parameter)
    return [{'params':decoder, 'lr':config['warmup_learning_rate'], 'name':'decoder'},
            {'params':encoder, 'lr':config['encoder_learning_rate'], 'name':'encoder'}]


def train_staged(model, manifest, config, output, epochs, deadline, device='cpu', resume=False,
                 initial_validation=None, checkpoint_every=100, examples=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = manifest_hash(manifest)
    train_records = manifest['train'][:config['limit']] if config.get('limit') else manifest['train']
    validation_records = manifest['validation'][:config['limit']] if config.get('limit') else manifest['validation']
    model.configure_encoder('frozen')
    model.to(device)
    optimizer = torch.optim.Adam(staged_parameter_groups(model, config))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.5, patience=config['lr_patience'], min_lr=[1e-6,1e-7])
    history, epoch, next_batch, stale = [], 1, 0, 0
    total, pixels, gradient_sum, edges = 0., 0, 0., 0
    best = float('inf')
    previous_stage = 'frozen'

    def snapshot(epoch_number, batch, complete, validation=None):
        return {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(), 'config':config, 'epoch':epoch_number,
            'epoch_complete':complete, 'next_batch':batch, 'train_loss_sum':total,
            'train_valid_pixels':pixels, 'train_gradient_sum':gradient_sum, 'train_valid_edges':edges,
            'history':history, 'best_loss':best, 'stale_epochs':stale,
            'encoder_stage':model.encoder_training,
            'manifest_hash':fingerprint, 'validation':validation, 'smoke_test':config['smoke_test'],
            'limit':config.get('limit'), 'torch_rng_state':torch.get_rng_state(), 'training_workflow':'v2'}

    if resume:
        saved = torch.load(output/'recovery.pt', map_location='cpu', weights_only=True)
        if saved['manifest_hash'] != fingerprint or saved['config'] != config:
            raise ValueError('Recovery config/manifest differs from this run')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history, best, stale = saved['history'], saved['best_loss'], saved['stale_epochs']
        previous_stage = saved.get('encoder_stage', 'frozen')
        epoch = saved['epoch']+int(saved['epoch_complete'])
        if not saved['epoch_complete']:
            next_batch = saved['next_batch']
            total, pixels = saved['train_loss_sum'], saved['train_valid_pixels']
            gradient_sum, edges = saved['train_gradient_sum'], saved['train_valid_edges']
        torch.set_rng_state(saved['torch_rng_state'])
        print(f'Resuming stage epoch {epoch}, next batch {next_batch+1}', flush=True)
    else:
        if any((output/f).exists() for f in ['best.pt','latest.pt','recovery.pt']):
            raise ValueError('Checkpoints already exist; use --resume or a new output')
        write_json(output/'config.json', config)
        if initial_validation is None:
            initial_validation = evaluate_canonical(model, validation_records, config, device)
        best = initial_validation['smooth_l1']
        write_json(output/'initial_validation.json', initial_validation)
        atomic_save(snapshot(0,0,True,initial_validation), output/'best.pt')
        scheduler.step(best)
        atomic_save(snapshot(0,0,True,initial_validation), output/'recovery.pt')
    stop_reason = 'epoch_budget'
    while epoch <= epochs:
        if time.monotonic() >= deadline:
            stop_reason='time_budget'
            break
        stage = encoder_stage(epoch, config)
        model.configure_encoder(stage)
        if stage != previous_stage and next_batch == 0:
            stale = 0
            if stage != 'frozen':
                optimizer.param_groups[0]['lr'] = min(optimizer.param_groups[0]['lr'], config['learning_rate'])
        model.train()
        print(f'Epoch {epoch}/{epochs}: encoder={stage}, rates={[g["lr"] for g in optimizer.param_groups]}',flush=True)
        order = epoch_records(train_records, config['seed'], epoch)
        batches=(len(order)+config['batch_size']-1)//config['batch_size']
        loader=make_loader(order[next_batch*config['batch_size']:], config, augment=True, epoch=epoch)
        timed_out=False
        epoch_start=time.monotonic()
        for step,(rgb,target,mask) in enumerate(loader,next_batch+1):
            if time.monotonic() >= deadline:
                timed_out=True
                break
            rgb,target,mask=rgb.to(device),target.to(device),mask.to(device)
            n=int(mask.sum().item())
            if n:
                optimizer.zero_grad(set_to_none=True)
                prediction=model(rgb)
                loss,smooth,grad_sum,valid_edges=depth_loss(prediction,target,mask,config['gradient_weight'])
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')
                loss.backward()
                optimizer.step()
                total+=smooth.item()*n
                pixels+=n
                gradient_sum+=grad_sum.item()
                edges+=valid_edges
            next_batch=step
            if step==1 or step%100==0:
                print(f'Epoch {epoch} batch {step}/{batches}: Smooth L1={total/pixels if pixels else 0:.5f}',flush=True)
                write_json(output/'progress.json', {'status':'training','epoch':epoch,'batch':step,'batches':batches,
                    'encoder_stage':stage,'train_smooth_l1':total/pixels if pixels else None,
                    'best_validation_smooth_l1':best,'completed_epochs':len(history)})
            if step%checkpoint_every==0:
                atomic_save(snapshot(epoch,next_batch,False), output/'recovery.pt')
        atomic_save(snapshot(epoch,next_batch,False), output/'recovery.pt')
        if timed_out:
            stop_reason='time_budget'
            break
        if not pixels:
            raise ValueError('Training epoch has no valid pixels')
        validation=evaluate_canonical(model, validation_records,config,device,
            output/'validation'/f'epoch_{epoch:03d}' if examples else None,examples)
        improved=validation['smooth_l1'] < best
        if improved:
            best,stale=validation['smooth_l1'],0
        else:
            stale+=1
        rates={g['name']:g['lr'] for g in optimizer.param_groups}
        scheduler.step(validation['smooth_l1'])
        history.append({'epoch':epoch,'encoder_stage':stage,'train_smooth_l1':total/pixels,
            'train_gradient_loss':gradient_sum/edges if edges else 0.,'learning_rates':rates,
            'validation':validation,'elapsed_seconds':time.monotonic()-epoch_start})
        if improved:
            atomic_save(snapshot(epoch,batches,True,validation),output/'best.pt')
        state=snapshot(epoch,batches,True,validation)
        atomic_save(state,output/'latest.pt')
        atomic_save(state,output/'recovery.pt')
        write_json(output/'history.json',history)
        print(f'Epoch {epoch}: canonical validation MAE={validation["mae_m"]:.4f} m, '
              f'RMSE={validation["rmse_m"]:.4f} m, Smooth L1={validation["smooth_l1"]:.5f}, '
              f'edge F1={validation["depth_edge_f1"]}; best={best:.5f}',flush=True)
        previous_stage=stage
        epoch+=1
        next_batch,total,pixels,gradient_sum,edges=0,0.,0,0.,0
        if stale>=config['patience']:
            stop_reason='early_stopping'
            break
    selected=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
    result=copy.deepcopy(selected['validation'])
    result.update(checkpoint=str((output/'best.pt').resolve()),checkpoint_epoch=selected['epoch'],
        manifest_hash=fingerprint,split='validation',smoke_test=config['smoke_test'],
        completed_epochs=len(history),stop_reason=stop_reason,initialization=config.get('initialization'))
    write_json(output/'validation_metrics.json',result)
    write_json(output/'progress.json', {'status':stop_reason,'completed_epochs':len(history),'best_epoch':selected['epoch']})
    write_json(output/'status.json',{'stop_reason':stop_reason,'completed_epochs':len(history),
        'best_epoch':selected['epoch'],'evaluation_protocol':PROTOCOL})
    return result
