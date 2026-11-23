import argparse
import copy
import json
import os
import sys
import shutil
import importlib.metadata
import time
from pathlib import Path

import torch

from model.runtime import file_hash, select_device
from dataloader.dataloader import audit_manifest, load_manifest
from model.canonical_evaluation import evaluate_canonical, PROTOCOL
from model.model import Model, transfer_checkpoint
from model.staged_training import train_staged
from script.CNN import ROOT, manifest_hash, write_json
from script.predict import default_checkpoint
from script.v2_report import write_v2_report


def promote_if_better(candidate, reference, checkpoint, registry):
    for key in ['evaluation_protocol','manifest_hash','images','valid_pixels','min_depth','max_depth']:
        if candidate.get(key) != reference.get(key):
            raise ValueError(f'Cannot compare validation results: {key} differs')
    if candidate.get('evaluation_protocol') != PROTOCOL:
        raise ValueError('Default selection requires canonical evaluation')
    if candidate.get('smoke_test', True) or candidate['smooth_l1'] >= reference['smooth_l1']:
        return False
    data = {'checkpoint':str(Path(checkpoint).resolve()), 'sha256':file_hash(checkpoint),
            'smoke_test':False, 'selected_on_validation':True, 'evaluation_protocol':PROTOCOL,
            'validation':candidate, 'previous_reference':reference}
    registry = Path(registry)
    temporary = registry.with_suffix('.tmp')
    write_json(temporary, data)
    temporary.replace(registry)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=default_checkpoint(),help='Trained source checkpoint for initialization')
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--manifest',type=Path,default=ROOT/'results/baseline/splits.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/highres_v2')
    parser.add_argument('--architecture',choices=['v1','v2'],default='v2')
    parser.add_argument('--size',type=int,nargs=2,default=[480,640],metavar=('HEIGHT','WIDTH'))
    parser.add_argument('--resize-mode',choices=['stretch','letterbox'],default='letterbox')
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=30,help='Total stage epochs; can be increased on resume')
    parser.add_argument('--hours',type=float,default=7.5,help='Wall-time allowance for training/preflight; final evaluation may add time')
    parser.add_argument('--warmup-epochs',type=int,default=3)
    parser.add_argument('--layer4-epochs',type=int,default=3)
    parser.add_argument('--warmup-lr',type=float,default=3e-4)
    parser.add_argument('--decoder-lr',type=float,default=1e-4)
    parser.add_argument('--encoder-lr',type=float,default=1e-5)
    parser.add_argument('--gradient-weight',type=float,default=.1)
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--lr-patience',type=int,default=2)
    parser.add_argument('--no-augmentation',action='store_true')
    parser.add_argument('--seed',type=int,default=174)
    parser.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    parser.add_argument('--resume',action='store_true',help='Restore recovery.pt and its saved configuration')
    parser.add_argument('--limit',type=int,help='Smoke test only: limit each processed split')
    parser.add_argument('--examples',type=int,default=4)
    parser.add_argument('--evaluate-test',action='store_true',help='Test the validation-selected model after training')
    args=parser.parse_args()
    if args.epochs<1 or args.hours<=0 or args.batch_size<1 or min(args.size)<32:
        parser.error('Use positive epochs, hours, batch size, and input dimensions >=32')
    if any(x%32 for x in args.size):
        parser.error('Input dimensions must be multiples of 32')
    if min(args.warmup_lr,args.decoder_lr,args.encoder_lr)<=0 or not 0<=args.gradient_weight<=1:
        parser.error('Require positive learning rates and gradient weight in [0,1]')
    if min(args.warmup_epochs,args.layer4_epochs,args.lr_patience,args.examples)<0 or args.patience<1 or (args.limit is not None and args.limit<1):
        parser.error('Invalid stage lengths, patience, example count, or smoke-test limit')
    started=time.monotonic()
    deadline=started+args.hours*3600
    manifest=load_manifest(args.data,args.manifest)
    fingerprint=manifest_hash(manifest)
    print('Split sizes:',{s:len(manifest[s]) for s in ['train','validation','test']},flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        saved=torch.load(args.output/'recovery.pt',map_location='cpu',weights_only=True)
        if saved.get('training_workflow')!='v2' or saved['manifest_hash']!=fingerprint:
            parser.error('Recovery checkpoint does not belong to this workflow/manifest')
        config=saved['config']
        reference=json.loads((args.output/'reference_validation.json').read_text())
        model=Model.from_config(config)
        model.load_state_dict(saved['model'])
        del saved
    else:
        if any((args.output/f).exists() for f in ['best.pt','recovery.pt','config.json']):
            parser.error('Output already contains a run; use --resume or another --output')
        source=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        if source['manifest_hash']!=fingerprint:
            parser.error('Initialization checkpoint uses a different manifest')
        if source['smoke_test'] and args.limit is None:
            parser.error('A smoke-test source requires --limit; it cannot produce a promoted model')
        config=copy.deepcopy(source['config'])
        config.update(architecture=args.architecture,size=args.size,resize_mode=args.resize_mode,encoder_training='staged',
            batch_size=args.batch_size,seed=args.seed,gradient_weight=args.gradient_weight,
            warmup_epochs=args.warmup_epochs,layer4_epochs=args.layer4_epochs,
            warmup_learning_rate=args.warmup_lr,learning_rate=args.decoder_lr,encoder_learning_rate=args.encoder_lr,
            patience=args.patience,lr_patience=args.lr_patience,evaluation_protocol=PROTOCOL,
            augmentation={} if args.no_augmentation else {'horizontal_flip':.5,'brightness':.1,'contrast':.1,'saturation':.1},
            limit=args.limit,smoke_test=args.limit is not None or source['smoke_test'],
            initialization={'checkpoint':str(args.checkpoint.resolve()),'sha256':file_hash(args.checkpoint),
                            'epoch':source['epoch'],'source_config':source['config']})
        torch.manual_seed(config['seed'])
        model=Model.from_config(config)
        transfer=transfer_checkpoint(model,source)
        write_json(args.output/'transfer.json',transfer)
        write_json(args.output/'reference_config.json',source['config'])
        reference=None
    write_json(args.output/'splits.json',manifest)
    write_json(args.output/'run_info.json', {'pid':os.getpid(),'started_unix':time.time(),
        'hours':args.hours,'target_epochs':args.epochs,'command':sys.argv,'resume':args.resume})
    if not args.resume:
        sources=[path for directory in ['model','script','dataloader'] for path in (ROOT/directory).glob('*.py')]
        for path in sources:
            target=args.output/'source_snapshot'/path.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
        write_json(args.output/'source_snapshot/hashes.json',{str(p.relative_to(ROOT)):file_hash(p) for p in sources})
        write_json(args.output/'environment.json', {name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy','Pillow','matplotlib']})
    selected_records={**manifest}
    if config.get('limit'):
        for split in ['train','validation','test']:
            selected_records[split]=manifest[split][:config['limit']]
    audit_manifest(selected_records,config['depth_encoding'],args.output/'integrity.json',('train','validation'))
    device=select_device(args.device)
    print('Device:',device,'input:',config['size'],'resize:',config['resize_mode'],flush=True)
    if reference is None:
        previous=Model.from_config(source['config']).to(device).eval()
        previous.load_state_dict(source['model'])
        reference=evaluate_canonical(previous,selected_records['validation'],source['config'],device)
        reference.update(manifest_hash=fingerprint,checkpoint=str(args.checkpoint.resolve()),smoke_test=config['smoke_test'])
        write_json(args.output/'reference_validation.json',reference)
        del previous,source
        if device=='mps':torch.mps.empty_cache()
    result=train_staged(model,manifest,config,args.output,args.epochs,deadline,device,
                        resume=args.resume,examples=args.examples)
    del model
    if device=='mps':torch.mps.empty_cache()
    promoted=promote_if_better(result,reference,args.output/'best.pt',ROOT/'results/selected_model.json')
    summary={'validation':result,'reference':reference,'promoted_to_default':promoted,
        'elapsed_minutes':(time.monotonic()-started)/60,'device':device,
        'limitations':'Single seeded image-level split; scene independence and internet-photo accuracy unverified. Previously observed test is not used for selection.'}
    write_json(args.output/'summary.json',summary)
    print('V2 validation:',result,'promoted:',promoted,flush=True)
    if args.evaluate_test:
        chosen=args.output/'best.pt' if promoted else Path(reference['checkpoint'])
        checkpoint=torch.load(chosen,map_location='cpu',weights_only=True)
        chosen_model=Model.from_config(checkpoint['config']).to(device).eval()
        chosen_model.load_state_dict(checkpoint['model'])
        audit_manifest(selected_records,config['depth_encoding'],args.output/'test_integrity.json',('test',))
        test=evaluate_canonical(chosen_model,selected_records['test'],checkpoint['config'],device,args.output/'test',args.examples)
        test.update(checkpoint=str(chosen.resolve()),manifest_hash=fingerprint,split='test',smoke_test=config['smoke_test'],checkpoint_epoch=checkpoint['epoch'])
        write_json(args.output/'test_metrics.json',test)
        print('Selected model test:',test,flush=True)
    summary['elapsed_minutes']=(time.monotonic()-started)/60
    write_json(args.output/'summary.json',summary)
    write_v2_report(args.output,summary)


if __name__=='__main__':
    main()
