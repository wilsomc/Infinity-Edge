import json
from pathlib import Path


def write_v2_report(output, summary):
    output=Path(output)
    candidate,reference=summary['validation'],summary['reference']
    history_path=output/'history.json'
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    config=json.loads((output/'config.json').read_text())
    def fmt(value,percent=False):
        return 'n/a' if value is None else f'{value*100:.2f}%' if percent else f'{value:.4f}'
    lines=['# High-resolution staged training', '',
        f"Architecture: {config['architecture']}; input {config['size']}; resize {config['resize_mode']}; batch {config['batch_size']}.",
        f"Completed epochs: {candidate['completed_epochs']}; best stage epoch: {candidate['checkpoint_epoch']}; stop reason: {candidate['stop_reason']}.",
        f"Selected as the local prediction default: {summary['promoted_to_default']}.", '',
        '## Comparable validation results', '',
        'Each model used its own recorded input preprocessing. Predictions were restored to the same 320×384 target grid, with identical ground-truth masks, range and global valid-pixel aggregation. Padding was removed before restoration. Raw validation Smooth L1 selected checkpoints; test scores were not used for selection.', '',
        '| Metric | Previous model | New candidate |', '| --- | ---: | ---: |']
    for key,label in [('mae_m','MAE (m)'),('rmse_m','RMSE (m)'),('abs_rel','AbsRel'),('delta_1','δ1'),
                      ('smooth_l1','Raw Smooth L1'),('depth_edge_f1','Depth-edge F1'),('boundary_gradient_mae_m','Boundary gradient MAE')]:
        lines.append(f"| {label} | {fmt(reference.get(key),key=='delta_1')} | {fmt(candidate.get(key),key=='delta_1')} |")
    lines+=['', 'Depth-edge metrics detect adjacent valid target depth jumps ≥0.1 m on the canonical grid. Matches are at the exact grid location. They are diagnostics, not a standard benchmark or proof that every thin object is recovered.', '',
        '## Training setup', '',
        f"Horizontal-flip and RGB-only jitter settings: {config['augmentation']}.",
        f"Frozen warm-up: {config['warmup_epochs']} epochs; layer4-only: {config['layer4_epochs']} epochs; then layer3+layer4. Encoder BatchNorm statistics stay fixed.",
        f"Decoder LR: {config['warmup_learning_rate']} during warm-up, at most {config['learning_rate']} thereafter; encoder LR {config['encoder_learning_rate']}. Plateau LR reduction and patience {config['patience']} govern continued training.",
        f"Gradient-loss weight: {config['gradient_weight']}. Augmentation is deterministic per record, epoch and seed for recovery.", '',
        '## Limits', '',
        '- Source checkpoint and manifest fingerprints are recorded in config.json. The existing damaged training pair remains excluded.',
        '- The same image-level split was reused; scene independence and internet-photo accuracy remain unverified.',
        '- This is a single-seed experiment. Test data were observed in earlier experiments; follow-up results are exploratory.',
        '- A smoke-test run is never promoted. Architecture/preprocessing changes together do not identify the contribution of each change; CLI switches support separate experiments.',
        '- MiDaS comparisons involving v2 use canonical_original_inputs_v2 with each model’s own preprocessing. Keep oracle-aligned relative-depth scores separate from meter-depth inference scores.']
    test_path=output/'test_metrics.json'
    if test_path.exists():
        test=json.loads(test_path.read_text())
        lines+=['', '## Selected-model test evaluation', '', f"Checkpoint: {test['checkpoint']}",
                f"Images: {test['images']}; MAE: {test['mae_m']:.4f} m; RMSE: {test['rmse_m']:.4f} m; δ1: {100*test['delta_1']:.2f}%."]
    lines+=['', f"Elapsed minutes for this invocation: {summary['elapsed_minutes']:.1f}."]
    (output/'training_report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    initial=json.loads((output/'initial_validation.json').read_text())
    x=[0]+[row['epoch'] for row in history]
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,key,label in zip(axes,['mae_m','depth_edge_f1'],['Validation MAE (m)','Validation depth-edge F1']):
        ax.plot(x,[initial.get(key)]+[row['validation'].get(key) for row in history],marker='o',label='New model')
        if reference.get(key) is not None:ax.axhline(reference[key],linestyle='--',color='gray',label='Previous model')
        for boundary in [config['warmup_epochs']+.5,config['warmup_epochs']+config['layer4_epochs']+.5]:
            if x[-1]>=boundary:ax.axvline(boundary,linestyle=':',color='silver')
        ax.set(xlabel='New stage epoch',ylabel=label)
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.savefig(output/'validation_curves.png',dpi=170)
    plt.close(fig)
