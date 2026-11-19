"""Compare models with different input geometry on one unchanged target grid."""
from pathlib import Path

import torch

from dataloader.custom_dataset import load_pair
from dataloader.preprocessing import prepare_rgb, prepare_depth, restore_prediction, spatial_geometry
from model.evaluation import DepthMetrics, save_panel

PROTOCOL = 'canonical_320x384_stretch_v1'
EVALUATION_SIZE = (320, 384)


def evaluate_canonical(model, records, config, device, output_dir=None, examples=4):
    model.eval()
    metrics = DepthMetrics(config['min_depth'], config['max_depth'])
    batch_size = config['batch_size']
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start:start+batch_size]
            inputs, geometries, targets, masks, previews = [], [], [], [], []
            for record in batch:
                image, depth = load_pair(record, config['depth_encoding'])
                rgb, geometry = prepare_rgb(image, config['size'], config.get('resize_mode','stretch'))
                target_geometry = spatial_geometry(depth.shape, EVALUATION_SIZE, 'stretch')
                target, mask = prepare_depth(depth, target_geometry, config['min_depth'], config['max_depth'])
                inputs.append(rgb)
                geometries.append(geometry)
                targets.append(target)
                masks.append(mask)
                if output_dir is not None and start+len(previews) < examples:
                    previews.append(prepare_rgb(image, EVALUATION_SIZE, 'stretch')[0])
            predictions = model(torch.stack(inputs).to(device)).cpu()
            predictions = torch.cat([restore_prediction(predictions[i:i+1], geometry, EVALUATION_SIZE)
                                     for i, geometry in enumerate(geometries)])
            targets, masks = torch.stack(targets), torch.stack(masks)
            metrics.update(predictions, targets, masks)
            if output_dir is not None:
                for i, rgb in enumerate(previews):
                    save_panel(rgb, targets[i], predictions[i], masks[i],
                               Path(output_dir)/f"{batch[i]['id']}.png", config['min_depth'], config['max_depth'])
    result = metrics.compute()
    weight = config.get('gradient_weight',0.)
    result.update(images=len(records), units='meters', aggregation='global valid pixels',
                  evaluation_protocol=PROTOCOL, evaluation_resolution=list(EVALUATION_SIZE),
                  input_resolution=config['size'], resize_mode=config.get('resize_mode','stretch'),
                  architecture=config.get('architecture','v1'), depth_encoding=config['depth_encoding'],
                  min_depth=config['min_depth'], max_depth=config['max_depth'], gradient_weight=weight,
                  total_loss=(1-weight)*result['smooth_l1']+weight*result['gradient_loss'],
                  prediction_policy='metrics use clamped depth; loss and boundary metrics use raw output',
                  boundary_policy='adjacent valid target depth jumps >=0.1m; exact-grid matches, not RGB edges')
    return result
