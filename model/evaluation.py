"""Dataset-wide, valid-pixel depth metrics and fixed qualitative examples."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.losses import gradient_error_sum, depth_edge_statistics


class DepthMetrics:
    def __init__(self, min_depth=0.001, max_depth=8.0, edge_threshold=0.1):
        self.min_depth, self.max_depth = min_depth, max_depth
        self.count = self.clipped = 0
        self.absolute = self.squared = self.relative = self.smooth_l1 = 0.0
        self.hits = [0, 0, 0]
        self.gradient_sum, self.valid_edges = 0.0, 0
        self.edge_threshold = edge_threshold
        self.edge_tp = self.edge_predicted = self.edge_target = 0
        self.boundary_error = 0.

    def update(self, prediction, target, mask):
        valid = mask.bool() & torch.isfinite(target) & (target >= self.min_depth) & (target <= self.max_depth)
        p, d = prediction[valid].double(), target[valid].double()
        if not torch.isfinite(p).all():
            raise ValueError('Non-finite model prediction on a valid target pixel')
        self.count += d.numel()
        if not d.numel():
            return
        if prediction.ndim >= 2:
            gradient_sum, edges = gradient_error_sum(prediction.double(), target.double(), valid)
            self.gradient_sum += gradient_sum.item()
            self.valid_edges += edges
            boundary = depth_edge_statistics(prediction.double(), target.double(), valid, self.edge_threshold)
            self.edge_tp += boundary['true_positive']
            self.edge_predicted += boundary['predicted']
            self.edge_target += boundary['target']
            self.boundary_error += boundary['boundary_error_sum']
        # Validation loss uses raw outputs, exactly as training does.
        self.smooth_l1 += F.smooth_l1_loss(p, d, reduction='sum').item()
        self.clipped += ((p < self.min_depth) | (p > self.max_depth)).sum().item()
        p = p.clamp(self.min_depth, self.max_depth)
        error = (p - d).abs()
        self.absolute += error.sum().item()
        self.squared += error.square().sum().item()
        self.relative += (error / d).sum().item()
        ratio = torch.maximum(p / d, d / p)
        for k in range(3):
            self.hits[k] += (ratio < 1.25 ** (k + 1)).sum().item()

    def compute(self):
        if self.count == 0:
            raise ValueError('Evaluation contains no valid depth pixels')
        return {'mae_m': self.absolute / self.count, 'rmse_m': (self.squared / self.count) ** 0.5,
                'abs_rel': self.relative / self.count,
                **{f'delta_{k+1}': self.hits[k] / self.count for k in range(3)},
                'smooth_l1': self.smooth_l1 / self.count, 'valid_pixels': self.count,
                'gradient_loss': self.gradient_sum / self.valid_edges if self.valid_edges else 0.0,
                'valid_edges': self.valid_edges,
                'edge_threshold_m': self.edge_threshold,
                'depth_edge_precision': self.edge_tp/self.edge_predicted if self.edge_predicted else None,
                'depth_edge_recall': self.edge_tp/self.edge_target if self.edge_target else None,
                'depth_edge_f1': 2*self.edge_tp/(self.edge_predicted+self.edge_target) if self.edge_predicted+self.edge_target else None,
                'target_boundary_edges': self.edge_target,
                'boundary_gradient_mae_m': self.boundary_error/self.edge_target if self.edge_target else None,
                'clipped_prediction_fraction': self.clipped / self.count}


def save_panel(rgb, depth, prediction, mask, path, min_depth, max_depth):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    image = rgb.cpu().permute(1, 2, 0).numpy()
    image = np.clip(image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    valid = mask.cpu().squeeze().numpy().astype(bool)
    truth = np.where(valid, depth.cpu().squeeze().numpy(), np.nan)
    pred = prediction.detach().cpu().squeeze().numpy().clip(min_depth, max_depth)
    error = np.where(valid, np.abs(pred - truth), np.nan)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title('RGB')
    for ax, values, title, cmap, low, high in [
        (axes[1], truth, 'Target (m)', 'viridis', min_depth, max_depth),
        (axes[2], pred, 'Prediction (m)', 'viridis', min_depth, max_depth),
        (axes[3], error, 'Absolute error (m)', 'magma', 0, max_depth),
    ]:
        im = ax.imshow(values, cmap=cmap, vmin=low, vmax=high)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.65)
    for ax in axes:
        ax.axis('off')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(model, loader, device, config, output_dir=None, examples=4):
    model.eval()
    metrics = DepthMetrics(config['min_depth'], config['max_depth'])
    seen = 0
    with torch.inference_mode():
        for rgb, target, mask in loader:
            rgb, target, mask = rgb.to(device), target.to(device), mask.to(device)
            prediction = model(rgb)
            # Float64 accumulation on CPU also supports MPS, which lacks float64.
            metrics.update(prediction.cpu(), target.cpu(), mask.cpu())
            if output_dir is not None:
                for i in range(min(rgb.shape[0], max(0, examples - seen))):
                    record = loader.dataset.records[seen + i]
                    save_panel(rgb[i], target[i], prediction[i], mask[i],
                               Path(output_dir) / f"{record['id']}.png", config['min_depth'], config['max_depth'])
            seen += rgb.shape[0]
    result = metrics.compute()
    weight = config.get('gradient_weight', 0.0)
    result['gradient_weight'] = weight
    result['total_loss'] = (1 - weight) * result['smooth_l1'] + weight * result['gradient_loss']
    result.update(images=seen, aggregation='global valid pixels', units='meters',
                  prediction_policy='clamp to configured depth range; raw outputs for loss components',
                  target_policy='exclude non-finite and out-of-range depths',
                  resolution=config['size'], depth_encoding=config['depth_encoding'],
                  min_depth=config['min_depth'], max_depth=config['max_depth'],
                  resize_mode=config.get('resize_mode', 'stretch'), evaluation_protocol='model_input_grid')
    return result
