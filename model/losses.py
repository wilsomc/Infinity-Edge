"""Depth and adjacent-pixel gradient losses on valid ground-truth pixels."""
import torch
import torch.nn.functional as F


def gradient_error_sum(prediction, target, mask):
    """Return absolute finite-difference error summed over valid x/y edges."""
    valid = mask.bool() & torch.isfinite(target)
    # Sanitize before subtraction: NaN times zero is still NaN.
    p = torch.where(valid, prediction, 0)
    d = torch.where(valid, target, 0)
    horizontal = valid[..., :, 1:] & valid[..., :, :-1]
    vertical = valid[..., 1:, :] & valid[..., :-1, :]
    dx = (p[..., :, 1:] - p[..., :, :-1]) - (d[..., :, 1:] - d[..., :, :-1])
    dy = (p[..., 1:, :] - p[..., :-1, :]) - (d[..., 1:, :] - d[..., :-1, :])
    error_sum = dx[horizontal].abs().sum() + dy[vertical].abs().sum()
    edges = int((horizontal.sum() + vertical.sum()).item())
    return error_sum, edges


def depth_loss(prediction, target, mask, gradient_weight=0.0):
    """Blend mean Smooth L1 and mean absolute gradient error (meters/pixel)."""
    if not 0 <= gradient_weight <= 1:
        raise ValueError('gradient_weight must lie in [0, 1]')
    valid = mask.bool() & torch.isfinite(target)
    if not valid.any():
        raise ValueError('Cannot compute depth loss without valid pixels')
    smooth_l1 = F.smooth_l1_loss(prediction[valid], target[valid])
    gradient_sum, edges = gradient_error_sum(prediction, target, valid)
    gradient = gradient_sum / edges if edges else gradient_sum
    total = (1 - gradient_weight) * smooth_l1 + gradient_weight * gradient
    return total, smooth_l1, gradient_sum, edges


def depth_edge_statistics(prediction, target, mask, threshold=0.1):
    """Exact-grid depth-jump detection; threshold is meters per adjacent pixel."""
    valid = mask.bool() & torch.isfinite(target)
    p, d = torch.where(valid, prediction, 0), torch.where(valid, target, 0)
    stats = {'true_positive':0, 'predicted':0, 'target':0, 'boundary_error_sum':0.}
    for axis in (-1, -2):
        right, left = [slice(None)]*p.ndim, [slice(None)]*p.ndim
        right[axis], left[axis] = slice(1,None), slice(None,-1)
        right, left = tuple(right), tuple(left)
        edges = valid[right] & valid[left]
        pd, dd = (p[right]-p[left])[edges], (d[right]-d[left])[edges]
        predicted_edges, target_edges = pd.abs() >= threshold, dd.abs() >= threshold
        stats['true_positive'] += int((predicted_edges & target_edges).sum().item())
        stats['predicted'] += int(predicted_edges.sum().item())
        stats['target'] += int(target_edges.sum().item())
        stats['boundary_error_sum'] += (pd-dd).abs()[target_edges].sum().item()
    return stats
