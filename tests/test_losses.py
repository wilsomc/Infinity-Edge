import unittest

import torch

from model.evaluation import DepthMetrics
from model.losses import depth_loss, gradient_error_sum


class GradientLossTests(unittest.TestCase):
    def test_known_horizontal_vertical_errors_and_blend(self):
        prediction = torch.tensor([[[[1., 2.], [2., 2.]]]], requires_grad=True)
        target = torch.tensor([[[[1., 1.], [1., 3.]]]])
        mask = torch.ones_like(target, dtype=torch.bool)
        loss, smooth, gradient_sum, edges = depth_loss(prediction, target, mask, .2)
        self.assertEqual(edges, 4)
        self.assertAlmostEqual(gradient_sum.item(), 6)
        self.assertAlmostEqual(smooth.item(), .375)
        self.assertAlmostEqual(loss.item(), .6)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_invalid_nan_pixels_exclude_touching_edges(self):
        prediction = torch.tensor([[[[1., float('nan')], [2., 2.]]]], requires_grad=True)
        target = torch.tensor([[[[1., float('nan')], [1., 3.]]]])
        mask = torch.tensor([[[[True, False], [True, True]]]])
        loss, _, error_sum, edges = depth_loss(prediction, target, mask, .2)
        self.assertEqual(edges, 2)
        self.assertEqual(error_sum.item(), 3)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(prediction.grad[0, 0, 0, 1].item(), 0)

    def test_no_adjacent_valid_pixels_is_safe(self):
        prediction = torch.ones(1, 1, 2, 2, requires_grad=True)
        target = torch.full_like(prediction, 2.)
        mask = torch.tensor([[[[True, False], [False, True]]]])
        loss, _, error_sum, edges = depth_loss(prediction, target, mask, .2)
        self.assertEqual(edges, 0)
        self.assertEqual(error_sum.item(), 0)
        self.assertAlmostEqual(loss.item(), .4)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_gradient_is_invariant_to_constant_offset(self):
        target = torch.tensor([[[[1., 2., 3.], [2., 4., 5.]]]])
        error, edges = gradient_error_sum(target + 10, target, torch.ones_like(target, dtype=torch.bool))
        self.assertEqual(edges, 7)
        self.assertEqual(error.item(), 0)

    def test_validation_aggregates_by_edges(self):
        metrics = DepthMetrics(max_depth=10)
        # First sample: one edge, absolute gradient error 1.
        metrics.update(torch.tensor([[[[1., 2.]]]]), torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2, dtype=torch.bool))
        # Second sample: two edges, each error 2. Overall (1+4)/3.
        metrics.update(torch.tensor([[[[1., 3., 5.]]]]), torch.ones(1, 1, 1, 3), torch.ones(1, 1, 1, 3, dtype=torch.bool))
        result = metrics.compute()
        self.assertEqual(result['valid_edges'], 3)
        self.assertAlmostEqual(result['gradient_loss'], 5/3)

    def test_zero_weight_matches_original_loss(self):
        prediction = torch.tensor([[[[1., 2.], [2., 2.]]]], requires_grad=True)
        target = torch.tensor([[[[1., 1.], [1., 3.]]]])
        mask = torch.ones_like(target, dtype=torch.bool)
        actual, smooth, _, _ = depth_loss(prediction, target, mask, 0)
        self.assertEqual(actual.item(), smooth.item())
        with self.assertRaises(ValueError):
            depth_loss(prediction, target, mask, 1.1)
