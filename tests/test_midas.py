import unittest
import numpy as np
from script.compare_midas import align_inverse_depth


class AlignmentTests(unittest.TestCase):
    def test_recovers_known_scale_shift(self):
        target = np.array([[1., 2.], [4., 8.]])
        inverse = (1 / target - .1) / 3
        actual, fit = align_inverse_depth(inverse, target, np.ones_like(target, bool), .001, 8)
        np.testing.assert_allclose(actual, target, rtol=1e-6)
        self.assertAlmostEqual(fit['scale'], 3)
        self.assertAlmostEqual(fit['shift'], .1)

    def test_mask_ignores_invalid_targets(self):
        target = np.array([[1., 2.], [np.nan, 0.]])
        inverse = np.array([[1., .5], [100., 200.]])
        actual, _ = align_inverse_depth(inverse, target, np.ones_like(target, bool), .001, 8)
        np.testing.assert_allclose(actual[0], target[0])

    def test_degenerate_and_reversed_predictions(self):
        target = np.array([[1., 2.], [4., 8.]])
        for inverse in [np.ones_like(target), -1/target]:
            actual, fit = align_inverse_depth(inverse, target, np.ones_like(target, bool), .001, 8)
            self.assertTrue(fit['constant_fit'])
            np.testing.assert_allclose(actual, 1/np.mean(1/target))

    def test_rejects_empty_or_nonfinite(self):
        target = np.ones((2, 2))
        with self.assertRaises(ValueError):
            align_inverse_depth(target, target, np.zeros_like(target, bool), .001, 8)
        with self.assertRaises(ValueError):
            align_inverse_depth(target*np.nan, target, np.ones_like(target, bool), .001, 8)
