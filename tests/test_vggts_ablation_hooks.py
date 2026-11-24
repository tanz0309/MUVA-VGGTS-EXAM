"""Feedback controls and protection of reusable inputs; no real model on CPU."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from vggts_ablation_hooks import (array_digest, clone_inputs, input_fingerprints,
                                  make_corrupted_feedback)
from vggts_ablation import build_ablation_case, validate_ablation_inputs


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.config = {'method': 'circular_shift', 'height_fraction': .5, 'width_fraction': .5}

    def test_shift_preserves_values_positive_area_dtype_and_shape(self):
        x = torch.arange(-32, 32, dtype=torch.float32).reshape(1, 1, 8, 8)
        before = x.clone()
        shifted, audit = make_corrupted_feedback(x, self.config)
        self.assertTrue(torch.equal(x, before))
        self.assertFalse(torch.equal(x, shifted))
        self.assertTrue(torch.equal(torch.sort(x.flatten()).values, torch.sort(shifted.flatten()).values))
        self.assertEqual(int((x > 0).sum()), int((shifted > 0).sum()))
        self.assertEqual(shifted.dtype, x.dtype)
        self.assertEqual(shifted.shape, x.shape)
        self.assertEqual(audit['shifts_yx'], [4, 4])
        self.assertFalse(audit['gt_used'])
        self.assertEqual(audit['input_sha256'], array_digest(x))

    def test_shift_is_repeatable_without_rng(self):
        x = torch.arange(15).reshape(1, 1, 3, 5).float()
        first, a = make_corrupted_feedback(x, self.config)
        second, b = make_corrupted_feedback(x, self.config)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(a, b)

    def test_invalid_transforms_fail(self):
        for fraction in [0, 1, -1, 2]:
            with self.assertRaises(ValueError):
                make_corrupted_feedback(torch.zeros(1, 1, 4, 4), {**self.config, 'height_fraction': fraction})
        with self.assertRaises(ValueError):
            make_corrupted_feedback(torch.zeros(4, 4), self.config)

    def test_ablation_copies_mutable_source_point_features(self):
        base = {'feat_map_b': torch.ones(1, 2, 1, 2, 2), 'src_points_feat_b': torch.ones(1, 5, 2),
                'src_mask_b': np.ones((1, 2, 2)), 'tar_mask_b': None, 'ret_logits': True}
        expected = input_fingerprints(base)
        call = build_ablation_case('A1', base, torch.zeros(1, 1, 2, 2))
        call['src_points_feat_b'].add_(30)
        validate_ablation_inputs(base, expected)
        self.assertIsNotNone(call['tar_mask_b'])
        self.assertIsNone(base['tar_mask_b'])
        base['feat_map_b'].add_(1)
        with self.assertRaises(RuntimeError): validate_ablation_inputs(base, expected)

    def test_recorded_inputs_are_independent(self):
        x = {'tensor': torch.ones(2), 'array': np.ones(2), 'flag': True}
        copied = clone_inputs(x)
        copied['tensor'].zero_()
        copied['array'][:] = 0
        self.assertTrue(torch.equal(x['tensor'], torch.ones(2)))
        np.testing.assert_array_equal(x['array'], np.ones(2))


if __name__ == '__main__': unittest.main()
