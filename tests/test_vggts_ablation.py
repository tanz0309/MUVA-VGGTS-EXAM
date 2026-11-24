"""Phase A runner contracts and orchestration; fake arithmetic, no CPU network."""
from contextlib import nullcontext
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_vggts_ablation import build_parser, select_rows
from vggts_ablation import capture_official_inputs, run_pass_sequence


class RunnerTests(unittest.TestCase):
    def test_test_split_and_later_phases_rejected(self):
        for args in [['--split', 'test'], ['--mode', 'prompt-ablation'], ['--mode', 'transformer-ablation']]:
            with self.assertRaises(SystemExit): build_parser().parse_args(['--output', '/tmp/unused', *args])

    def test_subset_is_seeded_unique_and_preserves_manifest_order(self):
        rows = [{'case_id': str(i), 'target_visibility': (i % 10)/10,
                 'target_components': 2+i % 5, 'group': str(i % 3)} for i in range(100)]
        first, second = select_rows(rows, 30, 42), select_rows(rows, 30, 42)
        self.assertEqual(first, second)
        self.assertEqual(len({r['case_id'] for r in first}), 30)
        ids = [int(r['case_id']) for r in first]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(select_rows(rows, 200, 42), rows)
        with self.assertRaises(ValueError): select_rows(rows, 0, 42)

    def test_four_calls_share_base_inputs_and_clone_source_features(self):
        class ArithmeticStandIn:
            calls = []
            def eval(self): return self
            def __call__(self, **kw):
                self.calls.append(kw['tar_mask_b'])
                kw['src_points_feat_b'].add_(1)  # stand-in for upstream in-place embedding
                value = kw['feat_map_b'][:, 0, :1].clone()
                if kw['tar_mask_b'] is not None: value = value+kw['tar_mask_b']
                return value, value.clone()
        class Adapter: predictor = ArithmeticStandIn()
        inputs = {'feat_map_b': torch.arange(32).reshape(1, 2, 1, 4, 4).float(),
                  'src_points_feat_b': torch.zeros(1, 5, 2), 'src_mask_b': np.ones((1, 4, 4)),
                  'tar_mask_b': None, 'ret_logits': True}
        config = {'corrupted_feedback': {'method': 'circular_shift', 'height_fraction': .5, 'width_fraction': .5},
                  'repeat_max_abs_tolerance': 1e-5}
        with patch('vggts_ablation.torch.autocast', return_value=nullcontext()):
            outputs, audit = run_pass_sequence(Adapter(), inputs, config)
        self.assertEqual(set(outputs), {'A0', 'A1', 'A2', 'A3'})
        self.assertEqual(len(Adapter.predictor.calls), 4)
        self.assertIsNone(Adapter.predictor.calls[0])
        self.assertIsNone(Adapter.predictor.calls[2])
        self.assertTrue(torch.equal(inputs['src_points_feat_b'], torch.zeros(1, 5, 2)))
        self.assertEqual(audit['repeat_none_max_abs_logit_difference'], 0)

    def test_capture_hook_removed_when_official_path_raises(self):
        class Adapter:
            predictor = torch.nn.Module()
            config = {}
            preprocessing = object()
            vggt = object()
        adapter = Adapter()
        with patch('vggts_official_data.OfficialEvaluator.predict', side_effect=RuntimeError('sentinel')):
            with self.assertRaisesRegex(RuntimeError, 'sentinel'): capture_official_inputs(adapter, ())
        self.assertEqual(len(adapter.predictor._forward_pre_hooks), 0)


if __name__ == '__main__': unittest.main()
