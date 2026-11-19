"""Data/coordinate invariants only; no neural-network forward passes on CPU."""
import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from common import load_rows, read_inputs
from vggts_official_data import MuvaPreprocessing, inputs_key
from run_vggts_official import validate_checkpoint, native_config


class OfficialDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'configs/vggts_official.json').read_text())
        cls.preprocessing = MuvaPreprocessing(cls.config)
        cls.inputs = read_inputs(load_rows('train')[0])
        # A compact valid target group in the first-stage coordinate system.
        cls.cached = {'tracks_518': np.column_stack((np.linspace(240, 260, 50),
                                                     np.linspace(130, 150, 50))).tolist()}

    def test_upstream_padding_is_bottom_right(self):
        a = np.full((1080, 1920, 3), 23, np.uint8)
        p = self.preprocessing.pad_frame(None, a, (518, 518), False)
        self.assertEqual(p.shape, (518, 518, 3))
        self.assertTrue((p[:294] == 23).all())
        self.assertTrue((p[294:] == 114).all())

    def test_native_small_object_crop_and_prompts(self):
        p = self.preprocessing.prepare(self.inputs, self.cached)
        self.assertEqual(p['meta']['crop_size'], 180)
        self.assertEqual(p['inliers'].shape, (40, 2))
        self.assertEqual(p['target_points'].shape, (5, 2))
        self.assertTrue(((p['target_points'] >= 0) & (p['target_points'] <= 517)).all())
        self.assertEqual(p['source_mask'].shape, (518, 518))

    def test_target_gt_does_not_affect_crop_or_prompts(self):
        with patch('vggts_official_data.read_gt', side_effect=AssertionError('GT read during inference preprocessing')):
            p = self.preprocessing.prepare(self.inputs, self.cached)
        saved = p['target_points'].copy(); meta = dict(p['meta'])
        zero, _, _ = self.preprocessing.target_supervision(np.zeros((1080, 1920), bool), p)
        one, _, _ = self.preprocessing.target_supervision(np.ones((1080, 1920), bool), p)
        self.assertEqual(int(zero.sum()), 0)
        self.assertEqual(int(one.sum()), 518*518)
        np.testing.assert_array_equal(saved, p['target_points'])
        self.assertEqual(meta, p['meta'])

    def test_960_coordinates_restore_to_original(self):
        p = self.preprocessing.prepare(self.inputs, self.cached)
        original = np.asarray(p['audit']['positive_prompts_original'])
        m = p['meta']
        recovered = (original/2-np.array([m['x1'], m['y1']]))*518/m['crop_size']
        np.testing.assert_allclose(recovered, p['target_points'], atol=1e-4)

    def test_input_identity_includes_reference_mask(self):
        from PIL import Image
        altered = list(self.inputs)
        altered[1] = Image.fromarray(np.zeros((1080, 1920), np.uint8))
        self.assertNotEqual(inputs_key(self.inputs), inputs_key(altered))

    def test_old_and_smoke_checkpoints_are_rejected(self):
        for scope in ['muva_fragment_training', 'smoke_only', 'official_muva_smoke_only']:
            with self.assertRaises(ValueError):
                validate_checkpoint({'scope': scope}, self.config)

    def test_training_keeps_author_refinement_and_loss_hyperparameters(self):
        c = native_config(self.config, 6224)
        self.assertEqual(c['exp']['refine_ratio'], .5)
        self.assertEqual(c['exp']['refine_time'], 1)
        self.assertEqual(c['exp']['lr'], 5e-5)
        self.assertEqual(c['exp']['lr_drops'], [8, 11])
        self.assertEqual(c['exp']['train_samples'], 6224)
        self.assertEqual(c['data']['train_batch_size'], self.config['training_batch_size'])
        self.assertEqual(self.config['training_batch_size'], 8)

    def test_missing_points_generated_once_and_reused_without_gt(self):
        p = copy.copy(self.preprocessing)
        p.point_cache_usage = {'generated_online': 0, 'reused': 0}
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as tmp:
            p.cache = Path(tmp)
            def generate(inputs, backbone, case_id):
                p.cache_path(inputs).write_text(json.dumps({**self.cached, 'recipe': p.recipe}))
                return True
            with patch.object(p, 'make_points', side_effect=generate) as generate_mock, \
                    patch('vggts_official_data.read_gt', side_effect=AssertionError('Unexpected GT access')):
                cold = p.prepare(self.inputs, vggt=object(), case_id='cold')
                warm = p.prepare(self.inputs, vggt=object(), case_id='warm')
            self.assertEqual(generate_mock.call_count, 1)
            self.assertEqual(p.point_cache_usage, {'generated_online': 1, 'reused': 1})
            self.assertEqual(cold['audit']['first_stage_point_source'], 'generated_online')
            self.assertEqual(warm['audit']['first_stage_point_source'], 'cache')
            np.testing.assert_array_equal(cold['target_points'], warm['target_points'])
            self.assertEqual(cold['meta'], warm['meta'])

    def test_invalid_cache_is_rejected_instead_of_silently_regenerated(self):
        p = copy.copy(self.preprocessing)
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as tmp:
            p.cache = Path(tmp)
            p.cache_path(self.inputs).write_text(json.dumps({'recipe': p.recipe, 'tracks_518': [[0, 0]]}))
            with patch.object(p, 'make_points') as generate_mock:
                with self.assertRaisesRegex(ValueError, 'Invalid point cache'):
                    p.resolve_points(self.inputs, object())
                generate_mock.assert_not_called()

    def test_generation_preserves_training_rng(self):
        p = copy.copy(self.preprocessing)
        original_fork = torch.random.fork_rng
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as tmp:
            p.cache = Path(tmp)
            def fake_generation(*args):
                random.random(); np.random.random(); torch.rand(1)
                p.cache_path(self.inputs).write_text(json.dumps({**self.cached, 'recipe': p.recipe}))
            random.seed(123); np.random.seed(123); torch.manual_seed(123)
            expected = (random.random(), np.random.random(), torch.rand(1))
            random.seed(123); np.random.seed(123); torch.manual_seed(123)
            # No model executes here. Actual CUDA RNG preservation is checked separately on GPU.
            with patch('torch.cuda.current_device', return_value=0), \
                    patch('torch.random.fork_rng', side_effect=lambda devices: original_fork(devices=[])), \
                    patch.object(p, '_generate_points', side_effect=fake_generation):
                self.assertTrue(p.make_points(self.inputs, object(), 'cold'))
                self.assertFalse(p.make_points(self.inputs, object(), 'warm'))
            actual = (random.random(), np.random.random(), torch.rand(1))
            self.assertEqual(actual[:2], expected[:2])
            self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_gpu_generation_rejects_worker_process(self):
        p = copy.copy(self.preprocessing)
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as tmp:
            p.cache = Path(tmp)
            with patch('torch.utils.data.get_worker_info', return_value=object()):
                with self.assertRaisesRegex(RuntimeError, 'num_workers=0'):
                    p.make_points(self.inputs, object(), 'worker')

    # ---- Phase 1 selector integration -------------------------------------------------
    def selector_preprocessing(self):
        return MuvaPreprocessing(self.config, selector_config={
            'method': 'block_first', 'budget': 5, 'eps': 0.05, 'min_samples': 2})

    def test_default_selector_matches_native_kmeans(self):
        from utils import cluster_points
        prepared = self.preprocessing.prepare(self.inputs, self.cached)
        points = np.asarray(prepared['audit']['point_selector']['candidate_points_crop'], np.float32)
        _, centers = cluster_points(points, K=5)
        np.testing.assert_array_equal(prepared['target_points'], centers.astype(np.float32))
        self.assertIsNone(prepared['audit']['point_selector']['selected_indices'])
        self.assertEqual(prepared['audit']['point_selector']['method_effective'], 'kmeans')

    def test_selector_switch_keeps_inputs_recipe_and_cache(self):
        block_first = self.selector_preprocessing()
        a = self.preprocessing.prepare(self.inputs, self.cached)
        b = block_first.prepare(self.inputs, self.cached)
        for key in ('source', 'source_mask', 'target', 'tracks', 'inliers'):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(a['meta'], b['meta'])
        self.assertEqual(a['audit']['selection_inputs'], b['audit']['selection_inputs'])
        self.assertEqual(self.preprocessing.recipe, block_first.recipe)
        self.assertEqual(self.preprocessing.cache_path(self.inputs),
                         block_first.cache_path(self.inputs))
        self.assertNotIn('selector', self.preprocessing.config)
        self.assertEqual(self.config, json.loads((ROOT/'configs/vggts_official.json').read_text()))

    def test_both_selectors_avoid_target_gt_during_prepare(self):
        block_first = self.selector_preprocessing()
        with patch('vggts_official_data.read_gt',
                   side_effect=AssertionError('GT read during inference preprocessing')):
            self.assertTrue(self.preprocessing.prepare(self.inputs, self.cached)['source_mask'].any())
            self.assertTrue(block_first.prepare(self.inputs, self.cached)['source_mask'].any())

    def test_block_first_prompts_restore_to_original_coordinates(self):
        prepared = self.selector_preprocessing().prepare(self.inputs, self.cached)
        original = np.asarray(prepared['audit']['positive_prompts_original'])
        meta = prepared['meta']
        recovered = (original/2 - np.array([meta['x1'], meta['y1']]))*518/meta['crop_size']
        np.testing.assert_allclose(recovered, prepared['target_points'], atol=1e-4)

    def test_block_first_cold_and_warm_select_the_same_points(self):
        p = copy.copy(self.selector_preprocessing())
        p.point_cache_usage = {'generated_online': 0, 'reused': 0}
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as tmp:
            p.cache = Path(tmp)
            def generate(inputs, backbone, case_id):
                p.cache_path(inputs).write_text(json.dumps({**self.cached, 'recipe': p.recipe}))
                return True
            with patch.object(p, 'make_points', side_effect=generate):
                cold = p.prepare(self.inputs, vggt=object(), case_id='cold')
                warm = p.prepare(self.inputs, vggt=object(), case_id='warm')
        np.testing.assert_array_equal(cold['target_points'], warm['target_points'])
        self.assertEqual(cold['audit']['point_selector']['selected_indices'],
                         warm['audit']['point_selector']['selected_indices'])

    def test_fallback_uses_the_original_40_candidate_points(self):
        from utils import cluster_points
        cached = {'tracks_518': np.tile(np.array([[240., 140.]], np.float32), (50, 1)).tolist()}
        prepared = self.selector_preprocessing().prepare(self.inputs, cached)
        selector = prepared['audit']['point_selector']
        self.assertEqual(selector['fallback_reason'], 'insufficient_unique_points')
        self.assertIsNone(selector['selected_indices'])
        points = np.asarray(selector['candidate_points_crop'], np.float32)
        _, centers = cluster_points(points, K=5)
        np.testing.assert_array_equal(prepared['target_points'], centers.astype(np.float32))


if __name__ == '__main__':
    unittest.main()
