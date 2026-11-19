"""Algorithm tests for the VGGT-S phase 1 Block-First selector (CPU only)."""
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from vggts_point_selector import (BLOCK_FIRST_VERSION, SelectorConfig,
                                  allocate_budget, choose_clusters,
                                  cluster_spatial_points, fps_indices,
                                  load_selector_config, medoid_index,
                                  normalize_points, select_target_points,
                                  stable_unique_points, validate_candidate_points,
                                  validate_selector_config)


class FakeKMeans:
    def __init__(self):
        self.calls = []

    def __call__(self, points, K):
        self.calls.append((np.array(points, copy=True), K))
        array = np.asarray(points, dtype=np.float32)
        return np.zeros(len(array), dtype=np.int64), array[:K].copy()


BF = SelectorConfig(method='block_first', budget=5, eps=0.05, min_samples=2)
IMAGE = 518


def three_clusters():
    a = [(50, 50), (52, 51), (48, 52), (51, 49), (53, 50), (49, 51), (50, 53), (47, 50)]
    b = [(300, 300), (302, 301), (298, 302), (301, 299), (303, 300), (299, 301), (300, 303), (297, 300)]
    c = [(100, 450), (102, 451), (98, 452), (101, 449)]
    return np.asarray(a + b + c, dtype=np.float32)


def run(points, config=BF, native=None):
    return select_target_points(points, image_size=IMAGE, config=config,
                                native_kmeans=native or FakeKMeans())


class ConfigTests(unittest.TestCase):
    def test_default_is_native_kmeans(self):
        config = load_selector_config(None, prompt_points=5)
        self.assertEqual(config.method, 'kmeans')
        self.assertEqual(config.budget, 5)

    def test_block_first_file(self):
        config = load_selector_config(ROOT/'configs/point_selectors/block_first_v1.json',
                                      prompt_points=5)
        self.assertEqual(config, BF)

    def test_rejects_unknown_field(self):
        with self.assertRaises(ValueError):
            validate_selector_config({'method': 'block_first', 'sample_weight': 1},
                                     prompt_points=5)

    def test_rejects_bad_values(self):
        for value in ({'method': 'grid'}, {'budget': 4}, {'eps': float('nan')},
                      {'eps': 0}, {'min_samples': 1}, {'min_samples': True},
                      {'schema_version': 2}):
            with self.assertRaises(ValueError, msg=str(value)):
                validate_selector_config(value, prompt_points=5)

    def test_phase1_budget_is_five(self):
        with self.assertRaises(ValueError):
            validate_selector_config({'budget': 5}, prompt_points=4)


class ValidationTests(unittest.TestCase):
    def test_candidate_validation(self):
        good = np.zeros((5, 2), np.float32)
        self.assertIsNone(validate_candidate_points(good, IMAGE, 5))
        bad = [np.zeros((4, 2)), np.zeros((6, 3)), np.full((6, 2), np.nan),
               np.zeros((6, 2)) + IMAGE]
        for points in bad:
            with self.assertRaises(ValueError, msg=str(points.shape)):
                validate_candidate_points(points, IMAGE, 5)


class DedupTests(unittest.TestCase):
    def test_exact_duplicates_preserve_first_occurrence(self):
        points = np.asarray([[1, 2], [3, 4], [1, 2], [5, 6]], np.float32)
        unique, to_candidate, to_unique = stable_unique_points(points)
        np.testing.assert_array_equal(unique, np.asarray([[1, 2], [3, 4], [5, 6]], np.float32))
        np.testing.assert_array_equal(to_candidate, [0, 1, 3])
        np.testing.assert_array_equal(to_unique, [0, 1, 0, 2])

    def test_near_points_are_not_merged(self):
        points = np.asarray([[1.0, 2.0], [1.0001, 2.0]], np.float32)
        self.assertEqual(len(stable_unique_points(points)[0]), 2)

    def test_normalization_divides_by_image_size(self):
        points = np.asarray([[259.0, 518.0]])
        np.testing.assert_allclose(normalize_points(points, 518), [[0.5, 1.0]])


class ClusterTests(unittest.TestCase):
    def test_two_point_cluster_with_min_samples_two(self):
        points = np.asarray([[0.0, 0.0], [0.004, 0.0]])
        labels, clusters, noise = cluster_spatial_points(points, BF, np.asarray([0, 1]))
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].unique_size, 2)
        self.assertEqual(len(noise), 0)
        self.assertTrue((labels == 0).all())

    def test_isolated_point_is_noise(self):
        points = np.asarray([[0.0, 0.0], [0.5, 0.5]])
        labels, clusters, noise = cluster_spatial_points(points, BF, np.asarray([0, 1]))
        self.assertEqual(clusters, [])
        np.testing.assert_array_equal(noise, [0, 1])
        self.assertTrue((labels == -1).all())

    def test_more_than_five_clusters_keep_five_and_record_discarded(self):
        pairs = []
        for index in range(6):
            pairs += [(30 + 70 * index, 50), (31 + 70 * index, 51)]
        result = run(np.asarray(pairs, np.float32))
        clusters = result.audit['clusters']
        self.assertEqual(len(clusters), 6)
        self.assertEqual(len(result.audit['discarded_cluster_ids']), 1)
        self.assertEqual(result.audit['discarded_cluster_ids'], [5])
        self.assertEqual(sum(cluster['quota'] for cluster in clusters), 5)


class MedoidTests(unittest.TestCase):
    def test_asymmetric_medoid_is_not_mean_nearest(self):
        # Brute force: sums are [22.399, 17.92, 16.027, 16.759]; mean-nearest is index 3.
        points = np.asarray([[0, 0], [4, 9], [4, 8], [3, 2]], dtype=np.float64) / 10
        index = medoid_index(np.arange(4), points, np.arange(4))
        self.assertEqual(index, 2)

    def test_medoid_tie_breaks_on_candidate_index(self):
        points = np.asarray([[0.0, 0.0], [0.2, 0.0], [-0.2, 0.0]])
        self.assertEqual(medoid_index(np.arange(3), points, np.arange(3)), 0)


class QuotaTests(unittest.TestCase):
    def test_fixed_example_is_two_two_one(self):
        self.assertEqual(allocate_budget([20, 12, 8], 5, [0, 1, 2]), {0: 2, 1: 2, 2: 1})

    def test_capacity_limits_are_respected(self):
        quota = allocate_budget([1, 4, 4], 5, [0, 1, 2])
        self.assertEqual(sum(quota.values()), 5)
        self.assertEqual(quota[0], 1)
        self.assertLessEqual(quota[1], 4)
        self.assertLessEqual(quota[2], 4)

    def test_insufficient_capacity_raises(self):
        with self.assertRaises(ValueError):
            allocate_budget([1, 2], 5, [0, 1])


class FpsTests(unittest.TestCase):
    def test_fps_starts_at_seed_and_is_deterministic(self):
        points = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        first = fps_indices([0, 1, 2, 3], 0, 3, points, np.arange(4))
        second = fps_indices([0, 1, 2, 3], 0, 3, points, np.arange(4))
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0)
        self.assertEqual(len(set(first)), 3)

    def test_fps_tie_breaks_on_candidate_index(self):
        points = np.asarray([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        self.assertEqual(fps_indices([0, 1, 2], 0, 2, points, np.arange(3)), [0, 1])


class SelectionTests(unittest.TestCase):
    def test_three_clusters_get_a_prompt_each(self):
        points = three_clusters()
        result = run(points)
        retained = [cluster for cluster in result.audit['clusters'] if cluster['retained']]
        self.assertEqual(len(retained), 3)
        self.assertTrue(all(cluster['quota'] >= 1 for cluster in retained))
        self.assertEqual(sum(cluster['quota'] for cluster in retained), 5)
        self.assertEqual(result.prompts_xy.shape, (5, 2))
        self.assertEqual(result.prompts_xy.dtype, np.float32)
        self.assertEqual(len(stable_unique_points(result.prompts_xy)[0]), 5)

    def test_single_cluster_uses_medoid_plus_fps(self):
        points = np.asarray([(50 + 1.5 * index, 50) for index in range(10)], np.float32)
        result = run(points)
        self.assertEqual(len(result.audit['clusters']), 1)
        self.assertEqual(result.audit['clusters'][0]['quota'], 5)
        self.assertEqual(int(result.selected_indices[0]),
                         result.audit['clusters'][0]['medoid_candidate_index'])

    def test_selected_points_come_from_original_candidates(self):
        points = three_clusters()
        result = run(points)
        candidates = {tuple(point) for point in points.tolist()}
        for point in result.selected_indices:
            self.assertIn(tuple(points[point].tolist()), candidates)
        self.assertEqual(result.selected_indices.dtype, np.int64)

    def test_input_is_not_modified(self):
        points = three_clusters()
        before = points.copy()
        selection_config = replace(BF)
        run(points, config=selection_config)
        np.testing.assert_array_equal(points, before)

    def test_repeated_runs_are_identical(self):
        points = three_clusters()
        first = run(points)
        second = run(points)
        np.testing.assert_array_equal(first.selected_indices, second.selected_indices)
        np.testing.assert_array_equal(first.prompts_xy, second.prompts_xy)

    def test_all_noise_falls_back_to_original_points(self):
        points = np.asarray([[10, 10], [110, 110], [210, 210], [310, 310],
                             [410, 410], [510, 20]], np.float32)
        native = FakeKMeans()
        result = run(points, config=replace(BF, eps=0.005), native=native)
        self.assertEqual(result.audit['fallback_reason'], 'all_noise')
        self.assertIsNone(result.selected_indices)
        self.assertEqual(native.calls[0][1], 5)
        np.testing.assert_array_equal(native.calls[0][0], points)
        np.testing.assert_array_equal(result.prompts_xy, points[:5])

    def test_insufficient_unique_points_falls_back(self):
        points = np.asarray([[50, 50], [50, 50], [50, 50], [60, 60], [70, 70], [80, 80]],
                            np.float32)
        native = FakeKMeans()
        result = run(points, native=native)
        self.assertEqual(result.audit['fallback_reason'], 'insufficient_unique_points')
        np.testing.assert_array_equal(native.calls[0][0], points)

    def test_insufficient_cluster_capacity_falls_back(self):
        points = np.asarray([[50, 50], [51, 50], [300, 300], [301, 300], [500, 500]],
                            np.float32)
        result = run(points)
        self.assertEqual(result.audit['fallback_reason'], 'insufficient_cluster_capacity')
        self.assertIsNone(result.selected_indices)

    def test_kmeans_branch_passes_original_points(self):
        points = np.asarray([[50, 50], [50, 50], [300, 300], [301, 300], [500, 500],
                             [100, 450]], np.float32)
        native = FakeKMeans()
        result = select_target_points(points, image_size=IMAGE,
                                      config=SelectorConfig(), native_kmeans=native)
        self.assertEqual(len(native.calls), 1)
        np.testing.assert_array_equal(native.calls[0][0], points)
        self.assertEqual(native.calls[0][1], 5)
        self.assertIsNone(result.selected_indices)
        self.assertEqual(result.audit['method_effective'], 'kmeans')
        np.testing.assert_array_equal(result.prompts_xy, points[:5].astype(np.float32))

    def test_audit_is_json_safe_and_versioned(self):
        result = run(three_clusters())
        json.dumps(result.audit, allow_nan=False)
        self.assertEqual(result.audit['selector_version'], BLOCK_FIRST_VERSION)
        self.assertEqual(result.audit['method_requested'], 'block_first')
        self.assertEqual(result.audit['method_effective'], 'block_first')
        self.assertEqual(result.audit['selected_indices'], result.selected_indices.tolist())
        self.assertEqual(result.audit['budget'], 5)
        for cluster in result.audit['clusters']:
            if cluster['retained']:
                self.assertIsNotNone(cluster['medoid_candidate_index'])
            else:
                self.assertIsNone(cluster['medoid_candidate_index'])
                self.assertEqual(cluster['quota'], 0)


if __name__ == '__main__':
    unittest.main()
