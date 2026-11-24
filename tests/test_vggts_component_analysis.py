"""Independent metric parity, support coordinates, transitions and macro/micro."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from common import metrics
from vggts_component_analysis import (analyze_saved_predictions, compare_component_passes,
    component_coverage, extract_significant_components, map_points_to_components,
    points_to_original, safe_rate, summarize_component_groups, summarize_transitions)


class ComponentTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((60, 80), bool)
        self.gt[2:7, 3:7] = True
        self.gt[20:25, 30:34] = True
        self.gt[35:45, 50:60] = True
        self.gt[55, 70] = True  # insignificant; excluded from CR/SCR

    def test_component_definition_and_coverage_match_main_metrics(self):
        labels, components = extract_significant_components(self.gt)
        self.assertEqual(len(components), 3)
        self.assertEqual(sum(c['is_smallest_component'] for c in components), 2)
        pred = np.zeros_like(self.gt)
        pred[2:7, 3:7] = True
        pred[20:22, 30:34] = True
        pred[35:40, 50:60] = True
        coverage = component_coverage(pred, labels, components)
        score = metrics(pred, self.gt)
        self.assertEqual(list(coverage.values()), score['component_coverages'])
        self.assertEqual(np.mean([v >= .5 for v in coverage.values()]), score['cr50'])
        smallest = [coverage[c['component_id']] >= .5 for c in components if c['is_smallest_component']]
        self.assertEqual(np.mean(smallest), score['scr50'])

    def test_diagonal_connectivity_and_relative_area_threshold(self):
        gt = np.zeros((120, 120), bool)
        gt[0:100, 0:100] = True
        gt[100:102, 100:105] = True  # diagonal -> same component under 8-neighbors
        gt[110:119, 110:119] = True  # 81 < ceil(10091*.01), excluded
        _, components = extract_significant_components(gt)
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]['component_area'], 10010)
        self.assertIsNone(components[0]['nearest_other_component_distance'])

    def test_nearest_distance_is_pixel_euclidean(self):
        gt = np.zeros((20, 20), bool)
        gt[1:5, 1:5] = True
        gt[8:12, 8:12] = True
        _, components = extract_significant_components(gt)
        for c in components: self.assertAlmostEqual(c['nearest_other_component_distance'], np.sqrt(32))

    def test_crop_points_map_to_original_with_correct_xy_scaling(self):
        labels = np.zeros((100, 200), np.int32)
        labels[40:50, 60:70] = 1
        meta = {'crop_size': 20, 'W': 10, 'H': 10, 'x1': 10, 'y1': 10, 'ori_h': 50, 'ori_w': 100}
        points = [[10, 5], [-10, -10], [1000, 1000]]
        original = points_to_original(points, meta, labels.shape)
        np.testing.assert_array_equal(original[0], [60, 40])
        self.assertEqual(map_points_to_components(points, labels, meta)[1], 1)
        # A negative point must not be clipped to a foreground border pixel.
        labels[0, 0] = 1
        self.assertEqual(map_points_to_components([[-1, -1], [200, 100]], labels)[1], 0)

    def test_recovery_and_regression_are_separate_and_denominators_explicit(self):
        components = [{'component_id': 1, 'has_candidate_support': False, 'has_selected_prompt': False},
                      {'component_id': 2, 'has_candidate_support': True, 'has_selected_prompt': True}]
        rows = compare_component_passes({1: .49, 2: .5}, {1: .5, 2: .49}, components, 'A1')
        self.assertTrue(rows[0]['newly_recovered'])
        self.assertTrue(rows[1]['regressed'])
        summary = summarize_transitions(rows)
        self.assertEqual(summary['all']['refinement_rescue_rate'], 1)
        self.assertEqual(summary['all']['refinement_regression_rate'], 1)
        self.assertIsNone(summary['no_candidate']['refinement_regression_rate'])
        self.assertIsNone(safe_rate(0, 0))

    def test_macro_and_micro_differ_for_unequal_component_counts(self):
        flags = [('one', True), ('two', False), ('two', False), ('two', False)]
        records = [{'sample_id': sample, 'recovered50': recovered, 'has_candidate_support': False,
                    'has_selected_prompt': False, 'is_smallest_component': True, 'candidate_support_count': 0}
                   for sample, recovered in flags]
        summary = summarize_component_groups(records)
        self.assertEqual(summary['all']['component_micro_recovery'], .25)
        self.assertEqual(summary['all']['per_sample_macro_recovery'], .5)
        self.assertIsNone(summary['with_candidate']['component_micro_recovery'])
        json.dumps(summary, allow_nan=False)

    def test_analysis_cannot_open_gt_before_run_wide_completion_marker(self):
        with tempfile.TemporaryDirectory() as path, patch('vggts_component_analysis.read_gt') as read:
            with self.assertRaises(FileNotFoundError): analyze_saved_predictions([], Path(path), {})
            read.assert_not_called()


if __name__ == '__main__': unittest.main()
