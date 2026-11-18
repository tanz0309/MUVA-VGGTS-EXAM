"""Selection contract tests; no model computation and no target GT inputs."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pccs import EXPERTS, cycle_score, select_expert


class PCCSTest(unittest.TestCase):
    def candidates(self):
        return {e: np.eye(5, dtype=bool) for e in EXPERTS}

    def test_selects_closest_return_and_keeps_candidate(self):
        masks = self.candidates()
        original = {e: m.copy() for e, m in masks.items()}
        selected, audit = select_expert(masks, {'anchor': [[80, 80]], 'visual': [[11, 10]], 'fusion': [[20, 10]]}, [[10, 10]], (100, 100))
        self.assertEqual(selected, 'visual')
        self.assertAlmostEqual(audit['normalized_cycle_distance']['visual'], 1/np.hypot(100, 100))
        for e in EXPERTS: np.testing.assert_array_equal(masks[e], original[e])
        self.assertFalse(audit['target_gt_used_for_selection'])

    def test_unequal_point_sets_and_scale(self):
        self.assertAlmostEqual(cycle_score([[3, 4], [10, 0]], [[0, 0], [10, 0]], (100, 100)), 2.5/np.hypot(100, 100))
        self.assertAlmostEqual(cycle_score([[6, 8], [20, 0]], [[0, 0], [20, 0]], (200, 200)), 2.5/np.hypot(100, 100))

    def test_empty_candidate_never_wins(self):
        masks = self.candidates(); masks['anchor'][:] = False
        selected, audit = select_expert(masks, {e: [[0, 0]] for e in EXPERTS}, [[0, 0]], (100, 100))
        self.assertEqual(selected, 'visual')
        self.assertIsNone(audit['normalized_cycle_distance']['anchor'])

    def test_invalid_returns_and_fallback(self):
        points = {'anchor': [[np.nan, 0]], 'visual': [[100, 100]], 'fusion': []}
        selected, audit = select_expert(self.candidates(), points, [[1, 1]], (100, 100))
        self.assertEqual(selected, 'fusion')
        self.assertTrue(audit['fallback_to_fusion'])
        self.assertEqual(list(audit['normalized_cycle_distance'].values()), [None]*3)

    def test_no_reference_and_fixed_tie(self):
        points = {e: [[1, 1]] for e in EXPERTS}
        self.assertEqual(select_expert(self.candidates(), points, [], (100, 100))[0], 'fusion')
        self.assertEqual(select_expert(self.candidates(), points, [[1, 1]], (100, 100))[0], 'anchor')

    def test_missing_expert_is_error(self):
        with self.assertRaises(ValueError): select_expert({}, {}, [], (100, 100))


if __name__ == '__main__': unittest.main()
