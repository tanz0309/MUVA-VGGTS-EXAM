import unittest
import numpy as np
import fragment_metrics as fm


class MetricsTests(unittest.TestCase):
    def test_missing_method_is_detected(self):
        self.assertTrue(callable(getattr(fm, 'mask_metrics', None)))

    def test_occluder_splits_one_complete_object(self):
        a = np.ones((20, 40), dtype=bool)
        v = a.copy(); v[:, 15:25] = False
        r = fm.mask_metrics(v, a)
        self.assertEqual(r['components_main'], 2)
        self.assertTrue(r['occlusion_split_main'])
        self.assertEqual(r['visible_area'], 600)

    def test_tiny_speck_is_not_a_significant_fragment(self):
        a = np.ones((30, 30), dtype=bool)
        v = np.zeros_like(a); v[2:12, 2:12] = True; v[25, 25] = True
        r = fm.mask_metrics(v, a)
        self.assertEqual(r['components_raw'], 2)
        self.assertEqual(r['components_main'], 1)
        self.assertFalse(r['occlusion_split_main'])

    def test_diagonal_contact_is_connected(self):
        v = np.eye(20, dtype=bool)
        r = fm.mask_metrics(v, v)
        self.assertEqual(r['components_raw'], 1)

    def test_preexisting_disconnected_shape_is_not_occluder_split(self):
        a = np.zeros((20, 40), dtype=bool)
        a[:, :10] = True; a[:, 30:] = True
        r = fm.mask_metrics(a, a)
        self.assertEqual(r['components_main'], 2)
        self.assertFalse(r['occlusion_split_main'])

    def test_empty_visible_mask_is_recorded(self):
        r = fm.mask_metrics(np.zeros((20, 20), bool), np.ones((20, 20), bool))
        self.assertEqual(r['visible_area'], 0)
        self.assertEqual(r['components_main'], 0)
        self.assertFalse(r['occlusion_split_main'])

    def test_reference_is_another_view_with_highest_visibility(self):
        records = [{'view_id': 1, 'visible_area': 10, 'amodal_area': 10},
                   {'view_id': 2, 'visible_area': 8, 'amodal_area': 10},
                   {'view_id': 3, 'visible_area': 5, 'amodal_area': 10}]
        self.assertEqual(fm.choose_reference(records[2], records)['view_id'], 1)
        self.assertEqual(fm.choose_reference(records[0], records)['view_id'], 2)
        self.assertIsNone(fm.choose_reference(records[0], records[:1]))


if __name__ == '__main__':
    unittest.main()
