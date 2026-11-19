import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import metrics


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((100, 100), bool)
        self.gt[10:30, 10:60] = True  # 1000 pixels
        self.gt[60:65, 60:70] = True  # 50 pixels

    def test_perfect(self):
        r = metrics(self.gt, self.gt)
        for k in ['visible_iou','dice','boundary_f','cr50','scr50','precision','recall']:
            self.assertEqual(r[k], 1.)

    def test_missed_small_fragment(self):
        p = self.gt.copy(); p[60:65,60:70] = False
        r = metrics(p,self.gt)
        self.assertAlmostEqual(r['visible_iou'],1000/1050)
        self.assertEqual(r['cr50'],.5); self.assertEqual(r['scr50'],0.)

    def test_empty(self):
        r=metrics(np.zeros_like(self.gt),self.gt)
        for k in ['visible_iou','dice','boundary_f','cr50','scr50','precision','recall']:
            self.assertEqual(r[k],0.)

    def test_oversegmentation_does_not_hide_in_iou(self):
        r=metrics(np.ones_like(self.gt),self.gt)
        self.assertEqual(r['cr50'],1.)
        self.assertLess(r['visible_iou'],.11)
        self.assertLess(r['precision'],.11)

    def test_smallest_ties(self):
        g=np.zeros((50,50),bool);g[2:6,2:6]=1;g[20:24,20:24]=1
        p=g.copy();p[2:6,2:6]=0
        self.assertEqual(metrics(p,g)['scr50'],.5)

    def test_threshold_does_not_remove_pixels_from_iou(self):
        g=self.gt.copy();g[90,90]=1
        r=metrics(self.gt,g)
        self.assertEqual(r['components'],2)
        self.assertLess(r['visible_iou'],1.)

    def test_70_threshold_is_stricter_than_50(self):
        p=np.zeros_like(self.gt)
        p[10:22,10:60]=True   # 60% of the 1000px component -> fails @70, passes @50
        p[60:65,60:70]=True   # smallest component fully covered
        r=metrics(p,self.gt)
        self.assertEqual(r['cr50'],1.)
        self.assertEqual(r['cr70'],.5)
        self.assertTrue(r['all_components_covered50'])
        self.assertFalse(r['all_components_covered70'])
        self.assertEqual(r['scr50'],1.)
        self.assertEqual(r['scr70'],1.)


if __name__ == '__main__': unittest.main()
