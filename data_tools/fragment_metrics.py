"""Full-resolution connected-component measurements for MUVA GT masks."""
import math
from collections import Counter
import numpy as np
from scipy import ndimage


def mask_metrics(visible, amodal):
    visible = np.asarray(visible, dtype=bool)
    amodal = np.asarray(amodal, dtype=bool)
    assert visible.shape == amodal.shape
    labels, raw = ndimage.label(visible, structure=np.ones((3, 3), np.uint8))
    areas = np.bincount(labels.ravel(), minlength=raw + 1)[1:]
    va, aa = int(areas.sum()), int(amodal.sum())
    thresholds = {'min10': 10, 'halfpct': max(10, math.ceil(va * .005)),
                  'main': max(10, math.ceil(va * .01)),
                  '2pct': max(10, math.ceil(va * .02))}
    significant = np.flatnonzero(areas >= thresholds['main']) + 1
    alabels, an = ndimage.label(amodal, structure=np.ones((3, 3), np.uint8))
    memberships = Counter()
    if len(significant) >= 2 and aa > va:
        # Count actual containment in an amodal connected region, rather than
        # treating a naturally disconnected amodal silhouette as occluder splitting.
        for label in significant:
            occupied = np.bincount(alabels[labels == label], minlength=an + 1)
            occupied[0] = 0
            best = int(occupied.argmax())
            if best and occupied[best] / areas[label - 1] >= .99:
                memberships[best] += 1
    r = {'visible_area': va, 'amodal_area': aa,
         'visibility_ratio': va / aa if aa else 0.0,
         'components_raw': int(raw), 'main_threshold_px': thresholds['main'],
         'component_areas_main': sorted([int(areas[x - 1]) for x in significant], reverse=True),
         'amodal_components_raw': int(an),
         'visible_outside_amodal_px': int(np.count_nonzero(visible & ~amodal)),
         'occlusion_split_main': any(n >= 2 for n in memberships.values())}
    for name, threshold in thresholds.items():
        r['components_' + name] = int(np.count_nonzero(areas >= threshold))
    return r


def choose_reference(target, records):
    candidates = [r for r in records if r['view_id'] != target['view_id']
                  and r['visible_area'] > 0 and r['amodal_area'] > 0]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (-r['visible_area'] / r['amodal_area'], r['view_id']))
