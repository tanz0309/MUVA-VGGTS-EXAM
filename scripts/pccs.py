"""Paper-based PCCS selection. No target label or metric is accepted here.

V2-SAM Sec.3.5 / Eq.8 describes inverse Anchor and average point distance,
but the publicly accessible release does not define unequal-set pairing.
We explicitly use mean nearest-reference Euclidean distance. With the native
one-point setting this is simply the distance between two points.
"""
import numpy as np

EXPERTS = ('anchor', 'visual', 'fusion')


def valid_points(points, image_size):
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    w, h = image_size
    if w <= 0 or h <= 0:
        raise ValueError('Invalid reference image size')
    ok = np.isfinite(xy).all(1) & (xy >= 0).all(1) & (xy[:, 0] < w) & (xy[:, 1] < h)
    return xy[ok]


def cycle_score(back_points, reference_points, image_size):
    back = valid_points(back_points, image_size)
    ref = valid_points(reference_points, image_size)
    if not len(back) or not len(ref):
        return None
    distances = np.linalg.norm(back[:, None] - ref[None], axis=-1)
    return float(distances.min(axis=1).mean() / np.hypot(*image_size))


def select_expert(predictions, back_points, reference_points, image_size):
    """Return one unmodified candidate and JSON-safe diagnostics, no extra mask."""
    if set(predictions) != set(EXPERTS) or set(back_points) != set(EXPERTS):
        raise ValueError('PCCS requires all three expert candidates')
    shapes = {np.asarray(predictions[e]).shape for e in EXPERTS}
    if len(shapes) != 1:
        raise ValueError('Expert candidate shapes differ')
    scores, invalid = {}, {}
    for e in EXPERTS:
        if not np.asarray(predictions[e]).any():
            scores[e] = None
            invalid[e] = 'empty_prediction'
        else:
            scores[e] = cycle_score(back_points[e], reference_points, image_size)
            if scores[e] is None:
                invalid[e] = 'no_valid_reference_or_return_points'
    valid = [e for e in EXPERTS if scores[e] is not None]
    # Explicit, fixed policies for cases that the paper does not specify.
    selected = min(valid, key=lambda e: scores[e]) if valid else 'fusion'
    return selected, {
        'selected_expert': selected,
        'normalized_cycle_distance': scores,
        'invalid_experts': invalid,
        'fallback_to_fusion': not bool(valid),
        'tie_priority': list(EXPERTS),
        'selector_implementation': 'paper_based_pccs_mean_nearest_reference_v1',
        'target_gt_used_for_selection': False,
    }
