"""Shared files and metrics for the agreed MUVA fragmented protocol."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_rows(split):
    if split not in ('train', 'val', 'test'):
        raise ValueError(split)
    with (ROOT / 'data/manifests' / f'{split}.jsonl').open() as f:
        return [json.loads(line) for line in f]


def read_inputs(row):
    # Explicit whitelist: no target labels enter either model adapter.
    inputs = row['inputs']
    with Image.open(inputs['reference_rgb']) as im:
        reference = im.convert('RGB').copy()
    with Image.open(inputs['reference_mask']) as im:
        mask = im.convert('L').copy()
    with Image.open(inputs['target_rgb']) as im:
        target = im.convert('RGB').copy()
    return reference, mask, target


def read_gt(row):
    with Image.open(row['supervision']['target_visible_mask']) as im:
        return np.asarray(im) > 0


def boundary(mask):
    # DAVIS seg2bmap at unchanged image resolution.
    e = np.zeros_like(mask); s = np.zeros_like(mask); se = np.zeros_like(mask)
    e[:, :-1] = mask[:, 1:]; s[:-1, :] = mask[1:, :]; se[:-1, :-1] = mask[1:, 1:]
    b = (mask ^ e) | (mask ^ s) | (mask ^ se)
    b[-1, :] = mask[-1, :] ^ e[-1, :]
    b[:, -1] = mask[:, -1] ^ s[:, -1]
    b[-1, -1] = False
    return b


def metrics(pred, gt):
    pred = np.asarray(pred, dtype=bool); gt = np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape or pred.ndim != 2 or not gt.any():
        raise ValueError(f'Invalid mask shapes/empty GT: {pred.shape}, {gt.shape}')
    intersection = int((pred & gt).sum()); pa = int(pred.sum()); ga = int(gt.sum())
    union = pa + ga - intersection
    labels, _ = ndi.label(gt, structure=np.ones((3, 3), np.uint8))
    areas = np.bincount(labels.ravel())
    threshold = max(10, int(np.ceil(ga * .01)))
    ids = np.flatnonzero(areas >= threshold); ids = ids[ids != 0]
    hits = np.bincount(labels[pred], minlength=len(areas))
    coverage = hits[ids] / areas[ids]
    successes = coverage >= .5
    pb, gb = boundary(pred), boundary(gt)
    tolerance = int(np.ceil(.008 * np.linalg.norm(gt.shape)))
    pn, gn = int(pb.sum()), int(gb.sum())
    if pn == 0 or gn == 0:
        bf = float(pn == gn)
    else:
        bp = float((ndi.distance_transform_edt(~gb)[pb] <= tolerance).mean())
        br = float((ndi.distance_transform_edt(~pb)[gb] <= tolerance).mean())
        bf = 2 * bp * br / (bp + br) if bp + br else 0.
    return {'visible_iou': intersection / union,
            'dice': 2 * intersection / (pa + ga), 'boundary_f': bf,
            'cr50': float(successes.mean()) if len(ids) else None,
            'scr50': float(successes[areas[ids] == areas[ids].min()].mean()) if len(ids) else None,
            'all_components_covered50': bool(successes.all()) if len(ids) else None,
            'precision': intersection / pa if pa else 0., 'recall': intersection / ga,
            'components': len(ids), 'component_coverages': coverage.tolist(),
            'component_min_area': threshold, 'boundary_tolerance_pixels': tolerance,
            'empty_prediction': pa == 0}


def summarize(rows):
    keys = ['visible_iou', 'dice', 'boundary_f', 'cr50', 'scr50',
            'all_components_covered50', 'precision', 'recall']
    means = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys}
    times = [r['inference_seconds'] for r in rows if not r.get('failed')]
    means.update({'count': len(rows), 'failures': sum(bool(r.get('failed')) for r in rows),
                  'empty_predictions': sum(r['empty_prediction'] for r in rows),
                  'mean_seconds_per_pair': float(np.mean(times)) if times else None,
                  'median_seconds_per_pair': float(np.median(times)) if times else None,
                  'p95_seconds_per_pair': float(np.percentile(times, 95)) if times else None,
                  'peak_allocated_gpu_mib': max(r['peak_allocated_gpu_mib'] for r in rows)})
    return means
