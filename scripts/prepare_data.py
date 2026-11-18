"""Export raw-image references and independent masks, never rendered overlays."""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mu
from scipy import ndimage as ndi

from common import ROOT, digest, write_json

RAW = Path('/root/datasets/MUVA')
SOURCE = Path('/root/datasets/MUVA_fragmented_dataset/pairs.csv')
EXPECTED = {'train': 6224, 'val': 2215, 'test': 2008}


def export_mask(job):
    split, ann = job
    rle = dict(ann['visible_segmentation'])
    if isinstance(rle['counts'], str): rle['counts'] = rle['counts'].encode('ascii')
    mask = mu.decode(rle).astype(bool)
    path = ROOT / 'data/masks' / split / f"{ann['id']}.png"
    if path.exists():
        with Image.open(path) as im:
            assert np.array_equal(np.asarray(im) > 0, mask), str(path)
    else:
        tmp = path.with_suffix('.png.part')
        Image.fromarray(mask.astype(np.uint8) * 255).save(tmp, format='PNG')
        tmp.replace(path)
    area = int(mask.sum())
    lab, _ = ndi.label(mask, structure=np.ones((3, 3), np.uint8))
    a = np.bincount(lab.ravel())[1:]
    return ann['id'], {'area': area, 'components': int((a >= max(10, np.ceil(.01 * area))).sum())}


def main():
    with SOURCE.open(encoding='utf-8-sig', newline='') as f: pairs = list(csv.DictReader(f))
    assert len(pairs) == len({p['case_id'] for p in pairs}) == 10447
    (ROOT / 'data/manifests').mkdir(parents=True, exist_ok=True)
    scene_sets = {}; image_sets = {}; counts = {}; mask_counts = {}; hashes = {}
    for split, expected in EXPECTED.items():
        src = RAW / 'annotations' / f'AiFi-MUVA_{split}.json'
        data = json.loads(src.read_text()); hashes[split] = digest(src)
        annotations = {a['id']: a for a in data['annotations']}
        images = {im['id']: im for im in data['images']}
        selected = [p for p in pairs if p['split'] == split]
        assert len(selected) == expected
        needed = {int(p[k]) for p in selected for k in ['reference_annotation_id', 'target_annotation_id']}
        (ROOT / 'data/masks' / split).mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            stats = dict(pool.map(export_mask, ((split, annotations[i]) for i in sorted(needed))))
        scene_sets[split] = set(); image_sets[split] = set(); rows = []
        for p in selected:
            ra = annotations[int(p['reference_annotation_id'])]; ta = annotations[int(p['target_annotation_id'])]
            ri, ti = images[ra['image_id']], images[ta['image_id']]
            assert ra['object_id'] == ta['object_id'] == int(p['object_id'])
            assert ri['planogram_id'] == ti['planogram_id'] == int(p['scene_id'])
            assert ri['view_id'] != ti['view_id']
            assert ri['file_name'] == p['reference_image'] and ti['file_name'] == p['target_image']
            assert (ri['width'], ri['height']) == (ti['width'], ti['height']) == (1920,1080)
            assert stats[ta['id']]['area'] == int(p['target_visible_area'])
            assert stats[ta['id']]['components'] == int(p['target_components_main']) >= 2
            for im in (ri, ti):
                assert (RAW / 'images' / im['file_name']).is_file()
                scene_sets[split].add(str(Path(im['file_name']).parent))
                image_sets[split].add(im['file_name'])
            rows.append({'case_id': p['case_id'], 'split': split, 'scene': str(Path(ti['file_name']).parent),
                         'object_id': int(p['object_id']), 'group': p['group'],
                         'reference_view': ri['view_id'], 'target_view': ti['view_id'],
                         'target_visibility': float(p['target_visibility']),
                         'target_components': int(p['target_components_main']),
                         'inputs': {'reference_rgb': str(RAW/'images'/ri['file_name']),
                                    'reference_mask': str(ROOT/'data/masks'/split/f"{ra['id']}.png"),
                                    'target_rgb': str(RAW/'images'/ti['file_name'])},
                         'supervision': {'target_visible_mask': str(ROOT/'data/masks'/split/f"{ta['id']}.png"),
                                         'target_annotation_id': ta['id'], 'reference_annotation_id': ra['id']}})
        path = ROOT/'data/manifests'/f'{split}.jsonl'
        with path.open('w') as f:
            for row in rows: f.write(json.dumps(row) + '\n')
        counts[split] = len(rows); mask_counts[split] = len(needed)
        print('PREPARED', split, len(rows), 'pairs', len(needed), 'masks', flush=True)
    overlaps = {}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        overlaps[a+'_'+b] = {'scene_overlap': len(scene_sets[a] & scene_sets[b]),
                             'image_overlap': len(image_sets[a] & image_sets[b])}
        assert not (scene_sets[a] & scene_sets[b]), (a,b,'scene leakage')
        assert not (image_sets[a] & image_sets[b]), (a,b,'image leakage')
    write_json(ROOT/'data/preparation_report.json', {'passed':True,'counts':counts,'mask_counts':mask_counts,
               'source_csv_sha256':digest(SOURCE), 'annotation_sha256':hashes,
               'split_overlap':overlaps,'raw_image_shape':[1080,1920], 'overlay_images_used':False})
    print('DATA_READY', counts, overlaps, flush=True)


if __name__ == '__main__': main()
