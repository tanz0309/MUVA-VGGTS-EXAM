"""Full MUVA GT fragmentation census and separate paired-mask renders.

This script never loads a neural network or changes source datasets.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mu
from scipy import ndimage

from fragment_metrics import mask_metrics, choose_reference

DATA = Path('/root/datasets/MUVA')
STATS = Path('/root/datasets/MUVA_case_statistics')
RENDERS = Path('/root/datasets/MUVA_case_renders')
STRUCT = np.ones((3, 3), np.uint8)


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def rle(value):
    return {'size': value['size'], 'counts': value['counts'].encode('ascii')
            if isinstance(value['counts'], str) else value['counts']}


def load_split(split):
    return json.loads((DATA / 'annotations' / f'AiFi-MUVA_{split}.json').read_text())


def analyze_one(job):
    split, image, ann = job
    vr, ar = rle(ann['visible_segmentation']), rle(ann['segmentation'])
    boxes = [mu.toBbox(vr), mu.toBbox(ar)]
    boxes = [b for b in boxes if b[2] > 0 and b[3] > 0]
    if boxes:
        x0 = max(0, math.floor(min(b[0] for b in boxes)))
        y0 = max(0, math.floor(min(b[1] for b in boxes)))
        x1 = min(image['width'], math.ceil(max(b[0] + b[2] for b in boxes)))
        y1 = min(image['height'], math.ceil(max(b[1] + b[3] for b in boxes)))
        visible = mu.decode(vr)[y0:y1, x0:x1].copy()
        amodal = mu.decode(ar)[y0:y1, x0:x1].copy()
    else:
        visible = amodal = np.zeros((1, 1), np.uint8)
    metrics = mask_metrics(visible, amodal)
    assert metrics['visible_area'] == int(mu.area(vr)), (split, ann['id'], 'visible')
    assert metrics['amodal_area'] == int(mu.area(ar)), (split, ann['id'], 'amodal')
    return {'split': split, 'scene_id': image['planogram_id'],
            'image_id': image['id'], 'annotation_id': ann['id'],
            'object_id': ann['object_id'], 'category_id': ann['category_id'],
            'view_id': image['view_id'], 'image_file': image['file_name'],
            'group': image['file_name'].split('/')[0], **metrics}


def case_name(target, ref):
    return (f"{target['split']}_{target['scene_id']}_obj{target['object_id']}"
            f"_v{ref['view_id']}_to_v{target['view_id']}")


def make_pairs(rows):
    objects = defaultdict(list)
    for row in rows:
        objects[(row['split'], row['scene_id'], row['object_id'])].append(row)
    pairs = []
    for records in objects.values():
        assert len({r['view_id'] for r in records}) == len(records), 'Duplicate object/view'
        for target in records:
            if target['visible_area'] <= 0 or target['amodal_area'] <= 0:
                continue
            ref = choose_reference(target, records)
            if ref is None:
                continue
            pairs.append({'case_id': case_name(target, ref), 'split': target['split'],
                          'scene_id': target['scene_id'], 'object_id': target['object_id'],
                          'group': target['group'], 'category_id': target['category_id'],
                          'reference_annotation_id': ref['annotation_id'],
                          'target_annotation_id': target['annotation_id'],
                          'reference_view_id': ref['view_id'], 'target_view_id': target['view_id'],
                          'reference_visibility': ref['visibility_ratio'],
                          'target_visibility': target['visibility_ratio'],
                          'reference_at_least_80pct': ref['visibility_ratio'] >= .8,
                          'target_components_raw': target['components_raw'],
                          'target_components_main': target['components_main'],
                          'target_occlusion_split_main': target['occlusion_split_main'],
                          'target_visible_area': target['visible_area'],
                          'target_component_areas_main': target['component_areas_main'],
                          'reference_image': ref['image_file'], 'target_image': target['image_file']})
    pairs.sort(key=lambda p: (p['split'], p['scene_id'], p['object_id'], p['target_view_id']))
    assert len({p['case_id'] for p in pairs}) == len(pairs)
    return pairs


def summarize(rows, pairs):
    visible = [r for r in rows if r['visible_area'] > 0 and r['amodal_area'] > 0]
    def rate(n, d):
        return round(100 * n / d, 6) if d else None
    groups = defaultdict(list)
    for r in rows:
        for key in [('all', 'all'), ('split', r['split']), ('group', r['group']),
                    ('split_group', r['split'] + '/' + r['group'])]:
            groups[key].append(r)
    tables = []
    for (kind, name), rs in groups.items():
        vs = [r for r in rs if r['visible_area'] > 0 and r['amodal_area'] > 0]
        multi = sum(r['components_main'] >= 2 for r in vs)
        occ = sum(r['occlusion_split_main'] for r in vs)
        tables.append({'grouping': kind, 'name': name, 'annotations': len(rs),
                       'visible_valid_instances': len(vs),
                       'multifragment_main_count': multi,
                       'multifragment_main_percent': rate(multi, len(vs)),
                       'occlusion_split_candidate_count': occ,
                       'occlusion_split_candidate_percent': rate(occ, len(vs))})
    sensitivity = []
    for key in ['raw', 'min10', 'halfpct', 'main', '2pct']:
        n = sum(r['components_' + key] >= 2 for r in visible)
        sensitivity.append({'threshold': key, 'multifragment_count': n,
                            'visible_valid_denominator': len(visible), 'percent': rate(n, len(visible))})
    objects = defaultdict(list); scenes = defaultdict(list); images = defaultdict(list)
    for r in visible:
        objects[(r['split'], r['scene_id'], r['object_id'])].append(r)
        scenes[(r['split'], r['scene_id'])].append(r)
        images[(r['split'], r['image_id'])].append(r)
    levels = []
    for label, grouping in [('unique_object_with_any_visible_view', objects),
                            ('scene_with_visible_instances', scenes), ('image_with_visible_instances', images)]:
        n = sum(any(r['components_main'] >= 2 for r in rs) for rs in grouping.values())
        levels.append({'unit': label, 'denominator': len(grouping),
                       'has_multifragment_count': n, 'percent': rate(n, len(grouping))})
    pair_multi = sum(p['target_components_main'] >= 2 for p in pairs)
    pair_occ = sum(p['target_occlusion_split_main'] for p in pairs)
    strong = [p for p in pairs if p['reference_at_least_80pct']]
    summary = {'annotation_count': len(rows), 'visible_valid_instances': len(visible),
               'empty_visible_annotations': sum(r['visible_area'] == 0 for r in rows),
               'zero_amodal_annotations': sum(r['amodal_area'] == 0 for r in rows),
               'visible_outside_amodal_annotations': sum(r['visible_outside_amodal_px'] > 0 for r in rows),
               'multifragment_count': sum(r['components_main'] >= 2 for r in visible),
               'multifragment_percent': rate(sum(r['components_main'] >= 2 for r in visible), len(visible)),
               'occlusion_split_candidate_count': sum(r['occlusion_split_main'] for r in visible),
               'occlusion_split_candidate_percent': rate(sum(r['occlusion_split_main'] for r in visible), len(visible)),
               'pair_count': len(pairs), 'unpaired_visible_instances': len(visible) - len(pairs),
               'paired_multifragment_count': pair_multi, 'paired_multifragment_percent': rate(pair_multi, len(pairs)),
               'paired_occlusion_split_candidate_count': pair_occ,
               'paired_occlusion_split_candidate_percent': rate(pair_occ, len(pairs)),
               'pairs_with_reference_80pct': len(strong),
               'strong_reference_multifragment_count': sum(p['target_components_main'] >= 2 for p in strong),
               'connectivity': 8, 'main_component_threshold': 'max(10, ceil(0.01 * visible_area))',
               'distribution_components_main': dict(sorted(Counter(r['components_main'] for r in visible).items())),
               'unit_summaries': levels, 'sensitivity': sensitivity}
    csv_write(STATS / 'summary_by_split_and_group.csv', tables)
    csv_write(STATS / 'threshold_sensitivity.csv', sensitivity)
    csv_write(STATS / 'summary_by_counting_unit.csv', levels)
    dump(STATS / 'summary.json', summary)
    return summary


def analyze(workers):
    started = time.monotonic(); rows = []
    with (STATS / 'instance_statistics.jsonl').open('w') as out, mp.Pool(workers) as pool:
        for split in ['train', 'val', 'test']:
            data = load_split(split)
            images = {im['id']: im for im in data['images']}
            jobs = ((split, images[a['image_id']], a) for a in data['annotations'])
            count = 0
            for row in pool.imap_unordered(analyze_one, jobs, chunksize=48):
                rows.append(row); out.write(json.dumps(row) + '\n'); count += 1
                if count % 5000 == 0:
                    print('ANALYZED', split, count, '/', len(data['annotations']),
                          'elapsed_seconds', round(time.monotonic() - started), flush=True)
            assert count == len(data['annotations'])
            print('SPLIT_DONE', split, count, flush=True)
            del data, images
    rows.sort(key=lambda r: (r['split'], r['scene_id'], r['object_id'], r['view_id']))
    csv_write(STATS / 'all_instance_statistics.csv', rows)
    pairs = make_pairs(rows)
    csv_write(STATS / 'all_pairs.csv', pairs)
    csv_write(STATS / 'multifragment_pairs.csv', [p for p in pairs if p['target_components_main'] >= 2])
    with (STATS / 'pairs.jsonl').open('w') as f:
        for p in pairs:
            f.write(json.dumps(p) + '\n')
    summary = summarize(rows, pairs)
    print('STATISTICS_DONE', json.dumps(summary), flush=True)


def make_overlay(base, ann, row, role):
    mask = mu.decode(rle(ann['visible_segmentation'])).astype(bool)
    x, y, w, h = map(int, mu.toBbox(rle(ann['visible_segmentation'])))
    image = base.copy()
    crop = image.crop((x, y, x + w, y + h))
    m = mask[y:y+h, x:x+w]
    fill = Image.fromarray((m.astype(np.uint8) * 92), 'L')
    crop.paste((25, 230, 130), (0, 0, w, h), fill)
    edge = m & ~ndimage.binary_erosion(m, structure=STRUCT, border_value=0)
    crop.paste((30, 255, 150), (0, 0, w, h), Image.fromarray(edge.astype(np.uint8) * 255, 'L'))
    image.paste(crop, (x, y))
    # Header lies outside original image pixels. Both files remain separate.
    canvas = Image.new('RGB', (image.width, image.height + 90), (20, 26, 36))
    canvas.paste(image, (0, 90))
    draw = ImageDraw.Draw(canvas); font = ImageFont.load_default(size=27)
    draw.text((20, 12), f"{role} | {row['split']} scene {row['scene_id']} | object {row['object_id']} | view {row['view_id']}", font=font, fill='white')
    draw.text((20, 48), f"Visible: {row['visibility_ratio']:.1%} | components: {row['components_main']} (raw {row['components_raw']}) | green = visible GT mask", font=font, fill=(160, 245, 200))
    draw.rectangle((max(0,x-4), max(90,y+86), min(image.width-1,x+w+4), min(canvas.height-1,y+h+94)), outline=(255, 195, 60), width=3)
    buf = io.BytesIO(); canvas.save(buf, 'JPEG', quality=90, subsampling=0)
    return buf.getvalue()


def render_scene(job):
    scene_pairs, ann_map, row_map = job
    base_images = {}; encoded = {}; completed = 0; reused = 0
    for pair in scene_pairs:
        folder = RENDERS / pair['case_id']; folder.mkdir(exist_ok=True)
        for role, key in [('reference', 'reference_annotation_id'), ('target', 'target_annotation_id')]:
            destination = folder / (role + '.jpg')
            if destination.is_file():
                reused += 1; continue
            aid = pair[key]; row = row_map[aid]; ann = ann_map[aid]
            cache_key = (aid, role)
            if cache_key not in encoded:
                image_key = row['image_file']
                if image_key not in base_images:
                    with Image.open(DATA / 'images' / image_key) as im:
                        base_images[image_key] = im.convert('RGB')
                encoded[cache_key] = make_overlay(base_images[image_key], ann, row, role.upper())
            temporary = destination.with_suffix('.jpg.part')
            temporary.write_bytes(encoded[cache_key]); temporary.replace(destination)
        completed += 1
    return completed, reused


def render(workers, limit_scenes):
    rows = [json.loads(l) for l in (STATS / 'instance_statistics.jsonl').open()]
    pairs = [json.loads(l) for l in (STATS / 'pairs.jsonl').open()]
    grouped_pairs = defaultdict(list)
    for p in pairs:
        grouped_pairs[(p['split'], p['scene_id'])].append(p)
    total = 0; total_scenes = 0; reused_files = 0; started = time.monotonic()
    with mp.Pool(workers) as pool:
        for split in ['train', 'val', 'test']:
            data = load_split(split); annotations = {a['id']: a for a in data['annotations']}
            lookup = {r['annotation_id']: r for r in rows if r['split'] == split}
            jobs = []
            for (s, scene), scene_pairs in sorted(grouped_pairs.items()):
                if s != split:
                    continue
                if limit_scenes and total_scenes + len(jobs) >= limit_scenes:
                    break
                aids = {p[k] for p in scene_pairs for k in ['reference_annotation_id', 'target_annotation_id']}
                jobs.append((scene_pairs, {i: annotations[i] for i in aids}, {i: lookup[i] for i in aids}))
            for count, reused in pool.imap_unordered(render_scene, jobs, chunksize=1):
                total += count; total_scenes += 1; reused_files += reused
                if total_scenes % 20 == 0:
                    print('RENDERED', total, '/', len(pairs), 'pairs; scenes', total_scenes,
                          'elapsed_seconds', round(time.monotonic() - started), flush=True)
                    dump(STATS / 'render_progress.json', {'complete_pairs': total, 'total_pairs': len(pairs),
                                                         'complete_scenes': total_scenes})
            if limit_scenes and total_scenes >= limit_scenes:
                break
    report = {'rendered_pairs': total, 'expected_pairs': len(pairs),
              'reused_files': reused_files, 'limited_preview': bool(limit_scenes),
              'elapsed_seconds': round(time.monotonic() - started), 'render_root': str(RENDERS)}
    dump(STATS / ('preview_report.json' if limit_scenes else 'render_report.json'), report)
    print('RENDER_DONE', json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['analyze', 'render'])
    p.add_argument('--workers', type=int, default=12); p.add_argument('--limit-scenes', type=int, default=0)
    args = p.parse_args()
    if args.stage == 'analyze': analyze(args.workers)
    else: render(args.workers, args.limit_scenes)


if __name__ == '__main__':
    main()
