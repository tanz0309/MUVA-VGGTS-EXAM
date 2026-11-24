"""Offline diagnostics for the VGGT-S phase 1 Block-First selector.

Read-only with respect to the formal algorithm, configs, checkpoints and
existing result directories. It reuses the 40 candidate points stored in an
existing run's prompts/*.json, re-runs only the selector at several eps
values, and uses GT masks only for post-hoc diagnostics.

Usage:
  /opt/conda/envs/vggts/bin/python tools/diagnose_block_first.py \
    --run outputs/vggts_official/phase1/test_block_first_run01 \
    --baseline outputs/vggts_official/evaluate/test_muva_seed42_run01 \
    --split test \
    --eps 0.05 0.04 0.03 0.02 0.01 \
    --output outputs/vggts_official/phase1/diagnostics_run01
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import load_rows  # noqa: E402
from vggts_point_selector import SelectorConfig, select_target_points  # noqa: E402


def official_kmeans():
    config = json.loads((ROOT / 'configs/vggts_official.json').read_text())
    repo_src = str(Path(config['repo']) / 'src')
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)
    from utils import cluster_points
    return cluster_points


def to_original(points_crop, meta, image_size):
    points = np.asarray(points_crop, dtype=np.float64)
    return (points * meta['crop_size'] / image_size
            + np.array([meta['x1'], meta['y1']], dtype=np.float64)) * 2.0


def gt_analysis(gt_path):
    with Image.open(gt_path) as image:
        gt = np.asarray(image.convert('L')) > 0
    labels, _ = ndi.label(gt, structure=np.ones((3, 3), np.uint8))
    areas = np.bincount(labels.ravel())
    visible = int(gt.sum())
    threshold = max(10, int(np.ceil(visible * .01)))
    ids = np.flatnonzero(areas >= threshold)
    ids = ids[ids != 0]
    order = ids[np.argsort(-areas[ids], kind='stable')]
    smallest_area = int(areas[ids].min()) if len(ids) else 0
    smallest = set(int(i) for i in ids if int(areas[i]) == smallest_area)
    return {'labels': labels, 'areas': areas, 'ids': [int(i) for i in ids],
            'ranked': [int(i) for i in order], 'smallest': smallest,
            'threshold': threshold, 'shape': gt.shape}


def lookup(labels, points_original, shape):
    height, width = shape
    xs = np.clip(points_original[:, 0].astype(np.int64), 0, width - 1)
    ys = np.clip(points_original[:, 1].astype(np.int64), 0, height - 1)
    return labels[ys, xs].astype(np.int64)


def prediction_coverage(pred_path, gt):
    with Image.open(pred_path) as image:
        pred = np.asarray(image.convert('L')) > 0
    if pred.shape != gt['shape']:
        return None
    if not pred.any():
        return {int(i): 0.0 for i in gt['ids']}
    hits = np.bincount(gt['labels'][pred], minlength=len(gt['areas']))
    return {int(i): float(hits[i] / gt['areas'][i]) for i in gt['ids']}


def quantiles(values):
    if not values:
        return None
    return {'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'min': float(np.min(values)), 'max': float(np.max(values))}


def distribution(counter):
    return {str(key): int(value) for key, value in sorted(counter.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True,
                        help='Existing Block-First run with prompts/*.json candidate points')
    parser.add_argument('--baseline', type=Path, default=None,
                        help='Optional KMeans run for prompt/mask comparison')
    parser.add_argument('--split', default='test')
    parser.add_argument('--eps', type=float, nargs='+', default=[0.05, 0.04, 0.03, 0.02, 0.01])
    parser.add_argument('--min-samples', type=int, default=2)
    parser.add_argument('--budget', type=int, default=5)
    parser.add_argument('--image-size', type=int, default=518)
    parser.add_argument('--viz-cases', type=int, default=12)
    parser.add_argument('--limit', type=int, default=0, help='Debug only; 0 means all cases')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run.resolve()
    out = args.output.resolve()
    if not (run_dir / 'prompts').is_dir():
        raise SystemExit(f'Missing prompts directory: {run_dir / "prompts"}')
    if out.exists():
        raise SystemExit(f'Output directory already exists: {out}')
    if args.limit < 0:
        raise SystemExit('--limit must be >= 0')

    run_config = json.loads((run_dir / 'run_config.json').read_text())
    selector_meta = run_config.get('point_selector') or {}
    effective = selector_meta.get('effective_config') or {}
    reference_eps = effective.get('eps')
    if effective.get('method') != 'block_first' or reference_eps is None:
        raise SystemExit('The --run directory is not a Block-First run with a recorded eps')
    reference_eps = float(reference_eps)

    rows = load_rows(args.split)
    if args.limit:
        rows = rows[:args.limit]
    gt_paths = {row['case_id']: row['supervision']['target_visible_mask'] for row in rows}
    native = official_kmeans()

    eps_values = sorted(set(float(e) for e in args.eps) | {reference_eps})
    out.mkdir(parents=True)
    per_eps_dirs = {}
    for eps in eps_values:
        directory = out / f'eps_{eps:.4f}'
        directory.mkdir()
        per_eps_dirs[eps] = directory

    # Per-eps accumulators and per-case records.
    diagnostics = {eps: [] for eps in eps_values}
    selector_rows = {eps: [] for eps in eps_values}
    correspondence = {eps: [] for eps in eps_values}
    fallbacks = {eps: defaultdict(list) for eps in eps_values}
    cluster_counts = defaultdict(dict)  # case_id -> eps -> k
    case_meta = {}
    missing_points = []
    reference_matches = {'match': 0, 'mismatch': 0, 'mismatch_cases': []}
    upstream_stats = Counter()
    prompt_vs_mask = {'block_first': Counter(), 'baseline': Counter()}
    baseline_available = args.baseline is not None and (Path(args.baseline) / 'prompts').is_dir()

    for index, row in enumerate(rows):
        case_id = row['case_id']
        prompt_path = run_dir / 'prompts' / f'{case_id}.json'
        if not prompt_path.is_file():
            missing_points.append({'case_id': case_id, 'reason': 'missing prompts json'})
            continue
        saved = json.loads(prompt_path.read_text())
        selector_saved = saved.get('point_selector') or {}
        candidate = selector_saved.get('candidate_points_crop')
        meta = saved.get('target_crop_960x540')
        if candidate is None or meta is None:
            missing_points.append({'case_id': case_id, 'reason': 'missing candidate_points_crop/meta'})
            continue
        candidate = np.asarray(candidate, dtype=np.float32)
        if candidate.ndim != 2 or candidate.shape[1] != 2 or len(candidate) < args.budget:
            missing_points.append({'case_id': case_id, 'reason': f'bad candidate shape {candidate.shape}'})
            continue

        gt = gt_analysis(gt_paths[case_id])
        original_points = to_original(candidate, meta, args.image_size)
        candidate_labels = lookup(gt['labels'], original_points, gt['shape'])
        significant = set(gt['ids'])
        no_support = [cid for cid in gt['ids'] if not np.any(candidate_labels == cid)]
        if no_support:
            upstream_stats['cases_with_missing_support'] += 1
        else:
            upstream_stats['cases_all_supported'] += 1
        upstream_stats['cases'] += 1
        upstream_stats['components_total'] += len(gt['ids'])
        upstream_stats['components_with_support'] += len(gt['ids']) - len(no_support)

        saved_pred_cov = None
        pred_path = run_dir / 'predictions' / f'{case_id}.png'
        if pred_path.is_file():
            saved_pred_cov = prediction_coverage(pred_path, gt)

        baseline_saved = None
        baseline_pred_cov = None
        if baseline_available:
            baseline_prompt = Path(args.baseline) / 'prompts' / f'{case_id}.json'
            baseline_pred = Path(args.baseline) / 'predictions' / f'{case_id}.png'
            if baseline_prompt.is_file():
                baseline_saved = json.loads(baseline_prompt.read_text())
            if baseline_pred.is_file():
                baseline_pred_cov = prediction_coverage(baseline_pred, gt)

        case_meta[case_id] = {'gt_path': gt_paths[case_id],
                              'gt_ids': list(gt['ids']),
                              'gt_areas': [int(gt['areas'][i]) for i in gt['ids']],
                              'candidate': candidate, 'original': original_points,
                              'meta': meta, 'candidate_labels': candidate_labels,
                              'saved_pred_cov': saved_pred_cov}

        for eps in eps_values:
            config = SelectorConfig(method='block_first', budget=args.budget,
                                    eps=eps, min_samples=args.min_samples)
            try:
                result = select_target_points(candidate, image_size=args.image_size,
                                              config=config, native_kmeans=native)
            except Exception as error:
                fallbacks[eps]['selector_error'].append(case_id)
                diagnostics[eps].append({'case_id': case_id, 'error': repr(error), 'eps': eps})
                continue
            audit = result.audit
            fallback = audit.get('fallback_reason')
            if fallback:
                fallbacks[eps][fallback].append(case_id)

            clusters = audit.get('clusters') or []
            labels_by_candidate = audit.get('cluster_labels_by_candidate')
            selected = result.selected_indices
            if eps == reference_eps:
                saved_indices = selector_saved.get('selected_indices')
                same = ((saved_indices is None and selected is None)
                        or (saved_indices is not None and selected is not None
                            and list(map(int, saved_indices)) == list(map(int, selected))))
                if same:
                    reference_matches['match'] += 1
                else:
                    reference_matches['mismatch'] += 1
                    reference_matches['mismatch_cases'].append(case_id)
            selector_rows[eps].append({'case_id': case_id, 'eps': eps,
                                       'method_effective': audit.get('method_effective'),
                                       'fallback_reason': fallback,
                                       'cluster_count': len(clusters),
                                       'unique_candidate_count': audit.get('unique_candidate_count'),
                                       'noise_count_unique': audit.get('noise_count_unique'),
                                       'noise_count_raw': audit.get('noise_count_raw'),
                                       'selected_indices': None if selected is None else list(map(int, selected)),
                                       'selected_points_crop': audit.get('selected_points_crop'),
                                       'selector_seconds': audit.get('selector_seconds')})

            cluster_count = len(clusters)
            if not fallback:
                cluster_counts[case_id][eps] = cluster_count
            unique_count = audit.get('unique_candidate_count') or candidate.shape[0]
            max_share = (max((c['unique_support'] for c in clusters), default=0) / unique_count
                         if clusters else 0.0)
            merged = 0
            for cluster in clusters:
                touched = sorted({int(candidate_labels[i]) for i in cluster['member_candidate_indices']}
                                 & significant)
                if len(touched) >= 2:
                    merged += 1
                correspondence[eps].append({
                    'record_type': 'cluster', 'case_id': case_id, 'cluster_id': cluster['cluster_id'],
                    'component_id': '', 'cluster_size': cluster['unique_support'],
                    'component_area': '', 'retained': cluster['retained'],
                    'quota': cluster['quota'], 'gt_components_touched': len(touched),
                    'gt_labels': ';'.join(map(str, touched)), 'merged': bool(len(touched) >= 2),
                    'clusters_touching': '', 'split': '', 'candidate_points': cluster['raw_support'],
                    'noise_points': '', 'retained_clusters': '',
                    'prompts_inside': '', 'pred_coverage': '', 'pred_covered': ''})

            cluster_of_candidate = (np.asarray(labels_by_candidate, dtype=np.int64)
                                    if labels_by_candidate is not None
                                    else np.full(len(candidate), -1, dtype=np.int64))
            retained_ids = {c['cluster_id'] for c in clusters if c['retained']}

            prompts_original = to_original(audit.get('selected_points_crop') or [], meta,
                                           args.image_size)
            prompt_labels = lookup(gt['labels'], prompts_original, gt['shape'])

            split_components = 0
            lost_discard = 0
            lost_noise = 0
            prompt_covered = 0
            for component_id in gt['ids']:
                support = np.flatnonzero(candidate_labels == component_id)
                noise_points = int((cluster_of_candidate[support] == -1).sum())
                touching = sorted({int(cluster_of_candidate[i]) for i in support} - {-1})
                retained_touching = [cid for cid in touching if cid in retained_ids]
                is_split = len(touching) >= 2
                if is_split:
                    split_components += 1
                if len(support) and not touching:
                    lost_noise += 1
                if len(support) and touching and not retained_touching:
                    lost_discard += 1
                prompts_inside = int((prompt_labels == component_id).sum())
                if prompts_inside:
                    prompt_covered += 1
                pred_cov = None if saved_pred_cov is None else saved_pred_cov.get(component_id)
                correspondence[eps].append({
                    'record_type': 'component', 'case_id': case_id, 'cluster_id': '',
                    'component_id': component_id, 'cluster_size': '',
                    'component_area': int(gt['areas'][component_id]), 'retained': '',
                    'quota': '', 'gt_components_touched': '',
                    'gt_labels': '', 'merged': '',
                    'clusters_touching': len(touching), 'split': bool(is_split),
                    'candidate_points': len(support), 'noise_points': noise_points,
                    'retained_clusters': len(retained_touching),
                    'prompts_inside': prompts_inside,
                    'pred_coverage': '' if pred_cov is None else round(pred_cov, 6),
                    'pred_covered': '' if pred_cov is None else bool(pred_cov >= .5)})

            quotas = [c['quota'] for c in clusters if c['retained']]
            largest = max(clusters, key=lambda c: c['unique_support']) if clusters else None
            largest_support_share = (largest['unique_support'] / unique_count) if largest else None
            largest_quota_share = (largest['quota'] / args.budget) if largest and largest['retained'] else None
            diagnostics[eps].append({
                'eps': eps, 'case_id': case_id, 'gt_components': len(gt['ids']),
                'candidate_count': int(len(candidate)),
                'unique_candidate_count': int(unique_count),
                'cluster_count': cluster_count,
                'retained_cluster_count': len(retained_ids),
                'max_cluster_share': max_share,
                'largest_cluster_support_share': largest_support_share,
                'largest_cluster_quota_share': largest_quota_share,
                'noise_unique': audit.get('noise_count_unique'),
                'noise_raw': audit.get('noise_count_raw'),
                'fallback_reason': fallback,
                'merged_clusters': merged,
                'split_components': split_components,
                'lost_to_discard': lost_discard,
                'lost_to_noise': lost_noise,
                'components_no_support': len(no_support),
                'prompt_covered_components': prompt_covered,
                'prompt_all_covered': prompt_covered == len(gt['ids']) if gt['ids'] else None,
                'mean_quota_retained': float(np.mean(quotas)) if quotas else None,
                'max_quota': int(max(quotas)) if quotas else None,
                'selector_seconds': audit.get('selector_seconds')})

            if eps == reference_eps:
                if saved_pred_cov is not None:
                    prompt_vs_mask['block_first']['cases_with_pred'] += 1
                prompts_all = bool(gt['ids']) and prompt_covered == len(gt['ids'])
                pred_all = (None if saved_pred_cov is None
                            else all(saved_pred_cov[i] >= .5 for i in gt['ids']))
                if prompts_all:
                    prompt_vs_mask['block_first']['prompts_all'] += 1
                    if pred_all:
                        prompt_vs_mask['block_first']['prompts_and_mask_all'] += 1
                    elif pred_all is False:
                        prompt_vs_mask['block_first']['prompts_all_mask_miss'] += 1
                if baseline_saved is not None:
                    if baseline_pred_cov is not None:
                        prompt_vs_mask['baseline']['cases_with_pred'] += 1
                    base_original = np.asarray(baseline_saved.get('positive_prompts_original'), float)
                    base_labels = lookup(gt['labels'], base_original, gt['shape'])
                    base_covered = sum(1 for i in gt['ids'] if np.any(base_labels == i))
                    base_prompts_all = bool(gt['ids']) and base_covered == len(gt['ids'])
                    base_pred_all = (None if baseline_pred_cov is None
                                     else all(baseline_pred_cov[i] >= .5 for i in gt['ids']))
                    if base_prompts_all:
                        prompt_vs_mask['baseline']['prompts_all'] += 1
                        if base_pred_all:
                            prompt_vs_mask['baseline']['prompts_and_mask_all'] += 1
                        elif base_pred_all is False:
                            prompt_vs_mask['baseline']['prompts_all_mask_miss'] += 1

        if (index + 1) % 200 == 0:
            print('DIAGNOSE', index + 1, '/', len(rows), flush=True)

    # Write per-eps files.
    sweep_rows = []
    for eps in eps_values:
        directory = per_eps_dirs[eps]
        diag_rows = diagnostics[eps]
        valid = [r for r in diag_rows if 'error' not in r and not r.get('fallback_reason')]
        all_valid = [r for r in diag_rows if 'error' not in r]
        fallback_count = sum(1 for r in all_valid if r.get('fallback_reason'))
        total_clusters = sum(r['cluster_count'] for r in valid)
        merged = sum(r['merged_clusters'] for r in valid)
        split = sum(r['split_components'] for r in valid)
        components = sum(r['gt_components'] for r in valid)
        cluster_dist = Counter(r['cluster_count'] for r in valid)
        noise_unique = sum((r['noise_unique'] or 0) for r in valid)
        unique_total = sum(r['unique_candidate_count'] for r in valid)
        shares = [r['max_cluster_share'] for r in valid if r['cluster_count']]
        quotas = [r['mean_quota_retained'] for r in valid if r['mean_quota_retained'] is not None]
        per_cluster_points = []
        for r in valid:
            if r['cluster_count']:
                per_cluster_points.append(r['unique_candidate_count'] / r['cluster_count'])
        prompt_all = [r for r in all_valid if r['prompt_all_covered'] is not None]
        prompt_frac = [r['prompt_covered_components'] / r['gt_components']
                       for r in all_valid if r['gt_components']]
        sweep_rows.append({
            'eps': eps,
            'cases': len(all_valid),
            'fallback_cases': fallback_count,
            'fallback_rate': fallback_count / len(all_valid) if all_valid else None,
            'mean_cluster_count': float(np.mean([r['cluster_count'] for r in valid])) if valid else None,
            'cluster_count_distribution': json.dumps(distribution(cluster_dist), sort_keys=True),
            'max_cluster_share_mean': float(np.mean(shares)) if shares else None,
            'max_cluster_share_median': float(np.median(shares)) if shares else None,
            'noise_unique_ratio': noise_unique / unique_total if unique_total else None,
            'merged_cluster_rate': merged / total_clusters if total_clusters else None,
            'split_component_rate': split / components if components else None,
            'components_no_support_rate': (sum(r['components_no_support'] for r in all_valid)
                                           / sum(r['gt_components'] for r in all_valid)
                                           if all_valid else None),
            'lost_to_discard_rate': (sum(r['lost_to_discard'] for r in valid) / components
                                     if components else None),
            'lost_to_noise_rate': (sum(r['lost_to_noise'] for r in valid) / components
                                   if components else None),
            'mean_points_per_cluster': float(np.mean(per_cluster_points)) if per_cluster_points else None,
            'mean_quota_retained': float(np.mean(quotas)) if quotas else None,
            'largest_cluster_support_share_mean': float(np.mean(
                [r['largest_cluster_support_share'] for r in valid
                 if r['largest_cluster_support_share'] is not None]))
            if any(r['largest_cluster_support_share'] is not None for r in valid) else None,
            'largest_cluster_quota_share_mean': float(np.mean(
                [r['largest_cluster_quota_share'] for r in valid
                 if r['largest_cluster_quota_share'] is not None]))
            if any(r['largest_cluster_quota_share'] is not None for r in valid) else None,
            'prompt_all_covered_rate': (sum(1 for r in prompt_all if r['prompt_all_covered'])
                                        / len(prompt_all) if prompt_all else None),
            'mean_prompt_covered_fraction': float(np.mean(prompt_frac)) if prompt_frac else None,
            'mean_selector_seconds': float(np.mean([r['selector_seconds'] for r in valid
                                                    if r['selector_seconds'] is not None]))
            if valid else None,
        })

        with (directory / 'cluster_diagnostics.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            fields = ['eps', 'case_id', 'gt_components', 'candidate_count', 'unique_candidate_count',
                      'cluster_count', 'retained_cluster_count', 'max_cluster_share',
                      'largest_cluster_support_share', 'largest_cluster_quota_share',
                      'noise_unique', 'noise_raw', 'fallback_reason', 'merged_clusters',
                      'split_components', 'lost_to_discard', 'lost_to_noise',
                      'components_no_support', 'prompt_covered_components', 'prompt_all_covered',
                      'mean_quota_retained', 'max_quota', 'selector_seconds']
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(diag_rows)
        with (directory / 'cluster_gt_correspondence.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            fields = ['record_type', 'case_id', 'cluster_id', 'component_id', 'cluster_size',
                      'component_area', 'retained', 'quota', 'gt_components_touched', 'gt_labels',
                      'merged', 'clusters_touching', 'split', 'candidate_points', 'noise_points',
                      'retained_clusters', 'prompts_inside', 'pred_coverage', 'pred_covered']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(correspondence[eps])
        fallback_payload = {'eps': eps, 'total': len(all_valid),
                            'fallback_total': fallback_count,
                            'fallback_rate': fallback_count / len(all_valid) if all_valid else None,
                            'by_reason': {reason: {'count': len(cases), 'case_ids': cases}
                                          for reason, cases in fallbacks[eps].items()},
                            'note': 'eps=reference 的 fallback 应与原始运行审计交叉核对'}
        (directory / 'fallback_statistics.json').write_text(
            json.dumps(fallback_payload, ensure_ascii=False, indent=2))
        with (directory / 'selections.jsonl').open('w') as handle:
            for item in selector_rows[eps]:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')

    with (out / 'eps_sweep.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)

    # Point-cloud visualisations per eps.
    def render(eps):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as error:  # pragma: no cover - environment dependent
            print('VISUALISATION_SKIPPED', repr(error), flush=True)
            return
        scores = []
        for case_id, per_eps in cluster_counts.items():
            values = [per_eps[e] for e in eps_values if e in per_eps]
            if not values:
                continue
            scores.append((max(values) - min(values), len(case_meta[case_id]['gt_ids']), case_id))
        scores.sort(reverse=True)
        chosen = [case_id for _, _, case_id in scores[:args.viz_cases]]
        if not chosen:
            return
        columns = 3
        rows_count = int(np.ceil(len(chosen) / columns))
        figure, axes = plt.subplots(rows_count, columns, figsize=(5 * columns, 3.6 * rows_count))
        axes = np.atleast_2d(axes)
        for axis in axes.ravel():
            axis.axis('off')
        for axis, case_id in zip(axes.ravel(), chosen):
            data = case_meta[case_id]
            meta = data['meta']
            gt = gt_analysis(data['gt_path'])
            labels = gt['labels']
            x1, x2 = int(meta['x1'] * 2) - 10, int(meta['x2'] * 2) + 10
            y1, y2 = int(meta['y1'] * 2) - 10, int(meta['y2'] * 2) + 10
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(labels.shape[1], x2), min(labels.shape[0], y2)
            axis.imshow(labels[y1:y2, x1:x2], cmap='Greys', alpha=.35)
            config = SelectorConfig(method='block_first', budget=args.budget,
                                    eps=eps, min_samples=args.min_samples)
            result = select_target_points(data['candidate'], image_size=args.image_size,
                                          config=config, native_kmeans=native)
            selected = result.audit.get('selected_points_crop')
            cluster_labels = result.audit.get('cluster_labels_by_candidate')
            cluster_count = len(result.audit.get('clusters') or [])
            points = data['original']
            axis.scatter(points[:, 0], points[:, 1], c=cluster_labels if cluster_labels is not None else 'k',
                         cmap='tab20', s=18, vmin=-1, vmax=max(1, cluster_count),
                         edgecolors='k', linewidths=.3)
            if selected:
                sp = to_original(selected, meta, args.image_size)
                axis.scatter(sp[:, 0], sp[:, 1], marker='*', s=140, c='red', edgecolors='white',
                             linewidths=.5, label='selected')
            axis.set_xlim(x1, x2)
            axis.set_ylim(y2, y1)
            axis.set_title(f'{case_id}\nk={cluster_count} '
                           f'fb={result.audit.get("fallback_reason")}', fontsize=7)
            axis.axis('on')
        figure.suptitle(f'eps={eps:.2f}  (grey=GT components, colour=DBSCAN cluster, star=selected)', fontsize=10)
        figure.tight_layout()
        figure.savefig(per_eps_dirs[eps] / 'pointcloud.png', dpi=110)
        plt.close(figure)

    for eps in eps_values:
        render(eps)

    # Report.
    sweep_by_eps = {row['eps']: row for row in sweep_rows}
    report = []
    report.append('# VGGT-S Block-First 第一阶段诊断报告')
    report.append('')
    report.append(f'- 数据来源：`{run_dir}`（split={args.split}，参考 eps={reference_eps}）')
    report.append(f'- 候选点：直接复用 prompts/*.json 中的 candidate_points_crop，未运行 VGGT，未重新生成候选点')
    report.append(f'- GT：仅用于离线对应分析，不参与 DBSCAN/Medoid/FPS/选点')
    report.append(f'- eps 扫描：{", ".join(f"{e:.2f}" for e in eps_values)}；min_samples={args.min_samples}，budget={args.budget}')
    report.append(f'- 缺失候选点的样本：{len(missing_points)}')
    if missing_points:
        report.append('')
        report.append('| case | 原因 |')
        report.append('| --- | --- |')
        for item in missing_points[:20]:
            report.append(f'| {item["case_id"]} | {item["reason"]} |')
    report.append('')
    report.append('## 1. 实现复现性')
    report.append('')
    report.append(f'参考 eps 下离线重算与已保存 selected_indices 的一致 / 不一致：'
                  f'**{reference_matches["match"]} / {reference_matches["mismatch"]}**。')
    if reference_matches['mismatch']:
        report.append('不一致样例：' + ', '.join(reference_matches['mismatch_cases'][:10]))
        report.append('→ 说明审计候选点不足或实现有漂移，后续 eps 结论需谨慎。')
    else:
        report.append('→ 离线复算逐样本一致，说明后续 eps 对比与原始运行使用同一候选点和同一算法实现。')
    report.append('')
    report.append('## 2. 上游候选点覆盖（与 eps 无关）')
    report.append('')
    cases = upstream_stats['cases'] or 1
    comps = upstream_stats['components_total'] or 1
    report.append(f'- 样本 {upstream_stats["cases"]} 个，显著 GT 分块 {upstream_stats["components_total"]} 个。')
    report.append(f'- 40 个候选点完全没有落在某显著分块内的样本：**{upstream_stats["cases_with_missing_support"]} '
                  f'({upstream_stats["cases_with_missing_support"]/cases*100:.1f}%)**。')
    report.append(f'- 有候选点支持的显著分块：{upstream_stats["components_with_support"]} / '
                  f'{upstream_stats["components_total"]} ({upstream_stats["components_with_support"]/comps*100:.1f}%)。')
    report.append(f'- 40 个候选点能覆盖**全部**显著分块的样本：{upstream_stats["cases_all_supported"]} '
                  f'({upstream_stats["cases_all_supported"]/cases*100:.1f}%)。'
                  '这同时是 5 点选择能做到“覆盖全部分块”的样本数上限。')
    report.append('→ 这些缺失分块在 40→5 之前就已丢失，任何 selector 都无法恢复。')
    report.append('')
    report.append('## 3. eps 扫描')
    report.append('')
    report.append('| eps | 平均簇数 | 最大簇占比(均值) | Noise比例 | 回退率 | 簇跨GT分块率 | 分块被切分率 | 分块无候选点 | 丢在簇筛选 | 丢在Noise | 5点覆盖全部分块 |')
    report.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for row in sweep_rows:
        def pct(value):
            return '—' if value is None else f'{value*100:.1f}%'
        mean_clusters = row["mean_cluster_count"]
        report.append(f'| {row["eps"]:.2f} | {"—" if mean_clusters is None else f"{mean_clusters:.2f}"} | '
                      f'{pct(row["max_cluster_share_mean"])} | {pct(row["noise_unique_ratio"])} | '
                      f'{pct(row["fallback_rate"])} | {pct(row["merged_cluster_rate"])} | '
                      f'{pct(row["split_component_rate"])} | {pct(row["components_no_support_rate"])} | '
                      f'{pct(row["lost_to_discard_rate"])} | {pct(row["lost_to_noise_rate"])} | '
                      f'{pct(row["prompt_all_covered_rate"])} |')
    report.append('')
    report.append('簇数分布：')
    for row in sweep_rows:
        report.append(f'- eps={row["eps"]:.2f}: `{row["cluster_count_distribution"]}`')
    report.append('')
    report.append('## 4. 提示点覆盖与最终 Mask 的关系（仅参考 eps）')
    report.append('')
    for key, label in (('baseline', 'KMeans 基线'), ('block_first', 'Block-First')):
        stats = prompt_vs_mask[key]
        with_pred = stats.get('cases_with_pred', 0)
        if not with_pred:
            report.append(f'- {label}：无可用 predictions 对照。')
            continue
        if stats.get('prompts_all'):
            report.append(f'- {label}：5 点覆盖全部分块的 {stats["prompts_all"]} 例中，最终 Mask 仍覆盖全部分块 '
                          f'{stats.get("prompts_and_mask_all", 0)} 例，仍遗漏分块 '
                          f'{stats.get("prompts_all_mask_miss", 0)} 例。')
        else:
            report.append(f'- {label}：有 Mask 对照的 {with_pred} 例中，没有任何一例的 5 个提示点'
                          f'同时落在全部显著 GT 分块内。')
    report.append('→ 提示点覆盖全部分块但 Mask 仍遗漏，说明瓶颈在解码/细化；'
                  '若提示点本身就无法覆盖全部分块，则瓶颈在选点或更上游。')
    reference_corr = correspondence[reference_eps]
    no_support = [row for row in reference_corr
                  if row['record_type'] == 'component' and int(row['candidate_points'] or 0) == 0]
    no_support_covered = [row for row in no_support if row['pred_covered'] is True]
    if no_support:
        report.append(f'- 无任何候选点支持的显著分块 {len(no_support)} 个，其中最终 Mask 仍覆盖 ≥50% 的 '
                      f'{len(no_support_covered)} 个 ({len(no_support_covered)/len(no_support)*100:.1f}%)。'
                      '→ 解码器能在没有提示点的分块上自行恢复，说明这些分块不构成选点收益的上限约束。')
    reference_diag = {row['case_id']: row for row in diagnostics[reference_eps]
                      if 'error' not in row}
    prompt_miss_cases = [row for row in reference_diag.values()
                         if row.get('prompt_all_covered') is False]
    mask_all_cases = 0
    for row in prompt_miss_cases:
        coverage = case_meta[row['case_id']]['saved_pred_cov']
        if coverage is None:
            continue
        gt_ids = case_meta[row['case_id']]['gt_ids']
        if gt_ids and all(coverage.get(i, 0.0) >= .5 for i in gt_ids):
            mask_all_cases += 1
    if prompt_miss_cases:
        report.append(f'- 5 点未覆盖全部分块的 {len(prompt_miss_cases)} 例中，最终 Mask 仍覆盖全部分块的 '
                      f'{mask_all_cases} 例 ({mask_all_cases/len(prompt_miss_cases)*100:.1f}%)。'
                      '→ 提示覆盖与最终 Mask 之间存在明显解耦。')
    report.append('')
    report.append('## 5. 结论判定')
    report.append('')
    def pct_or_dash(value):
        return '—' if value is None else f'{value*100:.1f}%'
    def find(key):
        return {row['eps']: row.get(key) for row in sweep_rows}
    merged = find('merged_cluster_rate')
    noise = find('noise_unique_ratio')
    fallback = find('fallback_rate')
    discard = find('lost_to_discard_rate')
    nosupport = find('components_no_support_rate')
    promptcov = find('prompt_all_covered_rate')
    reference_row = sweep_by_eps[reference_eps]
    max_share = reference_row.get('max_cluster_share_mean')
    mean_clusters = reference_row.get('mean_cluster_count')
    quota_share = reference_row.get('largest_cluster_quota_share_mean')
    support_share = reference_row.get('largest_cluster_support_share_mean')
    report.append('| 假设 | 证据 | 判定 |')
    report.append('| --- | --- | --- |')
    a_support = (max_share is not None and max_share >= .6) or max([v for v in merged.values() if v is not None] or [0]) >= .2
    report.append(f'| A. eps 过大导致簇合并 | 参考 eps：平均簇数 '
                  f'{"—" if mean_clusters is None else f"{mean_clusters:.2f}"}，'
                  f'最大簇平均占 {pct_or_dash(max_share)} 候选点；'
                  f'一个簇跨多个/多个真实分块的合并率：' +
                  '，'.join(f'eps {e:.2f}={pct_or_dash(v)}' for e, v in merged.items()) +
                  ' | ' + ('**支持**：候选点被并成极少的大簇' if a_support else '证据不足') + ' |')
    b_support = max([v for v in list(noise.values()) + list(fallback.values()) if v is not None] or [0]) >= .1
    report.append(f'| B. eps 过小导致 Noise/回退 | Noise比例：' +
                  '，'.join(f'eps {e:.2f}={pct_or_dash(v)}' for e, v in noise.items()) +
                  '；回退率：' +
                  '，'.join(f'eps {e:.2f}={pct_or_dash(v)}' for e, v in fallback.items()) +
                  ' | ' + ('**支持**' if b_support else '证据不足') + ' |')
    c_support = max([v for v in discard.values() if v is not None] or [0]) >= .02
    report.append(f'| C. 小簇在保留阶段被舍弃 | 候选点只落在被舍弃簇里的显著分块比例：' +
                  '，'.join(f'eps {e:.2f}={pct_or_dash(v)}' for e, v in discard.items()) +
                  ' | ' + ('**支持（参数相关）**' if c_support else '证据不足（比例很小）') + ' |')
    if quota_share is not None and support_share is not None:
        if quota_share + .05 < support_share:
            d_text = (f'最大簇平均获得 {pct_or_dash(quota_share)} 的配额，低于其候选点占比 '
                      f'{pct_or_dash(support_share)}；保底 1 点使小簇相对受益，'
                      '本数据中未出现大簇超额占用。')
            d_verdict = '**机制按比例分配，未观察到有害偏向**'
        elif quota_share > support_share + .05:
            d_text = (f'最大簇平均获得 {pct_or_dash(quota_share)} 的配额，高于其候选点占比 '
                      f'{pct_or_dash(support_share)}。')
            d_verdict = '**支持：大簇获得超额配额**'
        else:
            d_text = (f'最大簇平均获得 {pct_or_dash(quota_share)} 的配额，与其候选点占比 '
                      f'{pct_or_dash(support_share)} 基本一致。')
            d_verdict = '**接近按比例分配**'
    else:
        d_text = '无可用配额数据。'
        d_verdict = '无法判断'
    report.append(f'| D. 配额偏向大簇 | {d_text} | {d_verdict} |')
    e_support = max([v for v in nosupport.values() if v is not None] or [0]) >= .05
    report.append(f'| E. 上游候选点本身缺失 | 无任何候选点支持的显著分块比例：' +
                  '，'.join(f'eps {e:.2f}={pct_or_dash(v)}' for e, v in nosupport.items()) +
                  f'；{upstream_stats["cases_with_missing_support"]}/{upstream_stats["cases"]} 个样本至少缺一个分块 | '
                  + ('**强支持**' if e_support else '证据不足') + ' |')
    split = find('split_component_rate')
    f_evidence = (max([v for v in merged.values() if v is not None] or [0])
                  + max([v for v in split.values() if v is not None] or [0]))
    report.append(f'| F. 纯空间聚类无法区分真实分块 | 合并率与切分率见第 3 节（两者之和最高 '
                  f'{f_evidence*100:.1f}%）；5 点覆盖全部分块比例：' +
                  '，'.join(f'eps {e:.2f}={pct_or_dash(v)}' for e, v in promptcov.items()) +
                  ' | ' + ('**部分支持**' if f_evidence >= .1 else '证据不足') +
                  '；在 E 缺失严重时不能单独归因 |')
    report.append('')
    report.append('## 6. 解释边界')
    report.append('')
    report.append('- 本报告只分析已有候选点；没有重新运行 VGGT，也没有改变上游跟踪/过滤/裁剪。')
    report.append('- 最终预测 Mask 只在参考 eps 的既有运行中存在；其他 eps 的结论仅限提示点覆盖，不能推断 Mask 效果。')
    report.append('- eps 扫描结果不能代替完整 val/test；参数选择应在 val 上进行，test 保持冻结。')
    report.append('- 簇合并/切分以“候选点落在哪个 GT 分块”定义，不等于真实实例级对应。')
    report.append('')
    report.append('## 7. 文件')
    report.append('')
    report.append('- `eps_sweep.csv`：各 eps 汇总。')
    report.append('- `eps_*/cluster_diagnostics.csv`：逐样本统计。')
    report.append('- `eps_*/cluster_gt_correspondence.csv`：簇与 GT 分块逐条对应。')
    report.append('- `eps_*/fallback_statistics.json`：回退原因与样本。')
    report.append('- `eps_*/selections.jsonl`：每个 eps 的选点与审计摘要。')
    report.append('- `eps_*/pointcloud.png`：代表性样本点云（GT 灰度底图 + 簇颜色 + 选中点）。')
    (out / 'diagnostic_report.md').write_text('\n'.join(report) + '\n')
    print('DIAGNOSTICS_COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
