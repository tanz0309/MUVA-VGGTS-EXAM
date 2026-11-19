"""Compare two saved VGGT-S selectors runs without running the model.

Reads existing result directories, checks that they form a valid single-variable
comparison, and writes comparison tables. It never recomputes metrics with a new
definition: it reuses scripts/common.py::summarize.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import summarize

MAIN_METRICS = [
    ('visible_iou', 'Visible mIoU'),
    ('dice', 'Dice'),
    ('boundary_f', 'Boundary F'),
    ('cr50', 'CR@50'),
    ('scr50', 'SCR@50'),
    ('all_components_covered50', 'all_components_covered50'),
]
SUPPLEMENTARY_METRICS = [
    ('cr70', 'CR@70'),
    ('scr70', 'SCR@70'),
    ('all_components_covered70', 'all_components_covered70'),
]
REQUIRED_ROW_FIELDS = ('case_id', 'split', 'components', 'group', 'target_visibility',
                       'inference_seconds', 'peak_allocated_gpu_mib', 'failed')


def _read_json(path):
    if not path.is_file():
        raise ValueError(f'Missing file: {path}')
    return json.loads(path.read_text())


def load_run(run_dir, *, allow_smoke=False):
    run_dir = Path(run_dir)
    run_config = _read_json(run_dir / 'run_config.json')
    status = _read_json(run_dir / 'status.json')
    summary = _read_json(run_dir / 'summary.json')
    per_case = run_dir / 'per_case.jsonl'
    if not per_case.is_file():
        raise ValueError(f'Missing file: {per_case}')
    rows = []
    with per_case.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if status.get('status') != 'complete':
        raise ValueError(f'Run is not complete: {run_dir} -> {status.get("status")}')
    scope = summary.get('scope')
    if not allow_smoke and scope != 'full_split':
        raise ValueError(f'Not a full-split run: {run_dir} -> {scope}')
    if allow_smoke and scope not in ('full_split', 'smoke_only_not_benchmark'):
        raise ValueError(f'Unexpected run scope: {run_dir} -> {scope}')
    ids = [row['case_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate case ids: {run_dir}')
    for row in rows:
        for field in REQUIRED_ROW_FIELDS:
            if field not in row:
                raise ValueError(f'per_case is missing {field}: {row.get("case_id")}')
        for key, _label in MAIN_METRICS + SUPPLEMENTARY_METRICS:
            if key == 'all_components_covered50' or key == 'all_components_covered70':
                if key in row and not isinstance(row[key], (bool, type(None))):
                    raise ValueError(f'{key} must be bool or null: {row["case_id"]}')
                continue
            value = row.get(key)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                raise ValueError(f'Invalid {key}: {row["case_id"]} -> {value}')
    if summary.get('count') != len(rows):
        raise ValueError(f'summary count {summary.get("count")} != per_case {len(rows)}: {run_dir}')
    if summary.get('failures') != sum(bool(row.get('failed')) for row in rows):
        raise ValueError(f'failure count mismatch: {run_dir}')
    audits = {}
    prompts = run_dir / 'prompts'
    for case_id in ids:
        path = prompts / f'{case_id}.json'
        audits[case_id] = json.loads(path.read_text()) if path.is_file() else None
    return {'run_dir': run_dir, 'run_config': run_config, 'status': status,
            'summary': summary, 'rows': rows, 'row_by_id': {row['case_id']: row for row in rows},
            'audits': audits, 'scope': scope}


def _requested_method(loaded):
    selector = (loaded['run_config'].get('point_selector') or {})
    method = (selector.get('effective_config') or {}).get('method')
    if method is None:
        raise ValueError('run_config has no point_selector.effective_config.method; '
                         'this run predates the selector audit')
    return method


def _checkpoint_sha(loaded):
    return ((loaded['run_config'].get('evaluation_checkpoint') or {}).get('sha256'))


def validate_pair(baseline, candidate, *, allow_smoke=False):
    """Raise on hard incompatibilities; return soft completeness diagnostics."""
    errors, notes = [], []
    if _requested_method(baseline) != 'kmeans':
        raise ValueError('baseline must request method="kmeans"')
    if _requested_method(candidate) != 'block_first':
        raise ValueError('candidate must request method="block_first"')
    if baseline['scope'] != candidate['scope']:
        raise ValueError('baseline and candidate scopes differ')
    if not allow_smoke and baseline['scope'] != 'full_split':
        raise ValueError('formal comparison requires full_split runs')
    if allow_smoke and baseline['scope'] == 'smoke_only_not_benchmark' \
            and candidate['scope'] != 'smoke_only_not_benchmark':
        raise ValueError('refusing to mix smoke and full runs')
    checks = {
        'same_scope': baseline['scope'] == candidate['scope'],
        'same_manifests': baseline['run_config'].get('manifests') == candidate['run_config'].get('manifests'),
        'same_config': baseline['run_config'].get('config') == candidate['run_config'].get('config'),
        'same_metric_implementation': baseline['run_config'].get('metric_implementation_sha256')
        == candidate['run_config'].get('metric_implementation_sha256'),
        'same_checkpoint': _checkpoint_sha(baseline) is not None
        and _checkpoint_sha(baseline) == _checkpoint_sha(candidate),
    }
    for name, ok in checks.items():
        if not ok:
            errors.append(name)
    base_ids, cand_ids = list(baseline['row_by_id']), list(candidate['row_by_id'])
    if len(base_ids) != len(cand_ids):
        errors.append('different_sample_counts')
    if set(base_ids) != set(cand_ids):
        errors.append('different_case_id_sets')
    order_changed = base_ids != cand_ids[:len(base_ids)] if not errors else False
    if order_changed:
        notes.append('case order differs; aligned by case_id')
    if errors:
        raise ValueError('Comparison rejected: ' + ', '.join(errors))

    baseline_failures = sum(bool(row.get('failed')) for row in baseline['rows'])
    candidate_failures = sum(bool(row.get('failed')) for row in candidate['rows'])
    failures = {'baseline': baseline_failures, 'block_first': candidate_failures}

    audit_compared = audit_mismatch = audit_unavailable = 0
    for case_id in sorted(set(base_ids)):
        base_audit = baseline['audits'].get(case_id)
        cand_audit = candidate['audits'].get(case_id)
        base_failed = bool(baseline['row_by_id'][case_id].get('failed'))
        cand_failed = bool(candidate['row_by_id'][case_id].get('failed'))
        if base_failed or cand_failed:
            audit_unavailable += 1
            continue
        if not base_audit or not cand_audit:
            audit_unavailable += 1
            notes.append(f'missing audit for successful case {case_id}')
            continue
        base_inputs = base_audit.get('selection_inputs') or {}
        cand_inputs = cand_audit.get('selection_inputs') or {}
        if not base_inputs or not cand_inputs:
            audit_unavailable += 1
            notes.append(f'missing selection_inputs for {case_id}')
            continue
        audit_compared += 1
        if base_inputs != cand_inputs:
            audit_mismatch += 1
            notes.append(f'input fingerprint mismatch for {case_id}')
    if audit_mismatch:
        raise ValueError(f'candidate points/crop/meta mismatch in {audit_mismatch} cases; '
                         'not a single-variable comparison')

    clean = (failures['baseline'] == 0 and failures['block_first'] == 0
             and audit_unavailable == 0)
    return {'checks': checks, 'notes': notes, 'failures': failures,
            'audit_compared': audit_compared, 'audit_unavailable': audit_unavailable,
            'same_order': not order_changed, 'valid_clean_comparison': clean}


def compare_metrics(baseline_rows, candidate_rows):
    """Macro-average with the existing summarize(); verify stored summaries agree."""
    baseline_summary = summarize(baseline_rows)
    candidate_summary = summarize(candidate_rows)
    metrics = {}
    for key, label in MAIN_METRICS + SUPPLEMENTARY_METRICS:
        base_value = baseline_summary.get(key)
        cand_value = candidate_summary.get(key)
        if base_value is None or cand_value is None:
            continue
        base_value, cand_value = float(base_value), float(cand_value)
        metrics[key] = {'label': label, 'baseline': base_value, 'block_first': cand_value,
                        'delta': cand_value - base_value,
                        'delta_pp': 100.0 * (cand_value - base_value)}
    return metrics, baseline_summary, candidate_summary


def _group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups['components/' + ('2' if row['components'] == 2 else '3+')].append(row)
        groups['scene_setting/' + str(row['group'])].append(row)
        occlusion = 1 - row['target_visibility']
        name = '0-30%' if occlusion < .3 else '30-60%' if occlusion < .6 \
            else '60-90%' if occlusion < .9 else '90-100%'
        groups['occlusion/' + name].append(row)
    return groups


def _group_comparison(baseline_rows, candidate_rows):
    base_groups, cand_groups = _group_rows(baseline_rows), _group_rows(candidate_rows)
    output = {}
    for name in sorted(set(base_groups) & set(cand_groups)):
        base_sum = summarize(base_groups[name])
        cand_sum = summarize(cand_groups[name])
        output[name] = {'count': len(base_groups[name]),
                        **{key: {'baseline': float(base_sum[key]) if base_sum.get(key) is not None else None,
                                 'block_first': float(cand_sum[key]) if cand_sum.get(key) is not None else None}
                           for key, _ in MAIN_METRICS}}
    return output


def _selector_summary(loaded):
    audits = [audit for audit in loaded['audits'].values() if audit]
    selectors = [audit.get('point_selector') for audit in audits if audit.get('point_selector')]
    total = len(loaded['rows'])
    available = len(selectors)
    fallback_reasons = Counter(selector.get('fallback_reason') or 'none' for selector in selectors)
    fallbacks = sum(count for reason, count in fallback_reasons.items() if reason != 'none')
    cluster_counts = Counter(len(selector.get('clusters') or []) for selector in selectors
                             if selector.get('method_requested') == 'block_first'
                             and selector.get('fallback_reason') is None)
    seconds = [selector['selector_seconds'] for selector in selectors
               if selector.get('selector_seconds') is not None]
    unique_counts = [selector['unique_candidate_count'] for selector in selectors
                     if selector.get('unique_candidate_count') is not None]
    noise = [selector['noise_count_unique'] for selector in selectors
             if selector.get('noise_count_unique') is not None]
    return {
        'total_evaluated_samples': total,
        'audits_available': available,
        'audits_unknown': total - available,
        'fallback': {'numerator': fallbacks, 'denominator': available, 'total': total,
                     'rate_of_audited': fallbacks / available if available else None,
                     'rate_of_total': fallbacks / total if total else None},
        'fallback_reasons': dict(fallback_reasons),
        'method_effective': dict(Counter(selector.get('method_effective')
                                         for selector in selectors)),
        'cluster_count_distribution': dict(cluster_counts),
        'unique_candidate_count': {'mean': sum(unique_counts) / len(unique_counts)
                                   if unique_counts else None, 'min': min(unique_counts)
                                   if unique_counts else None},
        'noise_count_unique': {'mean': sum(noise) / len(noise) if noise else None,
                               'max': max(noise) if noise else None},
        'selector_seconds': {'mean': sum(seconds) / len(seconds) if seconds else None,
                             'max': max(seconds) if seconds else None},
    }


def write_comparison(out_dir, baseline, candidate, metrics, validation,
                     baseline_rows, candidate_rows):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    base_by_id = {row['case_id']: row for row in baseline_rows}
    cand_by_id = {row['case_id']: row for row in candidate_rows}
    ids = sorted(set(base_by_id) & set(cand_by_id))

    with (out / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'label', 'baseline', 'block_first', 'delta', 'delta_pp'])
        for key, label in MAIN_METRICS:
            item = metrics.get(key)
            if item:
                writer.writerow([key, label, item['baseline'], item['block_first'],
                                 item['delta'], item['delta_pp']])

    with (out / 'paired_deltas.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case_id', 'failed_baseline', 'failed_block_first'] +
                        [f'delta_{key}' for key, _ in MAIN_METRICS])
        for case_id in ids:
            base_row, cand_row = base_by_id[case_id], cand_by_id[case_id]
            deltas = []
            for key, _ in MAIN_METRICS:
                if base_row.get(key) is None or cand_row.get(key) is None:
                    deltas.append('')
                else:
                    deltas.append(float(cand_row[key]) - float(base_row[key]))
            writer.writerow([case_id, bool(base_row.get('failed')),
                             bool(cand_row.get('failed'))] + deltas)

    payload = {
        'baseline': {'run': str(baseline['run_dir']), 'scope': baseline['scope'],
                     'checkpoint': baseline['run_config'].get('evaluation_checkpoint'),
                     'selector': baseline['run_config'].get('point_selector')},
        'candidate': {'run': str(candidate['run_dir']), 'scope': candidate['scope'],
                      'checkpoint': candidate['run_config'].get('evaluation_checkpoint'),
                      'selector': candidate['run_config'].get('point_selector')},
        'checks': validation['checks'],
        'notes': validation['notes'],
        'failures': validation['failures'],
        'audit_compared': validation['audit_compared'],
        'audit_unavailable': validation['audit_unavailable'],
        'valid_clean_comparison': validation['valid_clean_comparison'],
        'main_metrics': {key: metrics[key] for key, _ in MAIN_METRICS if key in metrics},
        'supplementary_70': {key: metrics[key] for key, _ in SUPPLEMENTARY_METRICS
                             if key in metrics},
        'by_group': _group_comparison(baseline_rows, candidate_rows),
    }
    (out / 'comparison.json').write_text(json.dumps(payload, ensure_ascii=False,
                                                    indent=2, allow_nan=False))
    (out / 'selector_summary.json').write_text(json.dumps(
        {'baseline': _selector_summary(baseline), 'block_first': _selector_summary(candidate)},
        ensure_ascii=False, indent=2, allow_nan=False))

    lines = ['# VGGT-S selector A/B comparison', '',
             f'Baseline: `{baseline["run_dir"]}`', '',
             f'Candidate: `{candidate["run_dir"]}`', '',
             f'Comparability: `valid_clean_comparison={validation["valid_clean_comparison"]}`',
             '', f'Failures: baseline={validation["failures"]["baseline"]}, '
             f'block_first={validation["failures"]["block_first"]}', '',
             '| Method | Visible mIoU | Dice | Boundary F | CR@50 | SCR@50 | all_components_covered50 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, method in (('baseline', 'KMeans'), ('block_first', 'Block-First')):
        cells = []
        for key, _ in MAIN_METRICS:
            item = metrics.get(key)
            cells.append('—' if item is None else f'{item[name] * 100:.2f}')
        lines.append(f'| {method} | ' + ' | '.join(cells) + ' |')
    cells = []
    for key, _ in MAIN_METRICS:
        item = metrics.get(key)
        cells.append('—' if item is None else f'{item["delta_pp"]:+.2f}')
    lines.append('| 差值（百分点） | ' + ' | '.join(cells) + ' |')
    if not validation['valid_clean_comparison']:
        lines += ['', '> 注意：本次对照不是干净对照（存在失败样本或审计缺失），'
                  '表中保留全部样本，不得据此宣称方法胜出。']
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    baseline = load_run(args.baseline, allow_smoke=args.allow_smoke)
    candidate = load_run(args.candidate, allow_smoke=args.allow_smoke)
    validation = validate_pair(baseline, candidate, allow_smoke=args.allow_smoke)
    baseline_rows = [baseline['row_by_id'][case_id]
                     for case_id in sorted(baseline['row_by_id'])]
    candidate_rows = [candidate['row_by_id'][case_id]
                      for case_id in sorted(candidate['row_by_id'])]
    metrics, _base_summary, _cand_summary = compare_metrics(baseline_rows, candidate_rows)
    write_comparison(args.output, baseline, candidate, metrics, validation,
                     baseline_rows, candidate_rows)
    print('COMPARISON_WRITTEN', args.output, flush=True)
    if not validation['valid_clean_comparison']:
        print('WARNING not a clean comparison; see comparison.json', flush=True)


if __name__ == '__main__':
    main()
