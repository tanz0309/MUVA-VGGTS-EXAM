"""Post-hoc analysis of saved predictions; GT never enters inference modules."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from common import metrics, read_gt, write_json

MAIN_METRICS = ['visible_iou', 'dice', 'boundary_f', 'cr50', 'scr50',
                'all_components_covered50']
GROUPS = ['A0', 'A1', 'A2', 'A3']


def extract_significant_components(gt):
    """Same 8-connectivity, area threshold and tied minima as common.metrics."""
    gt = np.asarray(gt, dtype=bool)
    labels, _ = ndi.label(gt, structure=np.ones((3, 3), np.uint8))
    areas = np.bincount(labels.ravel())
    threshold = max(10, int(np.ceil(int(gt.sum())*.01)))
    ids = np.flatnonzero(areas >= threshold)
    ids = ids[ids != 0]
    smallest = int(areas[ids].min()) if len(ids) else None
    objects = ndi.find_objects(labels)
    boundaries = {}
    for ident in ids:
        sl = objects[ident-1]
        mask = labels[sl] == ident
        yy, xx = np.nonzero(mask & ~ndi.binary_erosion(mask, border_value=0))
        boundaries[int(ident)] = np.column_stack((yy+sl[0].start, xx+sl[1].start))
    components = []
    for ident in ids:
        ident = int(ident)
        others = [coords for key, coords in boundaries.items() if key != ident]
        distance = float(cKDTree(np.concatenate(others)).query(boundaries[ident])[0].min()) if others else None
        components.append({'component_id': ident, 'component_area': int(areas[ident]),
                           'component_area_ratio': float(areas[ident]/gt.sum()),
                           'is_smallest_component': int(areas[ident]) == smallest,
                           'nearest_other_component_distance': distance})
    return labels, components


def points_to_original(points, meta, original_shape):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    crop_scale = np.array([meta['crop_size']/meta['W'], meta['crop_size']/meta['H']])
    camera_to_original = np.array([original_shape[1]/meta['ori_w'], original_shape[0]/meta['ori_h']])
    return (points*crop_scale+np.array([meta['x1'], meta['y1']]))*camera_to_original


def map_points_to_components(points, component_labels, meta=None):
    """Map XY floats by floor; out-of-frame points count as background, not border hits."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if meta is not None:
        points = points_to_original(points, meta, component_labels.shape)
    height, width = component_labels.shape
    valid = np.isfinite(points).all(axis=1)
    valid &= (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
    hits = np.zeros(len(points), dtype=np.int64)
    xy = np.floor(points[valid]).astype(np.int64)
    hits[valid] = component_labels[xy[:, 1], xy[:, 0]]
    return np.bincount(hits, minlength=int(component_labels.max())+1)


def component_coverage(prediction, labels, components):
    prediction = np.asarray(prediction, dtype=bool)
    if prediction.shape != labels.shape:
        raise ValueError('Prediction and GT labels must share original-image coordinates')
    hits = np.bincount(labels[prediction], minlength=int(labels.max())+1)
    return {c['component_id']: float(hits[c['component_id']]/c['component_area']) for c in components}


def compare_component_passes(pass0, pass1, components, ablation):
    result = []
    for c in components:
        ident = c['component_id']
        before, after = pass0[ident], pass1[ident]
        result.append({**c, 'ablation': ablation, 'pass0_coverage': before,
                       'pass1_coverage': after, 'pass0_recovered50': before >= .5,
                       'pass1_recovered50': after >= .5,
                       'newly_recovered': before < .5 <= after,
                       'regressed': after < .5 <= before})
    return result


def safe_rate(numerator, denominator):
    return numerator/denominator if denominator else None


def summarize_component_groups(records):
    """Explicit counts + component micro and nonempty per-sample macro rates."""
    groups = {'all': lambda r: True,
              'no_candidate': lambda r: not r['has_candidate_support'],
              'with_candidate': lambda r: r['has_candidate_support'],
              'no_selected_prompt': lambda r: not r['has_selected_prompt'],
              'with_selected_prompt': lambda r: r['has_selected_prompt'],
              'smallest': lambda r: r['is_smallest_component'],
              'candidate_1': lambda r: r['candidate_support_count'] == 1,
              'candidate_ge2': lambda r: r['candidate_support_count'] >= 2}
    result = {}
    for name, include in groups.items():
        subset = [r for r in records if include(r)]
        per_sample = defaultdict(list)
        for r in subset:
            per_sample[r['sample_id']].append(bool(r['recovered50']))
        recovered = sum(bool(r['recovered50']) for r in subset)
        result[name] = {'component_count': len(subset), 'recovered_count': recovered,
                        'component_micro_recovery': safe_rate(recovered, len(subset)),
                        'sample_count_with_nonempty_group': len(per_sample),
                        'per_sample_macro_recovery': float(np.mean([np.mean(v) for v in per_sample.values()])) if per_sample else None}
    return result


def summarize_transitions(records):
    result = {}
    for name, subset in [('all', records), ('no_candidate', [r for r in records if not r['has_candidate_support']]),
                         ('no_selected_prompt', [r for r in records if not r['has_selected_prompt']])]:
        failed = sum(not r['pass0_recovered50'] for r in subset)
        succeeded = sum(r['pass0_recovered50'] for r in subset)
        new = sum(r['newly_recovered'] for r in subset)
        lost = sum(r['regressed'] for r in subset)
        result[name] = {'components': len(subset), 'pass0_failed': failed, 'pass0_recovered': succeeded,
                        'newly_recovered': new, 'regressed': lost,
                        'refinement_rescue_rate': safe_rate(new, failed),
                        'refinement_regression_rate': safe_rate(lost, succeeded)}
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No records for {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_component_visualization(path, target_path, gt, labels, components, audit, predictions):
    """Fixed eight-panel, post-hoc-only GT/point coloring."""
    size = (576, 324)
    target = Image.open(target_path).convert('RGB')
    colors = {c['component_id']: tuple(int(v) for v in color)
              for c, color in zip(components, np.random.default_rng(42).integers(60, 255, (len(components), 3)))}
    component_rgb = np.zeros((*gt.shape, 3), np.uint8)
    for ident, color in colors.items(): component_rgb[labels == ident] = color
    gt_panel = Image.blend(target, Image.fromarray(component_rgb), .55)
    draw = ImageDraw.Draw(gt_panel)
    for c in components:
        yy, xx = np.nonzero(labels == c['component_id'])
        draw.text((int(xx.mean()), int(yy.mean())), str(c['component_id']), fill='white')
    panels = [('Target RGB', target), ('GT components (post-hoc)', gt_panel)]
    for title, key in [('40 candidates (GT color only for display)', 'candidate_points_crop'),
                       ('5 selected prompts', 'selected_points_crop')]:
        panel = target.copy()
        draw = ImageDraw.Draw(panel)
        for x, y in points_to_original(audit[key], audit['target_crop_960x540'], gt.shape):
            xi, yi = int(np.floor(x)), int(np.floor(y))
            ident = int(labels[yi, xi]) if 0 <= xi < gt.shape[1] and 0 <= yi < gt.shape[0] else 0
            color = colors.get(ident, (255, 255, 0))
            draw.ellipse((x-7, y-7, x+7, y+7), fill=color, outline='black')
        panels.append((title, panel))
    for title, group in [('A0 Pass-0', 'A0'), ('A1 Native feedback', 'A1'), ('A3 Shifted feedback', 'A3')]:
        panels.append((title, Image.fromarray(predictions[group].astype(np.uint8)*255).convert('RGB')))
    difference = np.zeros((*gt.shape, 3), np.uint8)
    difference[predictions['A1'] & ~predictions['A0']] = (0, 220, 0)
    difference[predictions['A0'] & ~predictions['A1']] = (255, 50, 50)
    panels.append(('A1-A0: added green / removed red', Image.fromarray(difference)))
    canvas = Image.new('RGB', (size[0]*4, (size[1]+30)*2), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        x, y = index % 4*size[0], index // 4*(size[1]+30)
        draw.text((x+5, y+5), title, fill='black')
        canvas.paste(panel.resize(size, Image.Resampling.NEAREST), (x, y+30))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def analyze_saved_predictions(rows, out, config):
    """Called only after the run-wide inference_complete marker is committed."""
    import json
    out = Path(out)
    marker = json.loads((out/'inference_complete.json').read_text())
    if marker['sample_ids'] != [r['case_id'] for r in rows]:
        raise RuntimeError('Analysis sample order differs from completed inference')
    all_components, all_metrics, all_transitions, comparisons = [], [], [], []
    visualization_scores = []
    for row in rows:
        sample = row['case_id']
        directory = out/'pass_predictions'/sample
        audit = json.loads((directory/'audit.json').read_text())
        # This is the first target GT read in the entire run.
        gt = read_gt(row)
        labels, components = extract_significant_components(gt)
        if len(components) != row['target_components']:
            raise RuntimeError(f'Component definition differs from manifest: {sample}')
        candidates = map_points_to_components(audit['candidate_points_crop'], labels, audit['target_crop_960x540'])
        prompts = map_points_to_components(audit['selected_points_crop'], labels, audit['target_crop_960x540'])
        for c in components:
            ident = c['component_id']
            c.update({'sample_id': sample, 'candidate_support_count': int(candidates[ident]),
                      'selected_prompt_support_count': int(prompts[ident]),
                      'has_candidate_support': bool(candidates[ident]),
                      'has_selected_prompt': bool(prompts[ident]),
                      'sample_component_count': len(components)})
        predictions = {g: np.asarray(Image.open(directory/f'{g}_original.png')) > 0 for g in GROUPS}
        coverages, per_metrics = {}, {}
        comparison = {'sample_id': sample, 'no_candidate_components': sum(not c['has_candidate_support'] for c in components),
                      'no_prompt_components': sum(not c['has_selected_prompt'] for c in components),
                      'repeat_none_max_abs_logit_difference': audit['repeat_none_max_abs_logit_difference'],
                      'official_final_different_pixels': audit['official_final_different_pixels']}
        for group in GROUPS:
            pred = predictions[group]
            score = metrics(pred, gt)
            per_metrics[group] = score
            all_metrics.append({'sample_id': sample, 'ablation': group, 'pass_index': 0 if group == 'A0' else 1,
                                **{k: v for k, v in score.items() if k != 'component_coverages'},
                                'false_positive_pixels': int((pred & ~gt).sum()),
                                'prediction_area': int(pred.sum())})
            coverages[group] = component_coverage(pred, labels, components)
            group_records = [{**c, 'ablation': group, 'pass_index': 0 if group == 'A0' else 1,
                              'coverage': coverages[group][c['component_id']],
                              'recovered50': coverages[group][c['component_id']] >= .5} for c in components]
            if abs(float(np.mean([r['recovered50'] for r in group_records]))-score['cr50']) > 1e-12:
                raise RuntimeError('Post-hoc component recovery differs from main metric')
            all_components.extend(group_records)
            write_json(out/'component_records'/sample/f'{group}.json', group_records)
            for key in MAIN_METRICS: comparison[f'{group}_{key}'] = score[key]
            comparison[f'{group}_no_candidate_recovered'] = sum(r['recovered50'] and not r['has_candidate_support'] for r in group_records)
            comparison[f'{group}_no_prompt_recovered'] = sum(r['recovered50'] and not r['has_selected_prompt'] for r in group_records)
        for group in GROUPS[1:]:
            transitions = compare_component_passes(coverages['A0'], coverages[group], components, group)
            all_transitions.extend(transitions)
            comparison[f'{group}_newly_recovered'] = sum(r['newly_recovered'] for r in transitions)
            comparison[f'{group}_regressed'] = sum(r['regressed'] for r in transitions)
        comparisons.append(comparison)
        # Selection is post-hoc, prespecified ranking by transitions, then no-point count, then ID.
        visualization_scores.append((-(comparison['A1_newly_recovered']+comparison['A1_regressed']),
                                     -comparison['no_candidate_components'], sample, row))
        if len(comparisons) % 25 == 0 or len(comparisons) == len(rows):
            print('ANALYZED', len(comparisons), '/', len(rows), flush=True)
    area_values = [r['component_area_ratio'] for r in all_components if r['ablation'] == 'A0']
    distances = [r['nearest_other_component_distance'] for r in all_components
                 if r['ablation'] == 'A0' and r['nearest_other_component_distance'] is not None]
    area_cuts = np.quantile(area_values, [.25, .5, .75]).tolist()
    distance_cuts = np.quantile(distances, [.25, .5, .75]).tolist() if distances else []
    for r in all_components:
        r['area_ratio_quartile'] = int(np.searchsorted(area_cuts, r['component_area_ratio'], side='right'))+1
        d = r['nearest_other_component_distance']
        r['distance_quartile'] = int(np.searchsorted(distance_cuts, d, side='right'))+1 if d is not None else None
    summary = {'sample_count': len(rows), 'component_count': len(area_values),
               'scope': 'full_val' if len(rows) == marker['full_val_count'] else 'stratified_val_subset',
               'metrics': {}, 'component_groups': {}, 'transitions': {}, 'attribute_groups': {},
               'area_ratio_quartile_cuts': area_cuts, 'distance_quartile_cuts_pixels': distance_cuts,
               'official_max_different_pixels': max(r['official_final_different_pixels'] for r in comparisons),
               'repeat_none_max_abs_logit_difference': max(r['repeat_none_max_abs_logit_difference'] for r in comparisons),
               'inference_complete_before_gt': True}
    for group in GROUPS:
        scores = [r for r in all_metrics if r['ablation'] == group]
        summary['metrics'][group] = {key: float(np.mean([r[key] for r in scores if r[key] is not None]))
                                     for key in scores[0] if key not in ['sample_id', 'ablation', 'pass_index']}
        records = [r for r in all_components if r['ablation'] == group]
        summary['component_groups'][group] = summarize_component_groups(records)
        if group != 'A0':
            summary['transitions'][group] = summarize_transitions([r for r in all_transitions if r['ablation'] == group])
        attributes = {}
        for field in ['area_ratio_quartile', 'distance_quartile', 'sample_component_count']:
            for value in sorted({r[field] for r in records if r[field] is not None}):
                subset = [r for r in records if r[field] == value]
                attributes[f'{field}/{value}'] = summarize_component_groups(subset)
        summary['attribute_groups'][group] = attributes
    write_csv(out/'metrics/pass_metrics.csv', all_metrics)
    write_csv(out/'component_records.csv', all_components)
    write_csv(out/'recovery_transition.csv', all_transitions)
    write_csv(out/'sample_comparison.csv', comparisons)
    # Top-level delivery path requested by the plan.
    write_csv(out/'pass_metrics.csv', all_metrics)
    write_json(out/'summary.json', summary)
    for _, _, sample, row in sorted(visualization_scores)[:config['visualization_count']]:
        directory = out/'pass_predictions'/sample
        gt = read_gt(row)
        labels, components = extract_significant_components(gt)
        audit = json.loads((directory/'audit.json').read_text())
        preds = {g: np.asarray(Image.open(directory/f'{g}_original.png')) > 0 for g in GROUPS}
        render_component_visualization(out/'visualizations'/f'{sample}.png', row['inputs']['target_rgb'],
                                       gt, labels, components, audit, preds)
    write_diagnostic_report(out, summary)
    return summary


def write_diagnostic_report(out, summary):
    def pct(value): return 'null（分母为0）' if value is None else f'{100*value:.4f}%'
    lines = ['# VGGT-S Phase A：无提示点可见分块恢复机制', '',
             f"范围：val {summary['sample_count']}例，{summary['component_count']}个显著GT分块；{summary['scope']}。",
             '本报告只描述本次验证集结果；历史test诊断仅作为动机，不用于调参。', '',
             '## 实现与验收（已验证）', '',
             f"- A1与未修改OfficialEvaluator最终预测最大不同像素数：{summary['official_max_different_pixels']}。",
             f"- A2与A0最大绝对logit差：{summary['repeat_none_max_abs_logit_difference']}。",
             '- 每例只调用一次原始预处理/双视角VGGT，4组共享相同特征/源点/目标点/源点特征；每次前向输入指纹已保存。',
             '- A3固定将Pass-0 low logits在两个空间维循环平移半幅，精确保留数值多重集、尺寸、dtype和正值面积，不使用GT。',
             '- 所有样本预测和logits落盘并生成inference_complete.json后才读取目标GT；8邻域、显著面积阈值与六项指标复用正式口径。',
             '- vendor与正式权重文件不修改；无optimizer、backward或训练。checkpoint/source/hash证据见config.json和checkpoint_info.json。', '',
             '## 六项主指标（样本宏平均）', '', '| 组别 | IoU | Dice | Boundary F | CR@50 | SCR@50 | 全分块覆盖@50 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for group in GROUPS:
        lines.append('| '+group+' | '+' | '.join(pct(summary['metrics'][group][k]) for k in MAIN_METRICS)+' |')
    lines += ['', '## 无点分块恢复（分块微平均与样本宏平均）', '',
              '| 组别 | 支持分组 | 恢复/总数 | 分块微平均 | 非空样本宏平均 |', '|---|---|---:|---:|---:|']
    for group in GROUPS:
        for name in ['all', 'no_candidate', 'with_candidate', 'no_selected_prompt', 'with_selected_prompt', 'smallest']:
            r = summary['component_groups'][group][name]
            lines.append(f"| {group} | {name} | {r['recovered_count']}/{r['component_count']} | {pct(r['component_micro_recovery'])} | {pct(r['per_sample_macro_recovery'])} |")
    lines += ['', '## 相对Pass-0的新增恢复与退化', '',
              '| 对照组 | 分组 | 新恢复/初始失败 | 救回率 | 退化/初始成功 | 退化率 |', '|---|---|---:|---:|---:|---:|']
    for group in GROUPS[1:]:
        for name, r in summary['transitions'][group].items():
            lines.append(f"| {group} | {name} | {r['newly_recovered']}/{r['pass0_failed']} | {pct(r['refinement_rescue_rate'])} | {r['regressed']}/{r['pass0_recovered']} | {pct(r['refinement_regression_rate'])} |")
    before = summary['component_groups']['A0']['no_selected_prompt']
    after = summary['component_groups']['A1']['no_selected_prompt']
    m = summary['metrics']
    lines += ['', '## 研究问题与解释边界', '',
              f"1. 第一次预测是否恢复无提示分块：A0已恢复无selected-prompt分块 {before['recovered_count']}/{before['component_count']}（{pct(before['component_micro_recovery'])}）。这说明恢复可发生在feedback之前，尚不能定位到源Mask、跨视角attention或Mask Token中的哪一个。",
              f"2. 原生refinement后的无prompt恢复：{after['recovered_count']}/{after['component_count']}（{pct(after['component_micro_recovery'])}），新增与退化分开见上表；不能只用净变化掩盖二者。",
              f"3. A1相对A0：IoU变化 {(m['A1']['visible_iou']-m['A0']['visible_iou'])*100:+.4f}pp，Boundary F变化 {(m['A1']['boundary_f']-m['A0']['boundary_f'])*100:+.4f}pp，precision变化 {(m['A1']['precision']-m['A0']['precision'])*100:+.4f}pp。需要连同coverage和false_positive_pixels判断，不能把扩张等同于结构理解。",
              f"4. 空间反馈控制：A3的IoU {pct(m['A3']['visible_iou'])}、A1 {pct(m['A1']['visible_iou'])}；这是固定半幅平移的结果，不代表所有损坏方式。A2仅检查重复调用，不能当作额外机制。",
              '5. 本次不开展B/C，不将A组结果解释成源Mask、源特征或attention的因果证明；不承诺新方法收益。',
              '', '## 分块属性与可复核产物', '',
              '- 面积比例和最近分块欧氏像素距离的验证子集四分位数、各属性组的宏/微恢复率在summary.json。分组仅使用推理结束后的GT。',
              '- pass_metrics.csv：逐样本逐组原始六项指标及precision/recall/false positives。',
              '- component_records.csv：逐样本分块/组/轮的支持、面积、距离、coverage和recovered50。',
              '- recovery_transition.csv、sample_comparison.csv：新增恢复/退化与逐样本对照。',
              '- pass_predictions/：4组crop/low logits、crop及原图二值mask、原路径最终mask、输入审计。',
              '- visualizations/：按新增恢复/退化数量、无候选点分块数量及ID排序的代表性8面板图，仅用于解释，不用于筛选主指标样本。',
              '', '**Phase A执行结束，按方案暂停。待用户审阅报告后再决定Phase B。**', '']
    (Path(out)/'diagnostic_report.md').write_text('\n'.join(lines))
