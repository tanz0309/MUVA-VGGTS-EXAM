"""Phase 0 + A0/A1/A2/A3 only. No training, B/C, or test-split execution."""
import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from common import ROOT, digest, load_rows, read_inputs, write_json
from run_experiment import seed
from run_vggts_official import load_backbone, load_predictor, validate_checkpoint
from vggts_ablation import run_ablation_case
from vggts_component_analysis import analyze_saved_predictions
from vggts_official_data import MuvaPreprocessing, OfficialEvaluator


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['pass-analysis'], default='pass-analysis')
    parser.add_argument('--split', choices=['val'], default='val')
    parser.add_argument('--stage', choices=['all', 'inference', 'analysis'], default='all')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/vggts_official.json')
    parser.add_argument('--ablation-config', type=Path, default=ROOT/'configs/ablations/pass_analysis.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=300)
    return parser


def select_rows(rows, limit, seed_value):
    """Seeded stratification on existing manifest metadata, without opening GT.

    Balances component count (2/3/4+), existing visibility proxy and scene group.
    This is a mechanism subset, not an unbiased full-val benchmark estimate.
    """
    if limit is None or limit >= len(rows):
        return list(rows)
    if limit <= 0:
        raise ValueError('limit must be positive')
    strata = defaultdict(list)
    for index, row in enumerate(rows):
        visibility = row['target_visibility']
        visibility_bin = 0 if visibility < .1 else 1 if visibility < .4 else 2 if visibility < .7 else 3
        strata[(min(4, row['target_components']), visibility_bin, row['group'])].append(index)
    rng = random.Random(seed_value)
    for values in strata.values(): rng.shuffle(values)
    keys = sorted(strata)
    rng.shuffle(keys)
    chosen = []
    while len(chosen) < limit:
        for key in keys:
            if strata[key]: chosen.append(strata[key].pop())
            if len(chosen) == limit: break
    return [rows[i] for i in sorted(chosen)]


def source_hashes(config):
    files = ['scripts/common.py', 'scripts/vggts_official_data.py', 'scripts/run_vggts_official.py',
             'scripts/run_vggts_ablation.py', 'scripts/vggts_ablation.py',
             'scripts/vggts_ablation_hooks.py', 'scripts/vggts_component_analysis.py',
             'scripts/vggts_point_selector.py']
    result = {rel: digest(ROOT/rel) for rel in files}
    for name in ['predictor', 'prompt_encoder', 'mask_decoder', 'transformer']:
        path = Path(config['repo'])/'src/model'/f'{name}.py'
        result[str(path)] = digest(path)
    for rel in ['src/utils.py', 'src/main_exp/hybrid.py']:
        path = Path(config['repo'])/rel
        result[str(path)] = digest(path)
    return result


def model_fingerprint(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def initialize_run(args):
    config = json.loads(args.config.read_text())
    ablation = json.loads(args.ablation_config.read_text())
    if Path(sys.prefix).resolve() != Path(config['environment']).resolve():
        raise ValueError('Run in the VGGT-S Conda environment')
    if not args.checkpoint or not args.checkpoint.is_file():
        raise ValueError('A formal checkpoint is required')
    if args.limit <= 0:
        raise ValueError('--limit must be positive')
    if config['refine_time'] != 1 or config['seed'] != 42:
        raise ValueError('Phase A requires the unchanged seed42/two-pass protocol')
    if ablation['groups'] != ['A0', 'A1', 'A2', 'A3'] or ablation['mode'] != 'pass-analysis':
        raise ValueError('Only the four Phase A groups are implemented')
    if not torch.cuda.is_available():
        raise RuntimeError('Model execution requires CUDA')
    all_rows = load_rows('val')
    rows = select_rows(all_rows, args.limit, config['seed'])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'logs').mkdir()
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    diff = subprocess.check_output(['git', '-C', str(ROOT), 'diff', 'HEAD', '--binary'], text=True)
    (out/'logs/code_diff.patch').write_text(diff)
    info = {'path': str(args.checkpoint.resolve()), 'sha256_before': digest(args.checkpoint),
            'vggt_path': config['vggt_checkpoint'], 'vggt_sha256_before': digest(config['vggt_checkpoint'])}
    if info['vggt_sha256_before'] != config['vggt_checkpoint_sha256']:
        raise ValueError('Backbone SHA differs from the formal cache recipe')
    meta = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'agent': 'Codex',
            'mode': 'pass-analysis', 'groups': ablation['groups'], 'split': 'val',
            'base_config': config, 'ablation': ablation,
            'base_config_sha256': digest(args.config), 'ablation_config_sha256': digest(args.ablation_config),
            'project_commit': commit, 'working_tree_diff_sha256': digest(out/'logs/code_diff.patch'),
            'source_sha256': source_hashes(config), 'checkpoint': info,
            'manifest_sha256': {s: digest(ROOT/'data/manifests'/f'{s}.jsonl') for s in ['train', 'val', 'test']},
            'seed': config['seed'], 'refine_time': 1, 'autocast': 'cuda bfloat16',
            'python': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__,
            'gpu': torch.cuda.get_device_name(0), 'selected_case_ids': [r['case_id'] for r in rows],
            'full_val_count': len(all_rows), 'sample_count': len(rows),
            'selection': 'seed42 stratified by component-count(2,3,4+), visibility bins(0.1,0.4,0.7), scene group; manifest order retained',
            'selection_reads_gt': False, 'candidate_support_used_for_selection': False,
            'historical_test_used_for_selection': False, 'training': False,
            'zip_comparison': 'ZIP named in plan was not supplied; audited actual server code and pinned revisions',
            'b_c_groups': 'not implemented; awaiting Phase A report review'}
    write_json(out/'config.json', meta)
    write_json(out/'checkpoint_info.json', info)
    write_json(out/'sample_manifest.json', {'selection': meta['selection'],
        'selected_ids_sha256': hashlib.sha256(json.dumps(meta['selected_case_ids']).encode()).hexdigest(),
        'sample_ids': meta['selected_case_ids'], 'samples': rows})
    return config, ablation, rows, out, meta, info


def infer(args, config, ablation, rows, out, meta, info):
    preprocessing = MuvaPreprocessing(config)
    seed(config['seed'])
    backbone = load_backbone(config)
    predictor = load_predictor(config)
    saved = torch.load(args.checkpoint, map_location='cpu')
    validate_checkpoint(saved, config)
    predictor.load_state_dict(saved['model'], strict=True)
    info.update({'scope': saved['scope'], 'epoch': saved['epoch'],
                 'best_val_iou': saved['best_val_iou'], 'strict_validation': 'scope + exact config + all manifests + strict state_dict'})
    del saved
    predictor.eval().requires_grad_(False)
    info['predictor_state_sha256_before'] = model_fingerprint(predictor)
    write_json(out/'checkpoint_info.json', info)
    adapter = OfficialEvaluator(config, preprocessing, backbone, predictor)
    started = time.perf_counter()
    with (out/'logs/inference.jsonl').open('x', buffering=1) as log:
        for index, row in enumerate(rows, 1):
            torch.cuda.reset_peak_memory_stats()
            audit = run_ablation_case(adapter, read_inputs(row), ablation, out/'pass_predictions'/row['case_id'])
            progress = {'completed': index, 'total': len(rows), 'sample_id': row['case_id'],
                        'elapsed_seconds': time.perf_counter()-started,
                        'inference_seconds': audit['inference_seconds'],
                        'peak_gpu_mib': torch.cuda.max_memory_allocated()/1048576,
                        'official_different_pixels': audit['official_final_different_pixels'],
                        'repeat_max_abs_difference': audit['repeat_none_max_abs_logit_difference']}
            log.write(json.dumps(progress)+'\n')
            write_json(out/'status.json', {'status': 'inference_running', **progress})
            if index <= 8 or index % 10 == 0 or index == len(rows):
                print('INFERRED', json.dumps(progress), flush=True)
    info['sha256_after'] = digest(args.checkpoint)
    info['vggt_sha256_after'] = digest(config['vggt_checkpoint'])
    info['predictor_state_sha256_after'] = model_fingerprint(predictor)
    for a, b in [('sha256_before', 'sha256_after'), ('vggt_sha256_before', 'vggt_sha256_after'),
                 ('predictor_state_sha256_before', 'predictor_state_sha256_after')]:
        if info[a] != info[b]: raise RuntimeError(f'Weight/state changed: {a}')
    info['unchanged_verified'] = True
    write_json(out/'checkpoint_info.json', info)
    if source_hashes(config) != meta['source_sha256']:
        raise RuntimeError('Code changed concurrently during inference')
    write_json(out/'point_cache_usage.json', preprocessing.point_cache_usage)
    write_json(out/'inference_complete.json', {'sample_ids': [r['case_id'] for r in rows],
        'full_val_count': meta['full_val_count'], 'target_gt_read': False,
        'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
    write_json(out/'status.json', {'status': 'inference_complete', 'samples': len(rows), 'failures': 0})
    print('INFERENCE_COMPLETE', len(rows), 'target GT not yet read', flush=True)
    del adapter, predictor, backbone
    torch.cuda.empty_cache()


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.limit <= 0: parser.error('--limit must be positive')
    if args.stage != 'analysis' and args.checkpoint is None: parser.error('--checkpoint is required for inference')
    out = args.output.resolve()
    if args.stage == 'analysis':
        meta = json.loads((out/'config.json').read_text())
        ablation = meta['ablation']
        rows = json.loads((out/'sample_manifest.json').read_text())['samples']
        if json.loads((out/'status.json').read_text())['status'] == 'complete':
            raise RuntimeError('Already complete; refusing to overwrite analysis')
    else:
        config, ablation, rows, out, meta, info = initialize_run(args)
    try:
        if args.stage != 'analysis': infer(args, config, ablation, rows, out, meta, info)
        if args.stage != 'inference':
            write_json(out/'status.json', {'status': 'analysis_running', 'samples': len(rows)})
            result = analyze_saved_predictions(rows, out, ablation)
            write_json(out/'status.json', {'status': 'complete', 'samples': len(rows), 'failures': 0,
                'phase': 'A', 'scope': result['scope'], 'phase_B_C_started': False,
                'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
            print('PHASE_A_COMPLETE', json.dumps({k: result[k] for k in ['sample_count', 'component_count', 'metrics']}), flush=True)
    except BaseException as error:
        write_json(out/'status.json', {'status': 'incomplete_failed', 'error': f'{type(error).__name__}: {error}'})
        (out/'logs/error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
