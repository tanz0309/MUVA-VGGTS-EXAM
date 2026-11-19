"""Run the pinned authors' train_epoch with a MUVA Dataset, not a rewritten training loop."""
import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import sklearn
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from common import ROOT, digest, load_rows, read_inputs, write_json
from run_experiment import seed, evaluate
from vggts_official_data import (MuvaPreprocessing, MuvaOfficialDataset,
                                 OfficialEvaluator, protect_feature_inputs)
from vggts_point_selector import load_selector_config


class ModuleContainer(torch.nn.Module):
    """Single-GPU equivalent of the .module interface expected by upstream saving code."""
    def __init__(self, module):
        super().__init__(); self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class LossRecorder:
    def __init__(self, out, batch_size):
        self.file = (out/'training.jsonl').open('w', buffering=1)
        self.batch_size = batch_size; self.epoch = 0; self.last_loss = None

    def add_scalar(self, name, value, step):
        if not np.isfinite(value):
            raise RuntimeError('Non-finite training loss')
        self.last_loss = float(value)
        row = {'epoch': self.epoch, 'step': step+1, 'loss': float(value),
               'direct_samples': self.batch_size//2, 'refined_samples': self.batch_size//2,
               'source': 'unmodified upstream train_epoch'}
        self.file.write(json.dumps(row)+'\n')
        if (step+1) % 25 == 0 or step == 0:
            print('TRAIN_STEP', json.dumps(row), flush=True)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


def native_config(config, num_samples):
    import yaml
    cfg = yaml.safe_load((Path(config['repo'])/'src/main_exp/hybrid.yaml').read_text())
    cfg['exp'].update(mode='train', epochs=config['epochs'], resume_ckpt=None,
        train_samples=num_samples, refine_ratio=config['refine_ratio'], refine_time=config['refine_time'],
        lr=config['learning_rate'], weight_decay=config['weight_decay'],
        lr_drops=config['lr_milestones'], max_norm=config['max_grad_norm'])
    cfg['data'].update(train_batch_size=config['training_batch_size'], img_size=config['image_size'],
                       sa_tokens=config['sa_tokens'], src_R=config['source_crop_R'])
    cfg['vggt']['sample_num'] = config['prompt_points']
    cfg['debug']['vis_batch'] = 0  # Outputs belong to this experiment, not upstream source tree.
    return cfg


def load_backbone(config):
    from vggt.models.vggt import VGGT
    if digest(config['vggt_checkpoint']) != config['vggt_checkpoint_sha256']:
        raise ValueError('Pretrained VGGT checkpoint digest differs from the point-cache recipe')
    model = VGGT()
    state = torch.load(config['vggt_checkpoint'], map_location='cpu')
    model.load_state_dict(state, strict=True); del state
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gc.collect()
    return model


def load_predictor(config):
    from model.predictor import Predictor
    model = Predictor(img_size=config['image_size'], sa_tokens=config['sa_tokens'], resume_ckpt=None).cuda()
    protect_feature_inputs(model)
    return model


def point_preparation(config, preprocessing, rows, vggt, out):
    created = 0
    for i, row in enumerate(rows):
        created += preprocessing.make_points(read_inputs(row), vggt, row['case_id'])
        if i % 25 == 0 or i == len(rows)-1:
            print('PREPARE_POINTS', i+1, '/', len(rows), row['case_id'], flush=True)
    report = {'cases': len(rows), 'created': created, 'reused': len(rows)-created,
              'cache': str(preprocessing.cache), 'model_execution': 'GPU', 'target_gt_used': False}
    write_json(out/'point_preparation.json', report)
    return report


def checkpoint_metadata(config, scope, epoch, best):
    return {'scope': scope, 'config': config, 'epoch': epoch, 'best_val_iou': best,
            'manifest_sha256': {s: digest(ROOT/'data/manifests'/f'{s}.jsonl') for s in ['train', 'val', 'test']},
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}


def validate_checkpoint(checkpoint, config):
    if checkpoint.get('scope') != 'vggts_official_muva_training':
        raise ValueError('Only checkpoints from this official-code MUVA protocol are accepted; legacy/smoke checkpoints are separate experiments')
    if checkpoint.get('config') != config:
        raise ValueError('Checkpoint/config mismatch')
    for split, expected in checkpoint['manifest_sha256'].items():
        if digest(ROOT/'data/manifests'/f'{split}.jsonl') != expected:
            raise ValueError('Data manifest changed: '+split)


def train(config, preprocessing, vggt, predictor, out, args, synthetic=False):
    from main_exp import hybrid
    from dataloader import Egoexo4dData_collate_fn
    batch_size = config['training_batch_size']
    smoke = args.mode in ['smoke-train', 'smoke-head']
    rows = load_rows('train')
    if smoke:
        rows = rows[:batch_size]
    if synthetic:
        # This diagnostic tests upstream gradients and branch selection only. No VGGT
        # output or dataset accuracy is claimed. All network operations remain on CUDA.
        size = config['image_size']
        masks = np.zeros((size, size), np.uint8); masks[100:300, 100:300] = 1
        rgb = np.zeros((size, size, 3), np.uint8)
        points = np.array([[150, 150], [200, 150], [150, 200], [250, 200], [200, 250]], np.float32)
        item = (rgb, masks, rgb, masks, rgb, masks, rgb, masks, points, points, False, None)
        dataset = [item for _ in range(batch_size)]
        original_feat = hybrid.vggt_feat
        def synthetic_features(**kwargs):
            b = len(kwargs['src_img_b'])
            return (torch.as_tensor(np.stack([points]*b), device='cuda'),
                    torch.randn(b, 2, 128, size//2, size//2, device='cuda'))
        hybrid.vggt_feat = synthetic_features
    else:
        dataset = MuvaOfficialDataset(rows, preprocessing, training=True,
                                      audit_path=out/'training_retries.jsonl', vggt=vggt)
    native = native_config(config, len(rows))
    write_json(out/'effective_upstream_config.json', native)
    # Upstream train_epoch asks for world size and predictor.module, even for one GPU.
    original_excepthook = sys.excepthook
    dist.init_process_group('gloo', init_method='file://'+str(out/'distributed_init'), rank=0, world_size=1)
    wrapped = ModuleContainer(predictor)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['lr_milestones'], gamma=.1)
    ckpt_dir = out/'checkpoints'; ckpt_dir.mkdir()
    writer = LossRecorder(out, batch_size)
    start = 1; best = -1.; hybrid.train_glb_batch_idx = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu'); validate_checkpoint(saved, config)
        predictor.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        start = saved['epoch']+1; best = saved['best_val_iou']; hybrid.train_glb_batch_idx = saved['step']
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng']); del saved
    events = []
    event_hook = None
    if smoke:
        def record_call(_module, _args, kwargs):
            events.append({'batch': len(kwargs['src_mask_b']), 'grad_enabled': torch.is_grad_enabled(),
                           'has_refinement_mask': kwargs['tar_mask_b'] is not None})
        event_hook = predictor.register_forward_pre_hook(record_call, with_kwargs=True)
    probe = next(predictor.parameters())
    before = probe.detach().clone() if smoke else None
    try:
        for epoch in range(start, 2 if smoke else config['epochs']+1):
            writer.epoch = epoch
            order = list(range(len(rows))); random.Random(config['seed']+epoch-1).shuffle(order)
            loader = DataLoader(Subset(dataset, order), batch_size=batch_size, drop_last=True,
                                num_workers=0, collate_fn=Egoexo4dData_collate_fn)
            # Loss, 50/50 branch selection, final-pass gradient, clipping and optimizer
            # updates execute inside the unmodified author function.
            hybrid.train_epoch(epoch, native, torch.device('cuda'), str(out/'native_training.log'),
                str(ckpt_dir), vggt, wrapped, optimizer, scheduler, loader, writer)
            if smoke:
                assert bool((probe.detach() != before).any()), 'Predictor did not update'
                assert events == [
                    {'batch': batch_size//2, 'grad_enabled': True, 'has_refinement_mask': False},
                    {'batch': batch_size//2, 'grad_enabled': False, 'has_refinement_mask': False},
                    {'batch': batch_size//2, 'grad_enabled': True, 'has_refinement_mask': True}], events
                assert all(p.grad is None for p in vggt.parameters()) if vggt is not None else True
                scope = 'synthetic_head_smoke_only' if synthetic else 'official_muva_smoke_only'
                val_summary = None
            else:
                adapter = OfficialEvaluator(config, preprocessing, vggt, predictor)
                val_summary = evaluate(adapter, load_rows('val'), out/f'val_epoch_{epoch:02d}')
                if val_summary['failures']:
                    raise RuntimeError('Validation failures; refusing best-checkpoint selection')
                scope = 'vggts_official_muva_training'
            improved = val_summary is not None and val_summary['visible_iou'] > best
            if improved:
                best = val_summary['visible_iou']
            # Upstream saves before scheduler.step; our resumable wrapper saves its
            # post-epoch scheduler state explicitly, without altering training updates.
            saved = {'model': predictor.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'step': hybrid.train_glb_batch_idx,
                     **checkpoint_metadata(config, scope, epoch, best)}
            path = ckpt_dir/f'epoch_{epoch}.pth'
            torch.save(saved, path.with_suffix('.pth.part')); path.with_suffix('.pth.part').replace(path)
            import shutil
            shutil.copy2(path, out/('smoke_checkpoint.pth' if smoke else 'last.pth'))
            if improved:
                shutil.copy2(path, out/'best.pth')
            if smoke:
                reloaded = torch.load(out/'smoke_checkpoint.pth', map_location='cpu')
                predictor.load_state_dict(reloaded['model'], strict=True); del reloaded
        return {'scope': 'synthetic_gpu_head_check_only' if synthetic else 'smoke_only' if smoke else 'full_training',
                'upstream_train_epoch': True, 'steps': hybrid.train_glb_batch_idx,
                'refinement_calls': events if smoke else None, 'last_loss': writer.last_loss,
                'parameter_updated': True if smoke else None,
                'checkpoint_reload_passed': True if smoke else None,
                'peak_gpu_mib': torch.cuda.max_memory_allocated()/1048576,
                'full_vggt_checked': not synthetic}
    finally:
        writer.close()
        if event_hook is not None:
            event_hook.remove()
        if synthetic:
            hybrid.vggt_feat = original_feat
        dist.destroy_process_group()
        sys.excepthook = original_excepthook


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['check-data', 'prepare-points', 'smoke-head',
                                                        'smoke', 'smoke-train', 'train', 'evaluate',
                                                        'smoke-evaluate'])
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='val')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/vggts_official.json')
    parser.add_argument('--checkpoint', type=Path); parser.add_argument('--resume', type=Path)
    parser.add_argument('--limit', type=int); parser.add_argument('--output', type=Path)
    parser.add_argument('--selector-config', type=Path, default=None)
    return parser


def validate_run_arguments(parser, args):
    """Argument relationships only; environment/CUDA/config checks stay in main()."""
    if args.selector_config is not None and args.mode not in ('evaluate', 'smoke-evaluate'):
        parser.error('--selector-config is only supported for evaluate/smoke-evaluate in phase 1')
    if args.checkpoint is not None and args.mode not in ('evaluate', 'smoke-evaluate'):
        parser.error('--checkpoint is only supported for evaluate/smoke-evaluate')
    if args.resume is not None and args.mode != 'train':
        parser.error('--resume only applies to train')
    if args.mode == 'evaluate':
        if args.checkpoint is None:
            parser.error('evaluate requires --checkpoint')
        if args.split not in ('val', 'test'):
            parser.error('evaluate requires --split val|test')
        if args.limit is not None:
            parser.error('--limit is not allowed for a full evaluate')
    if args.mode == 'smoke-evaluate':
        if args.checkpoint is None:
            parser.error('smoke-evaluate requires --checkpoint')
        if args.split != 'val':
            parser.error('smoke-evaluate only supports --split val')
        if args.limit is not None and args.limit < 1:
            parser.error('--limit must be positive')
    if args.limit is not None and args.mode not in ('prepare-points', 'check-data', 'smoke-evaluate'):
        parser.error('--limit is only for point-cache/data diagnostics and smoke-evaluate')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')


def ordered_case_ids_sha256(rows):
    digest_ = hashlib.sha256()
    for row in rows:
        digest_.update(row['case_id'].encode('utf-8'))
        digest_.update(b'\n')
    return digest_.hexdigest()


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_run_arguments(parser, args)
    config = json.loads(args.config.read_text())
    if Path(sys.prefix).resolve() != Path(config['environment']).resolve():
        parser.error('Use the vggts Conda environment')
    if config['training_batch_size'] < 2 or config['training_batch_size'] % 2:
        parser.error('An even batch >=2 is required to preserve upstream 50/50 refinement')
    if config['refine_ratio'] != .5 or config['refine_time'] != 1:
        parser.error('Baseline protocol requires the upstream 0.5 ratio and one refinement')
    selector_config = load_selector_config(args.selector_config,
                                           prompt_points=config['prompt_points'])
    preprocessing = MuvaPreprocessing(config, selector_config=selector_config)
    if not torch.cuda.is_available() and args.mode != 'check-data':
        raise RuntimeError('GPU unavailable; CPU model execution is disabled')
    seed(config['seed'])
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = (args.output or ROOT/'outputs/vggts_official'/args.mode/stamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    run_meta = {'config': config, 'mode': args.mode, 'split': args.split,
        'python': sys.executable, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'manifests': {s: digest(ROOT/'data/manifests'/f'{s}.jsonl') for s in ['train', 'val', 'test']},
        'runner_sha256': digest(Path(__file__)), 'data_adapter_sha256': digest(Path(__file__).with_name('vggts_official_data.py')),
        'point_generation': 'on_demand_gpu_with_persistent_cache',
        'upstream_train_sha256': digest(Path(config['repo'])/'src/main_exp/hybrid.py'),
        'vggt_patch': subprocess.check_output(['git', '-C', config['vggt_repo'], 'diff', '--', 'vggt/heads/track_head.py'], text=True),
        'point_selector': {
            'effective_config': {field: getattr(selector_config, field) for field in
                                 ('schema_version', 'method', 'budget', 'eps', 'min_samples')},
            'config_file_sha256': digest(args.selector_config) if args.selector_config else None,
            'implementation_sha256': digest(Path(__file__).with_name('vggts_point_selector.py')),
            'version': 'block_first_v1' if selector_config.method == 'block_first' else 'native_kmeans'},
        'metric_implementation_sha256': digest(Path(__file__).with_name('common.py')),
        'point_cache': {'recipe': preprocessing.recipe,
                        'resolved_directory': str(preprocessing.cache)},
        'runtime_versions': {'numpy': np.__version__, 'sklearn': sklearn.__version__,
                             'torch': torch.__version__},
        'evaluation_checkpoint': None, 'evaluation_subset': None}
    write_json(out/'run_config.json', run_meta)
    print('OUTPUT_DIRECTORY', out, flush=True)
    try:
        if args.mode == 'check-data':
            splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
            report = {s: preprocessing.check_cache(load_rows(s)[:args.limit]) for s in splits}
            report['scope'] = 'data_and_cache_check_no_model_forward'
        elif args.mode == 'smoke-head':
            report = train(config, preprocessing, None, load_predictor(config), out, args, synthetic=True)
        else:
            vggt = load_backbone(config)
            if args.mode == 'prepare-points':
                splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
                rows = [r for s in splits for r in load_rows(s)]
                report = point_preparation(config, preprocessing, rows[:args.limit], vggt, out)
            else:
                predictor = load_predictor(config)
                if args.mode in ['smoke', 'smoke-train']:
                    rows = load_rows('train' if args.mode == 'smoke-train' else 'val')[:config['training_batch_size']]
                elif args.mode == 'smoke-evaluate':
                    rows = load_rows('val')[:args.limit if args.limit is not None else 8]
                if args.mode in ['train', 'smoke-train']:
                    report = train(config, preprocessing, vggt, predictor, out, args)
                else:
                    if args.checkpoint:
                        saved = torch.load(args.checkpoint, map_location='cpu'); validate_checkpoint(saved, config)
                        predictor.load_state_dict(saved['model'], strict=True)
                        run_meta['evaluation_checkpoint'] = {
                            'path': str(args.checkpoint.resolve()), 'sha256': digest(args.checkpoint),
                            'validated_scope': saved['scope']}
                        del saved
                    if args.mode not in ('smoke', 'smoke-evaluate'):
                        rows = load_rows(args.split)
                    run_meta['evaluation_subset'] = {
                        'kind': 'full_split' if args.mode == 'evaluate' else 'smoke_limited',
                        'requested_limit': args.limit, 'actual_count': len(rows),
                        'ordered_case_ids_sha256': ordered_case_ids_sha256(rows)}
                    write_json(out/'run_config.json', run_meta)
                    adapter = OfficialEvaluator(config, preprocessing, vggt, predictor)
                    report = evaluate(adapter, rows, out,
                                      smoke=args.mode in ('smoke', 'smoke-evaluate'))
        write_json(out/'summary.json', report)
        write_json(out/'status.json', {'status': 'complete', 'mode': args.mode})
        print('RUN_COMPLETE', json.dumps(report), flush=True)
    except BaseException as error:
        write_json(out/'status.json', {'status': 'incomplete_failed', 'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        write_json(out/'point_cache_usage.json', {
            **preprocessing.point_cache_usage, 'cache': str(preprocessing.cache),
            'scope': 'on-demand calls, including warmup and training validation; explicit prepare-points excluded'})


if __name__ == '__main__':
    main()
