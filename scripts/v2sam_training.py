"""MUVA iteration driver using the author's MMEngine optimizer and schedulers.

The native V2SAM.forward supplies every loss. The data iterator, checkpoint
format and common MUVA evaluation replace the Ego-Exo-specific runner IO.
"""
import json
import random
import shutil
import numpy as np
import torch
from common import ROOT, digest, load_rows, read_gt, read_inputs, write_json


def train_v2sam(adapter, config, out, args, evaluate):
    from mmengine.optim import AmpOptimWrapper, LinearLR, CosineAnnealingLR
    cfg = config['training']
    rows, val = load_rows('train'), load_rows('val')
    batch_size, accumulation = cfg['batch_size'], cfg['accumulative_counts']
    assert len(rows) % (batch_size * accumulation) == 0, 'Epoch checkpoints require completed accumulation windows'
    batches = len(rows) // batch_size
    total = batches * cfg['epochs']
    warmup = int(total * cfg['warmup_ratio'])
    params = [p for p in adapter.predictor.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], betas=(.9, .999), weight_decay=cfg['weight_decay'])
    wrapper = AmpOptimWrapper(optimizer=optimizer, accumulative_counts=accumulation,
        clip_grad=dict(max_norm=cfg['max_grad_norm'], error_if_nonfinite=False), loss_scale='dynamic', dtype='bfloat16')
    schedulers = [LinearLR(wrapper, start_factor=cfg['warmup_start_factor'], by_epoch=False, begin=0, end=warmup),
                  CosineAnnealingLR(wrapper, eta_min=0., by_epoch=False, begin=warmup, end=total, T_max=total-warmup)]
    epoch0, microstep, step, best = 0, 0, 0, -1.
    smoke = args.mode == 'smoke-train'
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu')
        assert ck['scope'] == 'muva_fragment_training' and ck['config'] == config, 'Incompatible checkpoint'
        assert ck['train_manifest_sha256'] == digest(ROOT/'data/manifests/train.jsonl')
        adapter.predictor.load_state_dict(ck['model'], strict=True)
        wrapper.load_state_dict(ck['optimizer_wrapper'])
        for sched, state in zip(schedulers, ck['schedulers']): sched.load_state_dict(state)
        epoch0, microstep, step, best = ck['epoch'], ck['microstep'], ck['step'], ck['best_val_iou']
        torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
        adapter.restore_training_state(ck['adapter_training_state'])
        del ck
    wrapper.initialize_count_status(adapter.model, microstep, accumulation if smoke else total)
    optimizer.zero_grad(set_to_none=True)
    # Count real optimizer calls, including AMP's possibility of skipping an update.
    updates = [step]
    hook = optimizer.register_step_post_hook(lambda *unused: updates.__setitem__(0, updates[0]+1))
    probe = params[0]
    before = probe.detach().clone() if smoke else None
    epochs = 1 if smoke else cfg['epochs']
    audit = None
    torch.cuda.reset_peak_memory_stats()
    with (out/'training.jsonl').open('w', buffering=1) as log:
        for epoch in range(epoch0, epochs):
            # Standard seed-based shuffle; no extra scene-aware negative sampling rule.
            order = list(range(len(rows))); random.Random(config['seed']+epoch).shuffle(order)
            if smoke: order = order[:batch_size*accumulation]
            for offset in range(0, len(order), batch_size):
                batch = [rows[i] for i in order[offset:offset+batch_size]]
                with wrapper.optim_context(adapter.model):
                    loss, details = adapter.training_loss_batch([read_inputs(r) for r in batch], [lambda r=r: read_gt(r) for r in batch])
                if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
                previous_updates = updates[0]
                wrapper.update_params(loss)
                microstep += 1
                for sched in schedulers: sched.step()
                item = {'epoch': epoch+1, 'microstep': microstep, 'step': updates[0],
                        'optimizer_updated': updates[0] > previous_updates,
                        'accumulative_counts': accumulation, 'batch_size': batch_size,
                        'case_ids': [r['case_id'] for r in batch], 'loss': float(loss.detach()),
                        'learning_rate_next_microstep': optimizer.param_groups[0]['lr'], **details}
                if item['optimizer_updated']:
                    norm = float(wrapper.message_hub.get_scalar('train/grad_norm').current())
                    if not np.isfinite(norm): raise RuntimeError('Non-finite gradient norm')
                    item['gradient_norm'] = norm
                log.write(json.dumps(item, allow_nan=False)+'\n')
                if microstep % 25 == 0 or smoke: print('TRAIN_MICROSTEP', json.dumps(item), flush=True)
                audit = item
                del loss
            val_summary = None
            if not smoke and (epoch+1) % cfg['validation_interval'] == 0:
                val_summary = evaluate(adapter, val, out/f'val_epoch_{epoch+1:02d}')
                if val_summary['failures']: raise RuntimeError('Validation failed; refusing checkpoint selection')
            improved = val_summary is not None and val_summary['visible_iou'] > best
            if improved: best = val_summary['visible_iou']
            ck = {'model': adapter.predictor.state_dict(), 'optimizer_wrapper': wrapper.state_dict(),
                  'schedulers': [s.state_dict() for s in schedulers], 'epoch': epoch+1,
                  'step': updates[0], 'microstep': microstep, 'best_val_iou': best,
                  'config': config, 'scope': 'smoke_only' if smoke else 'muva_fragment_training',
                  'adapter_training_state': adapter.training_state(),
                  'train_manifest_sha256': digest(ROOT/'data/manifests/train.jsonl'),
                  'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                  'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}
            path = out/('smoke_checkpoint.pth' if smoke else 'last.pth')
            tmp = path.with_suffix('.pth.part'); torch.save(ck, tmp); tmp.replace(path)
            if improved: shutil.copy2(path, out/'best.pth')
            if smoke:
                assert updates[0] == 1 and microstep == accumulation
                changed = bool((before != probe.detach()).any())
                assert changed, 'Trainable parameters did not change'
                assert all(p.grad is None for m in adapter.frozen_modules for p in m.parameters() if not p.requires_grad)
                loaded = torch.load(path, map_location='cpu')
                adapter.predictor.load_state_dict(loaded['model'], strict=True)
                # Verify AMP/scaler/scheduler and native contrastive counter can be restored.
                wrapper.load_state_dict(loaded['optimizer_wrapper'])
                for sched, state in zip(schedulers, loaded['schedulers']): sched.load_state_dict(state)
                adapter.restore_training_state(loaded['adapter_training_state'])
                audit.update(predictor_parameter_changed=True, frozen_backbones_no_grad=True, checkpoint_reload_passed=True)
                del loaded
    hook.remove()
    summary = {'status': 'complete', 'scope': 'smoke_only_not_training_experiment' if smoke else 'muva_supervised_training_author_modules',
               'steps': updates[0], 'microsteps': microstep, 'epochs': epochs, 'smoke_verification': audit if smoke else None,
               'best_validation_iou': best if not smoke else None, 'optimizer': 'native_mmengine_AmpOptimWrapper',
               'accumulative_counts': accumulation, 'effective_pairs_per_update': batch_size*accumulation,
               'peak_allocated_gpu_mib': torch.cuda.max_memory_allocated()/1048576}
    write_json(out/'summary.json', summary)
    return summary
