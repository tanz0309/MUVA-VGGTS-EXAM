"""GPU-only smoke, evaluation and new MUVA supervised training runs."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import ROOT, digest, load_rows, metrics, read_gt, read_inputs, summarize, write_json


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)


def score_one(adapter,row,out):
    images=read_inputs(row)  # Disk access excluded from inference timing.
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    started=time.perf_counter(); failed=False; error=None; audit={}
    try:
        prediction,audit=adapter.predict(*images)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
    except torch.cuda.OutOfMemoryError:
        raise  # Stop incomplete run; never turn hardware OOM into fake full evaluation.
    except Exception as exc:
        torch.cuda.synchronize(); elapsed=time.perf_counter()-started
        failed=True; error=f'{type(exc).__name__}: {exc}'
        prediction=np.zeros((images[2].height,images[2].width),bool)
        (out/'errors').mkdir(exist_ok=True)
        (out/'errors'/f"{row['case_id']}.txt").write_text(traceback.format_exc())
    peak=torch.cuda.max_memory_allocated()/1048576
    gt=read_gt(row)  # No target supervision is passed to predict().
    m=metrics(prediction,gt)
    assert m['components']==row['target_components']
    path=out/'predictions'/f"{row['case_id']}.png"
    path.parent.mkdir(exist_ok=True)
    tmp=path.with_suffix('.png.part')
    Image.fromarray(prediction.astype(np.uint8)*255).save(tmp,format='PNG');tmp.replace(path)
    if audit: write_json(out/'prompts'/f"{row['case_id']}.json",audit)
    return {'case_id':row['case_id'],'split':row['split'],'scene':row['scene'],
            'group':row['group'],'target_visibility':row['target_visibility'],**m,
            'inference_seconds':elapsed,'peak_allocated_gpu_mib':peak,'failed':failed,'error':error}


def evaluate(adapter,rows,out,smoke=False,warmup=True):
    out.mkdir(parents=True,exist_ok=True)
    if warmup:
        print('GPU_WARMUP',rows[0]['case_id'],flush=True)
        adapter.predict(*read_inputs(rows[0]));torch.cuda.synchronize()
    results=[]
    with (out/'per_case.jsonl').open('w',buffering=1) as f:
        for i,row in enumerate(rows):
            result=score_one(adapter,row,out);results.append(result)
            f.write(json.dumps(result,allow_nan=False)+'\n')
            if i%25==0 or i==len(rows)-1:
                print('EVALUATED',i+1,'/',len(rows),row['case_id'], 'failed',result['failed'],flush=True)
            if smoke and result['failed']: raise RuntimeError(result['error'])
    summary=summarize(results)
    summary.update({'scope':'smoke_only_not_benchmark' if smoke else 'full_split',
                    'status':'complete_with_failures' if summary['failures'] else 'complete',
                    'aggregation':'macro_per_target_case','metric_scale':'0_to_1',
                    'timing':getattr(adapter, 'timing_description',
                        'uncached complete pair; excludes image disk IO, initialization, GT scoring and output IO')})
    groups=defaultdict(list)
    for r in results:
        groups['components/'+('2' if r['components']==2 else '3+')].append(r)
        groups['scene_setting/'+r['group']].append(r)
        occ=1-r['target_visibility']
        name='0-30%' if occ<.3 else '30-60%' if occ<.6 else '60-90%' if occ<.9 else '90-100%'
        groups['occlusion/'+name].append(r)
    summary['by_group']={k:summarize(v) for k,v in groups.items()}
    write_json(out/'summary.json',summary)
    fields=[k for k in results[0] if k not in ['component_coverages']]
    with (out/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        flat={k:v for k,v in summary.items() if k!='by_group'}
        w=csv.DictWriter(f,fieldnames=list(flat));w.writeheader();w.writerow(flat)
    return summary


def train(adapter,config,out,args):
    if config['model'] in ('v2sam', 'v2sam-visual'):
        from v2sam_training import train_v2sam
        return train_v2sam(adapter, config, out, args, evaluate)
    cfg=config['training']; rows=load_rows('train'); val=load_rows('val')
    parameters=[p for p in adapter.predictor.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    batch_size=cfg.get('batch_size',1)
    batches_per_epoch=(len(rows)+batch_size-1)//batch_size
    scheduler_type=cfg.get('scheduler','multistep_epochs')
    if scheduler_type=='cosine_steps':
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs']*batches_per_epoch,eta_min=cfg['minimum_lr'])
    elif scheduler_type=='warmup_cosine_steps':
        import math
        total=cfg['epochs']*batches_per_epoch;warmup=max(1,int(total*cfg['warmup_ratio']))
        def rate(i):
            return max(1e-5,i/warmup) if i<warmup else .5*(1+math.cos(math.pi*min(1,(i-warmup)/(total-warmup))))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    else:
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=cfg['lr_milestones'],gamma=.1)
    start_epoch=0;best=-1.;step=0
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu')
        if ck.get('scope')!='muva_fragment_training': raise ValueError('Cannot resume from smoke/foreign checkpoint')
        assert ck['train_manifest_sha256']==digest(ROOT/'data/manifests/train.jsonl')
        assert ck['config']==config, 'Resume configuration differs from checkpoint'
        adapter.predictor.load_state_dict(ck['model'],strict=True)
        optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        start_epoch=ck['epoch'];best=ck['best_val_iou'];step=ck['step']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        if hasattr(adapter,'restore_training_state'): adapter.restore_training_state(ck.get('adapter_training_state',{}))
        del ck
    smoke=args.mode=='smoke-train'
    if smoke and args.resume: raise ValueError('Smoke training cannot resume real training')
    epochs=1 if smoke else cfg['epochs']; smoke_result=None
    probe=parameters[0]
    with (out/'training.jsonl').open('w',buffering=1) as log:
        for epoch in range(start_epoch,epochs):
            order=list(range(len(rows)));random.Random(config['seed']+epoch).shuffle(order)
            # Different scenes within a contrastive minibatch avoid same-instance false negatives.
            if batch_size==2:
                for j in range(0,len(order)-1,2):
                    if rows[order[j]]['scene']==rows[order[j+1]]['scene']:
                        replacement=next((z for z in range(j+2,len(order)) if rows[order[z]]['scene']!=rows[order[j]]['scene']),None)
                        if replacement is not None: order[j+1],order[replacement]=order[replacement],order[j+1]
            if smoke: order=order[:batch_size]
            adapter.predictor.train()
            for offset in range(0,len(order),batch_size):
                batch_rows=[rows[index] for index in order[offset:offset+batch_size]]
                row=batch_rows[0];before=probe.detach().clone() if smoke else None
                optimizer.zero_grad(set_to_none=True)
                if batch_size>1:
                    loss,details=adapter.training_loss_batch([read_inputs(r) for r in batch_rows],[lambda r=r:read_gt(r) for r in batch_rows])
                else:
                    loss,details=adapter.training_loss(*read_inputs(row),lambda: read_gt(row))
                if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
                loss.backward()
                if cfg.get('max_grad_norm') is None:
                    # O-MaMa's official update has no gradient clipping.
                    norm=torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in parameters if p.grad is not None]))
                else:
                    norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['max_grad_norm'])
                if not torch.isfinite(norm): raise RuntimeError('Non-finite gradients')
                optimizer.step();step+=1
                if scheduler_type.endswith('_steps'): scheduler.step()
                item={'epoch':epoch+1,'step':step,'case_id':row['case_id'],
                      'case_ids':[r['case_id'] for r in batch_rows],
                      'loss':float(loss.detach()),'gradient_norm':float(norm),**details}
                if smoke:
                    changed=bool((before!=probe.detach()).any())
                    assert changed and float(norm)>0
                    assert all(p.grad is None for module in adapter.frozen_modules for p in module.parameters() if not p.requires_grad)
                    item.update({'predictor_parameter_changed':changed,'frozen_backbones_no_grad':True})
                    smoke_result=item
                log.write(json.dumps(item)+'\n')
                if step%25==0 or smoke: print('TRAIN_STEP',json.dumps(item),flush=True)
                del loss
            if not scheduler_type.endswith('_steps'): scheduler.step()
            val_summary=None
            if not smoke:
                val_summary=evaluate(adapter,val,out/f'val_epoch_{epoch+1:02d}')
                if val_summary['failures']: raise RuntimeError('Validation failures: refusing best-checkpoint selection')
            improved=not smoke and val_summary['visible_iou']>best
            if improved: best=val_summary['visible_iou']
            checkpoint={'model':adapter.predictor.state_dict(),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'epoch':epoch+1,'step':step,'best_val_iou':best,
                'config':config,'scope':'smoke_only' if smoke else 'muva_fragment_training',
                'train_manifest_sha256':digest(ROOT/'data/manifests/train.jsonl'),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
            if hasattr(adapter,'training_state'): checkpoint['adapter_training_state']=adapter.training_state()
            path=out/('smoke_checkpoint.pth' if smoke else 'last.pth')
            tmp=path.with_suffix('.pth.part');torch.save(checkpoint,tmp);tmp.replace(path)
            if improved:
                import shutil
                shutil.copy2(path,out/'best.pth')
            if smoke:
                loaded=torch.load(path,map_location='cpu')
                adapter.predictor.load_state_dict(loaded['model'],strict=True)
                assert loaded['step']==1 and loaded['scope']=='smoke_only'
                smoke_result['checkpoint_reload_passed']=True
                del loaded
    summary={'scope':'smoke_only_not_training_experiment' if smoke else 'muva_supervised_training_from_random_head',
             'status':'complete','steps':step,'epochs':epochs,'smoke_verification':smoke_result,
             'best_validation_iou':best if not smoke else None,
             'peak_allocated_gpu_mib':torch.cuda.max_memory_allocated()/1048576}
    write_json(out/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['insid3','omama','v2sam','v2sam-visual','v2sam-anchor','v2sam-multi'],required=True)
    p.add_argument('--mode',choices=['smoke','evaluate','smoke-train','train'],default='smoke')
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--config',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--visual-checkpoint',type=Path)
    args=p.parse_args()
    training_free=args.model in ['insid3','v2sam-anchor']
    if training_free and 'train' in args.mode: p.error(f'{args.model} is training-free')
    if args.model=='v2sam-multi' and 'train' in args.mode:
        p.error('Train v2sam (Fusion) and v2sam-visual separately; multi only performs expert selection')
    if args.visual_checkpoint and args.model!='v2sam-multi': p.error('--visual-checkpoint is only for v2sam-multi')
    if args.resume and args.mode!='train': p.error('--resume requires --mode train')
    if args.checkpoint and args.mode in ['train','smoke-train']:
        p.error('New training cannot load a task checkpoint; use --resume to continue this experiment')
    if not training_free and args.mode=='evaluate' and not args.checkpoint:
        p.error('Trainable model formal evaluation requires --checkpoint from your MUVA training run')
    if args.model=='v2sam-multi' and args.mode=='evaluate' and not args.visual_checkpoint:
        p.error('v2sam-multi evaluation also requires --visual-checkpoint from your Visual training run')
    config=json.loads((args.config or ROOT/'configs'/f'{args.model}.json').read_text())
    assert Path(sys.prefix).resolve()==Path(config['environment']).resolve(), 'Use the model-specific Conda environment'
    if not torch.cuda.is_available(): raise RuntimeError('GPU unavailable; CPU fallback is disabled')
    if not training_free and config['checkpoint'] is not None:
        p.error('Keep config checkpoint null; select a validated task checkpoint with --checkpoint')
    if args.checkpoint:
        config['checkpoint']=str(args.checkpoint.resolve(strict=True))
        if not training_free:
            saved=torch.load(config['checkpoint'],map_location='cpu')
            if saved.get('scope')!='muva_fragment_training':
                p.error('Formal evaluation requires a real MUVA training checkpoint, not a smoke or official Predictor')
            assert saved['train_manifest_sha256']==digest(ROOT/'data/manifests/train.jsonl')
            expected=json.loads((ROOT/'configs/v2sam.json').read_text()) if args.model=='v2sam-multi' else config
            assert {k:v for k,v in saved['config'].items() if k!='checkpoint'} == {k:v for k,v in expected.items() if k!='checkpoint'}, 'Training/evaluation config mismatch'
            del saved
    if args.visual_checkpoint:
        config['visual_checkpoint']=str(args.visual_checkpoint.resolve(strict=True))
        saved=torch.load(config['visual_checkpoint'],map_location='cpu')
        assert saved['scope']=='muva_fragment_training','Visual checkpoint must be a real training run'
        assert saved['train_manifest_sha256']==digest(ROOT/'data/manifests/train.jsonl')
        assert saved['config']==json.loads((ROOT/'configs/v2sam-visual.json').read_text()),'Visual configuration mismatch'
        del saved
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    label='trained' if args.checkpoint else 'training_free' if training_free else 'new_task_head'
    run_split='train' if args.mode in ['train','smoke-train'] else args.split
    out=args.output or ROOT/'outputs'/args.model/args.mode/f'{run_split}_{label}_{stamp}'
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('OMP_NUM_THREADS','1');torch.set_num_threads(2)
    seed(config['seed'])
    info={'model':args.model,'mode':args.mode,'split':run_split,'config':config,
          'device':torch.cuda.get_device_name(0),'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),
          'python_executable':sys.executable,'torch':torch.__version__,'torch_cuda':torch.version.cuda,
          'started_utc':stamp,'checkpoint_sha256':digest(config['checkpoint']) if config['checkpoint'] else None,
          'manifest_sha256':{s:digest(ROOT/'data/manifests'/f'{s}.jsonl') for s in ['train','val','test']},
          'adapter_sha256':digest(Path(__file__).with_name('adapters.py')),
          'evaluator_sha256':digest(Path(__file__).with_name('common.py')),
          'target_gt_in_model_inputs':False,'seed':config['seed'],
          'cudnn_benchmark':False,'cudnn_deterministic':True,
          'deterministic_algorithms':'enabled with warnings for unsupported operators',
          'cublas_workspace_config':os.getenv('CUBLAS_WORKSPACE_CONFIG')}
    if args.model=='vggts': info['vggt_checkpoint_sha256']=digest(config['vggt_checkpoint'])
    for key in ['dinov2_checkpoint','fastsam_checkpoint','sam2_checkpoint','dinov3_checkpoint','visual_checkpoint']:
        if config.get(key): info[key+'_sha256']=digest(config[key])
    if args.model=='omama': info['model_adapter_sha256']=digest(Path(__file__).with_name('omama_adapter.py'))
    if args.model.startswith('v2sam'): info['model_adapter_sha256']=digest(Path(__file__).with_name('v2sam_adapter.py'))
    if args.model.startswith('v2sam'):
        info['selector_sha256']=digest(Path(__file__).with_name('pccs.py'))
        info['training_driver_sha256']=digest(Path(__file__).with_name('v2sam_training.py'))
    write_json(out/'run_config.json',info)
    print('OUTPUT_DIRECTORY',out,flush=True)
    from adapters import build_adapter
    try:
        adapter=build_adapter(args.model,config)
        model_details={'model':args.model,'seed':config['seed'],'gpu':torch.cuda.get_device_name(0)}
        if not training_free:
            probe=next(p for p in adapter.predictor.parameters() if p.requires_grad).detach().cpu().contiguous().numpy()
            model_details.update({'initial_predictor_probe_sha256':hashlib.sha256(probe.tobytes()).hexdigest(),
                'predictor_initialization':'checkpoint' if config['checkpoint'] else config.get('head_initialization','random'),
                'predictor_parameters':sum(p.numel() for p in adapter.predictor.parameters()),
                'predictor_trainable_parameters':sum(p.numel() for p in adapter.predictor.parameters() if p.requires_grad),
                'initialization_policy':config.get('head_initialization')})
        else:
            model_details['trainable_parameters']=sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
        write_json(out/'model_initialization.json',model_details)
        if args.mode in ['train','smoke-train']:
            summary=train(adapter,config,out,args)
        else:
            rows=load_rows(args.split)
            if args.mode=='smoke':
                first=rows[0]
                other=next((r for r in rows if r['target_components']>=3 and r['scene']!=first['scene']),rows[1])
                rows=[first,other]
            summary=evaluate(adapter,rows,out,smoke=args.mode=='smoke')
        write_json(out/'status.json',{'status':'complete','scope':summary['scope']})
        print('RUN_COMPLETE',json.dumps(summary),flush=True)
    except BaseException as exc:
        write_json(out/'status.json',{'status':'incomplete_failed','error':f'{type(exc).__name__}: {exc}'})
        raise


if __name__=='__main__': main()
