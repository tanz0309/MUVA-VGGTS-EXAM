"""Explicit GPU-only smoke: independent head loading and GT-placeholder isolation."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT, load_rows, read_inputs, write_json
from run_experiment import seed
from v2sam_adapter import V2SamMultiAdapter


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fusion',type=Path,required=True)
    p.add_argument('--visual',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'logs/v2sam_gpu_integration_check.json')
    a=p.parse_args()
    cfg=json.loads((ROOT/'configs/v2sam-multi.json').read_text())
    assert Path(sys.prefix).resolve()==Path(cfg['environment']).resolve()
    assert torch.cuda.is_available(), 'GPU required; CPU execution disabled'
    torch.set_num_threads(2);seed(42)
    states={}
    for e,path in [('fusion',a.fusion),('visual',a.visual)]:
        ck=torch.load(path,map_location='cpu')
        assert ck['scope']=='smoke_only', 'This check only accepts smoke checkpoints'
        expected=json.loads((ROOT/'configs'/('v2sam.json' if e=='fusion' else 'v2sam-visual.json')).read_text())
        assert ck['config']==expected
        states[e]=ck['model']
    cfg.update(checkpoint=str(a.fusion),visual_checkpoint=str(a.visual))
    adapter=V2SamMultiAdapter(cfg)
    for expert in states:
        loaded=getattr(adapter,expert).predictor.state_dict()
        for k,v in states[expert].items(): assert torch.equal(loaded[k].cpu(),v), (expert,k)
    foundation=torch.load(cfg['sam2_checkpoint'],map_location='cpu')['model']
    anchor=adapter.fusion.anchor_decoder.state_dict()
    for k,v in anchor.items(): assert torch.equal(v.cpu(),foundation['sam_mask_decoder.'+k]),k
    del states,foundation
    images=read_inputs(load_rows('val')[0])  # Deliberately never read target GT.
    report={'scope':'gpu_integration_smoke_only','gpu':torch.cuda.get_device_name(0),
            'independent_expert_weights_loaded':True,'anchor_retains_foundation_decoder':True,
            'placeholder_value_invariance':{}}
    for name in ('visual','fusion'):
        expert=getattr(adapter,name);expert.model.eval()
        outputs=[]
        for value in (0,1):
            seed(42)
            batch=expert.data([images]);batch['masks'][0].fill_(value)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                outputs.append(expert.model.predict(batch)[0]['pred_masks'].cpu())
        assert torch.equal(*outputs), name+' output depends on target placeholder values'
        report['placeholder_value_invariance'][name]=True
    seed(42)
    prediction,audit=adapter.predict(*images)
    assert prediction.shape==(images[2].height,images[2].width)
    assert audit['target_gt_used_for_selection'] is False
    report.update(status='pass',selected_expert=audit['selected_expert'],
                  final_mask_shape=list(prediction.shape),peak_gpu_mib=torch.cuda.max_memory_allocated()/1048576,
                  fusion_smoke_checkpoint=str(a.fusion),visual_smoke_checkpoint=str(a.visual),pccs_audit=audit)
    write_json(a.output,report)
    print('GPU_INTEGRATION_PASS',json.dumps({k:v for k,v in report.items() if k!='pccs_audit'}),flush=True)

if __name__=='__main__':main()
