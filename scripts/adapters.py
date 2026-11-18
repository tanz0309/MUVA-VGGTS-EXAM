"""Thin model adapters. predict() only accepts reference RGB/mask and target RGB."""
import sys
from pathlib import Path

import numpy as np
import torch


class Insid3Adapter:
    def __init__(self, config):
        from source_audit import verify_source
        verify_source(config)
        repo = Path(config['repo']); sys.path.insert(0, str(repo))
        from models.insid3 import INSID3
        encoder = torch.hub.load(config['dinov3_source'], 'dinov3_vitl16', source='local',
                                 weights=config['checkpoint'])
        self.model = INSID3(encoder=encoder, image_size=config['image_size'],
                           svd_components=config['svd_components'], tau=config['tau'],
                           merge_threshold=config['merge_threshold'], mask_refiner='bilinear',
                           resize_to_orig_size=True, device='cuda:0')
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def predict(self, reference, reference_mask, target):
        self.model.reset_state()
        self.model.set_reference(reference, reference_mask)
        self.model.set_target(target)
        prediction = self.model.segment()
        return prediction.detach().cpu().numpy().astype(bool), {}


def build_adapter(name, config):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; CPU model execution is disabled')
    if name=='insid3': return Insid3Adapter(config)
    if name=='vggts': raise ValueError('Use run.sh train/evaluate vggts: the legacy adapter was retired')
    if name=='omama':
        from omama_adapter import OmamaAdapter
        return OmamaAdapter(config)
    if name.startswith('v2sam'):
        from v2sam_adapter import V2SamAdapter,V2SamMultiAdapter
        return V2SamMultiAdapter(config) if name=='v2sam-multi' else V2SamAdapter(config)
    raise ValueError(name)
