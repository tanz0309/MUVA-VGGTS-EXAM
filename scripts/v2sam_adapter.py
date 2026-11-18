"""Author V²-SAM experts, MUVA IO, and an explicitly paper-based PCCS selector."""
import copy
import json
import sys
from pathlib import Path
from types import MethodType
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pccs import EXPERTS, select_expert
from source_audit import verify_source


class V2SamAdapter:
    def __init__(self, config):
        verify_source(config)
        self.config = config
        self.variant = config['variant']
        sys.path.insert(0, config['repo'])
        from projects.v2sam.models.sam2_train import SAM2TrainRunner
        from projects.v2sam.models.v2sam import V2SAM
        from projects.v2sam.models.sparse_correspondence import SparseCorrespondenceMatcher, load_dinov3_model
        from projects.v2sam.models.preprocess.image_resize import DirectResize
        from third_parts.mmdet.models.losses import CrossEntropyLoss, DiceLoss
        self.resize = DirectResize(1024)
        self.point_prep = V2SAM._prep_sparse_correspondence_points
        self.fill_holes = V2SAM.fill_holes_in_masks
        if self.variant == 'anchor':
            runner = SAM2TrainRunner(ckpt_path=config['sam2_checkpoint'])
            dino = load_dinov3_model(repo_path=config['dinov3_source'], weights_path=config['dinov3_checkpoint'])
            matcher = SparseCorrespondenceMatcher(dino, **config['sparse_correspondence'])
            self.model = torch.nn.ModuleDict({'runner': runner, 'anchor': matcher}).cuda().eval().requires_grad_(False)
            self.runner, self.anchor = runner, matcher
            self.frozen_modules = [self.model]
        else:
            kwargs = {}
            runner_kwargs = {}
            if self.variant == 'visual':
                from projects.v2sam_visual.models.v2sam import V2SAM
                from projects.v2sam_visual.models.sam2_train import SAM2TrainRunner
                # Keep the Visual extension, even when Fusion is imported in the same process.
                runner_kwargs['hydra_overrides_extra'] = ['++model._target_=projects.v2sam_visual.models.extension.SAM2Base']
            else:
                kwargs = {'dinov3_cfg': {'repo_path': config['dinov3_source'], 'weights_path': config['dinov3_checkpoint']},
                          'sparse_corr_cfg': config['sparse_correspondence']}
            self.model = V2SAM(
                grounding_encoder=dict(type=SAM2TrainRunner, ckpt_path=config['sam2_checkpoint'], **runner_kwargs),
                loss_mask=dict(type=CrossEntropyLoss, use_sigmoid=True, reduction='mean', loss_weight=2.),
                loss_dice=dict(type=DiceLoss, use_sigmoid=True, activate=True, reduction='mean', naive_dice=True, eps=1., loss_weight=.5),
                frozen_sam2_decoder=False, pretrained_pth=None, loss_sample_points=True,
                bs=config['training']['batch_size'], **kwargs).cuda().eval()
            self.runner = self.model.grounding_encoder
            self.anchor = getattr(self.model, 'sparse_correspondence', None)
            self.predictor = torch.nn.ModuleDict({'matcher': self.model.matcher,
                'constr_prompt_fcs': self.model.constr_prompt_fcs, 'decoder': self.runner.sam2_model.sam_mask_decoder})
            if config.get('keep_anchor_decoder'):
                self.anchor_decoder = copy.deepcopy(self.runner.sam2_model.sam_mask_decoder).eval().requires_grad_(False)
            if config['checkpoint']:
                self.predictor.load_state_dict(torch.load(config['checkpoint'], map_location='cpu')['model'], strict=True)
            self.frozen_modules = [self.runner] + ([self.anchor] if self.anchor is not None else [])
        if self.anchor is not None:
            original_stratify = self.anchor._stratify_points
            def stable_stratify(module, points, threshold):
                # BF16 can round threshold+1 back to threshold, making the native loop infinite.
                with torch.autocast('cuda', enabled=False):
                    return original_stratify(points.float(), threshold)
            self.anchor._stratify_points = MethodType(stable_stratify, self.anchor)
        self.last_audit = {}

    def training_state(self):
        return {'contrastive_step': getattr(type(self.model), '_constr_step', 0)}

    def restore_training_state(self, state):
        type(self.model)._constr_step = state.get('contrastive_step', 0)

    def image_tensor(self, image):
        a = self.resize.apply_image(np.asarray(image)).copy()
        return torch.from_numpy(a).permute(2, 0, 1).cuda().float()

    def mask_tensor(self, mask):
        a = np.asarray(Image.fromarray((np.asarray(mask) > 0).astype(np.uint8)).resize((1024, 1024), Image.Resampling.NEAREST)).copy()
        return torch.from_numpy(a)[None].cuda()

    def data(self, images, gt_loaders=None):
        refs, rms, tars = zip(*images)
        data = {'g_pixel_values': [self.image_tensor(im) for im in tars],
                'prompt_g_pixel_values': [self.image_tensor(im) for im in refs],
                'prompt_masks': [self.mask_tensor(m) for m in rms],
                'frames_per_batch': [1] * len(images),
                'query_img_path': [[im] for im in refs], 'target_img_path': [[im] for im in tars],
                'raw_prompt_masks': [torch.from_numpy((np.asarray(m) > 0).astype(np.uint8))[None] for m in rms]}
        # Native predict only uses target-mask shape; zero placeholders contain no target labels.
        data['masks'] = ([self.mask_tensor(fn()) for fn in gt_loaders] if gt_loaders is not None
                         else [torch.zeros((1, 1024, 1024), dtype=torch.uint8, device='cuda') for _ in images])
        return data

    def training_loss_batch(self, images, gt_loaders):
        self.model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses = self.model(self.data(images, gt_loaders), mode='loss')
            loss = sum(losses.values())
        return loss, {k: float(v.detach()) for k, v in losses.items()}

    @torch.no_grad()
    def correspondence(self, reference, reference_mask, target):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.anchor(image_ego=reference, image_exo=target,
                mask_ego=(np.asarray(reference_mask) > 0).astype(np.uint8), mask_exo=None)

    @staticmethod
    def restore_mask(mask, target):
        restored = F.interpolate(mask.reshape(1, 1, *mask.shape[-2:]).float(),
                                 size=(target.height, target.width), mode='nearest')[0, 0]
        return restored.cpu().numpy().astype(bool)

    @torch.no_grad()
    def predict_anchor(self, reference, reference_mask, target):
        self.model.eval()
        result = self.correspondence(reference, reference_mask, target)
        audit = {'expert': 'anchor', 'reference_points': result['points_ego'][0].tolist(),
                 'positive_prompts_original': result['points_exo'][0].tolist()}
        if not result['num_matches'][0]:
            return np.zeros((target.height, target.width), bool), audit
        sparse = self.point_prep(self, result['points_exo'], (target.height, target.width), 'cuda:0')
        saved = self.runner.sam2_model.sam_mask_decoder
        if hasattr(self, 'anchor_decoder'):
            self.runner.sam2_model.sam_mask_decoder = self.anchor_decoder
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                states, _, _ = self.runner.get_sam2_embeddings(
                    self.runner.preprocess_image(self.image_tensor(target))[None],
                    self.runner.preprocess_image(self.image_tensor(reference))[None])
                low = self.runner.inject_language_embd(states, None, sparse, nf_nobj=(1, 1))
                logits = F.interpolate(low.reshape(1, 1, *low.shape[-2:]), size=(1024, 1024),
                                       mode='bilinear', align_corners=False)[0]
                mask = self.fill_holes(self, (logits.sigmoid() > .5).to(torch.uint8))
            return self.restore_mask(mask, target), audit
        finally:
            self.runner.sam2_model.sam_mask_decoder = saved

    @torch.no_grad()
    def predict(self, reference, reference_mask, target):
        if self.variant == 'anchor':
            return self.predict_anchor(reference, reference_mask, target)
        self.model.eval()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = self.model.predict(self.data([(reference, reference_mask, target)]))
        return self.restore_mask(result[0]['pred_masks'], target), {
            'expert': self.variant, 'prediction_entry': type(self.model).__module__ + '.V2SAM.predict',
            'target_mask_placeholder': 'zeros_1024x1024_shape_only', 'target_gt_in_prediction': False}


class V2SamMultiAdapter:
    """Separate native Visual/Fusion networks; sequential experts; one DINO matcher."""
    def __init__(self, config):
        cfg_root = Path(__file__).parents[1] / 'configs'
        fc = json.loads((cfg_root / 'v2sam.json').read_text())
        vc = json.loads((cfg_root / 'v2sam-visual.json').read_text())
        for cfg in (fc, vc):
            for k in ('repo', 'repo_commit', 'sam2_checkpoint', 'dinov3_source', 'dinov3_checkpoint', 'seed'):
                cfg[k] = config[k]
        fc.update(checkpoint=config.get('checkpoint'), keep_anchor_decoder=True)
        vc.update(checkpoint=config.get('visual_checkpoint'))
        self.fusion = V2SamAdapter(fc)
        self.visual = V2SamAdapter(vc)
        self.predictor = torch.nn.ModuleDict({'fusion': self.fusion.predictor, 'visual': self.visual.predictor})
        self.config = config

    @torch.no_grad()
    def predict(self, reference, reference_mask, target):
        results = {'anchor': self.fusion.predict_anchor(reference, reference_mask, target),
                   'visual': self.visual.predict(reference, reference_mask, target),
                   'fusion': self.fusion.predict(reference, reference_mask, target)}
        reference_points = results['anchor'][1].get('reference_points', [])
        returned = {}
        for expert in EXPERTS:
            prediction = results[expert][0]
            if not prediction.any() or not reference_points:
                returned[expert] = []
                continue
            reverse = self.fusion.correspondence(target, Image.fromarray(prediction.astype(np.uint8) * 255), reference)
            returned[expert] = reverse['points_exo'][0].tolist()
        predictions = {e: results[e][0] for e in EXPERTS}
        selected, audit = select_expert(predictions, returned, reference_points, reference.size)
        audit.update(expert='multi_pccs', reference_points=reference_points,
                     back_projected_points=returned, experts={e: results[e][1] for e in EXPERTS},
                     candidate_foreground_pixels={e: int(predictions[e].sum()) for e in EXPERTS})
        return predictions[selected], audit
