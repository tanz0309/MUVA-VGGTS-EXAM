"""MUVA IO/coordinate adapter; network, crop functions and losses come from pinned upstream."""
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import ROOT, load_rows, read_inputs, read_gt, write_json
from vggts_point_selector import (array_fingerprint, select_target_points,
                                  validate_selector_config)


def setup_sources(config):
    for key, commit_key in [('repo', 'repo_commit'), ('vggt_repo', 'vggt_commit')]:
        repo = Path(config[key])
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != config[commit_key]:
            raise RuntimeError(f'Unexpected upstream revision: {repo}: {commit}')
    changed = subprocess.check_output(['git', '-C', config['repo'], 'diff', 'HEAD', '--name-only'], text=True)
    if changed.strip():
        raise RuntimeError('Pinned VGGT-S author files have been edited: '+changed)
    # Never import the historically modified /root/VGGT-S-main/src.
    sys.path[:0] = [str(Path(config['repo'])/'src'), config['vggt_repo']]
    path = Path(config['repo'])/'data/extract_point_g2x.py'
    spec = importlib.util.spec_from_file_location('official_extract_points', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.G2xCropImage.pad_frame


def inputs_key(inputs):
    h = hashlib.sha256()
    for image in inputs:
        a = np.asarray(image)
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class MuvaPreprocessing:
    def __init__(self, config, selector_config=None):
        self.config = config
        # Selector settings stay out of the base config so old checkpoints and
        # the first-stage cache recipe remain strictly compatible.
        self.selector_config = validate_selector_config(
            selector_config if selector_config is not None else {},
            prompt_points=config['prompt_points'])
        self.pad_frame = setup_sources(config)
        # Batch size and epochs do not change frozen first-stage point generation.
        recipe = {k: config[k] for k in ['protocol', 'repo_commit', 'vggt_commit',
                   'seed', 'tracking_queries', 'image_size', 'vggt_checkpoint_sha256']}
        self.recipe = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        self.cache = Path(config['point_cache'])/self.recipe[:16]
        self.point_cache_usage = {'generated_online': 0, 'reused': 0}

    def cache_path(self, inputs):
        return self.cache/(inputs_key(inputs)+'.json')

    def read_points(self, inputs):
        path = self.cache_path(inputs)
        if not path.exists():
            raise FileNotFoundError(f'Missing official first-stage points: {path}')
        value = json.loads(path.read_text())
        if value['recipe'] != self.recipe:
            raise ValueError(f'Point cache protocol mismatch: {path}')
        tracks = np.asarray(value['tracks_518'], dtype=np.float32)
        if tracks.shape != (self.config['tracking_queries'], 2) or not np.isfinite(tracks).all():
            raise ValueError(f'Invalid point cache: {path}')
        return value

    @torch.no_grad()
    def make_points(self, inputs, vggt, case_id):
        path = self.cache_path(inputs)
        if path.exists():
            self.read_points(inputs)
            return False
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError('On-demand GPU points require DataLoader num_workers=0')
        if vggt is None:
            raise RuntimeError('A frozen GPU VGGT backbone is required to generate missing points')
        # Cache misses must not change the random stream used by Predictor training.
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            try:
                self._generate_points(inputs, vggt, case_id, path)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
        return True

    def _generate_points(self, inputs, vggt, case_id, path):
        from utils import mask_sampling
        size = self.config['image_size']
        reference, mask, target = [np.asarray(x) for x in inputs]
        mask = (mask > 0).astype(np.uint8)
        ref = self.pad_frame(None, reference, (size, size), False)
        m = self.pad_frame(None, mask, (size, size), True)
        tar = self.pad_frame(None, target, (size, size), False)
        if not m.any():
            raise ValueError(f'Reference mask vanished in official preprocessing: {case_id}')
        queries = mask_sampling(m[None, None], self.config['tracking_queries'])[:, 0]
        pair = torch.from_numpy(np.stack([ref, tar])).permute(0, 3, 1, 2)[None].cuda().float()/255.
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tokens, patch = vggt.aggregator(pair)
            coords, vis, conf = vggt.track_head(tokens, images=pair, patch_start_idx=patch,
                                              query_points=torch.as_tensor(queries, device='cuda'))
        value = {'recipe': self.recipe, 'case_id_at_creation': case_id,
                 'tracks_518': coords[-1][0, 1].float().clamp(0, size-1).cpu().tolist(),
                 'visibility': vis[0, 1].float().cpu().tolist(),
                 'confidence': conf[0, 1].float().cpu().tolist()}
        write_json(path, value)

    def resolve_points(self, inputs, vggt, case_id=None):
        try:
            value = self.read_points(inputs)
            source = 'cache'
            self.point_cache_usage['reused'] += 1
        except FileNotFoundError:
            case_id = case_id or 'pair_' + inputs_key(inputs)[:16]
            self.make_points(inputs, vggt, case_id)
            value = self.read_points(inputs)
            source = 'generated_online'
            self.point_cache_usage['generated_online'] += 1
            count = self.point_cache_usage['generated_online']
            if count == 1 or count % 25 == 0:
                print('POINTS_GENERATED', count, case_id, flush=True)
        return value, source

    def prepare(self, inputs, cached=None, *, vggt=None, case_id=None):
        from utils import (clip_exo_mean, crop_and_resize_by_mask_safe, remove_outliers_median,
                           cal_crop_size_3x, clip_exo_box_center_wp, cluster_points)
        cfg = self.config; size = cfg['image_size']
        ref, ref_mask, tar = [np.asarray(x) for x in inputs]
        if ref.shape[:2] != (1080, 1920) or tar.shape[:2] != (1080, 1920):
            raise ValueError('This explicit MUVA 1920x1080 coordinate adapter does not accept other resolutions')
        camera_size = tuple(cfg['native_camera_size'])
        ref_camera = cv2.resize(ref, camera_size, interpolation=cv2.INTER_AREA)
        mask_camera = cv2.resize((ref_mask > 0).astype(np.uint8), camera_size, interpolation=cv2.INTER_NEAREST)
        tar_camera = cv2.resize(tar, camera_size, interpolation=cv2.INTER_AREA)
        # MUVA cameras are external views: use upstream exo_src and exo_tar primitives.
        src, smask = clip_exo_mean(ref_camera, mask_camera, size)
        src_cropped, smask_cropped = crop_and_resize_by_mask_safe(src, smask, cfg['source_crop_R'])
        if smask_cropped.any():
            src, smask = src_cropped, smask_cropped
        if not smask.any():
            raise ValueError('Reference mask empty after upstream source crop')
        if cached is None:
            cached, point_source = self.resolve_points(inputs, vggt, case_id)
        else:
            point_source = 'provided'
        tracks = np.asarray(cached['tracks_518'], dtype=np.float32).copy()
        # Upstream padding is bottom/right. Its short side is rounded to a multiple of 14.
        resized_h = round(tar.shape[0]*(size/tar.shape[1])/14)*14
        tracks *= np.array([camera_size[0]/size, camera_size[1]/resized_h], np.float32)
        tracks[:, 0] = tracks[:, 0].clip(0, camera_size[0]-1)
        tracks[:, 1] = tracks[:, 1].clip(0, camera_size[1]-1)
        inliers, _ = remove_outliers_median(tracks, K=10)
        crop_size = cal_crop_size_3x(max(np.ptp(inliers[:, 0]), np.ptp(inliers[:, 1])))
        # A dummy mask keeps target GT out of crop and prompt construction, including evaluation.
        target, _, points, _, meta = clip_exo_box_center_wp(
            inliers, tar_camera, np.zeros(tar_camera.shape[:2], np.uint8), size, size, crop_size)
        # Phase 1 changes only this 40->5 choice; crop and cache stay untouched.
        selection = select_target_points(
            points, image_size=size, config=self.selector_config,
            native_kmeans=cluster_points)
        prompts = selection.prompts_xy
        candidate_points = np.ascontiguousarray(points, dtype=np.float32)
        return {'source': src, 'source_mask': smask, 'target': target,
                'target_points': prompts.astype(np.float32), 'meta': meta,
                'original_shape': tar.shape[:2], 'tracks': tracks, 'inliers': inliers,
                'audit': {'official_commit': cfg['repo_commit'], 'target_crop_960x540': meta,
                    'first_stage_point_source': point_source,
                    'tracking_points_original': (tracks*2).tolist(),
                    'filtered_points_original': (inliers*2).tolist(),
                    'positive_prompts_original': ((prompts*crop_size/size+
                        np.array([meta['x1'], meta['y1']]))*2).tolist(),
                    'point_selector': selection.audit,
                    'selection_inputs': {
                        'candidate_points_sha256': array_fingerprint(candidate_points),
                        'target_crop_rgb_sha256': array_fingerprint(target),
                        'source_crop_rgb_sha256': array_fingerprint(src),
                        'source_crop_mask_sha256': array_fingerprint(smask),
                        'meta_sha256': hashlib.sha256(json.dumps(
                            meta, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}}}

    def target_supervision(self, gt, prepared):
        mask = cv2.resize(gt.astype(np.uint8), tuple(self.config['native_camera_size']),
                          interpolation=cv2.INTER_NEAREST)
        m = prepared['meta']; size = self.config['image_size']
        cropped = mask[m['y1']:m['y2'], m['x1']:m['x2']]
        crop = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_NEAREST)
        return crop, mask, bool(cropped.sum() < mask.sum())

    def check_cache(self, rows):
        missing = []
        for row in rows:
            try:
                self.read_points(read_inputs(row))
            except FileNotFoundError:
                missing.append(row['case_id'])
        return {'cases': len(rows), 'missing': len(missing), 'first_missing': missing[:5]}


class MuvaOfficialDataset(Dataset):
    def __init__(self, rows, preprocessing, training=False, audit_path=None, vggt=None):
        self.rows = rows; self.preprocessing = preprocessing; self.training = training
        self.audit_path = audit_path; self.last_case = None
        self.vggt = vggt

    def __len__(self):
        return len(self.rows)

    def process(self, index):
        row = self.rows[index]; inputs = read_inputs(row)
        p = self.preprocessing.prepare(inputs, vggt=self.vggt, case_id=row['case_id'])
        # Supervision is read only after the crop and points have been determined.
        gt = read_gt(row)
        crop_gt, original_gt, truncated = self.preprocessing.target_supervision(gt, p)
        item = (p['source'], p['source_mask'], np.asarray(inputs[0]), np.asarray(inputs[1]),
                p['target'], crop_gt, np.asarray(inputs[2]), original_gt,
                p['inliers'], p['target_points'], truncated, p['meta'])
        return item

    def __getitem__(self, index):
        if not self.training:
            return self.process(index)
        # Preserve upstream training validity/retry policy; never filter evaluation cases.
        for retry in range(10):
            used = (index+retry) % len(self)
            item = self.process(used)
            if item[5].sum() > 0 and item[1].sum() >= self.preprocessing.config['image_size']**2*.001:
                self.last_case = (used, item)
                if retry:
                    self.record_retry(index, used, retry)
                return item
        if self.last_case is not None:
            used, item = self.last_case
            self.record_retry(index, used, 10)
            return item
        raise ValueError('No valid training case in upstream 10-retry window; no silent invalid-target training')

    def record_retry(self, requested, used, retries):
        if self.audit_path:
            with Path(self.audit_path).open('a') as f:
                f.write(json.dumps({'requested': self.rows[requested]['case_id'],
                                    'used': self.rows[used]['case_id'], 'retries': retries})+'\n')


def protect_feature_inputs(module):
    # Upstream PromptEncoder mutates source point features in place. Reused views from
    # the no_grad refinement can otherwise fail autograd or accumulate positional embeddings.
    def clone_input(_module, args, kwargs):
        kwargs = dict(kwargs)
        kwargs['src_points_feat_b'] = kwargs['src_points_feat_b'].clone()
        return args, kwargs
    return module.register_forward_pre_hook(clone_input, with_kwargs=True)


class OfficialEvaluator:
    timing_description = ('on-demand point cache: misses include first-stage generation and cache IO; '
                          'hits reuse first-stage points; mixed cache timing is not uncached end-to-end; '
                          'excludes image disk IO, initialization, GT scoring and output IO')

    def __init__(self, config, preprocessing, vggt, predictor):
        self.config = config; self.preprocessing = preprocessing
        self.vggt = vggt; self.predictor = predictor

    @torch.no_grad()
    def predict(self, reference, reference_mask, target):
        from main_exp.hybrid import vggt_feat
        from utils import sample_feat_at_points, recover_square_crop_tensor
        self.predictor.eval()
        p = self.preprocessing.prepare((reference, reference_mask, target), vggt=self.vggt)
        size = self.config['image_size']
        with torch.autocast('cuda', dtype=torch.bfloat16):
            src_points, features = vggt_feat(self.vggt, p['source'][None], p['source_mask'][None],
                                            p['target'][None], self.config['prompt_points'], torch.device('cuda'))
            src_points = src_points.float(); features = features.float()
            src_features = sample_feat_at_points(features[:, 0], src_points, size, size).float()
            target_points = torch.as_tensor(p['target_points'][None], device='cuda')
            refinement = None
            for _ in range(self.config['refine_time']+1):
                logits, low = self.predictor(feat_map_b=features, src_points_b=src_points,
                    tar_points_b=target_points, src_mask_b=p['source_mask'][None],
                    src_points_feat_b=src_features, tar_mask_b=refinement, ret_logits=True)
                refinement = low.float()
        recovered, _ = recover_square_crop_tensor(logits[0], target_points[0], p['meta'])
        restored = torch.nn.functional.interpolate(recovered[None, None].float(),
            size=p['original_shape'], mode='bilinear', align_corners=False)[0, 0]
        return (restored > 0).cpu().numpy(), p['audit']
