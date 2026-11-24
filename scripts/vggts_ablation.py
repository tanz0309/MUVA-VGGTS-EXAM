"""Inference-only Phase A, sharing the actual official call's prepared inputs.

This module deliberately has no GT reader or access to manifest supervision.
The official call is recorded once; all four independent experimental forwards
reuse a protected snapshot of its features and points. Temporary hooks are
removed even on failure. The original feature-protection hook remains installed.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import write_json
from vggts_ablation_hooks import (array_digest, clone_inputs, input_fingerprints,
                                  make_corrupted_feedback)

GROUPS = {'A0': 0, 'A1': 1, 'A2': 1, 'A3': 1}


class PreparedRecorder:
    def __init__(self, preprocessing):
        self.preprocessing = preprocessing
        self.prepared = None

    def prepare(self, *args, **kwargs):
        if self.prepared is not None:
            raise RuntimeError('Expected exactly one preprocessing call per sample')
        self.prepared = self.preprocessing.prepare(*args, **kwargs)
        return self.prepared


@torch.no_grad()
def capture_official_inputs(evaluator, images):
    """Record the unchanged OfficialEvaluator, before its feature clone hook."""
    from vggts_official_data import OfficialEvaluator
    recorder = PreparedRecorder(evaluator.preprocessing)
    official = OfficialEvaluator(evaluator.config, recorder, evaluator.vggt, evaluator.predictor)
    base, signatures, feedback = {}, [], []

    def before(_module, args, kwargs):
        if args:
            raise RuntimeError('Official Predictor inputs must be keyword arguments')
        if not base:
            base.update(clone_inputs(kwargs))
        signatures.append(input_fingerprints(kwargs))
        mask = kwargs['tar_mask_b']
        feedback.append(None if mask is None else array_digest(mask))

    hook = evaluator.predictor.register_forward_pre_hook(before, with_kwargs=True, prepend=True)
    try:
        prediction, audit = official.predict(*images)
    finally:
        hook.remove()
    if len(signatures) != 2 or signatures[0] != signatures[1]:
        raise RuntimeError('Official inputs changed between the two native passes')
    if feedback[0] is not None or feedback[1] is None:
        raise RuntimeError('Unexpected official feedback sequence')
    if input_fingerprints(base) != signatures[0]:
        raise RuntimeError('Recorded input copy differs from the official inputs')
    return recorder.prepared, base, prediction, audit, feedback


def validate_ablation_inputs(base_inputs, expected):
    if input_fingerprints(base_inputs) != expected:
        raise RuntimeError('A model call mutated shared experimental inputs')


def build_ablation_case(name, base_inputs, feedback=None):
    if name not in GROUPS:
        raise ValueError(name)
    # Protect source-point features independently of the upstream protection hook.
    # The feature map itself is shared and fingerprinted before and after all calls.
    case = dict(base_inputs)
    case['src_points_feat_b'] = base_inputs['src_points_feat_b'].clone()
    case['tar_mask_b'] = feedback
    return case


@torch.no_grad()
def run_pass_sequence(evaluator, base_inputs, config):
    predictor = evaluator.predictor
    predictor.eval()
    expected = input_fingerprints(base_inputs)
    outputs, audits = {}, {}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits, low = predictor(**build_ablation_case('A0', base_inputs))
        outputs['A0'] = (logits.detach().clone(), low.detach().float().clone())
        corrupted, transform = make_corrupted_feedback(outputs['A0'][1], config['corrupted_feedback'])
        for group, feedback in [('A1', outputs['A0'][1]), ('A2', None), ('A3', corrupted)]:
            outputs[group] = tuple(x.detach().clone() for x in predictor(
                **build_ablation_case(group, base_inputs, feedback)))
            audits[group] = {'base_inputs_sha256': expected,
                              'feedback_sha256': None if feedback is None else array_digest(feedback)}
    validate_ablation_inputs(base_inputs, expected)
    difference = float((outputs['A0'][0].float()-outputs['A2'][0].float()).abs().max())
    low_difference = float((outputs['A0'][1].float()-outputs['A2'][1].float()).abs().max())
    if max(difference, low_difference) > config['repeat_max_abs_tolerance']:
        raise RuntimeError(f'Repeat-none control failed: high={difference}, low={low_difference}')
    audits['A0'] = {'base_inputs_sha256': expected, 'feedback_sha256': None}
    return outputs, {'inputs_per_pass': audits, 'corrupted_feedback': transform,
                     'repeat_none_max_abs_logit_difference': difference,
                     'repeat_none_max_abs_low_difference': low_difference}


@torch.no_grad()
def restore_prediction(logits, target_points, prepared):
    from utils import recover_square_crop_tensor
    recovered, _ = recover_square_crop_tensor(logits[0], target_points[0], prepared['meta'])
    restored = torch.nn.functional.interpolate(recovered[None, None].float(),
        size=prepared['original_shape'], mode='bilinear', align_corners=False)[0, 0]
    return (restored > 0).cpu().numpy()


def save_prediction_artifacts(directory, outputs, predictions, official_prediction, audit):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for group, (logits, low) in outputs.items():
        # Float32 preserves the exact values returned under BF16 autocast.
        np.save(directory/f'{group}_logits.npy', logits.float().cpu().numpy(), allow_pickle=False)
        np.save(directory/f'{group}_low_logits.npy', low.float().cpu().numpy(), allow_pickle=False)
        Image.fromarray((logits[0, 0] > 0).cpu().numpy().astype(np.uint8)*255).save(directory/f'{group}_crop.png')
        Image.fromarray(predictions[group].astype(np.uint8)*255).save(directory/f'{group}_original.png')
    Image.fromarray(official_prediction.astype(np.uint8)*255).save(directory/'official_original.png')
    write_json(directory/'audit.json', audit)


@torch.no_grad()
def run_ablation_case(evaluator, images, config, directory):
    started = time.perf_counter()
    prepared, base, official_prediction, audit, official_feedback = capture_official_inputs(evaluator, images)
    outputs, checks = run_pass_sequence(evaluator, base, config)
    predictions = {group: restore_prediction(values[0], base['tar_points_b'], prepared)
                   for group, values in outputs.items()}
    different_pixels = int(np.count_nonzero(predictions['A1'] != official_prediction))
    if different_pixels:
        raise RuntimeError(f'Independent A1 differs from OfficialEvaluator: {different_pixels} pixels')
    if official_feedback[1] != checks['inputs_per_pass']['A1']['feedback_sha256']:
        raise RuntimeError('Independent Pass-0 low logits differ from native feedback')
    checks.update({'official_final_different_pixels': different_pixels,
                   'official_feedback_sha256': official_feedback,
                   'candidate_points_crop': audit['point_selector']['candidate_points_crop'],
                   'selected_points_crop': prepared['target_points'].tolist(),
                   'source_points_crop': base['src_points_b'].float().cpu().tolist(),
                   'target_crop_960x540': prepared['meta'],
                   'original_shape': list(prepared['original_shape']),
                   'point_audit': audit,
                   'inference_seconds': time.perf_counter()-started,
                   'target_gt_used': False})
    save_prediction_artifacts(directory, outputs, predictions, official_prediction, checks)
    return checks
