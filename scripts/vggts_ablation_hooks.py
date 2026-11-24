"""Phase A feedback control. No vendor patches or persistent model hooks."""
import hashlib

import numpy as np
import torch


def array_digest(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.shape, str(value.dtype))).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def input_fingerprints(inputs):
    return {key: array_digest(value) for key, value in inputs.items()
            if isinstance(value, (torch.Tensor, np.ndarray)) and key != 'tar_mask_b'}


def clone_inputs(inputs):
    return {key: value.detach().clone() if isinstance(value, torch.Tensor)
            else value.copy() if isinstance(value, np.ndarray) else value
            for key, value in inputs.items()}


def make_corrupted_feedback(low_logits, config):
    """Area- and histogram-preserving circular shift, independent of GT."""
    if config['method'] != 'circular_shift':
        raise ValueError('Phase A supports only a prespecified circular shift')
    if low_logits.ndim != 4 or min(low_logits.shape[-2:]) < 2:
        raise ValueError('Expected BCHW low logits with spatial dimensions >= 2')
    shifts = []
    for size, key in zip(low_logits.shape[-2:], ['height_fraction', 'width_fraction']):
        fraction = config[key]
        if not 0 < fraction < 1:
            raise ValueError('Shift fraction must be strictly between zero and one')
        shifts.append(max(1, min(size-1, int(size*fraction))))
    result = torch.roll(low_logits, shifts=tuple(shifts), dims=(-2, -1)).clone()
    return result, {'method': 'circular_shift', 'shifts_yx': shifts,
                    'input_sha256': array_digest(low_logits),
                    'output_sha256': array_digest(result),
                    'preserves': ['shape', 'dtype', 'value_multiset', 'positive_area'],
                    'gt_used': False}
