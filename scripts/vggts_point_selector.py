"""Block-First target point selection for VGGT-S phase 1.

Pure NumPy/scikit-learn. This module must not import VGGT, CUDA modules,
dataset readers or GT helpers so the algorithm stays unit-testable.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sklearn.cluster import DBSCAN

SELECTOR_SCHEMA_VERSION = 1
BLOCK_FIRST_VERSION = 'block_first_v1'
NATIVE_KMEANS_VERSION = 'native_kmeans'
PHASE1_BUDGET = 5
_ALLOWED_FIELDS = {'schema_version', 'method', 'budget', 'eps', 'min_samples'}
_METHODS = {'kmeans', 'block_first'}
_DEFAULTS = {'schema_version': SELECTOR_SCHEMA_VERSION, 'method': 'kmeans',
             'budget': PHASE1_BUDGET, 'eps': 0.05, 'min_samples': 2}


@dataclass(frozen=True)
class SelectorConfig:
    schema_version: int = SELECTOR_SCHEMA_VERSION
    method: str = 'kmeans'
    budget: int = PHASE1_BUDGET
    eps: float = 0.05
    min_samples: int = 2


@dataclass
class PointSelectionResult:
    prompts_xy: np.ndarray
    selected_indices: np.ndarray | None
    audit: dict


@dataclass(eq=False)
class _Cluster:
    cluster_id: int
    member_unique: np.ndarray
    member_candidates: np.ndarray
    raw_support: int
    unique_size: int
    min_candidate_index: int
    retained: bool = False
    quota: int = 0
    medoid_unique: int | None = None


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_selector_config(value, *, prompt_points):
    """Return a validated SelectorConfig; reject unknown fields and bad values."""
    if isinstance(value, SelectorConfig):
        data = asdict(value)
    elif isinstance(value, dict):
        data = dict(value)
    else:
        raise TypeError('selector config must be a mapping or SelectorConfig')
    unknown = set(data) - _ALLOWED_FIELDS
    if unknown:
        raise ValueError('Unknown selector config fields: ' + ', '.join(sorted(unknown)))
    merged = dict(_DEFAULTS)
    merged.update(data)
    if not _is_int(merged['schema_version']) or merged['schema_version'] != SELECTOR_SCHEMA_VERSION:
        raise ValueError('selector schema_version must be 1')
    if merged['method'] not in _METHODS:
        raise ValueError("selector method must be 'kmeans' or 'block_first'")
    if not _is_int(merged['budget']) or merged['budget'] < 1:
        raise ValueError('selector budget must be a positive integer')
    if not _is_int(prompt_points) or merged['budget'] != prompt_points:
        raise ValueError('selector budget must equal the base config prompt_points')
    if prompt_points != PHASE1_BUDGET:
        raise ValueError('Phase 1 fixes the prompt budget at 5')
    eps = merged['eps']
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise ValueError('selector eps must be a finite positive number')
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('selector eps must be a finite positive number')
    merged['eps'] = eps
    if not _is_int(merged['min_samples']) or merged['min_samples'] < 2:
        raise ValueError('selector min_samples must be an integer >= 2')
    return SelectorConfig(**merged)


def load_selector_config(path, *, prompt_points):
    """None keeps the native KMeans baseline; a path loads and validates JSON."""
    if path is None:
        return validate_selector_config({}, prompt_points=prompt_points)
    value = json.loads(Path(path).read_text())
    return validate_selector_config(value, prompt_points=prompt_points)


def validate_candidate_points(points_xy, image_size, budget):
    """Validate the cropped candidate array without modifying or reordering it."""
    points = np.asarray(points_xy)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f'candidate points must have shape [N,2], got {points.shape}')
    if points.shape[0] < budget:
        raise ValueError(f'need at least {budget} candidate points, got {points.shape[0]}')
    if not np.issubdtype(points.dtype, np.number):
        raise ValueError('candidate points must be numeric')
    if not np.isfinite(points).all():
        raise ValueError('candidate points contain NaN/Inf')
    if (points < 0).any() or (points > image_size - 1).any():
        raise ValueError(f'candidate points fall outside [0,{image_size - 1}]')


def stable_unique_points(points_xy):
    """Exact float32 dedupe preserving first occurrence.

    Returns (unique_points, unique_to_candidate, candidate_to_unique).
    """
    points = np.ascontiguousarray(points_xy, dtype=np.float32)
    seen = {}
    unique = []
    unique_to_candidate = []
    candidate_to_unique = np.empty(len(points), dtype=np.int64)
    for index, (x, y) in enumerate(points):
        key = (float(x), float(y))
        slot = seen.get(key)
        if slot is None:
            slot = len(unique)
            seen[key] = slot
            unique.append((x, y))
            unique_to_candidate.append(index)
        candidate_to_unique[index] = slot
    return (np.asarray(unique, dtype=np.float32),
            np.asarray(unique_to_candidate, dtype=np.int64),
            candidate_to_unique)


def normalize_points(points_xy, image_size):
    """Distance coordinates: divide by image_size (not size-1, no offsets)."""
    if not np.isfinite(image_size) or image_size <= 0:
        raise ValueError('image_size must be positive')
    return np.asarray(points_xy, dtype=np.float64) / float(image_size)


def cluster_spatial_points(points_norm, config, unique_to_candidate):
    """DBSCAN on unique normalized points; renumber non-noise clusters by first index."""
    labels_raw = DBSCAN(eps=config.eps, min_samples=config.min_samples,
                        metric='euclidean', algorithm='brute', n_jobs=1).fit_predict(points_norm)
    noise_unique = np.flatnonzero(labels_raw == -1).astype(np.int64)
    clusters = []
    for label in np.unique(labels_raw):
        if label == -1:
            continue
        members = np.flatnonzero(labels_raw == label).astype(np.int64)
        members = members[np.argsort(unique_to_candidate[members], kind='stable')]
        candidates = unique_to_candidate[members]
        clusters.append(_Cluster(cluster_id=-1, member_unique=members,
                                 member_candidates=candidates, raw_support=len(candidates),
                                 unique_size=len(members),
                                 min_candidate_index=int(candidates.min())))
    clusters.sort(key=lambda cluster: cluster.min_candidate_index)
    labels = np.full(len(labels_raw), -1, dtype=np.int64)
    for cluster_id, cluster in enumerate(clusters):
        cluster.cluster_id = cluster_id
        labels[cluster.member_unique] = cluster_id
    return labels, clusters, noise_unique


def choose_clusters(clusters, budget):
    """Keep at most `budget` clusters: unique support desc, then first candidate index."""
    ordered = sorted(clusters, key=lambda cluster: (-cluster.unique_size,
                                                    cluster.min_candidate_index))
    retained_ids = {cluster.cluster_id for cluster in ordered[:budget]}
    for cluster in clusters:
        cluster.retained = cluster.cluster_id in retained_ids
    retained = [cluster for cluster in clusters if cluster.retained]
    discarded = [cluster for cluster in clusters if not cluster.retained]
    return retained, discarded


def medoid_index(member_indices, points_norm, unique_to_candidate):
    """Return the unique-point index minimizing summed Euclidean distance."""
    members = np.asarray(member_indices, dtype=np.int64)
    if len(members) == 0:
        raise ValueError('cannot choose a medoid from an empty cluster')
    pts = points_norm[members]
    distances = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2).sum(axis=1)
    candidates = np.asarray(unique_to_candidate, dtype=np.int64)[members]
    order = np.lexsort((candidates, distances))
    return int(members[order[0]])


def allocate_budget(cluster_sizes: Sequence[int], budget: int, cluster_ids: Sequence[int]):
    """Minimum of one per cluster, then capacity-constrained largest remainder."""
    if len(cluster_sizes) != len(cluster_ids) or not cluster_sizes:
        raise ValueError('cluster_sizes and cluster_ids must be non-empty and equal length')
    if budget < len(cluster_sizes):
        raise ValueError('budget cannot be smaller than the retained cluster count')
    sizes = {int(cid): int(size) for cid, size in zip(cluster_ids, cluster_sizes)}
    if any(size < 1 for size in sizes.values()):
        raise ValueError('every retained cluster must have positive capacity')
    if sum(sizes.values()) < budget:
        raise ValueError('retained clusters cannot hold the full budget')
    quota = {cid: 1 for cid in sizes}
    remaining = budget - len(sizes)
    guard = 0
    while remaining > 0:
        guard += 1
        if guard > 100000:
            raise RuntimeError('quota allocation stalled')
        available = [cid for cid in sizes if sizes[cid] - quota[cid] > 0]
        if not available:
            raise ValueError('retained clusters cannot hold the full budget')
        total_weight = sum(sizes[cid] for cid in available)
        theoretical = {cid: remaining * sizes[cid] / total_weight for cid in available}
        added = 0
        for cid in available:
            capacity = sizes[cid] - quota[cid]
            extra = min(capacity, int(np.floor(theoretical[cid])))
            if extra:
                quota[cid] += extra
                added += extra
        remaining -= added
        if remaining == 0:
            break
        # One extra point per still-available cluster this round.
        order = []
        for cid in available:
            if sizes[cid] - quota[cid] <= 0:
                continue
            fraction = theoretical[cid] - np.floor(theoretical[cid])
            order.append((fraction, sizes[cid], cid))
        order.sort(key=lambda item: (-item[0], -item[1], item[2]))
        progressed = False
        for fraction, _size, cid in order:
            if remaining == 0:
                break
            if sizes[cid] - quota[cid] > 0:
                quota[cid] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError('quota allocation stalled')
    assert sum(quota.values()) == budget
    for cid in sizes:
        assert 1 <= quota[cid] <= sizes[cid]
    return quota


def fps_indices(member_indices, seed_index, quota, points_norm, unique_to_candidate):
    """Cluster-local farthest point sampling starting from the medoid."""
    members = [int(index) for index in member_indices]
    if quota < 1:
        raise ValueError('fps quota must be positive')
    if seed_index not in members:
        raise ValueError('fps seed must belong to the cluster')
    candidates = np.asarray(unique_to_candidate, dtype=np.int64)
    selected = [int(seed_index)]
    while len(selected) < quota:
        remaining = np.asarray([index for index in members if index not in selected], dtype=np.int64)
        if len(remaining) == 0:
            raise ValueError('cluster capacity below quota')
        distances = np.linalg.norm(points_norm[remaining][:, None, :]
                                   - points_norm[selected][None, :, :], axis=2).min(axis=1)
        order = np.lexsort((candidates[remaining], -distances))
        selected.append(int(remaining[order[0]]))
    return selected


def select_native_kmeans(points_xy, budget, native_kmeans):
    """The only baseline/fallback path: unchanged upstream call on original points."""
    _, centers = native_kmeans(points_xy, K=budget)
    prompts = np.asarray(centers, dtype=np.float32)
    if prompts.shape != (budget, 2):
        raise ValueError(f'native KMeans returned {prompts.shape}, expected {(budget, 2)}')
    return prompts


def array_fingerprint(array):
    """Stable SHA-256 over shape, dtype and C-order bytes."""
    data = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps({'shape': list(data.shape), 'dtype': str(data.dtype)},
                             sort_keys=True).encode())
    digest.update(data.tobytes())
    return digest.hexdigest()


def _base_audit(config, points_xy, image_size):
    points = np.asarray(points_xy)
    return {
        'schema_version': config.schema_version,
        'selector_version': NATIVE_KMEANS_VERSION if config.method == 'kmeans' else BLOCK_FIRST_VERSION,
        'method_requested': config.method,
        'method_effective': 'kmeans',
        'budget': config.budget,
        'eps': None,
        'min_samples': None,
        'normalization': None,
        'image_size': image_size,
        'candidate_count': int(points.shape[0]),
        'candidate_points_crop': points.astype(np.float64).tolist(),
        'unique_candidate_count': None,
        'duplicate_count': None,
        'unique_to_candidate': None,
        'candidate_to_unique': None,
        'cluster_labels_by_candidate': None,
        'clusters': None,
        'noise_count_raw': None,
        'noise_count_unique': None,
        'discarded_cluster_ids': None,
        'selected_indices': None,
        'selected_points_crop': None,
        'fallback_reason': None,
        'selector_seconds': None,
    }


def _cluster_payload(cluster, unique_to_candidate):
    medoid_candidate = (None if cluster.medoid_unique is None
                        else int(unique_to_candidate[cluster.medoid_unique]))
    return {
        'cluster_id': int(cluster.cluster_id),
        'member_candidate_indices': [int(index) for index in cluster.member_candidates],
        'member_unique_indices': [int(index) for index in cluster.member_unique],
        'raw_support': int(cluster.raw_support),
        'unique_support': int(cluster.unique_size),
        'retained': bool(cluster.retained),
        'quota': int(cluster.quota),
        'medoid_unique_index': None if cluster.medoid_unique is None else int(cluster.medoid_unique),
        'medoid_candidate_index': medoid_candidate,
    }


def _native_result(points_xy, config, image_size, native_kmeans, audit, reason):
    prompts = select_native_kmeans(points_xy, config.budget, native_kmeans)
    if audit is None:
        audit = _base_audit(config, points_xy, image_size)
    audit['method_effective'] = 'kmeans'
    audit['fallback_reason'] = reason
    audit['selected_indices'] = None
    audit['selected_points_crop'] = prompts.astype(np.float64).tolist()
    return PointSelectionResult(prompts_xy=prompts, selected_indices=None, audit=audit)


def select_target_points(points_xy, *, image_size, config, native_kmeans):
    """Dispatch kmeans/block_first and return prompts plus a JSON-safe audit."""
    started = time.perf_counter()
    validate_candidate_points(points_xy, image_size, config.budget)
    original = np.asarray(points_xy)

    if config.method == 'kmeans':
        audit = _base_audit(config, original, image_size)
        result = _native_result(original, config, image_size, native_kmeans, audit, None)
        result.audit['selector_seconds'] = time.perf_counter() - started
        return result

    audit = _base_audit(config, original, image_size)
    audit['method_effective'] = 'block_first'
    audit['eps'] = float(config.eps)
    audit['min_samples'] = int(config.min_samples)
    audit['normalization'] = 'divide_by_image_size'

    unique, unique_to_candidate, candidate_to_unique = stable_unique_points(original)
    audit['unique_candidate_count'] = int(len(unique))
    audit['duplicate_count'] = int(len(original) - len(unique))
    audit['unique_to_candidate'] = [int(index) for index in unique_to_candidate]
    audit['candidate_to_unique'] = [int(index) for index in candidate_to_unique]
    if len(unique) < config.budget:
        result = _native_result(original, config, image_size, native_kmeans, audit,
                                'insufficient_unique_points')
        result.audit['selector_seconds'] = time.perf_counter() - started
        return result

    normalized = normalize_points(unique, image_size)
    labels, clusters, noise_unique = cluster_spatial_points(normalized, config,
                                                            unique_to_candidate)
    noise_set = set(int(index) for index in noise_unique)
    audit['noise_count_unique'] = len(noise_unique)
    audit['noise_count_raw'] = int(sum(1 for label in candidate_to_unique if int(label) in noise_set))
    audit['cluster_labels_by_candidate'] = [int(labels[slot]) for slot in candidate_to_unique]
    if not clusters:
        audit['clusters'] = []
        audit['discarded_cluster_ids'] = []
        result = _native_result(original, config, image_size, native_kmeans, audit, 'all_noise')
        result.audit['selector_seconds'] = time.perf_counter() - started
        return result

    retained, discarded = choose_clusters(clusters, config.budget)
    if sum(cluster.unique_size for cluster in retained) < config.budget:
        audit['clusters'] = [_cluster_payload(cluster, unique_to_candidate) for cluster in clusters]
        audit['discarded_cluster_ids'] = [int(cluster.cluster_id) for cluster in discarded]
        result = _native_result(original, config, image_size, native_kmeans, audit,
                                'insufficient_cluster_capacity')
        result.audit['selector_seconds'] = time.perf_counter() - started
        return result

    quotas = allocate_budget([cluster.unique_size for cluster in retained], config.budget,
                             [cluster.cluster_id for cluster in retained])
    selected_unique = []
    for cluster in sorted(retained, key=lambda item: item.cluster_id):
        cluster.quota = quotas[cluster.cluster_id]
        medoid = medoid_index(cluster.member_unique, normalized, unique_to_candidate)
        cluster.medoid_unique = medoid
        chosen = fps_indices(cluster.member_unique, medoid, cluster.quota, normalized,
                             unique_to_candidate)
        selected_unique.extend(chosen)
    selected = unique_to_candidate[np.asarray(selected_unique, dtype=np.int64)]
    prompts = original[selected].astype(np.float32)
    if prompts.shape != (config.budget, 2):
        raise RuntimeError(f'selected {prompts.shape[0]} prompts, expected {config.budget}')
    if len(stable_unique_points(prompts)[0]) != config.budget:
        raise RuntimeError('selected prompts are not unique')

    audit['clusters'] = [_cluster_payload(cluster, unique_to_candidate) for cluster in clusters]
    audit['discarded_cluster_ids'] = [int(cluster.cluster_id) for cluster in discarded]
    audit['selected_indices'] = [int(index) for index in selected]
    audit['selected_points_crop'] = prompts.astype(np.float64).tolist()
    audit['selector_seconds'] = time.perf_counter() - started
    return PointSelectionResult(prompts_xy=prompts,
                                selected_indices=np.asarray(selected, dtype=np.int64),
                                audit=audit)
