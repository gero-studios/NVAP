from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt


_CC_STRUCTURE = ndi.generate_binary_structure(3, 2).astype(np.uint8, copy=False)


def _empty_labels(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(shape, dtype=np.int32),
        np.empty((0,), dtype=np.int32),
        np.zeros((1,), dtype=np.int64),
    )


def compute_component_labels(
    volume: np.ndarray,
    threshold: float,
    min_voxels: int = 2,
    max_components: int = 512,
    smooth_sigma: tuple[float, float, float] = (0.0, 0.0, 0.0),
    branch_sensitivity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute connected microglia-like components for isolate-view rendering.

    Returns (labels, component_ids_sorted_by_size_desc, bincount_sizes).
    """
    arr = np.asarray(volume, dtype=np.float32)
    if arr.size == 0:
        return _empty_labels(arr.shape)
    t = float(np.clip(threshold, 0.0, 1.0))
    branch_sense = float(np.clip(branch_sensitivity, 0.4, 2.0))
    min_keep = max(2, int(min_voxels))
    max_keep = max(1, int(max_components))

    working = arr
    if any(float(s) > 0.0 for s in smooth_sigma):
        working = ndi.gaussian_filter(arr, sigma=smooth_sigma, mode="nearest")
    finite = np.isfinite(working)
    if not np.any(finite):
        return _empty_labels(arr.shape)

    positive = working[finite & (working > 0.0)]
    if positive.size <= 0:
        return _empty_labels(arr.shape)
    high_quantile = float(np.clip(0.84 - (0.06 * (branch_sense - 1.0)), 0.70, 0.92))
    high_q = float(np.quantile(positive, high_quantile))
    high_t = float(np.clip(max(t, high_q), 0.0, 0.999))
    seed = finite & (working >= high_t)
    if not np.any(seed):
        seed = finite & (working >= max(t, high_t * 0.92))
    if not np.any(seed):
        return _empty_labels(arr.shape)

    # Start branch-friendly and tighten only if components become unrealistically merged.
    low_scale = float(np.clip(0.58 - (0.16 * (branch_sense - 1.0)), 0.30, 0.80))
    core_scale = float(np.clip(0.40 - (0.08 * (branch_sense - 1.0)), 0.20, 0.55))
    low_start = float(
        np.clip(
            max(0.01, min(t * low_scale, high_t * core_scale)),
            0.0,
            max(0.0, high_t - 1.0e-4),
        )
    )
    low_candidates: list[float] = []
    for val in (
        low_start,
        max(low_start, max(t * 0.80, high_t * 0.50)),
        max(low_start, max(t, high_t * 0.65)),
        high_t,
    ):
        v = float(np.clip(val, 0.0, max(0.0, high_t)))
        if not low_candidates or abs(v - low_candidates[-1]) > 1.0e-6:
            low_candidates.append(v)

    raw_labels = None
    raw_sizes = None
    core_counts = None
    visible_counts = None
    keep = np.empty((0,), dtype=np.int32)
    component_ids = np.empty((0,), dtype=np.int32)
    min_core_voxels = max(1, int(min_keep // 8))
    min_visible_voxels = max(2, min_core_voxels)
    min_visible_ratio = float(np.clip(0.10 - (0.03 * (branch_sense - 1.0)), 0.05, 0.12))
    max_component_ratio = float(np.clip(0.06 + (0.02 * (branch_sense - 1.0)), 0.04, 0.10))
    max_component_voxels = max(int(min_keep) * 512, int(arr.size * max_component_ratio))

    for low_t in low_candidates:
        candidate = finite & (working >= low_t)
        if not np.any(candidate):
            continue

        # Drop isolated low-threshold noise while preserving high-confidence core voxels.
        support = ndi.convolve(candidate.astype(np.uint8), _CC_STRUCTURE, mode="constant", cval=0)
        support_min = 2 if branch_sense >= 1.15 else 3
        candidate = (candidate & (support >= support_min)) | seed
        grown = ndi.binary_propagation(seed, structure=_CC_STRUCTURE, mask=candidate)
        if not np.any(grown):
            continue

        # Label seeds first so nearby cells with overlapping low-threshold
        # regions stay separate.  Then assign every grown voxel to the
        # nearest seed label (Voronoi-style) via distance_transform_edt.
        seed_labels, _n_seeds = ndi.label(seed, structure=_CC_STRUCTURE)
        if _n_seeds <= 0:
            continue
        not_seed = (seed_labels == 0).astype(np.uint8)
        _dist, nearest = distance_transform_edt(not_seed, return_distances=True, return_indices=True)
        trial_labels = np.zeros(seed_labels.shape, dtype=np.int32)
        trial_labels[grown] = seed_labels[
            nearest[0][grown], nearest[1][grown], nearest[2][grown]
        ]
        trial_labels = np.asarray(trial_labels, dtype=np.int32)
        trial_sizes = np.bincount(trial_labels.ravel()).astype(np.int64, copy=False)
        if trial_sizes.size <= 1:
            continue
        trial_core_counts = np.bincount(
            trial_labels[seed],
            minlength=trial_sizes.size,
        ).astype(np.int64, copy=False)
        trial_visible_counts = np.bincount(
            trial_labels[arr >= t],
            minlength=trial_sizes.size,
        ).astype(np.int64, copy=False)

        trial_component_ids = np.arange(1, trial_sizes.size, dtype=np.int32)
        trial_visible_ratio = trial_visible_counts[1:] / np.maximum(1, trial_sizes[1:])
        keep_mask = (
            (trial_sizes[1:] >= min_keep)
            & (trial_sizes[1:] <= max_component_voxels)
            & (trial_core_counts[1:] >= min_core_voxels)
            & (trial_visible_counts[1:] >= min_visible_voxels)
            & (trial_visible_ratio >= min_visible_ratio)
        )
        trial_keep = trial_component_ids[keep_mask]

        raw_labels = trial_labels
        raw_sizes = trial_sizes
        core_counts = trial_core_counts
        visible_counts = trial_visible_counts
        component_ids = trial_component_ids
        keep = trial_keep
        if keep.size > 0:
            break

    if (
        raw_labels is None
        or raw_sizes is None
        or core_counts is None
        or visible_counts is None
        or component_ids.size == 0
    ):
        return _empty_labels(arr.shape)
    if keep.size == 0:
        # Fallback: relax core/max-size constraints if data is very sparse.
        visible_ratio = visible_counts[1:] / np.maximum(1, raw_sizes[1:])
        keep = component_ids[
            (raw_sizes[1:] >= min_keep)
            & (raw_sizes[1:] <= max(int(arr.size * 0.18), int(min_keep) * 1024))
            & (core_counts[1:] > 0)
            & (visible_counts[1:] >= min_visible_voxels)
            & (visible_ratio >= (min_visible_ratio * 0.5))
        ]
    if keep.size == 0:
        return _empty_labels(arr.shape)

    score_scale = int(raw_sizes.max()) + 1
    ranking = (visible_counts[keep].astype(np.int64) * score_scale) + raw_sizes[keep].astype(np.int64)
    keep_sorted = keep[np.argsort(ranking)[::-1]][:max_keep]
    labels = np.zeros(arr.shape, dtype=np.int32)
    sizes = np.zeros((int(keep_sorted.size) + 1,), dtype=np.int64)
    for new_id, old_id in enumerate(keep_sorted, start=1):
        labels[raw_labels == int(old_id)] = int(new_id)
        sizes[new_id] = int(raw_sizes[int(old_id)])

    order = np.arange(1, int(keep_sorted.size) + 1, dtype=np.int32)
    return labels, order, sizes


def isolate_component(
    volume: np.ndarray,
    labels: np.ndarray,
    component_id: int,
) -> np.ndarray:
    """Keep only a single component in the returned volume."""
    arr = np.asarray(volume, dtype=np.float32)
    lbl = np.asarray(labels, dtype=np.int32)
    if arr.shape != lbl.shape:
        raise ValueError("volume and labels must have the same shape.")
    component = int(component_id)
    if component <= 0:
        return arr.copy()
    out = np.zeros_like(arr, dtype=np.float32)
    mask = lbl == component
    out[mask] = arr[mask]
    return out
