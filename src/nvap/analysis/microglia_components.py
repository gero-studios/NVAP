from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed as _watershed


_CC_STRUCTURE = ndi.generate_binary_structure(3, 2).astype(np.uint8, copy=False)


def _empty_labels(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(shape, dtype=np.int32),
        np.empty((0,), dtype=np.int32),
        np.zeros((1,), dtype=np.int64),
    )


def _mask_bounds(mask: np.ndarray) -> tuple[slice, slice, slice] | None:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 3 or arr.size == 0:
        return None

    z_any = np.any(arr, axis=(1, 2))
    if not np.any(z_any):
        return None
    y_any = np.any(arr, axis=(0, 2))
    x_any = np.any(arr, axis=(0, 1))

    z0 = int(np.argmax(z_any))
    z1 = int(z_any.size - np.argmax(z_any[::-1]))
    y0 = int(np.argmax(y_any))
    y1 = int(y_any.size - np.argmax(y_any[::-1]))
    x0 = int(np.argmax(x_any))
    x1 = int(x_any.size - np.argmax(x_any[::-1]))
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def _select_seed_centers(
    arr: np.ndarray,
    centers: np.ndarray,
    *,
    min_sep: int,
    max_centers: int,
) -> np.ndarray:
    pts = np.asarray(centers, dtype=np.int32)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    pts = pts.reshape(-1, 3)
    pts = np.unique(pts, axis=0)
    values = arr[pts[:, 0], pts[:, 1], pts[:, 2]]
    order = np.argsort(values)[::-1]

    selected: list[tuple[int, int, int]] = []
    min_sep2 = int(max(0, min_sep) ** 2)
    for idx in order:
        z, y, x = (int(v) for v in pts[int(idx)])
        keep = True
        if min_sep2 > 0:
            for sz, sy, sx in selected:
                dz = sz - z
                dy = sy - y
                dx = sx - x
                if (dz * dz + dy * dy + dx * dx) < min_sep2:
                    keep = False
                    break
        if not keep:
            continue
        selected.append((z, y, x))
        if len(selected) >= int(max(1, max_centers)):
            break

    if not selected:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(selected, dtype=np.int32)


def _detect_soma_blobs(
    volume: np.ndarray,
    *,
    threshold: float,
    branch_sensitivity: float,
    spacing: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Detect soma-like seed centers for watershed-based component splitting."""
    arr = np.asarray(volume, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=bool)
    finite = np.isfinite(arr)
    positive = arr[finite & (arr > 0.0)]
    if positive.size <= 0:
        return np.zeros(arr.shape, dtype=bool)

    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
    spacing_zyx = np.maximum(np.asarray(spacing, dtype=np.float32), 1.0e-6)

    sensitivity_scale = float(np.clip(branch_sensitivity, 0.4, 2.0))

    # Slight smoothing suppresses noisy pixel-level peaks while preserving somas.
    smooth_sigma = (
        float(np.clip(0.20 / float(spacing_zyx[0]), 0.0, 1.2)),
        float(np.clip(0.75 / float(spacing_zyx[1]), 0.0, 2.5)),
        float(np.clip(0.75 / float(spacing_zyx[2]), 0.0, 2.5)),
    )
    detect = ndi.gaussian_filter(arr, sigma=smooth_sigma, mode="nearest")
    detect_positive = detect[finite & (detect > 0.0)]
    if detect_positive.size <= 0:
        return np.zeros(arr.shape, dtype=bool)

    peak_quantile = float(np.clip(0.88 - (0.05 * (sensitivity_scale - 1.0)), 0.76, 0.92))
    peak_floor = float(max(threshold, np.quantile(detect_positive, peak_quantile)))
    max_size = (
        3,
        int(max(3, round(7.0 / sensitivity_scale))),
        int(max(3, round(7.0 / sensitivity_scale))),
    )
    peak_mask = finite & (detect >= peak_floor)
    if np.any(peak_mask):
        peak_max = ndi.maximum_filter(detect, size=max_size, mode="nearest")
        peak_mask &= detect >= (peak_max - 1.0e-6)

    peak_labels, n_peaks = ndi.label(peak_mask, structure=_CC_STRUCTURE)
    # Vectorised: find one max-intensity centre per peak region — O(N) not O(N×K).
    if n_peaks > 0:
        positions = ndi.maximum_position(detect, peak_labels, list(range(1, n_peaks + 1)))
        if not isinstance(positions, list):
            positions = [positions]
        local_centers = np.array(positions, dtype=np.int32)
    else:
        local_centers = np.empty((0, 3), dtype=np.int32)

    max_centers = int(np.clip((arr.shape[1] * arr.shape[2]) // 36, 32, 1024))
    min_sep = int(max(2, round(3.0 / sensitivity_scale)))
    selected = _select_seed_centers(
        detect,
        local_centers,
        min_sep=min_sep,
        max_centers=max_centers,
    )

    seed_mask = np.zeros(arr.shape, dtype=bool)
    if selected.size > 0:
        support_floor = float(max(threshold, peak_floor * 0.85))
        for zc, yc, xc in selected:
            z0 = max(0, int(zc) - 1)
            z1 = min(arr.shape[0], int(zc) + 2)
            y0 = max(0, int(yc) - 1)
            y1 = min(arr.shape[1], int(yc) + 2)
            x0 = max(0, int(xc) - 1)
            x1 = min(arr.shape[2], int(xc) + 2)
            local = finite[z0:z1, y0:y1, x0:x1] & (detect[z0:z1, y0:y1, x0:x1] >= support_floor)
            if np.any(local):
                seed_mask[z0:z1, y0:y1, x0:x1] |= local
            else:
                seed_mask[int(zc), int(yc), int(xc)] = True
        return seed_mask

    # Final fallback for very weak or flat volumes.
    fallback_floor = float(max(threshold, np.quantile(detect_positive, 0.84)))
    fallback_mask = finite & (detect >= fallback_floor)
    if np.any(fallback_mask):
        fallback_max = ndi.maximum_filter(detect, size=(3, 5, 5), mode="nearest")
        fallback_mask &= detect >= (fallback_max - 1.0e-6)
    if np.any(fallback_mask):
        fallback_labels, n_fallback = ndi.label(fallback_mask, structure=_CC_STRUCTURE)
        # Vectorised: one max-intensity centre per fallback region.
        if n_fallback > 0:
            positions = ndi.maximum_position(detect, fallback_labels, list(range(1, n_fallback + 1)))
            if not isinstance(positions, list):
                positions = [positions]
            for pos in positions:
                seed_mask[int(pos[0]), int(pos[1]), int(pos[2])] = True
        if np.any(seed_mask):
            return seed_mask

    # Last-resort single seed at global maximum.
    max_idx = np.unravel_index(int(np.argmax(detect)), detect.shape)
    seed_mask[int(max_idx[0]), int(max_idx[1]), int(max_idx[2])] = True

    return seed_mask


def _seed_labels_from_soma_candidates(
    seed: np.ndarray,
    working: np.ndarray,
    finite: np.ndarray,
    *,
    low_floor: float,
    high_t: float,
    branch_sensitivity: float,
    min_keep: int,
) -> np.ndarray:
    merge_scale = float(np.clip(0.72 - (0.08 * (branch_sensitivity - 1.0)), 0.56, 0.80))
    merge_t = float(np.clip(max(low_floor, high_t * merge_scale), 0.0, max(high_t, low_floor)))
    soma_mask = finite & (working >= merge_t)
    if np.any(soma_mask):
        merge_mask = soma_mask
        if int(np.count_nonzero(soma_mask)) >= 64:
            merge_mask = ndi.binary_closing(soma_mask, structure=_CC_STRUCTURE, iterations=1)
        merged_seed = ndi.binary_propagation(seed, structure=_CC_STRUCTURE, mask=merge_mask)
        if np.any(merged_seed):
            seed = np.asarray(merged_seed, dtype=bool)

        # Guarantee one seed in each disconnected high-confidence island.
        soma_labels, n_soma = ndi.label(soma_mask, structure=_CC_STRUCTURE)
        if n_soma > 0:
            min_island_voxels = max(2, int(min_keep // 2))
            island_sizes = np.bincount(soma_labels.ravel(), minlength=n_soma + 1)
            # Count seed voxels per island in one vectorised pass — O(N) not O(N×K).
            seed_per_island = np.bincount(
                soma_labels[seed].ravel(), minlength=n_soma + 1
            )
            need_ids = [
                i for i in range(1, n_soma + 1)
                if island_sizes[i] >= min_island_voxels and seed_per_island[i] == 0
            ]
            if need_ids:
                positions = ndi.maximum_position(working, soma_labels, need_ids)
                if not isinstance(positions, list):
                    positions = [positions]
                for pos in positions:
                    seed[int(pos[0]), int(pos[1]), int(pos[2])] = True

    seed_labels, _ = ndi.label(seed, structure=_CC_STRUCTURE)
    return np.asarray(seed_labels, dtype=np.int32)


def compute_component_labels(
    volume: np.ndarray,
    threshold: float,
    min_voxels: int = 2,
    max_components: int = 512,
    smooth_sigma: tuple[float, float, float] = (0.0, 0.0, 0.0),
    branch_sensitivity: float = 1.0,
    spacing: tuple[float, float, float] | None = None,
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

    # Detect soma seed centres — one compact region per microglia.
    seed = _detect_soma_blobs(
        working,
        threshold=t,
        branch_sensitivity=branch_sense,
        spacing=spacing,
    )
    if not np.any(seed):
        seed = finite & (working >= high_t)
        if not np.any(seed):
            seed = finite & (working >= max(t, high_t * 0.92))
        if not np.any(seed):
            return _empty_labels(arr.shape)

    # Single low threshold for the watershed growth mask.  The watershed
    # itself handles where adjacent regions should be split, so we no longer
    # need to iterate over multiple candidate thresholds.
    low_scale = float(np.clip(0.46 - (0.16 * (branch_sense - 1.0)), 0.22, 0.68))
    core_scale = float(np.clip(0.32 - (0.08 * (branch_sense - 1.0)), 0.16, 0.46))
    low_t = float(
        np.clip(
            max(0.01, min(t * low_scale, high_t * core_scale)),
            0.0,
            max(0.0, high_t - 1.0e-4),
        )
    )

    active_bounds = _mask_bounds(finite & (working >= low_t))
    if active_bounds is None:
        active_bounds = _mask_bounds(seed)
    if active_bounds is None:
        return _empty_labels(arr.shape)

    arr_roi = arr[active_bounds]
    working_roi = working[active_bounds]
    finite_roi = finite[active_bounds]
    seed_roi = seed[active_bounds]

    seed_labels = _seed_labels_from_soma_candidates(
        seed_roi,
        working_roi,
        finite_roi,
        low_floor=low_t,
        high_t=high_t,
        branch_sensitivity=branch_sense,
        min_keep=min_keep,
    )
    if not np.any(seed_labels):
        return _empty_labels(arr.shape)

    # Build the growth mask: above low threshold with sufficient local support.
    support_min = 1 if min_keep <= 8 else (2 if branch_sense >= 1.30 else 1)
    candidate = finite_roi & (working_roi >= low_t)
    if np.any(candidate):
        support = ndi.convolve(
            candidate.astype(np.uint8), _CC_STRUCTURE, mode="constant", cval=0
        )
        # Seeds are always included regardless of neighbour support.
        candidate = (candidate & (support >= support_min)) | (seed_labels > 0)
    else:
        candidate = seed_labels > 0

    # Watershed: marker seeds grow through the candidate mask.
    # Inverting working_roi makes bright somas low-elevation (fill first), so
    # cells are naturally split at intensity saddle-points — no multi-pass needed.
    raw_labels = np.asarray(
        _watershed(-working_roi, markers=seed_labels, mask=candidate, connectivity=2),
        dtype=np.int32,
    )

    # --- Quality filtering ---
    raw_sizes = np.bincount(raw_labels.ravel()).astype(np.int64)
    if raw_sizes.size <= 1:
        return _empty_labels(arr.shape)
    n_labels = int(raw_sizes.size)

    core_counts = np.bincount(
        raw_labels[seed_labels > 0].ravel(), minlength=n_labels
    ).astype(np.int64)
    visible_counts = np.bincount(
        raw_labels[arr_roi >= t].ravel(), minlength=n_labels
    ).astype(np.int64)

    component_ids = np.arange(1, n_labels, dtype=np.int32)
    min_core_voxels = max(1, int(min_keep // 16))
    min_visible_voxels = max(2, min_core_voxels)
    min_visible_ratio = float(np.clip(0.06 - (0.02 * (branch_sense - 1.0)), 0.03, 0.08))
    max_component_ratio = float(np.clip(0.08 + (0.02 * (branch_sense - 1.0)), 0.06, 0.14))
    max_component_voxels = max(int(min_keep) * 512, int(arr.size * max_component_ratio))

    visible_ratio = visible_counts[1:] / np.maximum(1, raw_sizes[1:])
    keep = component_ids[
        (raw_sizes[1:] >= min_keep)
        & (raw_sizes[1:] <= max_component_voxels)
        & (core_counts[1:] >= min_core_voxels)
        & (visible_counts[1:] >= min_visible_voxels)
        & (visible_ratio >= min_visible_ratio)
    ]

    if keep.size == 0:
        # Relaxed fallback for sparse / very dim data.
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

    # Rank and select top components.
    score_scale = int(raw_sizes.max()) + 1
    ranking = (
        visible_counts[keep].astype(np.int64) * score_scale
        + raw_sizes[keep].astype(np.int64)
    )
    keep_sorted = keep[np.argsort(ranking)[::-1]][:max_keep]

    # Vectorised label remapping via LUT — O(N) instead of O(N × K).
    lut = np.zeros(n_labels, dtype=np.int32)
    for new_id, old_id in enumerate(keep_sorted, start=1):
        lut[int(old_id)] = int(new_id)

    labels = np.zeros(arr.shape, dtype=np.int32)
    labels[active_bounds] = lut[raw_labels]

    sizes = np.zeros(int(keep_sorted.size) + 1, dtype=np.int64)
    for new_id, old_id in enumerate(keep_sorted, start=1):
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
