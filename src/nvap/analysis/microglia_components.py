from __future__ import annotations

import logging

import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed as _watershed


logger = logging.getLogger(__name__)
_CC_STRUCTURE = ndi.generate_binary_structure(3, 2).astype(np.uint8, copy=False)
_MAX_PEAK_CENTER_CANDIDATES = 16384
PREFERRED_VISIBLE_MICROGLIA_MIN_VOXELS = 15_000


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


def _peak_centers_from_mask(
    values: np.ndarray,
    peak_mask: np.ndarray,
    *,
    max_candidates: int = _MAX_PEAK_CENTER_CANDIDATES,
) -> np.ndarray:
    coords = np.argwhere(np.asarray(peak_mask, dtype=bool))
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.int32)

    candidate_count = int(coords.shape[0])
    if candidate_count > int(max_candidates):
        peak_values = np.asarray(values[tuple(coords.T)], dtype=np.float32)
        keep_count = int(max(1, max_candidates))
        keep_idx = np.argpartition(peak_values, -keep_count)[-keep_count:]
        coords = coords[keep_idx]
        logger.info(
            "Microglia seed detection: capped peak candidates %d -> %d.",
            candidate_count,
            int(coords.shape[0]),
        )
    return np.asarray(coords, dtype=np.int32)


def _max_positions_for_labels(
    values: np.ndarray,
    labels: np.ndarray,
    label_ids: list[int],
) -> list[tuple[int, int, int]]:
    if not label_ids:
        return []
    objects = ndi.find_objects(np.asarray(labels))
    positions: list[tuple[int, int, int]] = []
    for label_id in label_ids:
        obj_idx = int(label_id) - 1
        if obj_idx < 0 or obj_idx >= len(objects):
            continue
        comp_slice = objects[obj_idx]
        if comp_slice is None:
            continue
        local_mask = np.asarray(labels[comp_slice] == int(label_id), dtype=bool)
        if not np.any(local_mask):
            continue
        local_values = np.asarray(values[comp_slice], dtype=np.float32)
        masked_values = np.where(local_mask, local_values, -np.inf)
        local_pos = np.unravel_index(int(np.argmax(masked_values)), masked_values.shape)
        positions.append(
            (
                int(local_pos[0]) + int(comp_slice[0].start or 0),
                int(local_pos[1]) + int(comp_slice[1].start or 0),
                int(local_pos[2]) + int(comp_slice[2].start or 0),
            )
        )
    return positions


def _merge_close_soma_marker_positions(
    positions: list[tuple[int, int, int]],
    working: np.ndarray,
    *,
    low_floor: float,
    high_t: float,
    branch_sensitivity: float,
) -> list[tuple[int, int, int]]:
    if len(positions) <= 1:
        return positions

    pts = np.asarray(positions, dtype=np.int32)
    body_t = float(max(low_floor, high_t * 0.45))
    body_mask = np.asarray(working >= body_t, dtype=bool)
    body_labels, _ = ndi.label(body_mask, structure=_CC_STRUCTURE)
    body_ids = body_labels[pts[:, 0], pts[:, 1], pts[:, 2]]

    merge_radius = float(np.clip(7.0 / max(0.5, branch_sensitivity), 4.5, 9.0))
    parent = list(range(len(positions)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(len(positions)):
        if int(body_ids[i]) <= 0:
            continue
        for j in range(i + 1, len(positions)):
            if int(body_ids[i]) != int(body_ids[j]):
                continue
            dist = float(np.linalg.norm(pts[i].astype(np.float32) - pts[j].astype(np.float32)))
            if dist <= merge_radius:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(len(positions)):
        groups.setdefault(find(idx), []).append(idx)
    if all(len(group) == 1 for group in groups.values()):
        return positions

    merged: list[tuple[int, int, int]] = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(positions[group[0]])
            continue
        group_pts = pts[group]
        group_values = working[group_pts[:, 0], group_pts[:, 1], group_pts[:, 2]]
        best = int(group[int(np.argmax(group_values))])
        merged.append(positions[best])

    logger.info(
        "Microglia seed merge: collapsed %d close soma peak marker(s) into %d marker(s).",
        len(positions),
        len(merged),
    )
    return merged


def _assign_low_confidence_branches_to_one_owner(
    labels: np.ndarray,
    working: np.ndarray,
    *,
    low_t: float,
    high_t: float,
    max_branch_voxels: int,
) -> np.ndarray:
    branch_mask = (labels > 0) & (working >= low_t) & (working < high_t)
    if not np.any(branch_mask):
        return labels

    branch_labels, n_branch = ndi.label(branch_mask, structure=_CC_STRUCTURE)
    if n_branch <= 0:
        return labels

    out = np.asarray(labels, dtype=np.int32).copy()
    objects = ndi.find_objects(branch_labels)
    reassigned = 0
    for branch_id, comp_slice in enumerate(objects, start=1):
        if comp_slice is None:
            continue
        local_branch = branch_labels[comp_slice] == int(branch_id)
        branch_voxels = int(np.count_nonzero(local_branch))
        if branch_voxels <= 0 or branch_voxels > int(max_branch_voxels):
            continue

        local_labels = out[comp_slice]
        current_ids = np.unique(local_labels[local_branch])
        current_ids = current_ids[current_ids > 0]
        if current_ids.size <= 1:
            continue

        # Score branch ownership by integrated intensity already assigned to
        # each label; this favors the soma with the strongest 3D connection.
        scores: list[tuple[float, int]] = []
        local_values = np.asarray(working[comp_slice], dtype=np.float32)
        for label_id in current_ids.tolist():
            owned = local_branch & (local_labels == int(label_id))
            if np.any(owned):
                scores.append((float(np.sum(local_values[owned])), int(label_id)))
        if not scores:
            continue
        owner = max(scores)[1]
        local_labels[local_branch] = owner
        out[comp_slice] = local_labels
        reassigned += 1

    if reassigned > 0:
        logger.info(
            "Microglia branch ownership: reassigned %d faint branch segment(s) to one soma owner.",
            reassigned,
        )
    return out


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
        np.subtract(peak_max, 1.0e-6, out=peak_max)
        np.greater_equal(detect, peak_max, out=peak_mask, where=peak_mask)

    max_centers = int(np.clip((arr.shape[1] * arr.shape[2]) // 36, 32, 1024))
    local_centers = _peak_centers_from_mask(
        detect,
        peak_mask,
        max_candidates=max(_MAX_PEAK_CENTER_CANDIDATES, max_centers * 8),
    )
    logger.info(
        "Microglia seed detection: volume_shape=%s threshold=%.5f peak_floor=%.5f "
        "peak_candidates=%d max_centers=%d spacing=%s",
        arr.shape,
        float(threshold),
        peak_floor,
        int(local_centers.shape[0]),
        max_centers,
        spacing,
    )
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
        np.subtract(fallback_max, 1.0e-6, out=fallback_max)
        np.greater_equal(detect, fallback_max, out=fallback_mask, where=fallback_mask)
    if np.any(fallback_mask):
        fallback_candidates = _peak_centers_from_mask(
            detect,
            fallback_mask,
            max_candidates=max(_MAX_PEAK_CENTER_CANDIDATES, max_centers * 8),
        )
        fallback_centers = _select_seed_centers(
            detect,
            fallback_candidates,
            min_sep=min_sep,
            max_centers=max_centers,
        )
        logger.info(
            "Microglia seed fallback: candidates=%d selected=%d.",
            int(fallback_candidates.shape[0]),
            int(fallback_centers.shape[0]),
        )
        for pos in fallback_centers:
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
    marker_mask = np.asarray(seed, dtype=bool).copy()
    if np.any(soma_mask):
        # Keep one marker per disconnected high-confidence soma island. Faint
        # bridges are assigned later by watershed instead of being allowed to
        # merge distinct somas during marker construction.
        soma_labels, n_soma = ndi.label(soma_mask, structure=_CC_STRUCTURE)
        if n_soma > 0:
            min_island_voxels = max(2, int(min_keep // 2))
            island_sizes = np.bincount(soma_labels.ravel(), minlength=n_soma + 1)
            valid_ids = [
                i for i in range(1, n_soma + 1)
                if island_sizes[i] >= min_island_voxels
            ]
            if valid_ids:
                positions = _max_positions_for_labels(working, soma_labels, valid_ids)
                positions = _merge_close_soma_marker_positions(
                    positions,
                    working,
                    low_floor=low_floor,
                    high_t=high_t,
                    branch_sensitivity=branch_sensitivity,
                )
                logger.info(
                    "Microglia seed merge: using %d high-confidence soma island marker(s).",
                    len(positions),
                )
                marker_mask &= ~soma_mask
                for pos in positions:
                    marker_mask[int(pos[0]), int(pos[1]), int(pos[2])] = True

    seed_labels, _ = ndi.label(marker_mask, structure=_CC_STRUCTURE)
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
    # Use a data-driven floor so near-zero thresholds don't make the growth
    # mask include background noise and merge adjacent cells.  The floor is
    # kept modest (5th-10th percentile of positive values) to avoid trimming
    # legitimate dim branches.
    data_floor_quantile = float(np.clip(0.05 - 0.02 * (branch_sense - 1.0), 0.03, 0.10))
    data_floor = float(np.quantile(positive, data_floor_quantile))
    low_t = float(
        np.clip(
            max(0.01, data_floor, min(t * low_scale, high_t * core_scale)),
            0.0,
            max(0.0, high_t - 1.0e-4),
        )
    )
    merge_scale = float(np.clip(0.72 - (0.08 * (branch_sense - 1.0)), 0.56, 0.80))
    merge_t = float(np.clip(max(low_t, high_t * merge_scale), 0.0, max(high_t, low_t)))

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
    raw_labels = _assign_low_confidence_branches_to_one_owner(
        raw_labels,
        working_roi,
        low_t=low_t,
        high_t=merge_t,
        max_branch_voxels=max(int(min_keep) * 128, int(arr.size * 0.015)),
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

    # Use bounding box to avoid allocating a full-volume zeros array.
    bounds = _mask_bounds(lbl == component)
    if bounds is None:
        return np.zeros_like(arr, dtype=np.float32)

    out = np.zeros_like(arr, dtype=np.float32)
    local_mask = lbl[bounds] == component
    out[bounds][local_mask] = arr[bounds][local_mask]
    return out


def filter_components_by_preferred_voxel_floor(
    labels: np.ndarray,
    order: np.ndarray,
    sizes: np.ndarray,
    *,
    preferred_min_voxels: int = PREFERRED_VISIBLE_MICROGLIA_MIN_VOXELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop small components when clearly visible larger components already exist."""
    lbl = np.asarray(labels, dtype=np.int32)
    ordered = np.asarray(order, dtype=np.int32)
    comp_sizes = np.asarray(sizes, dtype=np.int64)
    preferred_floor = max(1, int(preferred_min_voxels))

    if ordered.size <= 0 or comp_sizes.size <= 1:
        return lbl, ordered, comp_sizes

    keep_mask = np.array(
        [
            int(component_id) < int(comp_sizes.shape[0])
            and int(comp_sizes[int(component_id)]) >= preferred_floor
            for component_id in ordered
        ],
        dtype=bool,
    )
    if not np.any(keep_mask):
        return lbl, ordered, comp_sizes

    keep_ids = ordered[keep_mask]
    if keep_ids.size >= 8:
        keep_sizes = np.asarray(comp_sizes[keep_ids], dtype=np.float64)
        median_size = float(np.median(keep_sizes))
        q25, q75 = np.percentile(keep_sizes, [25.0, 75.0])
        iqr = float(max(1.0, q75 - q25))
        giant_floor = float(
            max(
                preferred_floor * 3.0,
                median_size * 2.75,
                q75 + (2.5 * iqr),
            )
        )
        giant_mask = keep_sizes > giant_floor
        if np.any(giant_mask) and np.any(~giant_mask):
            normal_ids = keep_ids[~giant_mask]
            giant_ids = keep_ids[giant_mask]
            normal_ids = normal_ids[np.argsort(comp_sizes[normal_ids])[::-1]]
            giant_ids = giant_ids[np.argsort(comp_sizes[giant_ids])[::-1]]
            keep_ids = np.concatenate([normal_ids, giant_ids]).astype(np.int32, copy=False)
            logger.info(
                "Microglia ranking: moved %d unusually large component(s) after "
                "typical cells (median=%d giant_floor=%d).",
                int(np.count_nonzero(giant_mask)),
                int(round(median_size)),
                int(round(giant_floor)),
            )

    lut = np.zeros(int(comp_sizes.shape[0]), dtype=np.int32)
    filtered_sizes = np.zeros(int(keep_ids.size) + 1, dtype=np.int64)
    for new_id, old_id in enumerate(keep_ids, start=1):
        lut[int(old_id)] = int(new_id)
        filtered_sizes[new_id] = int(comp_sizes[int(old_id)])

    filtered_labels = lut[lbl]
    filtered_order = np.arange(1, int(keep_ids.size) + 1, dtype=np.int32)
    return filtered_labels, filtered_order, filtered_sizes
