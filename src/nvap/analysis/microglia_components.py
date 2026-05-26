from __future__ import annotations

import logging
import time

import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed as _watershed


logger = logging.getLogger(__name__)
_CC_STRUCTURE = ndi.generate_binary_structure(3, 2).astype(np.uint8, copy=False)
_CUBIC_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)
_AXIAL_STRUCTURE = ndi.generate_binary_structure(3, 1).astype(np.uint8, copy=False)
_MAX_PEAK_CENTER_CANDIDATES = 16384
_MAX_DISTANCE_EDT_VOXELS = 24_000_000
_MAX_DISTANCE_COMPONENT_LABEL_VOXELS = 96_000_000
PREFERRED_VISIBLE_MICROGLIA_MIN_VOXELS = 15_000
_DENSE_FIELD_NONZERO_RATIO = 0.035
_DENSE_FIELD_MAX_SEED_CENTERS = 4096
_LARGE_VOLUME_VOXELS = 160_000_000
_EXTREME_VOLUME_VOXELS = 280_000_000
_LARGE_VOLUME_MARKER_CAP = 1024
_EXTREME_VOLUME_MARKER_CAP = 768
_MAX_SOMA_EDT_VOXELS = 72_000_000
_MAX_SOMA_CORE_LABEL_VOXELS = 120_000_000
_MAX_BRANCH_REASSIGN_VOXELS = 120_000_000
_MAX_BRANCH_REASSIGN_LABELS = 2048


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


def _dense_microglia_scene(nonzero_ratio: float, occupied_z_ratio: float = 1.0) -> bool:
    ratio = float(nonzero_ratio)
    z_ratio = float(occupied_z_ratio)
    return bool(
        ratio >= float(_DENSE_FIELD_NONZERO_RATIO)
        and (z_ratio >= 0.5 or ratio >= float(_DENSE_FIELD_NONZERO_RATIO) * 2.0)
    )


def _values_above_floor(values: np.ndarray, floor: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    subset = arr[arr >= float(floor)]
    if subset.size <= 0:
        return arr
    return np.asarray(subset, dtype=np.float32)


def _signal_occupancy_ratio(values: np.ndarray, finite: np.ndarray, floor: float) -> float:
    finite_mask = np.asarray(finite, dtype=bool)
    if finite_mask.size == 0:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(np.count_nonzero(finite_mask & (arr >= float(floor)))) / max(1, int(finite_mask.size))


def _signal_occupied_z_ratio(values: np.ndarray, finite: np.ndarray, floor: float) -> float:
    finite_mask = np.asarray(finite, dtype=bool)
    if finite_mask.ndim != 3 or finite_mask.size == 0:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    signal = finite_mask & (arr >= float(floor))
    if not np.any(signal):
        return 0.0
    return float(np.count_nonzero(np.any(signal, axis=(1, 2)))) / max(1, int(signal.shape[0]))


def _spacing_aware_structure(spacing_zyx: np.ndarray) -> np.ndarray:
    spacing = np.maximum(np.asarray(spacing_zyx, dtype=np.float32), 1.0e-6)
    xy_floor = float(max(1.0e-6, min(float(spacing[1]), float(spacing[2]))))
    z_ratio = float(spacing[0] / xy_floor)
    if z_ratio <= 1.45:
        return _CUBIC_STRUCTURE
    if z_ratio <= 2.2:
        return _CC_STRUCTURE
    return _AXIAL_STRUCTURE


def _local_contrast_response(
    values: np.ndarray,
    spacing_zyx: np.ndarray,
    *,
    sigma_xy_um: float,
    sigma_z_um: float,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    spacing = np.maximum(np.asarray(spacing_zyx, dtype=np.float32), 1.0e-6)
    sigma = (
        float(np.clip(sigma_z_um / float(spacing[0]), 0.0, 2.0)),
        float(np.clip(sigma_xy_um / float(spacing[1]), 0.0, 8.0)),
        float(np.clip(sigma_xy_um / float(spacing[2]), 0.0, 8.0)),
    )
    local_bg = ndi.gaussian_filter(arr, sigma=sigma, mode="nearest")
    residual = np.asarray(arr - local_bg, dtype=np.float32)
    residual[~np.isfinite(residual)] = 0.0
    return np.maximum(residual, 0.0)


def _select_seed_centers(
    arr: np.ndarray,
    centers: np.ndarray,
    *,
    min_sep_um: float,
    spacing_zyx: np.ndarray,
    max_centers: int,
) -> np.ndarray:
    pts = np.asarray(centers, dtype=np.int32)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    pts = pts.reshape(-1, 3)
    pts = np.unique(pts, axis=0)
    values = arr[pts[:, 0], pts[:, 1], pts[:, 2]]
    order = np.argsort(values)[::-1]
    spacing = np.maximum(np.asarray(spacing_zyx, dtype=np.float32).reshape(1, 3), 1.0e-6)
    pts_um = pts.astype(np.float32, copy=False) * spacing
    min_sep = float(max(0.0, min_sep_um))
    min_sep2 = float(min_sep * min_sep)
    max_keep = int(max(1, max_centers))

    selected: list[tuple[int, int, int]] = []
    selected_um: list[np.ndarray] = []
    if min_sep2 > 0.0:
        cell = float(max(min_sep, 1.0e-6))
        neighborhood = (
            (-1, -1, -1),
            (-1, -1, 0),
            (-1, -1, 1),
            (-1, 0, -1),
            (-1, 0, 0),
            (-1, 0, 1),
            (-1, 1, -1),
            (-1, 1, 0),
            (-1, 1, 1),
            (0, -1, -1),
            (0, -1, 0),
            (0, -1, 1),
            (0, 0, -1),
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, -1),
            (0, 1, 0),
            (0, 1, 1),
            (1, -1, -1),
            (1, -1, 0),
            (1, -1, 1),
            (1, 0, -1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, -1),
            (1, 1, 0),
            (1, 1, 1),
        )
        buckets: dict[tuple[int, int, int], list[np.ndarray]] = {}
    else:
        cell = 1.0
        neighborhood = ((0, 0, 0),)
        buckets = {}

    for idx in order:
        z, y, x = (int(v) for v in pts[int(idx)])
        keep = True
        candidate_um = np.asarray(pts_um[int(idx)], dtype=np.float32)
        if min_sep2 > 0.0 and selected_um:
            cell_key = tuple(int(v) for v in np.floor(candidate_um / cell).astype(np.int32).tolist())
            for dz, dy, dx in neighborhood:
                neighbor_key = (
                    int(cell_key[0] + dz),
                    int(cell_key[1] + dy),
                    int(cell_key[2] + dx),
                )
                for existing_um in buckets.get(neighbor_key, []):
                    delta = existing_um - candidate_um
                    if float(np.dot(delta, delta)) < min_sep2:
                        keep = False
                        break
                if not keep:
                    break
        if not keep:
            continue
        selected.append((z, y, x))
        selected_um.append(candidate_um)
        if min_sep2 > 0.0:
            cell_key = tuple(int(v) for v in np.floor(candidate_um / cell).astype(np.int32).tolist())
            buckets.setdefault(cell_key, []).append(candidate_um)
        if len(selected) >= max_keep:
            break

    if not selected:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(selected, dtype=np.int32)


def _distance_peak_centers_from_soma_mask(
    soma_mask: np.ndarray,
    spacing_zyx: np.ndarray,
    *,
    max_candidates: int = _MAX_PEAK_CENTER_CANDIDATES,
) -> np.ndarray:
    mask = np.asarray(soma_mask, dtype=bool)
    bounds = _mask_bounds(mask)
    if bounds is None:
        return np.empty((0, 3), dtype=np.int32)

    local_mask = mask[bounds]
    if not np.any(local_mask):
        return np.empty((0, 3), dtype=np.int32)

    spacing = np.maximum(np.asarray(spacing_zyx, dtype=np.float32), 1.0e-6)

    def _component_distance_peak_centers(
        component_mask: np.ndarray,
        *,
        global_offset_zyx: np.ndarray,
        peak_cap: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        comp_mask = np.asarray(component_mask, dtype=bool)
        if not np.any(comp_mask):
            return (
                np.empty((0, 3), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        comp_shape = np.asarray(comp_mask.shape, dtype=np.int64)
        comp_voxels = int(np.prod(comp_shape))
        step = int(max(1, np.ceil((comp_voxels / float(_MAX_DISTANCE_EDT_VOXELS)) ** (1.0 / 3.0))))
        step_zyx = np.asarray([step, step, step], dtype=np.int32)
        if step > 1:
            eval_mask = np.asarray(comp_mask[::step, ::step, ::step], dtype=bool)
            logger.info(
                "Microglia seed EDT: downsampled component bbox %s by step=%d for memory safety.",
                tuple(int(v) for v in comp_shape.tolist()),
                int(step),
            )
        else:
            eval_mask = comp_mask

        if not np.any(eval_mask):
            return (
                np.empty((0, 3), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        eval_spacing = spacing * step_zyx.astype(np.float32)
        try:
            dist = ndi.distance_transform_edt(
                eval_mask,
                sampling=(float(eval_spacing[0]), float(eval_spacing[1]), float(eval_spacing[2])),
            )
        except MemoryError:
            logger.warning(
                "Microglia seed EDT: skipping component bbox %s due memory pressure.",
                tuple(int(v) for v in comp_shape.tolist()),
            )
            return (
                np.empty((0, 3), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        local_dist = np.asarray(dist[eval_mask], dtype=np.float32)
        if local_dist.size <= 0:
            return (
                np.empty((0, 3), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        floor = float(max(np.quantile(local_dist, 0.60), 0.75 * float(np.min(eval_spacing))))
        peak_mask = eval_mask & (dist >= floor)
        if not np.any(peak_mask):
            return (
                np.empty((0, 3), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )

        peak_radius_um = 1.2
        max_size = tuple(
            int(
                max(
                    3,
                    (2 * int(np.ceil(peak_radius_um / max(1.0e-6, float(eval_spacing[i]))))) + 1,
                )
            )
            for i in range(3)
        )
        peak_max = ndi.maximum_filter(dist, size=max_size, mode="nearest")
        np.subtract(peak_max, 1.0e-6, out=peak_max)
        np.greater_equal(dist, peak_max, out=peak_mask, where=peak_mask)

        eval_centers = _peak_centers_from_mask(dist, peak_mask, max_candidates=int(max(1, peak_cap)))
        if eval_centers.size == 0:
            return (
                np.empty((0, 3), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )
        eval_centers = np.asarray(eval_centers, dtype=np.int32)
        eval_scores = np.asarray(dist[tuple(eval_centers.T)], dtype=np.float32)

        if step > 1:
            local_centers = eval_centers * step_zyx
            np.minimum(local_centers, comp_shape.astype(np.int32) - 1, out=local_centers)
        else:
            local_centers = eval_centers

        global_centers = local_centers + np.asarray(global_offset_zyx, dtype=np.int32)
        return np.asarray(global_centers, dtype=np.int32), eval_scores

    max_candidates = int(max(1, max_candidates))
    local_offsets = np.asarray(
        [
            int(bounds[0].start or 0),
            int(bounds[1].start or 0),
            int(bounds[2].start or 0),
        ],
        dtype=np.int32,
    )
    candidate_centers: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []

    if int(local_mask.size) > int(_MAX_DISTANCE_COMPONENT_LABEL_VOXELS):
        logger.info(
            "Microglia seed EDT: bypassing full connected-component labeling for large mask shape=%s.",
            tuple(int(v) for v in local_mask.shape),
        )
        centers, scores = _component_distance_peak_centers(
            local_mask,
            global_offset_zyx=local_offsets,
            peak_cap=max_candidates,
        )
        if centers.size > 0:
            candidate_centers.append(centers)
            candidate_scores.append(scores)
    else:
        component_labels, n_components = ndi.label(local_mask, structure=_CC_STRUCTURE)
        if n_components <= 0:
            return np.empty((0, 3), dtype=np.int32)

        component_objects = ndi.find_objects(component_labels)
        per_component_cap = int(max(32, max_candidates // max(1, n_components)))
        for component_id, comp_slice in enumerate(component_objects, start=1):
            if comp_slice is None:
                continue
            comp_mask = np.asarray(component_labels[comp_slice] == int(component_id), dtype=bool)
            if not np.any(comp_mask):
                continue
            component_offset = np.asarray(
                [
                    int(comp_slice[0].start or 0),
                    int(comp_slice[1].start or 0),
                    int(comp_slice[2].start or 0),
                ],
                dtype=np.int32,
            )
            centers, scores = _component_distance_peak_centers(
                comp_mask,
                global_offset_zyx=component_offset + local_offsets,
                peak_cap=per_component_cap,
            )
            if centers.size > 0:
                candidate_centers.append(centers)
                candidate_scores.append(scores)

    if not candidate_centers:
        return np.empty((0, 3), dtype=np.int32)

    centers = np.vstack(candidate_centers)
    scores = np.concatenate(candidate_scores)
    if centers.shape[0] > max_candidates:
        keep_idx = np.argpartition(scores, -max_candidates)[-max_candidates:]
        centers = centers[keep_idx]
        scores = scores[keep_idx]

    unique_centers, inverse = np.unique(centers, axis=0, return_inverse=True)
    if unique_centers.shape[0] != centers.shape[0]:
        unique_scores = np.full((unique_centers.shape[0],), -np.inf, dtype=np.float32)
        np.maximum.at(unique_scores, inverse, scores)
        centers = unique_centers
        scores = unique_scores
        if centers.shape[0] > max_candidates:
            keep_idx = np.argpartition(scores, -max_candidates)[-max_candidates:]
            centers = centers[keep_idx]

    return np.asarray(centers, dtype=np.int32)


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


def _line_min_intensity_between_points(
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    left_pt = np.asarray(left, dtype=np.int32)
    right_pt = np.asarray(right, dtype=np.int32)
    steps = int(np.max(np.abs(right_pt - left_pt))) + 1
    if steps <= 1:
        return float(values[int(left_pt[0]), int(left_pt[1]), int(left_pt[2])])
    z = np.rint(np.linspace(float(left_pt[0]), float(right_pt[0]), num=steps)).astype(np.int32)
    y = np.rint(np.linspace(float(left_pt[1]), float(right_pt[1]), num=steps)).astype(np.int32)
    x = np.rint(np.linspace(float(left_pt[2]), float(right_pt[2]), num=steps)).astype(np.int32)
    return float(np.min(values[z, y, x]))


def _is_multi_soma_island(
    local_values: np.ndarray,
    centers: np.ndarray,
    *,
    spacing_zyx: np.ndarray,
    branch_sensitivity: float,
) -> bool:
    pts = np.asarray(centers, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return False

    vals = np.asarray(local_values[tuple(pts.T)], dtype=np.float32)
    order = np.argsort(vals)[::-1]
    keep = order[: min(int(order.size), 6)]
    pts = pts[keep]
    vals = vals[keep]

    spacing = np.maximum(np.asarray(spacing_zyx, dtype=np.float32).reshape(1, 3), 1.0e-6)
    pts_um = pts.astype(np.float32, copy=False) * spacing

    min_sep_um = float(np.clip(1.8 / max(0.5, branch_sensitivity), 1.1, 2.6))
    # Require a fairly deep valley between candidate peaks before treating an
    # island as multi-soma; this avoids splitting one soma with patchy lobes.
    dip_ratio_limit = float(np.clip(0.42 + (0.04 * (branch_sensitivity - 1.0)), 0.38, 0.50))
    for i in range(pts.shape[0]):
        for j in range(i + 1, pts.shape[0]):
            if float(np.linalg.norm(pts_um[i] - pts_um[j])) < min_sep_um:
                continue
            base = float(min(vals[i], vals[j]))
            if base <= 1.0e-6:
                continue
            line_min = _line_min_intensity_between_points(local_values, pts[i], pts[j])
            dip_ratio = line_min / base
            if dip_ratio <= dip_ratio_limit:
                return True
    return False


def _single_island_multi_marker_positions(
    working: np.ndarray,
    island_mask: np.ndarray,
    *,
    high_t: float,
    branch_sensitivity: float,
    spacing_zyx: np.ndarray,
) -> list[tuple[int, int, int]]:
    mask = np.asarray(island_mask, dtype=bool)
    bounds = _mask_bounds(mask)
    if bounds is None:
        return []

    local_mask = np.asarray(mask[bounds], dtype=bool)
    if not np.any(local_mask):
        return []
    local_values = np.asarray(working[bounds], dtype=np.float32)
    spacing = np.maximum(np.asarray(spacing_zyx, dtype=np.float32), 1.0e-6)

    candidate_arrays: list[np.ndarray] = []

    peak_floor = float(max(high_t * 0.96, np.quantile(local_values[local_mask], 0.90)))
    peak_mask = local_mask & (local_values >= peak_floor)
    if np.any(peak_mask):
        peak_radius_um = float(np.clip(1.5 / max(0.5, branch_sensitivity), 0.9, 2.4))
        peak_size = tuple(
            int(max(3, (2 * int(np.ceil(peak_radius_um / float(spacing[i])))) + 1))
            for i in range(3)
        )
        peak_max = ndi.maximum_filter(local_values, size=peak_size, mode="nearest")
        np.subtract(peak_max, 1.0e-6, out=peak_max)
        np.greater_equal(local_values, peak_max, out=peak_mask, where=peak_mask)
        if np.any(peak_mask):
            peak_labels, n_peak = ndi.label(peak_mask, structure=_CC_STRUCTURE)
            if n_peak > 0:
                peak_ids = list(range(1, n_peak + 1))
                peak_positions = _max_positions_for_labels(local_values, peak_labels, peak_ids)
                if peak_positions:
                    candidate_arrays.append(np.asarray(peak_positions, dtype=np.int32))

    distance_centers = _distance_peak_centers_from_soma_mask(
        local_mask,
        spacing,
        max_candidates=64,
    )
    if distance_centers.size > 0:
        candidate_arrays.append(np.asarray(distance_centers, dtype=np.int32))

    if not candidate_arrays:
        return []

    candidates = np.vstack(candidate_arrays)
    candidates = np.unique(np.asarray(candidates, dtype=np.int32), axis=0)
    if candidates.shape[0] < 2:
        return []

    selected = _select_seed_centers(
        local_values,
        candidates,
        min_sep_um=float(np.clip(1.8 / max(0.5, branch_sensitivity), 1.1, 2.8)),
        spacing_zyx=spacing,
        max_centers=4,
    )
    if selected.shape[0] < 2:
        return []
    if not _is_multi_soma_island(
        local_values,
        selected,
        spacing_zyx=spacing,
        branch_sensitivity=branch_sensitivity,
    ):
        return []

    offset = np.asarray(
        [
            int(bounds[0].start or 0),
            int(bounds[1].start or 0),
            int(bounds[2].start or 0),
        ],
        dtype=np.int32,
    )
    global_selected = np.asarray(selected, dtype=np.int32) + offset
    return [tuple(int(v) for v in pos) for pos in global_selected.tolist()]


def _merge_close_soma_marker_positions(
    positions: list[tuple[int, int, int]],
    working: np.ndarray,
    *,
    low_floor: float,
    high_t: float,
    branch_sensitivity: float,
    spacing_zyx: np.ndarray,
) -> list[tuple[int, int, int]]:
    if len(positions) <= 1:
        return positions

    if int(working.size) >= int(_LARGE_VOLUME_VOXELS) or len(positions) > int(_LARGE_VOLUME_MARKER_CAP * 2):
        merge_radius_um = float(np.clip(2.2 / max(0.5, branch_sensitivity), 1.1, 3.0))
        cap = int(
            min(
                len(positions),
                _EXTREME_VOLUME_MARKER_CAP
                if int(working.size) >= int(_EXTREME_VOLUME_VOXELS)
                else _LARGE_VOLUME_MARKER_CAP,
            )
        )
        reduced = _select_seed_centers(
            np.asarray(working, dtype=np.float32),
            np.asarray(positions, dtype=np.int32),
            min_sep_um=max(merge_radius_um, 1.4),
            spacing_zyx=spacing_zyx,
            max_centers=cap,
        )
        if reduced.size > 0:
            merged_fast = [tuple(int(v) for v in row) for row in reduced.tolist()]
            logger.info(
                "Microglia seed merge: fast-path reduced markers %d -> %d for large volume.",
                len(positions),
                len(merged_fast),
            )
            return merged_fast

    pts = np.asarray(positions, dtype=np.int32)
    pts_um = pts.astype(np.float32, copy=False) * np.maximum(
        np.asarray(spacing_zyx, dtype=np.float32).reshape(1, 3),
        1.0e-6,
    )
    body_scale = float(np.clip(0.60 + (0.03 * (branch_sensitivity - 1.0)), 0.54, 0.70))
    body_t = float(max(low_floor, high_t * body_scale))
    body_mask = np.asarray(working >= body_t, dtype=bool)
    body_labels, _ = ndi.label(body_mask, structure=_CC_STRUCTURE)
    body_ids = body_labels[pts[:, 0], pts[:, 1], pts[:, 2]]

    merge_radius_um = float(np.clip(2.2 / max(0.5, branch_sensitivity), 1.1, 3.0))
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
            dist_um = float(np.linalg.norm(pts_um[i] - pts_um[j]))
            if dist_um <= merge_radius_um:
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
    structure: np.ndarray = _CUBIC_STRUCTURE,
) -> np.ndarray:
    branch_mask = (labels > 0) & (working >= low_t) & (working < high_t)
    if not np.any(branch_mask):
        return labels

    branch_labels, n_branch = ndi.label(branch_mask, structure=structure)
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

        owner_candidates = np.unique(
            local_labels[
                ndi.binary_dilation(local_branch, structure=structure) & (~local_branch)
            ]
        )
        owner_candidates = owner_candidates[owner_candidates > 0]
        if owner_candidates.size == 1:
            local_labels[local_branch] = int(owner_candidates[0])
            out[comp_slice] = local_labels
            reassigned += 1
            continue

        if owner_candidates.size >= 2:
            owner_frontier_markers = np.zeros(local_branch.shape, dtype=np.int32)
            owner_map: dict[int, int] = {}
            marker_id = 0
            for owner_id in owner_candidates.tolist():
                owner_touch = local_branch & ndi.binary_dilation(
                    local_labels == int(owner_id),
                    structure=structure,
                )
                if not np.any(owner_touch):
                    continue
                marker_id += 1
                owner_map[marker_id] = int(owner_id)
                owner_frontier_markers[owner_touch] = marker_id

            if marker_id >= 2:
                local_values = np.asarray(working[comp_slice], dtype=np.float32)
                branch_owner_markers = np.asarray(
                    _watershed(
                        -local_values,
                        markers=owner_frontier_markers,
                        mask=local_branch,
                        connectivity=structure,
                    ),
                    dtype=np.int32,
                )
                if np.any(branch_owner_markers > 0):
                    for marker_idx, owner_id in owner_map.items():
                        local_labels[branch_owner_markers == int(marker_idx)] = int(owner_id)
                    out[comp_slice] = local_labels
                    reassigned += 1
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


def _segment_soma_markers_from_reduced_threshold(
    seed: np.ndarray,
    working: np.ndarray,
    finite: np.ndarray,
    *,
    low_floor: float,
    high_t: float,
    branch_sensitivity: float,
    min_keep: int,
    spacing_zyx: np.ndarray,
    structure: np.ndarray,
) -> np.ndarray:
    sensitivity = float(np.clip(branch_sensitivity, 0.4, 2.0))
    dense_floor = max(0.02, low_floor, high_t * 0.25)
    dense_mode = _dense_microglia_scene(
        _signal_occupancy_ratio(working, finite, dense_floor),
        _signal_occupied_z_ratio(working, finite, dense_floor),
    )
    reduce_scale = float(np.clip(0.58 - (0.08 * (sensitivity - 1.0)), 0.46, 0.72))
    if dense_mode:
        reduce_scale = float(np.clip(reduce_scale + 0.06, 0.46, 0.82))
    soma_t = float(np.clip(max(low_floor, high_t * reduce_scale), 0.0, max(high_t, low_floor)))
    soma_candidate = np.asarray(finite & (working >= soma_t), dtype=bool)
    if not np.any(soma_candidate):
        return np.zeros(seed.shape, dtype=np.int32)

    soma_voxels = int(np.count_nonzero(soma_candidate))
    use_full_dist = bool(
        int(soma_candidate.size) <= int(_MAX_SOMA_EDT_VOXELS)
        and soma_voxels <= int(_MAX_SOMA_EDT_VOXELS)
    )
    if use_full_dist:
        try:
            dist = ndi.distance_transform_edt(
                soma_candidate,
                sampling=(float(spacing_zyx[0]), float(spacing_zyx[1]), float(spacing_zyx[2])),
            )
        except MemoryError:
            dist = np.zeros_like(working, dtype=np.float32)
            use_full_dist = False
    else:
        dist = np.zeros_like(working, dtype=np.float32)
        logger.info(
            "Microglia soma EDT: skipped full-resolution EDT for large candidate voxels=%d shape=%s.",
            soma_voxels,
            tuple(int(v) for v in soma_candidate.shape),
        )

    soma_core_mask = soma_candidate
    dist_on_candidate = np.asarray(dist[soma_candidate], dtype=np.float32)
    if use_full_dist and dist_on_candidate.size > 0 and np.any(dist_on_candidate > 0.0):
        core_quantile = float(np.clip(0.48 + (0.05 * (sensitivity - 1.0)), 0.40, 0.60))
        core_floor = float(np.quantile(dist_on_candidate, core_quantile))
        soma_core_mask = soma_candidate & (dist >= max(0.0, core_floor))
        if not np.any(soma_core_mask):
            soma_core_mask = soma_candidate

    marker_positions: list[tuple[int, int, int]] = []
    seed_inside = np.asarray(seed, dtype=bool) & soma_candidate

    run_core_labeling = bool(
        use_full_dist
        or int(soma_candidate.size) <= int(_MAX_SOMA_CORE_LABEL_VOXELS)
        or soma_voxels <= int(_MAX_SOMA_CORE_LABEL_VOXELS)
    )
    if run_core_labeling:
        core_labels, n_core = ndi.label(soma_core_mask, structure=structure)
        if n_core > 0:
            core_sizes = np.bincount(core_labels.ravel(), minlength=n_core + 1)
            min_core_voxels = max(2, int(min_keep // 3))
            min_core_radius_um = float(np.clip(0.56 - (0.06 * (sensitivity - 1.0)), 0.40, 0.70))
            valid_ids = [
                i
                for i in range(1, n_core + 1)
                if (
                    int(core_sizes[i]) >= min_core_voxels
                    and (
                        (not use_full_dist)
                        or float(np.max(dist[core_labels == int(i)])) >= min_core_radius_um
                    )
                )
            ]
            if valid_ids:
                marker_positions.extend(_max_positions_for_labels(working, core_labels, valid_ids))
                if use_full_dist and len(valid_ids) == 1:
                    island_mask = np.asarray(core_labels == int(valid_ids[0]), dtype=bool)
                    split_positions = _single_island_multi_marker_positions(
                        working,
                        island_mask,
                        high_t=high_t,
                        branch_sensitivity=sensitivity,
                        spacing_zyx=spacing_zyx,
                    )
                    if len(split_positions) >= 2:
                        marker_positions = split_positions
    elif np.any(seed_inside):
        logger.info(
            "Microglia soma markers: skipped full candidate connected-component labeling for large volume.",
        )

    if not marker_positions and np.any(seed_inside):
        seed_labels, n_seed = ndi.label(seed_inside, structure=structure)
        if n_seed > 0:
            marker_positions = _max_positions_for_labels(working, seed_labels, list(range(1, n_seed + 1)))

    if marker_positions:
        marker_positions = _merge_close_soma_marker_positions(
            marker_positions,
            working,
            low_floor=low_floor,
            high_t=high_t,
            branch_sensitivity=sensitivity,
            spacing_zyx=spacing_zyx,
        )

    if marker_positions:
        marker_cap = int(
            _EXTREME_VOLUME_MARKER_CAP
            if int(working.size) >= int(_EXTREME_VOLUME_VOXELS)
            else _LARGE_VOLUME_MARKER_CAP
        )
        if int(working.size) >= int(_LARGE_VOLUME_VOXELS) and len(marker_positions) > marker_cap:
            capped = _select_seed_centers(
                np.asarray(working, dtype=np.float32),
                np.asarray(marker_positions, dtype=np.int32),
                min_sep_um=float(np.clip(2.3 / max(0.5, sensitivity), 1.2, 3.2)),
                spacing_zyx=spacing_zyx,
                max_centers=marker_cap,
            )
            if capped.size > 0:
                logger.info(
                    "Microglia soma marker cap: reduced markers %d -> %d (cap=%d).",
                    len(marker_positions),
                    int(capped.shape[0]),
                    marker_cap,
                )
                marker_positions = [tuple(int(v) for v in row) for row in capped.tolist()]

    if not marker_positions and np.any(seed_inside):
        marker_labels, _ = ndi.label(seed_inside, structure=structure)
    elif marker_positions:
        marker_mask = np.zeros(seed.shape, dtype=bool)
        for zc, yc, xc in marker_positions:
            marker_mask[int(zc), int(yc), int(xc)] = True
        marker_labels, _ = ndi.label(marker_mask, structure=structure)
    else:
        marker_labels = np.zeros(seed.shape, dtype=np.int32)

    if not np.any(marker_labels):
        return np.zeros(seed.shape, dtype=np.int32)

    soma_elevation = -np.asarray(working, dtype=np.float32)
    if np.any(dist > 0.0):
        max_dist = float(np.max(dist))
        if max_dist > 1.0e-6:
            soma_elevation = soma_elevation - (0.16 * (dist / max_dist).astype(np.float32, copy=False))

    soma_labels = np.asarray(
        _watershed(
            soma_elevation,
            markers=marker_labels,
            mask=soma_candidate,
            connectivity=structure,
        ),
        dtype=np.int32,
    )
    if not np.any(soma_labels):
        return np.asarray(marker_labels, dtype=np.int32)

    soma_sizes = np.bincount(soma_labels.ravel()).astype(np.int64)
    keep_ids = np.asarray(
        [
            i
            for i in range(1, soma_sizes.size)
            if int(soma_sizes[i]) >= max(2, int(min_keep // 3))
        ],
        dtype=np.int32,
    )
    if keep_ids.size <= 0:
        return np.asarray(marker_labels, dtype=np.int32)

    lut = np.zeros(int(soma_sizes.size), dtype=np.int32)
    for new_id, old_id in enumerate(keep_ids.tolist(), start=1):
        lut[int(old_id)] = int(new_id)
    return np.asarray(lut[soma_labels], dtype=np.int32)


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
    volume_voxels = int(arr.size)
    large_mode = bool(volume_voxels >= int(_LARGE_VOLUME_VOXELS))
    extreme_mode = bool(volume_voxels >= int(_EXTREME_VOLUME_VOXELS))
    dense_floor = max(0.02, float(threshold) * 0.25)
    nonzero_ratio = _signal_occupancy_ratio(arr, finite, dense_floor)
    dense_mode = _dense_microglia_scene(
        nonzero_ratio,
        _signal_occupied_z_ratio(arr, finite, dense_floor),
    )

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

    if extreme_mode:
        contrast = np.zeros_like(detect, dtype=np.float32)
        contrast_positive = np.empty((0,), dtype=np.float32)
        contrast_norm = np.zeros_like(detect, dtype=np.float32)
        seed_score = np.asarray(detect, dtype=np.float32)
    else:
        contrast = _local_contrast_response(
            detect,
            spacing_zyx,
            sigma_xy_um=3.1 if dense_mode else 2.6,
            sigma_z_um=0.30,
        )
        contrast_positive = contrast[finite & (contrast > 0.0)]
        if contrast_positive.size > 0:
            contrast_norm = np.asarray(
                contrast / max(1.0e-6, float(np.quantile(contrast_positive, 0.98))),
                dtype=np.float32,
            )
            contrast_norm *= float(0.42 if dense_mode else 0.28)
            contrast_norm += np.asarray(detect, dtype=np.float32)
            seed_score = contrast_norm
        else:
            seed_score = np.asarray(detect, dtype=np.float32)

    peak_quantile = float(np.clip(0.85 - (0.07 * (sensitivity_scale - 1.0)), 0.70, 0.90))
    peak_floor = float(max(threshold, np.quantile(detect_positive, peak_quantile)))
    peak_radius_um = float(np.clip(1.6 / sensitivity_scale, 0.9, 2.4))
    max_size = (
        int(max(3, (2 * int(np.ceil(peak_radius_um / float(spacing_zyx[0])))) + 1)),
        int(max(3, (2 * int(np.ceil(peak_radius_um / float(spacing_zyx[1])))) + 1)),
        int(max(3, (2 * int(np.ceil(peak_radius_um / float(spacing_zyx[2])))) + 1)),
    )
    peak_mask = finite & (detect >= peak_floor)
    if np.any(peak_mask):
        peak_max = ndi.maximum_filter(detect, size=max_size, mode="nearest")
        np.subtract(peak_max, 1.0e-6, out=peak_max)
        np.greater_equal(detect, peak_max, out=peak_mask, where=peak_mask)

    max_center_cap = int(_DENSE_FIELD_MAX_SEED_CENTERS if dense_mode else 1024)
    if large_mode:
        max_center_cap = min(max_center_cap, int(_LARGE_VOLUME_MARKER_CAP))
    if extreme_mode:
        max_center_cap = min(max_center_cap, int(_EXTREME_VOLUME_MARKER_CAP))
    max_centers = int(
        np.clip(
            (arr.shape[1] * arr.shape[2]) // (26 if dense_mode else 36),
            32,
            max_center_cap,
        )
    )
    peak_candidate_cap = int(np.clip(max_centers * (6 if dense_mode else 8), 512, _MAX_PEAK_CENTER_CANDIDATES))
    local_centers = _peak_centers_from_mask(
        seed_score,
        peak_mask,
        max_candidates=peak_candidate_cap,
    )

    if contrast_positive.size > 0:
        contrast_q = float(np.clip(0.84 - (0.03 * (sensitivity_scale - 1.0)), 0.74, 0.90))
        contrast_floor = float(np.quantile(contrast_positive, contrast_q))
        contrast_peak_mask = finite & (contrast >= contrast_floor)
        if np.any(contrast_peak_mask):
            peak_max = ndi.maximum_filter(seed_score, size=max_size, mode="nearest")
            np.subtract(peak_max, 1.0e-6, out=peak_max)
            np.greater_equal(seed_score, peak_max, out=contrast_peak_mask, where=contrast_peak_mask)
            contrast_centers = _peak_centers_from_mask(
                seed_score,
                contrast_peak_mask,
                max_candidates=max(256, int(peak_candidate_cap * 3 // 4)),
            )
            if contrast_centers.size > 0:
                if local_centers.size > 0:
                    local_centers = np.vstack([local_centers, contrast_centers])
                else:
                    local_centers = contrast_centers
                local_centers = np.unique(np.asarray(local_centers, dtype=np.int32), axis=0)

    soma_body_floor = float(max(threshold, peak_floor * (0.74 if dense_mode else 0.70)))
    soma_body_mask = finite & (detect >= soma_body_floor)
    if contrast_positive.size > 0:
        soma_body_mask |= contrast >= float(np.quantile(contrast_positive, 0.62 if dense_mode else 0.68))
    if extreme_mode:
        distance_centers = np.empty((0, 3), dtype=np.int32)
        logger.info(
            "Microglia seed detection: skipped EDT-based distance centers for extreme volume voxels=%d.",
            volume_voxels,
        )
    else:
        distance_centers = _distance_peak_centers_from_soma_mask(
            soma_body_mask,
            spacing_zyx,
            max_candidates=max(512, int(peak_candidate_cap * 3 // 4)),
        )
    if distance_centers.size > 0:
        if local_centers.size > 0:
            local_centers = np.vstack([local_centers, distance_centers])
        else:
            local_centers = distance_centers
        local_centers = np.unique(np.asarray(local_centers, dtype=np.int32), axis=0)

    logger.info(
        "Microglia seed detection: volume_shape=%s threshold=%.5f peak_floor=%.5f "
        "peak_candidates=%d max_centers=%d dense_mode=%s large_mode=%s spacing=%s",
        arr.shape,
        float(threshold),
        peak_floor,
        int(local_centers.shape[0]),
        max_centers,
        dense_mode,
        large_mode,
        spacing,
    )
    min_sep_um = float(np.clip((1.55 if dense_mode else 1.8) / sensitivity_scale, 0.75, 2.6))
    selected = _select_seed_centers(
        seed_score,
        local_centers,
        min_sep_um=min_sep_um,
        spacing_zyx=spacing_zyx,
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
            seed_score,
            fallback_candidates,
            min_sep_um=min_sep_um,
            spacing_zyx=spacing_zyx,
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
    spacing_zyx: np.ndarray,
    structure: np.ndarray,
) -> np.ndarray:
    soma_seed_labels = _segment_soma_markers_from_reduced_threshold(
        seed,
        working,
        finite,
        low_floor=low_floor,
        high_t=high_t,
        branch_sensitivity=branch_sensitivity,
        min_keep=min_keep,
        spacing_zyx=spacing_zyx,
        structure=structure,
    )
    if np.any(soma_seed_labels):
        logger.info(
            "Microglia soma segmentation: segmented %d soma marker region(s) with reduced threshold.",
            int(np.max(soma_seed_labels)),
        )
        return np.asarray(soma_seed_labels, dtype=np.int32)

    seed_labels, _ = ndi.label(np.asarray(seed, dtype=bool), structure=structure)
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
    stage_t0 = time.perf_counter()

    working = arr
    if any(float(s) > 0.0 for s in smooth_sigma):
        working = ndi.gaussian_filter(arr, sigma=smooth_sigma, mode="nearest")
    finite = np.isfinite(working)
    if not np.any(finite):
        return _empty_labels(arr.shape)

    positive = working[finite & (working > 0.0)]
    if positive.size <= 0:
        return _empty_labels(arr.shape)
    dense_floor = max(0.02, t * 0.25)
    nonzero_ratio = _signal_occupancy_ratio(working, finite, dense_floor)
    dense_mode = _dense_microglia_scene(
        nonzero_ratio,
        _signal_occupied_z_ratio(working, finite, dense_floor),
    )
    positive_q75 = float(np.quantile(positive, 0.75))
    dense_signal_floor = float(max(0.02, t * 0.45, positive_q75 * 1.05)) if dense_mode else 0.0
    dense_signal_values = _values_above_floor(positive, dense_signal_floor) if dense_mode else positive
    high_quantile = float(np.clip(0.84 - (0.06 * (branch_sense - 1.0)), 0.70, 0.92))
    high_q = float(np.quantile(dense_signal_values if dense_mode else positive, high_quantile))
    high_t = float(np.clip(max(t, high_q), 0.0, 0.999))
    spacing_zyx = np.maximum(
        np.asarray(spacing if spacing is not None else (1.0, 1.0, 1.0), dtype=np.float32),
        1.0e-6,
    )
    structure = _spacing_aware_structure(spacing_zyx)

    # Detect soma seed centres — one compact region per microglia.
    seed = _detect_soma_blobs(
        working,
        threshold=t,
        branch_sensitivity=branch_sense,
        spacing=(float(spacing_zyx[0]), float(spacing_zyx[1]), float(spacing_zyx[2])),
    )
    logger.info(
        "Microglia segmentation stage=seed_detection dt=%.2fs seed_voxels=%d.",
        time.perf_counter() - stage_t0,
        int(np.count_nonzero(seed)),
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
    if dense_mode:
        low_scale = float(np.clip(low_scale + 0.05, 0.22, 0.72))
    # Use a data-driven floor so near-zero thresholds don't make the growth
    # mask include background noise and merge adjacent cells.  The floor is
    # kept modest (5th-10th percentile of positive values) to avoid trimming
    # legitimate dim branches.
    data_floor_quantile = float(np.clip(0.05 - 0.02 * (branch_sense - 1.0), 0.03, 0.10))
    data_floor_values = (
        _values_above_floor(positive, max(0.02, t * 0.25, positive_q75 * 0.75))
        if dense_mode
        else positive
    )
    data_floor = float(np.quantile(data_floor_values, data_floor_quantile))
    low_t = float(
        np.clip(
            max(0.01, data_floor, min(t * low_scale, high_t * core_scale)),
            0.0,
            max(0.0, high_t - 1.0e-4),
        )
    )
    if dense_mode:
        dense_floor_values = dense_signal_values
        low_t = float(
            max(
                low_t,
                min(
                    high_t * 0.62,
                    float(
                        np.quantile(
                            dense_floor_values,
                            float(np.clip(0.11 - 0.02 * (branch_sense - 1.0), 0.07, 0.14)),
                        )
                    ),
                ),
            )
        )
        logger.info(
            "Microglia dense thresholds: signal_floor=%.5f q75=%.5f high_t=%.5f low_t=%.5f.",
            dense_signal_floor,
            positive_q75,
            high_t,
            low_t,
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
        spacing_zyx=spacing_zyx,
        structure=structure,
    )
    if not np.any(seed_labels):
        return _empty_labels(arr.shape)

    seed_count = int(np.max(seed_labels))
    seed_cap = int(max(256, min(2048, max_keep * 6)))
    if int(arr.size) >= int(_LARGE_VOLUME_VOXELS):
        seed_cap = min(seed_cap, int(_LARGE_VOLUME_MARKER_CAP))
    if int(arr.size) >= int(_EXTREME_VOLUME_VOXELS):
        seed_cap = min(seed_cap, int(_EXTREME_VOLUME_MARKER_CAP))
    if seed_count > seed_cap:
        score_by_label: list[tuple[float, int]] = []
        objects = ndi.find_objects(seed_labels)
        for label_id in range(1, seed_count + 1):
            obj_idx = int(label_id) - 1
            if obj_idx < 0 or obj_idx >= len(objects):
                continue
            comp_slice = objects[obj_idx]
            if comp_slice is None:
                continue
            local = np.asarray(seed_labels[comp_slice] == int(label_id), dtype=bool)
            if not np.any(local):
                continue
            local_values = np.asarray(working_roi[comp_slice], dtype=np.float32)
            score_by_label.append((float(np.max(local_values[local])), int(label_id)))
        if score_by_label:
            score_by_label.sort(key=lambda item: item[0], reverse=True)
            keep_ids = [label_id for _, label_id in score_by_label[:seed_cap]]
            lut = np.zeros(seed_count + 1, dtype=np.int32)
            for new_id, old_id in enumerate(keep_ids, start=1):
                lut[int(old_id)] = int(new_id)
            seed_labels = np.asarray(lut[seed_labels], dtype=np.int32)
            logger.info(
                "Microglia seed cap: reduced markers %d -> %d (cap=%d).",
                seed_count,
                int(np.max(seed_labels)),
                seed_cap,
            )

    # Build the growth mask: above low threshold with sufficient local support.
    support_min = 1 if min_keep <= 8 else (2 if branch_sense >= 1.30 else 1)
    if dense_mode:
        support_min = max(2, int(support_min))
    candidate = finite_roi & (working_roi >= low_t)
    if np.any(candidate):
        support = ndi.convolve(
            candidate.astype(np.uint8), structure, mode="constant", cval=0
        )
        # Seeds are always included regardless of neighbour support.
        candidate = (candidate & (support >= support_min)) | (seed_labels > 0)
    else:
        candidate = seed_labels > 0

    # Watershed: marker seeds grow through the candidate mask.
    # Inverting working_roi makes bright somas low-elevation (fill first), so
    # cells are naturally split at intensity saddle-points — no multi-pass needed.
    raw_labels = np.asarray(
        _watershed(
            -working_roi,
            markers=seed_labels,
            mask=candidate,
            connectivity=structure,
        ),
        dtype=np.int32,
    )
    labeled_voxels = int(np.count_nonzero(raw_labels))
    label_count = int(np.max(raw_labels)) if labeled_voxels > 0 else 0
    if labeled_voxels > int(_MAX_BRANCH_REASSIGN_VOXELS) or label_count > int(_MAX_BRANCH_REASSIGN_LABELS):
        logger.info(
            "Microglia branch ownership: skipped reassignment (voxels=%d labels=%d).",
            labeled_voxels,
            label_count,
        )
    else:
        raw_labels = _assign_low_confidence_branches_to_one_owner(
            raw_labels,
            working_roi,
            low_t=low_t,
            high_t=merge_t,
            max_branch_voxels=max(int(min_keep) * 128, int(arr.size * 0.015)),
            structure=structure,
        )
    logger.info(
        "Microglia segmentation stage=watershed_and_assignment dt=%.2fs labels=%d voxels=%d.",
        time.perf_counter() - stage_t0,
        label_count,
        labeled_voxels,
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
    # Marker seeds are now compact and may be single-voxel; requiring more
    # than one core voxel can incorrectly drop valid soma components.
    min_core_voxels = 1
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
