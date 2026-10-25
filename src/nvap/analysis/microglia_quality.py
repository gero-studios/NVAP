"""Shared single-soma eligibility for cell rendering and quantitative analysis."""
from __future__ import annotations

import logging
import numpy as np
import scipy.ndimage as ndi

logger = logging.getLogger(__name__)
_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)
_MAX_LOCAL_EDT_VOXELS = 8_000_000
_MIN_SOMA_RADIUS_UM = 1.0
_MIN_SOMA_CORE_VOLUME_UM3 = 1.0


def analyzable_component_ids(
    volume: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
    *,
    threshold: float,
    spacing: tuple[float, float, float],
    trim_first_slices: int = 0,
    trim_last_slices: int = 0,
) -> np.ndarray:
    """Keep complete component IDs with exactly one supported visible soma.

    A soma needs a physically thick core with at least 1 cubic micron of
    support. Multiple supported cores are ambiguous merged cells. Evaluate
    only visible, untrimmed signal, preserving the entire accepted arbor.
    """
    ordered = np.asarray(order, dtype=np.int32)
    if not ordered.size:
        return ordered.copy()
    spacing_zyx = np.maximum(np.asarray(spacing, dtype=np.float64), 1e-6)
    objects = ndi.find_objects(labels)
    first = max(0, int(trim_first_slices))
    stop = max(first, volume.shape[0] - max(0, int(trim_last_slices)))
    accepted = []
    for cid in ordered.tolist():
        if cid <= 0 or cid > len(objects) or objects[cid - 1] is None:
            continue
        bounds = list(objects[cid - 1])
        bounds[0] = slice(max(first, bounds[0].start), min(stop, bounds[0].stop))
        if bounds[0].start >= bounds[0].stop:
            continue
        bounds = tuple(bounds)
        mask = (labels[bounds] == cid) & (volume[bounds] >= threshold)
        steps = np.ones(3, dtype=np.int32)
        while int(np.prod((np.asarray(mask.shape) + steps - 1) // steps + 2)) > _MAX_LOCAL_EDT_VOXELS:
            available = np.asarray(mask.shape) > steps
            if not np.any(available):
                break
            axis = int(np.argmin(np.where(available, spacing_zyx * steps, np.inf)))
            steps[axis] *= 2
        sample = tuple(slice(None, None, int(s)) for s in steps)
        local = mask[sample]
        physical_spacing = spacing_zyx * steps
        distance = ndi.distance_transform_edt(
            np.pad(local, 1), sampling=tuple(physical_spacing),
        )[1:-1, 1:-1, 1:-1]
        radius = float(distance.max()) if distance.size else 0.0
        if radius < _MIN_SOMA_RADIUS_UM:
            continue
        cores, count = ndi.label(
            local & (distance >= max(_MIN_SOMA_RADIUS_UM, radius * 0.45)),
            structure=_STRUCTURE,
        )
        core_sizes = np.bincount(cores.ravel(), minlength=count + 1)[1:]
        soma_count = int(np.count_nonzero(
            core_sizes * float(np.prod(physical_spacing)) >= _MIN_SOMA_CORE_VOLUME_UM3
        ))
        if soma_count == 1:
            accepted.append(cid)
    logger.info("Microglia soma quality: accepted %d/%d single-soma cells.", len(accepted), len(ordered))
    return np.asarray(accepted, dtype=np.int32)


def filter_analyzable_components(
    volume: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
    sizes: np.ndarray,
    *,
    threshold: float,
    spacing: tuple[float, float, float],
    trim_first_slices: int = 0,
    trim_last_slices: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove rejected labels and compact IDs for display and subsequent analysis."""
    keep = analyzable_component_ids(
        volume, labels, order, threshold=threshold, spacing=spacing,
        trim_first_slices=trim_first_slices, trim_last_slices=trim_last_slices,
    )
    lut = np.zeros(len(sizes), dtype=np.int32)
    lut[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    filtered = lut[labels]
    return (
        filtered,
        np.arange(1, len(keep) + 1, dtype=np.int32),
        np.concatenate((np.array([0], dtype=np.int64), np.asarray(sizes)[keep])),
    )
