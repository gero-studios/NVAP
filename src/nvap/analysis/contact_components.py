"""Visible-contact segmentation with supported soma anchors."""
from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import skeletonize
from nvap.analysis.microglia_quality import _soma_geometry
from nvap.config.types import SomaCriteria

logger = logging.getLogger(__name__)
_CONTACT = ndi.generate_binary_structure(3, 1)
DEFAULT_INTERIOR_MIN_VOXELS = 128


# Physical floors apply only to the soma-free exception, not supported cells.
INTERIOR_MIN_VOLUME_UM3 = 20.0
INTERIOR_MIN_LENGTH_UM = 30.0
INTERIOR_MIN_ARM_UM = 3.0


def _skeleton_length(skeleton, spacing):
    length = 0.0
    for offset in np.ndindex(3, 3, 3):
        delta = np.asarray(offset) - 1
        if tuple(delta) <= (0, 0, 0):
            continue
        a = tuple(slice(max(0, -int(d)), min(n, n - int(d))) for n, d in zip(skeleton.shape, delta))
        b = tuple(slice(max(0, int(d)), min(n, n + int(d))) for n, d in zip(skeleton.shape, delta))
        length += np.count_nonzero(skeleton[a] & skeleton[b]) * float(np.linalg.norm(delta * spacing))
    return length


def _substantial_arbor(mask, spacing):
    """Require physical extent and several meaningful arms, not a lone twig.

    These are conservative inclusion heuristics, not soma detection. Skeleton
    neighbors measure morphology only; component ownership remains face-based.
    """
    physical = np.maximum(np.asarray(spacing, dtype=np.float64), 1e-6)
    if np.count_nonzero(mask) * float(np.prod(physical)) < INTERIOR_MIN_VOLUME_UM3:
        return False
    points = np.nonzero(mask)
    spans = np.asarray([int(p.max()) - int(p.min()) + 1 for p in points]) * physical
    if np.count_nonzero(spans >= 8.0) < 2:
        return False
    skeleton = skeletonize(np.ascontiguousarray(mask))
    degree = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3, 3), dtype=np.uint8), mode='constant') - skeleton
    if np.count_nonzero(skeleton & (degree == 1)) < 3:
        return False
    if _skeleton_length(skeleton, physical) < INTERIOR_MIN_LENGTH_UM:
        return False
    # Remove junctions, then count terminal arms with meaningful physical length.
    arms, _ = ndi.label(skeleton & (degree < 3), structure=np.ones((3, 3, 3)))
    terminal_ids = np.unique(arms[skeleton & (degree == 1)])
    objects = ndi.find_objects(arms)
    meaningful = 0
    for arm_id in terminal_ids:
        if arm_id and _skeleton_length(arms[objects[arm_id - 1]] == arm_id, physical) >= INTERIOR_MIN_ARM_UM:
            meaningful += 1
            if meaningful >= 3:
                return True
    return False


def _z_interior(mask, z_start, first, stop, minimum, spacing):
    if np.count_nonzero(mask) < max(2, int(minimum)):
        return False
    occupied = np.flatnonzero(np.any(mask, axis=(1, 2)))
    return bool(occupied.size and z_start + occupied[0] > first and z_start + occupied[-1] < stop - 1
                and _substantial_arbor(mask, spacing))


@dataclass(frozen=True)
class ComponentClassification:
    soma_ids: np.ndarray
    interior_ids: np.ndarray
    cell_ids: np.ndarray


def classify_components(volume, labels, order, *, threshold, spacing,
                        trim_first_slices=0, trim_last_slices=0,
                        interior_min_voxels=DEFAULT_INTERIOR_MIN_VOXELS, soma_criteria: SomaCriteria = SomaCriteria()):
    """Soma support or the user-requested Z-interior exception, never proximity.

    Only the first/last retained Z planes count for the interior rule. Lateral
    boundaries are deliberately not part of this user-defined inference. The
    soma-free exception additionally requires a substantial physical arbor.
    """
    first = min(volume.shape[0], max(0, int(trim_first_slices)))
    stop = max(first, volume.shape[0] - max(0, int(trim_last_slices)))
    if stop <= first:
        empty = np.empty(0, dtype=np.int32)
        return ComponentClassification(empty, empty, empty)
    values, local_labels = volume[first:stop], labels[first:stop]
    somas = soma_component_ids(values, local_labels, order, threshold=threshold, spacing=spacing, soma_criteria=soma_criteria)
    supported = set(somas.tolist())
    inferred, accepted = [], []
    objects = ndi.find_objects(local_labels)
    for cid in np.asarray(order).tolist():
        if cid in supported:
            accepted.append(cid)
            continue
        if cid <= 0 or cid > len(objects) or objects[cid - 1] is None:
            continue
        bounds = objects[cid - 1]
        mask = (local_labels[bounds] == cid) & np.isfinite(values[bounds]) & (values[bounds] >= threshold)
        if (not soma_criteria.required and _z_interior(mask, bounds[0].start, 0, len(values), interior_min_voxels, spacing)
            and ndi.label(mask, structure=_CONTACT)[1] == 1):
            inferred.append(cid)
            accepted.append(cid)
    return ComponentClassification(somas, np.asarray(inferred, dtype=np.int32), np.asarray(accepted, dtype=np.int32))


def prepare_contact_components(volume, *, threshold, spacing, labels=None, order=None,
                               trim_first_slices=0, trim_last_slices=0, compact_ids=True,
                               require_soma=True, interior_min_voxels=DEFAULT_INTERIOR_MIN_VOXELS, soma_criteria: SomaCriteria = SomaCriteria()):
    """Retain every visible object, split touching somas and enforce one island/ID.

    Contact means shared voxel faces at the displayed intensity cutoff.
    No gap closing, faint-path growth, proximity attachment or top-N truncation
    is applied here. With require_soma, accept supported somas or Z-interior
    candidates meeting interior_min_voxels. Otherwise retain all visible islands.
    Speck removal is a separate preprocessing operation.
    """
    values = np.asarray(volume)
    visible = np.isfinite(values) & (values >= float(threshold))
    first = min(values.shape[0], max(0, int(trim_first_slices)))
    stop = max(first, values.shape[0] - max(0, int(trim_last_slices)))
    visible[:first] = False
    visible[stop:] = False
    if labels is None:
        source, count = ndi.label(visible, structure=_CONTACT)
        ordered = np.arange(1, count + 1, dtype=np.int32)
    else:
        source = np.asarray(labels)
        if source.shape != values.shape:
            raise ValueError('labels and volume must share the same shape')
        ordered = np.asarray(order if order is not None else np.unique(source[source > 0]), dtype=np.int32)
    objects = ndi.find_objects(source)
    result = np.zeros(values.shape, dtype=np.int32)
    next_id = int(source.max())
    kept = []
    physical = np.maximum(np.asarray(spacing, dtype=np.float64), 1e-6)
    for cid in ordered.tolist():
        if cid <= 0 or cid > len(objects) or objects[cid - 1] is None:
            continue
        bounds = objects[cid - 1]
        owned = (source[bounds] == cid) & visible[bounds]
        islands, count = ndi.label(owned, structure=_CONTACT)
        sizes = np.bincount(islands.ravel())
        island_objects = ndi.find_objects(islands)
        first_piece = True
        for island_id in sorted(range(1, count + 1), key=lambda i: -sizes[i]):
            island_bounds = island_objects[island_id - 1]
            mask = islands[island_bounds] == island_id
            positions, cores, steps = ([], None, None)
            # Thin pieces cannot contain a supported body, but are still kept.
            if np.all(np.asarray(mask.shape) * physical >= 2.0 * soma_criteria.min_radius_um):
                positions, cores, steps = _soma_geometry(mask, physical, soma_criteria)
            interior = (require_soma and not soma_criteria.required and not positions and
                        _z_interior(mask, bounds[0].start + island_bounds[0].start,
                                    first, stop, interior_min_voxels, physical))
            if require_soma and not positions and not interior:
                continue
            if len(positions) > 1:
                index = np.ix_(*(np.minimum(np.arange(mask.shape[i]) // steps[i], cores.shape[i] - 1) for i in range(3)))
                markers = cores[index].copy()
                markers[~mask] = 0
                local_values = values[bounds][island_bounds]
                parts = watershed(-local_values, markers=markers, mask=mask, connectivity=_CONTACT)
            else:
                parts = mask.astype(np.int32)
            target = result[bounds][island_bounds]
            for part_id in range(1, int(parts.max()) + 1):
                pieces, piece_count = ndi.label(parts == part_id, structure=_CONTACT)
                for piece_id in range(1, piece_count + 1):
                    if require_soma and len(positions) > 1 and not np.any((pieces == piece_id) & (markers > 0)):
                        if soma_criteria.required or not _z_interior(pieces == piece_id, bounds[0].start + island_bounds[0].start,
                                           first, stop, interior_min_voxels, physical):
                            continue
                    if first_piece:
                        output_id = cid
                        first_piece = False
                    else:
                        next_id += 1
                        output_id = next_id
                    target[pieces == piece_id] = output_id
                    kept.append(output_id)
    sizes = np.bincount(result.ravel(), minlength=next_id + 1).astype(np.int64)
    sizes[0] = 0
    # Stable size order makes the largest components easiest to navigate.
    kept = np.asarray(sorted(kept, key=lambda cid: -sizes[cid]), dtype=np.int32)
    if compact_ids:
        remap = np.zeros(len(sizes), dtype=np.int32)
        remap[kept] = np.arange(1, len(kept) + 1)
        result = remap[result]
        sizes = np.r_[0, sizes[kept]].astype(np.int64)
        kept = np.arange(1, len(kept) + 1, dtype=np.int32)
    logger.info('Visible-contact segmentation: %d components, %d visible voxels retained; require_soma=%s.', len(kept), int(np.count_nonzero(result)), require_soma)
    return result, kept, sizes


def soma_component_ids(volume, labels, order, *, threshold, spacing, soma_criteria: SomaCriteria = SomaCriteria()):
    """Classify contact components for coloring without changing their IDs."""
    objects = ndi.find_objects(labels)
    physical = np.maximum(np.asarray(spacing, dtype=np.float64), 1e-6)
    supported = []
    for cid in np.asarray(order).tolist():
        if cid <= 0 or cid > len(objects) or objects[cid - 1] is None:
            continue
        bounds = objects[cid - 1]
        if np.any(np.asarray([s.stop - s.start for s in bounds]) * physical < 2.0 * soma_criteria.min_radius_um):
            continue
        mask = (labels[bounds] == cid) & (volume[bounds] >= threshold)
        if ndi.label(mask, structure=_CONTACT)[1] != 1:
            continue
        if _soma_geometry(mask, physical, soma_criteria)[0]:
            supported.append(cid)
    return np.asarray(supported, dtype=np.int32)
