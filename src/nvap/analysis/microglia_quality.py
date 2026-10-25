"""Shared soma eligibility and repair for cell rendering and analysis."""
from __future__ import annotations
import logging
import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import h_maxima

from nvap.config.types import SomaCriteria

logger = logging.getLogger(__name__)
_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)
_MAX_LOCAL_EDT_VOXELS = 8_000_000


def _supported_soma_core(core: np.ndarray, spacing: np.ndarray, criteria: SomaCriteria = SomaCriteria()) -> bool:
    """Reject tiny or essentially planar/filamentary thickness peaks."""
    coords = np.argwhere(core)
    if len(coords) < 3 or len(coords) * float(np.prod(spacing)) < criteria.min_core_volume_um3:
        return False
    physical = coords * spacing
    if np.any(np.ptp(coords, axis=0) == 0):
        return False
    eigenvalues = np.linalg.eigvalsh(np.cov(physical, rowvar=False))
    return bool(eigenvalues[-1] > 0 and eigenvalues[0] >= 0.15**2 * eigenvalues[-1])


def _soma_geometry(mask: np.ndarray, spacing: np.ndarray, criteria: SomaCriteria = SomaCriteria()):
    """Join patchy core lobes within one supported soma body."""
    steps = np.ones(3, dtype=np.int32)
    while int(np.prod((np.asarray(mask.shape) + steps - 1) // steps + 2)) > _MAX_LOCAL_EDT_VOXELS:
        available = np.asarray(mask.shape) > steps
        if not np.any(available):
            break
        axis = int(np.argmin(np.where(available, spacing * steps, np.inf)))
        steps[axis] *= 2
    sample = tuple(slice(None, None, int(s)) for s in steps)
    local = mask[sample]
    physical_spacing = spacing * steps
    distance = ndi.distance_transform_edt(np.pad(local, 1), sampling=tuple(physical_spacing))[1:-1, 1:-1, 1:-1]
    radius = float(distance.max()) if distance.size else 0.0
    if radius < criteria.min_radius_um:
        return [], None, steps
    # A shared body envelope can connect distinct somas through a thick
    # process. Use prominent thickness peaks instead, and merge only peaks
    # whose inscribed soma interiors overlap (patchy lobes of one body).
    envelopes, envelope_count = ndi.label(
        local & (distance >= criteria.min_radius_um * 0.7), structure=_STRUCTURE,
    )
    # Retain smaller 3D bodies, but reject single-plane knots and tiny peaks.
    minimum_core_support = max(criteria.min_core_volume_um3, 3 * float(np.prod(physical_spacing)))
    support = np.bincount(envelopes[distance >= criteria.min_radius_um], minlength=envelope_count + 1)
    envelope_ids = np.flatnonzero(support * float(np.prod(physical_spacing)) >= minimum_core_support)
    envelope_ids = envelope_ids[envelope_ids > 0]
    objects = ndi.find_objects(envelopes)
    candidates = []
    candidate_envelopes = []
    # Peak reconstruction belongs inside compact body envelopes, not the
    # entire large bounding box occupied by thin distant processes.
    for envelope_id in envelope_ids:
        bounds = objects[envelope_id - 1]
        values = np.where(envelopes[bounds] == envelope_id, distance[bounds], 0.0)
        peaks, peak_count = ndi.label(
            h_maxima(np.pad(values, 1), criteria.min_radius_um * 0.5)[1:-1, 1:-1, 1:-1]
            & (values >= criteria.min_radius_um), structure=_STRUCTURE,
        )
        positions = ndi.maximum_position(values, labels=peaks, index=np.arange(1, peak_count + 1))
        offset = np.array([s.start for s in bounds])
        candidates.extend(tuple(np.asarray(position) + offset) for position in positions)
        candidate_envelopes.extend([int(envelope_id)] * len(positions))
    if not candidates:
        return [], None, steps
    radii = np.asarray([distance[position] for position in candidates])
    selected = []
    for index in np.argsort(-radii, kind='stable'):
        center = np.asarray(candidates[index]) * physical_spacing
        if any(
            np.linalg.norm(center - np.asarray(candidates[other]) * physical_spacing)
            < 0.98 * (radii[index] + radii[other])
            for other in selected
        ):
            continue
        selected.append(int(index))
    bodies = np.zeros(local.shape, dtype=np.int32)
    for envelope_id in envelope_ids:
        bounds = objects[envelope_id - 1]
        mask = envelopes[bounds] == envelope_id
        selected_ids = [(label_id, index) for label_id, index in enumerate(selected, start=1) if candidate_envelopes[index] == envelope_id]
        if len(selected_ids) == 1:
            bodies[bounds][mask] = selected_ids[0][0]
        elif selected_ids:
            seeds = np.zeros(mask.shape, dtype=np.int32)
            offset = np.array([s.start for s in bounds])
            for label_id, index in selected_ids:
                seeds[tuple(np.asarray(candidates[index]) - offset)] = label_id
            parts = watershed(-distance[bounds], markers=seeds, mask=mask, connectivity=_STRUCTURE)
            bodies[bounds][mask] = parts[mask]
    count = len(selected)
    # Each soma uses its own radius; a large neighboring body must not suppress
    # a smaller real soma through a component-wide relative thickness cutoff.
    floors = np.maximum(criteria.min_radius_um, np.r_[0.0, radii[selected]] * 0.45)
    core = (bodies > 0) & (distance >= floors[bodies])
    support = np.bincount(bodies[core], minlength=count + 1)
    valid = np.flatnonzero(support * float(np.prod(physical_spacing)) >= minimum_core_support)
    valid = valid[valid > 0]
    valid = np.asarray([cid for cid in valid if _supported_soma_core(core & (bodies == cid), physical_spacing, criteria)], dtype=np.int32)
    if not valid.size:
        return [], None, steps
    positions = [candidates[selected[label_id - 1]] for label_id in valid]
    positions = [tuple(int(x) for x in np.asarray(position) * steps) for position in positions]
    lut = np.zeros(count + 1, dtype=np.int32)
    lut[valid] = np.arange(1, len(valid) + 1)
    markers = lut[bodies]
    markers[~core] = 0
    return positions, markers, steps


def _soma_positions(mask: np.ndarray, spacing: np.ndarray) -> list[tuple[int, int, int]]:
    return _soma_geometry(mask, spacing)[0]


def _component_bodies(volume, labels, order, threshold, spacing, trim_first_slices, trim_last_slices, *, include_cores=False, stable_ids=frozenset()):
    objects = ndi.find_objects(labels)
    first = max(0, int(trim_first_slices))
    stop = max(first, volume.shape[0] - max(0, int(trim_last_slices)))
    for cid in np.asarray(order, dtype=np.int32).tolist():
        if cid <= 0 or cid > len(objects) or objects[cid - 1] is None:
            continue
        bounds = list(objects[cid - 1])
        bounds[0] = slice(max(first, bounds[0].start), min(stop, bounds[0].stop))
        if bounds[0].start >= bounds[0].stop:
            continue
        bounds = tuple(bounds)
        if include_cores and cid in stable_ids:
            yield cid, bounds, None, None, None
            continue
        visible = (labels[bounds] == cid) & (volume[bounds] >= threshold)
        if include_cores:
            positions, cores, steps = _soma_geometry(visible, spacing)
            yield cid, bounds, positions, cores, steps
        else:
            yield cid, bounds, _soma_positions(visible, spacing)


def analyzable_component_ids(volume, labels, order, *, threshold, spacing, trim_first_slices=0, trim_last_slices=0,
                            interior_min_voxels=128, soma_criteria: SomaCriteria = SomaCriteria()):
    """Existing IDs eligible by soma support or the Z-interior inference rule."""
    from nvap.analysis.contact_components import classify_components
    return classify_components(volume, labels, order, threshold=threshold, spacing=spacing,
        trim_first_slices=trim_first_slices, trim_last_slices=trim_last_slices,
        interior_min_voxels=interior_min_voxels, soma_criteria=soma_criteria).cell_ids


def filter_analyzable_components(volume, labels, order, sizes, *, threshold, spacing,
                                 trim_first_slices=0, trim_last_slices=0, compact_ids=True,
                                 interior_min_voxels=128, soma_criteria: SomaCriteria = SomaCriteria()):
    """Contact repair with soma support or the Z-interior inclusion exception."""
    from nvap.analysis.contact_components import prepare_contact_components
    return prepare_contact_components(
        volume, labels=labels, order=order, threshold=threshold, spacing=spacing,
        trim_first_slices=trim_first_slices, trim_last_slices=trim_last_slices,
        compact_ids=compact_ids,
        interior_min_voxels=interior_min_voxels, soma_criteria=soma_criteria,
    )
