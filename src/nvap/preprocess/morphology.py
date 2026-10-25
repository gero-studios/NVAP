"""Binary preprocessing kernels with explicit, neutral image-border handling.

The common odd-footprint path is the same SciPy operation used by skimage,
without deprecated wrappers or repeated warning/stack inspection per slice.
Unusual footprints retain the library implementation and its origin rules.
"""
from __future__ import annotations

import inspect
import numpy as np
import scipy.ndimage as ndi
from skimage import morphology as _reference

_OBJECTS_HAS_MAX_SIZE = 'max_size' in inspect.signature(_reference.remove_small_objects).parameters


def _simple(footprint) -> bool:
    return footprint is None or (
        isinstance(footprint, np.ndarray)
        and all(size % 2 for size in footprint.shape)
    )


def binary_dilation(image, footprint=None, out=None):
    if not _simple(footprint):
        return _reference.binary_dilation(image, footprint=footprint, out=out)
    return ndi.binary_dilation(image, structure=footprint, output=out, border_value=0)


def binary_opening(image, footprint=None, out=None):
    if not _simple(footprint):
        return _reference.binary_opening(image, footprint=footprint, out=out)
    eroded = ndi.binary_erosion(image, structure=footprint, border_value=1)
    return ndi.binary_dilation(eroded, structure=footprint, output=out, border_value=0)


def binary_closing(image, footprint=None, out=None):
    if not _simple(footprint):
        return _reference.binary_closing(image, footprint=footprint, out=out)
    dilated = ndi.binary_dilation(image, structure=footprint, border_value=0)
    return ndi.binary_erosion(dilated, structure=footprint, output=out, border_value=1)


def remove_small_objects(ar, min_size=64, connectivity=1, *, out=None):
    # Match the installed library's deprecated-keyword translation exactly.
    # Older skimage versions without max_size retain their original min_size.
    kwargs = {'max_size': int(min_size)} if _OBJECTS_HAS_MAX_SIZE else {'min_size': min_size}
    return _reference.remove_small_objects(ar, connectivity=connectivity, out=out, **kwargs)


def remove_small_holes(ar, area_threshold=64, connectivity=1, *, out=None):
    if np.asarray(ar).dtype != bool:
        return _reference.remove_small_holes(ar, area_threshold=area_threshold, connectivity=connectivity, out=out)
    if out is None:
        out = np.empty_like(ar, dtype=bool)
    elif out.dtype != bool:
        raise TypeError('out dtype must be bool')
    np.logical_not(ar, out=out)
    remove_small_objects(out, min_size=area_threshold, connectivity=connectivity, out=out)
    np.logical_not(out, out=out)
    return out
