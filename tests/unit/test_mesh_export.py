"""Tests for 3D mesh reconstruction and export."""
from __future__ import annotations

import numpy as np
import pytest

from nvap.config.types import ChannelVolume, DatasetVolume, MeshExportConfig, VoxelSpacing
from nvap.export.mesh_export import (
    export_mesh,
    extract_isosurface,
    laplacian_smooth,
    reconstruct_channel_mesh,
)


def _make_sphere_volume(radius: int = 8, size: int = 24) -> np.ndarray:
    """Create a 3D sphere volume for testing."""
    z, y, x = np.mgrid[:size, :size, :size]
    center = size // 2
    dist = np.sqrt((z - center)**2 + (y - center)**2 + (x - center)**2)
    volume = np.clip(1.0 - dist / radius, 0.0, 1.0).astype(np.float32)
    return volume


def test_extract_isosurface_produces_mesh() -> None:
    volume = _make_sphere_volume()
    spacing = VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0)
    verts, faces, normals = extract_isosurface(volume, spacing, iso_level=0.3)
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert len(verts) > 10
    assert len(faces) > 10


def test_laplacian_smooth_does_not_explode() -> None:
    volume = _make_sphere_volume()
    spacing = VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0)
    verts, faces, _ = extract_isosurface(volume, spacing, iso_level=0.3)
    smoothed = laplacian_smooth(verts, faces, iterations=5, relaxation=0.1)
    assert smoothed.shape == verts.shape
    assert np.all(np.isfinite(smoothed))
    # Smoothed should not move vertices drastically
    max_displacement = np.max(np.linalg.norm(smoothed - verts, axis=1))
    assert max_displacement < float(np.max(verts) - np.min(verts))


def test_export_mesh_ply(tmp_path) -> None:
    volume = _make_sphere_volume()
    spacing = VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0)
    verts, faces, normals = extract_isosurface(volume, spacing, iso_level=0.3)
    path = export_mesh(verts, faces, normals, tmp_path / "test.ply", fmt="ply")
    assert path.exists()
    assert path.stat().st_size > 100


def test_export_mesh_obj(tmp_path) -> None:
    volume = _make_sphere_volume()
    spacing = VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0)
    verts, faces, normals = extract_isosurface(volume, spacing, iso_level=0.3)
    path = export_mesh(verts, faces, normals, tmp_path / "test.obj", fmt="obj")
    assert path.exists()
    content = path.read_text()
    assert "v " in content
    assert "f " in content


def test_export_mesh_stl(tmp_path) -> None:
    volume = _make_sphere_volume()
    spacing = VoxelSpacing(x_um=1.0, y_um=1.0, z_um=1.0)
    verts, faces, normals = extract_isosurface(volume, spacing, iso_level=0.3)
    path = export_mesh(verts, faces, normals, tmp_path / "test.stl", fmt="stl")
    assert path.exists()
    assert path.stat().st_size > 100


def test_reconstruct_channel_mesh() -> None:
    volume = _make_sphere_volume()
    spacing = VoxelSpacing(x_um=0.331, y_um=0.331, z_um=0.4)
    channel = ChannelVolume("green", volume, list(range(volume.shape[0])), spacing)
    config = MeshExportConfig(smooth_iterations=3, decimate_fraction=1.0)
    verts, faces, normals = reconstruct_channel_mesh(channel, config)
    assert len(verts) > 0
    assert len(faces) > 0
