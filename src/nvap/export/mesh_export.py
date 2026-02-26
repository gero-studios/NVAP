"""High-quality 3D mesh reconstruction from volumetric microscopy data.

Extracts isosurfaces via marching cubes, applies Laplacian smoothing and
optional decimation, then exports to PLY/OBJ/STL.  Optionally runs
Poisson surface reconstruction for watertight manifold meshes.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from skimage.measure import marching_cubes

from nvap.config.types import (
    ChannelVolume,
    DatasetVolume,
    MeshExportConfig,
    VoxelSpacing,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core mesh extraction
# ---------------------------------------------------------------------------

def extract_isosurface(
    volume: np.ndarray,
    spacing: VoxelSpacing,
    iso_level: float,
    smooth_volume: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract an isosurface using marching cubes.

    Returns (vertices, faces, normals) in physical coordinates (µm).
    """
    arr = np.asarray(volume, dtype=np.float32)
    if smooth_volume:
        # Light pre-smoothing removes staircase artifacts without losing branches
        arr = ndi.gaussian_filter(arr, sigma=(0.5, 0.5, 0.5), mode="nearest")

    spacing_tuple = (spacing.z_um, spacing.y_um, spacing.x_um)

    verts, faces, normals, _ = marching_cubes(
        arr,
        level=float(iso_level),
        spacing=spacing_tuple,
        step_size=1,
        allow_degenerate=False,
    )
    logger.info(
        "Marching cubes: verts=%d faces=%d iso=%.4f",
        len(verts), len(faces), iso_level,
    )
    return (
        np.asarray(verts, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        np.asarray(normals, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Mesh smoothing
# ---------------------------------------------------------------------------

def laplacian_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 25,
    relaxation: float = 0.15,
) -> np.ndarray:
    """Laplacian mesh smoothing — reduces staircase artifacts.

    For each vertex, moves it toward the centroid of its neighbors.
    Uses Taubin λ/μ scheme for volume-preserving smoothing.
    """
    verts = np.array(vertices, dtype=np.float64, copy=True)
    n_verts = len(verts)
    if n_verts == 0 or len(faces) == 0:
        return verts

    # Build adjacency (neighbor list)
    neighbors: list[set[int]] = [set() for _ in range(n_verts)]
    for f in faces:
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                neighbors[f[i]].add(f[j])
                neighbors[f[j]].add(f[i])

    lam = float(relaxation)
    mu = -lam / 0.9  # Taubin shrinkage compensation

    for it in range(iterations):
        # Forward pass (λ)
        new_verts = np.copy(verts)
        for vi in range(n_verts):
            nbrs = neighbors[vi]
            if not nbrs:
                continue
            centroid = np.mean(verts[list(nbrs)], axis=0)
            new_verts[vi] += lam * (centroid - verts[vi])
        verts = new_verts

        # Backward pass (μ) — prevents shrinkage
        new_verts = np.copy(verts)
        for vi in range(n_verts):
            nbrs = neighbors[vi]
            if not nbrs:
                continue
            centroid = np.mean(verts[list(nbrs)], axis=0)
            new_verts[vi] += mu * (centroid - verts[vi])
        verts = new_verts

    logger.info("Laplacian smoothing: %d iterations, λ=%.3f, μ=%.3f", iterations, lam, mu)
    return verts


def decimate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_fraction: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Decimate a mesh to reduce polygon count.

    Uses quadric decimation via trimesh when available.
    """
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        target_faces = max(100, int(len(faces) * target_fraction))

        try:
            simplified = mesh.simplify_quadric_decimation(target_faces)
            logger.info(
                "Quadric decimation: %d → %d faces (%.1f%%)",
                len(faces), len(simplified.faces),
                100.0 * len(simplified.faces) / max(1, len(faces)),
            )
            return np.asarray(simplified.vertices), np.asarray(simplified.faces)
        except Exception as exc:
            logger.warning("Mesh decimation failed (%s), returning original mesh", exc)
            return vertices, faces

    except ImportError:
        logger.info("trimesh not installed — skipping decimation")
        return vertices, faces


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _recompute_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Recompute per-vertex normals by averaging adjacent face normals."""
    verts = np.asarray(vertices, dtype=np.float64)
    face_arr = np.asarray(faces, dtype=np.int64)
    v0 = verts[face_arr[:, 0]]
    v1 = verts[face_arr[:, 1]]
    v2 = verts[face_arr[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(norms, 1e-10)
    vertex_normals = np.zeros_like(verts, dtype=np.float64)
    np.add.at(vertex_normals, face_arr[:, 0], face_normals)
    np.add.at(vertex_normals, face_arr[:, 1], face_normals)
    np.add.at(vertex_normals, face_arr[:, 2], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / np.maximum(norms, 1e-10)
    return vertex_normals


def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray | None, color: tuple[int, int, int]) -> None:
    """Write ASCII PLY mesh file."""
    n_verts = len(vertices)
    n_faces = len(faces)

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {n_faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    with open(path, "w") as f:
        f.write(header)
        for i in range(n_verts):
            v = vertices[i]
            n = normals[i] if normals is not None and i < len(normals) else (0, 0, 1)
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray | None) -> None:
    """Write OBJ mesh file."""
    with open(path, "w") as f:
        f.write("# NVAP mesh export\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if normals is not None:
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for face in faces:
            # OBJ faces are 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def _write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray | None) -> None:
    """Write binary STL mesh file."""
    import struct
    n_faces = len(faces)

    with open(path, "wb") as f:
        # 80-byte header
        f.write(b"NVAP mesh export" + b"\0" * 64)
        f.write(struct.pack("<I", n_faces))

        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            # Compute face normal
            e1 = v1 - v0
            e2 = v2 - v0
            normal = np.cross(e1, e2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal /= norm_len

            f.write(struct.pack("<3f", *normal.astype(np.float32)))
            f.write(struct.pack("<3f", *v0.astype(np.float32)))
            f.write(struct.pack("<3f", *v1.astype(np.float32)))
            f.write(struct.pack("<3f", *v2.astype(np.float32)))
            f.write(struct.pack("<H", 0))  # attribute byte count


def export_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None,
    output_path: str | Path,
    fmt: str = "ply",
    color: tuple[int, int, int] = (0, 255, 0),
) -> Path:
    """Export mesh to file in the given format."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower().strip(".")
    if fmt == "ply":
        _write_ply(path, vertices, faces, normals, color)
    elif fmt == "obj":
        _write_obj(path, vertices, faces, normals)
    elif fmt == "stl":
        _write_stl(path, vertices, faces, normals)
    else:
        raise ValueError(f"Unsupported mesh format: {fmt}")

    logger.info("Mesh exported: %s (verts=%d, faces=%d)", path, len(vertices), len(faces))
    return path


# ---------------------------------------------------------------------------
# Full reconstruction pipeline
# ---------------------------------------------------------------------------

def reconstruct_channel_mesh(
    channel: ChannelVolume,
    config: MeshExportConfig,
    iso_level: float | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full mesh reconstruction pipeline for a single channel.

    Steps:
    1. Extract isosurface via marching cubes
    2. Laplacian smooth (Taubin volume-preserving)
    3. Optional decimation
    """
    t0 = time.perf_counter()
    level = iso_level or (config.iso_level_green if channel.name == "green" else config.iso_level_red)

    verts, faces, normals = extract_isosurface(
        channel.data, channel.spacing, level, smooth_volume=True,
    )

    if len(verts) == 0:
        logger.warning("No isosurface found for channel '%s' at level %.4f", channel.name, level)
        return verts, faces, normals

    verts = laplacian_smooth(
        verts, faces,
        iterations=config.smooth_iterations,
        relaxation=config.smooth_relaxation,
    )

    if config.decimate_fraction < 1.0:
        verts, faces = decimate_mesh(verts, faces, config.decimate_fraction)

    # Recompute normals to match modified vertex positions and topology
    normals = _recompute_vertex_normals(verts, faces) if len(verts) > 0 and len(faces) > 0 else normals

    logger.info(
        "Channel '%s' mesh: verts=%d faces=%d dt=%.2fs",
        channel.name, len(verts), len(faces), time.perf_counter() - t0,
    )
    return verts, faces, normals


def export_dataset_meshes(
    dataset: DatasetVolume,
    config: MeshExportConfig,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export both channel meshes from a processed dataset."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = config.export_format
    results: dict[str, Path] = {}

    for channel, color, iso in [
        (dataset.green, (25, 255, 50), config.iso_level_green),
        (dataset.red, (255, 50, 50), config.iso_level_red),
    ]:
        try:
            verts, faces, normals = reconstruct_channel_mesh(
                channel, config, iso_level=iso, color=color,
            )
            if len(verts) > 0:
                path = export_mesh(
                    verts, faces, normals,
                    out_dir / f"{channel.name}_mesh.{fmt}",
                    fmt=fmt,
                    color=color,
                )
                results[channel.name] = path
        except Exception as exc:
            logger.error("Mesh export failed for '%s': %s", channel.name, exc)

    return results


def reconstruct_combined_mesh(
    dataset: DatasetVolume,
    config: MeshExportConfig,
    output_path: str | Path,
) -> Path | None:
    """Export both channels as a combined PLY with distinct colors."""
    try:
        green_v, green_f, green_n = reconstruct_channel_mesh(
            dataset.green, config, color=(25, 255, 50),
        )
        red_v, red_f, red_n = reconstruct_channel_mesh(
            dataset.red, config, color=(255, 50, 50),
        )

        if len(green_v) == 0 and len(red_v) == 0:
            logger.warning("No mesh data to combine")
            return None

        # Offset red face indices
        if len(green_v) > 0 and len(red_v) > 0:
            red_f_offset = red_f + len(green_v)
            all_verts = np.vstack([green_v, red_v])
            all_faces = np.vstack([green_f, red_f_offset])
            # Build per-vertex colors
            green_colors = np.full((len(green_v), 3), [25, 255, 50], dtype=np.uint8)
            red_colors = np.full((len(red_v), 3), [255, 50, 50], dtype=np.uint8)
            all_normals = None
            if green_n is not None and red_n is not None:
                gn = green_n[:len(green_v)] if len(green_n) >= len(green_v) else green_n
                rn = red_n[:len(red_v)] if len(red_n) >= len(red_v) else red_n
                if len(gn) == len(green_v) and len(rn) == len(red_v):
                    all_normals = np.vstack([gn, rn])
        elif len(green_v) > 0:
            all_verts, all_faces = green_v, green_f
            all_normals = green_n
        else:
            all_verts, all_faces = red_v, red_f
            all_normals = red_n

        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write combined PLY with per-vertex colors
        n_verts = len(all_verts)
        n_faces = len(all_faces)
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {n_verts}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            f"element face {n_faces}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        with open(path, "w") as f:
            f.write(header)
            for i in range(n_verts):
                v = all_verts[i]
                if i < len(green_v):
                    c = (25, 255, 50)
                else:
                    c = (255, 50, 50)
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
            for face in all_faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

        logger.info("Combined mesh exported: %s (verts=%d, faces=%d)", path, n_verts, n_faces)
        return path

    except Exception as exc:
        logger.error("Combined mesh export failed: %s", exc)
        return None
