from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class VoxelSpacing:
    x_um: float = 0.331
    y_um: float = 0.331
    z_um: float = 0.4

    @property
    def voxel_volume_um3(self) -> float:
        return self.x_um * self.y_um * self.z_um


DEFAULT_SPACING = VoxelSpacing()


@dataclass
class ChannelVolume:
    name: Literal["green", "red"]
    data: np.ndarray
    z_indices: list[int]
    spacing: VoxelSpacing = DEFAULT_SPACING

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError("ChannelVolume.data must be 3D in (z, y, x) order.")
        if self.data.shape[0] != len(self.z_indices):
            raise ValueError("z_indices length must match data z dimension.")


@dataclass
class DatasetVolume:
    green: ChannelVolume
    red: ChannelVolume
    shared_z_range: tuple[int, int]


@dataclass(frozen=True)
class PSFConfig:
    enabled: bool = True
    sigma_xy_um: float = 0.35
    sigma_z_um: float = 0.7
    iterations: int = 5
    use_measured_psf: bool = True
    measured_psf_path: str = ""
    regularization_lambda: float = 0.002
    tv_regularization: bool = True
    tv_weight: float = 0.015


@dataclass(frozen=True)
class PreprocessConfig:
    enabled: bool = True
    flatfield_sigma_xy: float = 32.0
    contrast_low_pct: float = 0.5
    contrast_high_pct: float = 99.8
    denoise_method: Literal["anisotropic", "bilateral", "non_local_means", "wavelet", "none"] = "wavelet"
    denoise_strength: float = 0.012
    green_denoise_multiplier: float = 1.9
    preserve_branches: bool = True
    green_denoise_strategy: Literal[
        "microglia_masking",
        "hybrid_auto",
        "classical_branch_aware",
        "pixel2voxel_no_psf",
        "bm4d",
        "noise2void",
        "legacy_anisotropic",
    ] = "pixel2voxel_no_psf"
    green_noise_model: Literal["auto", "poisson_gaussian", "gaussian"] = "auto"
    green_branch_protection: float = 0.72
    green_nlm_patch_size: int = 3
    green_nlm_patch_distance: int = 4
    green_nlm_h_factor: float = 0.9
    green_apply_vst: bool = True
    green_pre_deconv_strength: float = 0.65
    green_post_deconv_strength: float = 0.15
    green_speckle_min_voxels: int = 10
    green_speckle_attenuation: float = 0.12
    green_noise2void_model_path: str = ""
    green_chunked_processing: bool = True
    green_chunk_depth: int = 48
    green_chunk_overlap: int = 4
    cpu_worker_threads: int = 0
    resample_for_mesh: bool = True
    mesh_target_z_um: float = 0.331
    wavelet_level: int = 0
    wavelet_name: str = "db4"


@dataclass(frozen=True)
class RenderConfig:
    threshold_green: float = 0.15
    threshold_red: float = 0.15
    opacity_green: float = 0.35
    opacity_red: float = 0.35
    iso_green: float = 0.25
    iso_red: float = 0.25
    display_z_scale: float = 2.0 / 3.0
    trim_first_slices: int = 20
    trim_last_slices: int = 20
    offset_x_um: float = 0.0
    offset_y_um: float = 0.0
    offset_z_um: float = 0.0
    show_green: bool = True
    show_red: bool = True
    show_iso_green: bool = False
    show_iso_red: bool = False


@dataclass(frozen=True)
class MeshExportConfig:
    """Configuration for 3D mesh export."""
    enabled: bool = True
    iso_level_green: float = 0.18
    iso_level_red: float = 0.18
    smooth_iterations: int = 25
    smooth_relaxation: float = 0.15
    decimate_fraction: float = 0.35
    export_format: Literal["ply", "obj", "stl"] = "ply"
    poisson_depth: int = 8
    use_poisson: bool = False


@dataclass
class MetricsResult:
    channel: str
    voxel_count: int
    volume_um3: float
    component_count: int
    largest_component_voxels: int
    overlap_voxel_count: int = 0
    overlap_volume_um3: float = 0.0


@dataclass
class MetricsComputation:
    channel_results: list[MetricsResult] = field(default_factory=list)
    overlap_voxel_count: int = 0
    overlap_volume_um3: float = 0.0
