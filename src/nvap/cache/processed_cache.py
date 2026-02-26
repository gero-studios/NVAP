from __future__ import annotations

from hashlib import sha256
import logging
from pathlib import Path

import numpy as np

from nvap.config.types import ChannelVolume, DatasetVolume, PSFConfig, PreprocessConfig, VoxelSpacing

logger = logging.getLogger(__name__)

CACHE_VERSION = 4
CACHE_DIRNAME = ".nvap_cache"


def cache_dir(base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    path = root / CACHE_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_processed_cache(base_dir: str | Path | None = None) -> tuple[int, Path]:
    """Delete only NVAP processed cache files inside .nvap_cache.

    Safety rules:
    - Never traverses outside ``<base_dir>/.nvap_cache``.
    - Never deletes directories.
    - Deletes only files matching ``processed_*.npz``.
    - Skips symlinks.
    """
    root = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    target = (root / CACHE_DIRNAME).resolve()
    if target.name != CACHE_DIRNAME:
        raise ValueError(f"Refusing to clear cache outside '{CACHE_DIRNAME}': {target}")

    if not target.exists():
        return 0, target
    if not target.is_dir():
        raise ValueError(f"Cache path is not a directory: {target}")
    if target.is_symlink():
        raise ValueError(f"Refusing to clear cache symlink directory: {target}")

    removed = 0
    for child in target.iterdir():
        if child.is_symlink():
            continue
        if child.is_file() and child.suffix.lower() == ".npz" and child.name.startswith("processed_"):
            child.unlink()
            removed += 1
    logger.info("Cleared processed cache files: removed=%d path=%s", removed, target)
    return removed, target


def _directory_signature(channel_dir: Path) -> str:
    files = sorted(channel_dir.glob("*.png"), key=lambda p: p.name.lower())
    h = sha256()
    for file_path in files:
        stat = file_path.stat()
        h.update(file_path.name.encode("utf-8", errors="ignore"))
        h.update(str(stat.st_size).encode("ascii"))
        h.update(str(stat.st_mtime_ns).encode("ascii"))
    return h.hexdigest()


def build_dataset_signature(channel_dirs: dict[str, Path]) -> str:
    h = sha256()
    for channel_name in ("green", "red"):
        if channel_name not in channel_dirs:
            raise ValueError("channel_dirs must include both 'green' and 'red'.")
        channel_dir = channel_dirs[channel_name].resolve()
        h.update(channel_name.encode("ascii"))
        h.update(str(channel_dir).encode("utf-8", errors="ignore"))
        h.update(_directory_signature(channel_dir).encode("ascii"))
    return h.hexdigest()


def build_processed_cache_key(
    dataset_signature: str,
    spacing: VoxelSpacing,
    psf_config: PSFConfig,
    preprocess_config: PreprocessConfig | None = None,
) -> str:
    h = sha256()
    h.update(f"v{CACHE_VERSION}".encode("ascii"))
    h.update(dataset_signature.encode("ascii"))
    h.update(f"{spacing.x_um:.9f},{spacing.y_um:.9f},{spacing.z_um:.9f}".encode("ascii"))
    h.update(
        f"{int(psf_config.enabled)},{psf_config.sigma_xy_um:.9f},"
        f"{psf_config.sigma_z_um:.9f},{int(psf_config.iterations)},"
        f"{int(psf_config.use_measured_psf)},{psf_config.measured_psf_path},"
        f"{psf_config.regularization_lambda:.9f},"
        f"{int(getattr(psf_config, 'tv_regularization', False))},"
        f"{getattr(psf_config, 'tv_weight', 0.015):.9f}".encode("utf-8", errors="ignore")
    )
    if preprocess_config is not None:
        h.update(
            (
                f"{int(preprocess_config.enabled)},"
                f"{preprocess_config.flatfield_sigma_xy:.6f},"
                f"{preprocess_config.contrast_low_pct:.6f},"
                f"{preprocess_config.contrast_high_pct:.6f},"
                f"{preprocess_config.denoise_method},"
                f"{preprocess_config.denoise_strength:.6f},"
                f"{preprocess_config.green_denoise_multiplier:.6f},"
                f"{int(preprocess_config.preserve_branches)},"
                f"{preprocess_config.green_denoise_strategy},"
                f"{preprocess_config.green_noise_model},"
                f"{preprocess_config.green_branch_protection:.6f},"
                f"{int(preprocess_config.green_nlm_patch_size)},"
                f"{int(preprocess_config.green_nlm_patch_distance)},"
                f"{preprocess_config.green_nlm_h_factor:.6f},"
                f"{int(preprocess_config.green_apply_vst)},"
                f"{preprocess_config.green_pre_deconv_strength:.6f},"
                f"{preprocess_config.green_post_deconv_strength:.6f},"
                f"{int(preprocess_config.green_speckle_min_voxels)},"
                f"{preprocess_config.green_speckle_attenuation:.6f},"
                f"{preprocess_config.green_noise2void_model_path},"
                f"{int(preprocess_config.green_chunked_processing)},"
                f"{int(preprocess_config.green_chunk_depth)},"
                f"{int(preprocess_config.green_chunk_overlap)},"
                f"{int(preprocess_config.cpu_worker_threads)},"
                f"{int(preprocess_config.resample_for_mesh)},"
                f"{preprocess_config.mesh_target_z_um:.6f},"
                f"{int(preprocess_config.wavelet_level)},"
                f"{preprocess_config.wavelet_name}"
            ).encode("utf-8", errors="ignore")
        )
    return h.hexdigest()[:24]


def _cache_file_path(cache_key: str, base_dir: str | Path | None = None) -> Path:
    return cache_dir(base_dir) / f"processed_{cache_key}.npz"


def has_processed_cache(cache_key: str, base_dir: str | Path | None = None) -> bool:
    return _cache_file_path(cache_key, base_dir=base_dir).exists()


def save_processed_dataset(
    cache_key: str,
    dataset: DatasetVolume,
    base_dir: str | Path | None = None,
) -> Path:
    path = _cache_file_path(cache_key, base_dir=base_dir)
    np.savez_compressed(
        path,
        cache_version=np.int32(CACHE_VERSION),
        green_data=dataset.green.data.astype(np.float32, copy=False),
        red_data=dataset.red.data.astype(np.float32, copy=False),
        green_z=np.asarray(dataset.green.z_indices, dtype=np.int32),
        red_z=np.asarray(dataset.red.z_indices, dtype=np.int32),
        shared_z=np.asarray(dataset.shared_z_range, dtype=np.int32),
    )
    logger.info("Saved processed dataset cache: %s", path)
    return path


def load_processed_dataset(
    cache_key: str,
    spacing: VoxelSpacing,
    base_dir: str | Path | None = None,
) -> DatasetVolume | None:
    path = _cache_file_path(cache_key, base_dir=base_dir)
    if not path.exists():
        return None

    try:
        with np.load(path) as data:
            version = int(data["cache_version"])
            if version != CACHE_VERSION:
                logger.info("Ignoring stale cache version in %s", path)
                return None

            green = ChannelVolume(
                name="green",
                data=np.asarray(data["green_data"], dtype=np.float32),
                z_indices=[int(v) for v in np.asarray(data["green_z"], dtype=np.int32)],
                spacing=spacing,
            )
            red = ChannelVolume(
                name="red",
                data=np.asarray(data["red_data"], dtype=np.float32),
                z_indices=[int(v) for v in np.asarray(data["red_z"], dtype=np.int32)],
                spacing=spacing,
            )
            shared = np.asarray(data["shared_z"], dtype=np.int32)
            result = DatasetVolume(
                green=green,
                red=red,
                shared_z_range=(int(shared[0]), int(shared[1])),
            )
            logger.info("Loaded processed dataset cache: %s", path)
            return result
    except Exception as exc:
        logger.warning("Failed to load cache %s: %s", path, exc)
        return None
