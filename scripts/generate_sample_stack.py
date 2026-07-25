from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np


SAMPLE_SHAPE_ZYX = (17, 55, 55)
DEFAULT_SPACING_UM = {
    "x_um": 0.331,
    "y_um": 0.331,
    "z_um": 0.4,
}


def build_sample_volumes() -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic green microglia phantom and red vessel markers."""
    green = np.zeros(SAMPLE_SHAPE_ZYX, dtype=np.float32)
    red = np.zeros(SAMPLE_SHAPE_ZYX, dtype=np.float32)

    # One compact soma with two straight processes. With NVAP's default spacing,
    # each process is long enough to pass the 3 um terminal-branch gate.
    green[6:11, 23:32, 23:32] = 1.0
    green[8, 27, 32:48] = 1.0
    green[8, 27, 7:23] = 1.0

    # Two tiny vessel-wall markers, each one x-voxel from a process tip. They
    # span three z-slices so the cleaned wall-mask path keeps them as signal
    # rather than treating them as isolated one-voxel red specks.
    red[7:10, 27, 6] = 1.0
    red[7:10, 27, 48] = 1.0
    return green, red


def expected_summary() -> dict[str, float | int | dict[str, float]]:
    green, red = build_sample_volumes()
    voxel_volume = (
        DEFAULT_SPACING_UM["x_um"]
        * DEFAULT_SPACING_UM["y_um"]
        * DEFAULT_SPACING_UM["z_um"]
    )
    green_voxels = int(np.count_nonzero(green >= 0.5))
    red_voxels = int(np.count_nonzero(red >= 0.5))
    return {
        "shape_zyx": {
            "z": int(SAMPLE_SHAPE_ZYX[0]),
            "y": int(SAMPLE_SHAPE_ZYX[1]),
            "x": int(SAMPLE_SHAPE_ZYX[2]),
        },
        "spacing_um": DEFAULT_SPACING_UM,
        "green_voxel_count": green_voxels,
        "green_component_count": 1,
        "green_volume_um3": green_voxels * voxel_volume,
        "red_voxel_count": red_voxels,
        "red_component_count": 2,
        "red_volume_um3": red_voxels * voxel_volume,
        "overlap_voxel_count": 0,
        "expected_tip_to_vessel_um": DEFAULT_SPACING_UM["x_um"],
        "expected_cell_to_vessel_um": DEFAULT_SPACING_UM["x_um"],
        "expected_microglia_count": 1,
        "expected_branch_count": 2,
        "expected_tip_count": 2,
    }


def _write_channel_slices(volume: np.ndarray, out_dir: Path, channel_suffix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for z_idx, plane in enumerate(np.asarray(volume), start=1):
        image = np.asarray(np.clip(plane, 0.0, 1.0) * 255.0, dtype=np.uint8)
        iio.imwrite(out_dir / f"nvap_synthetic_z{z_idx:03d}{channel_suffix}.png", image)


def _write_preview(green: np.ndarray, red: np.ndarray, out_path: Path) -> None:
    green_proj = np.max(green, axis=0)
    red_proj = np.max(red, axis=0)
    preview = np.zeros((green.shape[1], green.shape[2], 3), dtype=np.uint8)
    preview[..., 0] = np.asarray(np.clip(red_proj, 0.0, 1.0) * 255.0, dtype=np.uint8)
    preview[..., 1] = np.asarray(np.clip(green_proj, 0.0, 1.0) * 255.0, dtype=np.uint8)
    iio.imwrite(out_path, preview)


def write_sample_stack(output_root: str | Path) -> Path:
    root = Path(output_root)
    green, red = build_sample_volumes()
    segmented = root / "Input" / "Segmented"

    _write_channel_slices(green, segmented / "Green", "c1")
    _write_channel_slices(red, segmented / "Red", "c2")
    _write_preview(green, red, root / "preview_max_projection.png")
    (root / "expected_analytics.json").write_text(
        json.dumps(expected_summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a small NVAP sample image stack.")
    parser.add_argument(
        "--output",
        default="manual_test_outputs/nvap_synthetic_sample",
        help="Output folder that will contain Input/Segmented/Green and Red.",
    )
    args = parser.parse_args()
    root = write_sample_stack(args.output)
    print(root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
