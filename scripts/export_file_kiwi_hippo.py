from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.export_final_ppa_saline as final_export
from nvap.analysis.vascular_analysis import build_vascular_masks
from nvap.config.types import DatasetVolume, PreprocessConfig, RenderConfig, VoxelSpacing
from nvap.io.czi_loader import load_czi_channels as load_czi_channels_original
from nvap.pipeline import automatic_thresholds
from nvap.preprocess.enhancement import preprocess_dataset, wipe_vasculature_blobs


CONTROL_SPACING = VoxelSpacing(
    x_um=0.1315305679563492,
    y_um=0.1315305679563492,
    z_um=0.4,
)
VASCULAR_BLOB_MAX_VOXELS = 2_048


def _load_with_spacing_override(path: str | Path):
    source = Path(path)
    spacing = CONTROL_SPACING if source.name == "7013-1_M_CL_hippo_63X.czi" else None
    return load_czi_channels_original(source, spacing=spacing)


def _prepare_clean_wall_mask(
    sample: str,
    config: dict[str, Any],
    *,
    previous_root: Path,
) -> None:
    source = Path(config["source"])
    green, red, spacing = _load_with_spacing_override(source)
    raw = DatasetVolume(
        green=green,
        red=red,
        shared_z_range=(1, int(green.data.shape[0])),
    )
    preprocess = PreprocessConfig()
    processed = preprocess_dataset(raw, preprocess)
    threshold_green, threshold_red = automatic_thresholds(
        processed,
        green_fallback=0.80,
        red_fallback=0.60,
    )
    cleaned_red = wipe_vasculature_blobs(
        processed.red.data,
        threshold=float(threshold_red),
        max_voxels=VASCULAR_BLOB_MAX_VOXELS,
    )
    render = RenderConfig(
        threshold_green=float(threshold_green),
        threshold_red=float(threshold_red),
    )
    masks = build_vascular_masks(
        cleaned_red,
        threshold=float(threshold_red),
        spacing=spacing,
        render=render,
    )
    sample_dir = previous_root / sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sample_dir / f"{sample}_vascular_masks.npz",
        vascular_wall_mask=masks.wall_mask.astype(np.uint8),
        vascular_solid_mask=masks.solid_mask.astype(np.uint8),
    )
    config["threshold_green"] = float(threshold_green)
    config["threshold_red"] = float(threshold_red)
    config["spacing_um"] = {
        "x": float(spacing.x_um),
        "y": float(spacing.y_um),
        "z": float(spacing.z_um),
    }
    print(
        f"prepared {sample}: thresholds=(green={threshold_green:.6f}, "
        f"red={threshold_red:.6f}) spacing_um={config['spacing_um']}",
        flush=True,
    )


def _write_zip(output_root: Path) -> Path:
    zip_path = output_root / "control_ppa_hippo_csv_and_viewports.zip"
    include = [
        output_root / "control_ppa_hippo_complete_metrics.csv",
        output_root / "control_ppa_hippo_summary.csv",
        output_root / "manifest.json",
        output_root / "qc_report.json",
    ]
    for sample in ("control", "ppa"):
        sample_dir = output_root / sample
        include.extend(
            [
                sample_dir / f"{sample}_complete_metrics.csv",
                sample_dir / f"{sample}_microglia_cells_with_coordinates.csv",
                sample_dir / f"{sample}_vascular_metrics.csv",
                sample_dir / f"{sample}_neurovascular_metrics.csv",
                sample_dir / f"{sample}_basic_metrics.csv",
                sample_dir / f"{sample}_viewport_final.png",
            ]
        )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in include:
            archive.write(path, path.relative_to(output_root))
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "analysis_inputs" / "file_kiwi_20260722",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "analysis_outputs" / "file_kiwi_hippo_20260722",
    )
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output.resolve()
    previous_root = output_root / "_prepared"
    output_root.mkdir(parents=True, exist_ok=True)

    samples: dict[str, dict[str, Any]] = {
        "control": {
            "source": input_root / "7013-1_M_CL_hippo_63X.czi",
            "channel_mapping": {"green_microglia": "C0 EGFP", "red_vasculature": "C1 Texas Red"},
            "spacing_metadata_note": (
                "CZI DefaultUnitFormat is a display unit; meter-valued distances "
                "were converted to micrometers."
            ),
        },
        "ppa": {
            "source": input_root / "7013-2__CCI_PPA_CL_hippo_63xC.czi",
            "channel_mapping": {"green_microglia": "C1 EGFP", "red_vasculature": "C0 Texas Red"},
            "spacing_metadata_note": "CZI meter-valued distances converted to micrometers.",
        },
    }
    for sample, config in samples.items():
        if not Path(config["source"]).is_file():
            raise FileNotFoundError(config["source"])
        _prepare_clean_wall_mask(sample, config, previous_root=previous_root)

    final_export.load_czi_channels = _load_with_spacing_override
    combined: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for sample, config in samples.items():
        rows, summary = final_export._process_sample(
            sample,
            config,
            previous_root=previous_root,
            output_root=output_root,
        )
        combined.extend(rows)
        summary["spacing_um"] = config["spacing_um"]
        summary["spacing_metadata_note"] = config["spacing_metadata_note"]
        summary["channel_mapping"] = config["channel_mapping"]
        vascular_values = {
            str(row["metric"]): row["value"]
            for row in rows
            if row.get("metric_family") == "vascular"
        }
        summary["vascular_qc"] = {
            key: vascular_values.get(key)
            for key in (
                "anatomical_radius_reliable",
                "radius_estimator_ratio",
                "max_principal_slice_fraction",
                "sheet_like_mask",
                "reconstructed_lumen_fill_fraction",
                "solid_to_wall_volume_ratio",
                "solid_component_count",
            )
        }
        summaries.append(summary)
        print(f"analyzed {sample}", flush=True)

    final_export.export_metrics_csv(
        combined,
        output_root / "control_ppa_hippo_complete_metrics.csv",
        columns=(
            "sample",
            "source_file",
            "metric_family",
            "component_id",
            "metric",
            "value",
            "unit",
        ),
    )
    final_export.export_metrics_csv(
        summaries,
        output_root / "control_ppa_hippo_summary.csv",
    )
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "analysis": "NVAP control-versus-PPA hippocampus final export",
        "parameters": {
            "automatic_thresholds": True,
            "green_speck_wipe_min_voxels": final_export.GREEN_WIPE_MIN_VOXELS,
            "vascular_blob_max_voxels": VASCULAR_BLOB_MAX_VOXELS,
            "microglia_component_min_voxels": final_export.MICROGLIA_COMPONENT_MIN_VOXELS,
            "microglia_enhancement_method": "microscopy_clean",
            "branch_sensitivity": final_export.BRANCH_SENSITIVITY,
            "radius_reliability": (
                "requires reconstructed solid mask; low-fill masks additionally require "
                "EDT-to-volume/length radius ratio between 0.75 and 1.25; obvious "
                "acquisition-plane sheets are rejected"
            ),
        },
        "samples": summaries,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    qc_report = {
        "generated_at": manifest["generated_at"],
        "accepted": all(
            bool(summary["vascular_qc"]["anatomical_radius_reliable"])
            and not bool(summary["vascular_qc"]["sheet_like_mask"])
            for summary in summaries
        ),
        "acceptance_requirements": {
            "metadata_channel_mapping_verified": True,
            "anatomical_radius_reliable": True,
            "sheet_like_mask": False,
            "viewport_review_required": True,
        },
        "samples": [
            {
                "sample": summary["sample"],
                "channel_mapping": summary["channel_mapping"],
                **summary["vascular_qc"],
            }
            for summary in summaries
        ],
    }
    (output_root / "qc_report.json").write_text(
        json.dumps(qc_report, indent=2) + "\n",
        encoding="utf-8",
    )
    zip_path = _write_zip(output_root)
    print(zip_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
