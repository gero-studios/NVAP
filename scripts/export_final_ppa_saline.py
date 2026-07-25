from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkLight,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderer,
    vtkWindowToImageFilter,
)
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 - registers OpenGL rendering

from nvap.analysis.metrics import compute_metrics, metrics_to_csv_rows
from nvap.analysis.microglia_analysis import (
    analyze_microglia_cells,
    microglia_analysis_to_csv_rows,
)
from nvap.analysis.microglia_components import (
    compute_component_labels,
    filter_components_by_preferred_voxel_floor,
)
from nvap.analysis.neurovascular import (
    neurovascular_association_to_csv_rows,
    summarize_neurovascular_association,
)
from nvap.analysis.vascular_analysis import (
    analyze_vasculature,
    build_vascular_masks,
    vascular_analysis_to_csv_rows,
)
from nvap.config.types import (
    ChannelVolume,
    DatasetVolume,
    PreprocessConfig,
    RenderConfig,
    VoxelSpacing,
)
from nvap.export.exporters import export_metrics_csv
from nvap.io.czi_loader import load_czi_channels
from nvap.pipeline import prepare_dataset_for_mesh
from nvap.preprocess.enhancement import (
    enhance_microglia_background,
    preprocess_dataset,
    wipe_small_specks,
)


SAMPLES = {
    "ppa": {
        "source": Path(r"C:\Users\giaco\Downloads\New folder (2)\7015-2_CCI_PPA_CL_cortex_63x.czi"),
        "threshold_green": 0.33111512660980225,
        "threshold_red": 0.4275728762149811,
    },
    "saline": {
        "source": Path(r"C:\Users\giaco\Downloads\New folder (2)\7015-1_CCI_saline_CL_cortex_63x.czi"),
        "threshold_green": 0.353515625,
        "threshold_red": 0.44470643997192383,
    },
}

GREEN_WIPE_MIN_VOXELS = 128
MICROGLIA_COMPONENT_MIN_VOXELS = 53
BRANCH_SENSITIVITY = 1.0


def _unit(metric: str) -> str:
    name = str(metric)
    if name.endswith("_um3"):
        return "um^3"
    if name.endswith("_um2"):
        return "um^2"
    if name.endswith("_um"):
        return "um"
    if name.endswith("_per_mm3"):
        return "1/mm^3"
    if name.endswith("_per_um"):
        return "1/um"
    if "fraction" in name or name.endswith("_roundness"):
        return "ratio"
    if name.endswith("_vox") or "voxel_count" in name:
        return "voxels"
    return ""


def _normalise_rows(
    *,
    sample: str,
    source: Path,
    family: str,
    rows: list[dict[str, Any]],
    component_key: str | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_row in rows:
        component_id = source_row.get(component_key) if component_key else None
        if set(source_row).issuperset({"metric", "value"}):
            metric_items = [(str(source_row["metric"]), source_row["value"])]
        else:
            metric_items = [
                (str(key), value)
                for key, value in source_row.items()
                if key != component_key
            ]
        for metric, value in metric_items:
            output.append(
                {
                    "sample": sample,
                    "source_file": str(source),
                    "metric_family": family,
                    "component_id": component_id,
                    "metric": metric,
                    "value": value,
                    "unit": _unit(metric),
                }
            )
    return output


def _provenance_rows(
    sample: str,
    source: Path,
    spacing: VoxelSpacing,
    threshold_green: float,
    threshold_red: float,
) -> list[dict[str, Any]]:
    values = {
        "source_path": str(source),
        "source_last_write_time": datetime.fromtimestamp(source.stat().st_mtime).isoformat(),
        "spacing_x_um": spacing.x_um,
        "spacing_y_um": spacing.y_um,
        "spacing_z_um": spacing.z_um,
        "threshold_green": threshold_green,
        "threshold_red_source": threshold_red,
        "vascular_wall_input": "saved cleaned thresholded wall mask",
        "vascular_cross_section_gap_close_um": 2.0,
        "vascular_3d_gap_close_cap_um": 1.5,
        "vascular_radius_ridge_search_um": 1.5,
        "green_speck_wipe_min_voxels": GREEN_WIPE_MIN_VOXELS,
        "microglia_component_min_voxels": MICROGLIA_COMPONENT_MIN_VOXELS,
        "microglia_enhancement_method": "microscopy_clean",
        "branch_sensitivity": BRANCH_SENSITIVITY,
        "coordinate_convention": (
            "zero-based z,y,x voxel centers; physical coordinates are x,y,z um from origin 0,0,0"
        ),
    }
    return [
        {
            "sample": sample,
            "source_file": str(source),
            "metric_family": "provenance",
            "component_id": None,
            "metric": key,
            "value": value,
            "unit": _unit(key),
        }
        for key, value in values.items()
    ]


def _vtk_image(volume: np.ndarray, spacing: VoxelSpacing) -> vtkImageData:
    arr = np.ascontiguousarray(np.asarray(volume, dtype=np.float32))
    image = vtkImageData()
    image.SetDimensions(int(arr.shape[2]), int(arr.shape[1]), int(arr.shape[0]))
    image.SetSpacing(float(spacing.x_um), float(spacing.y_um), float(spacing.z_um))
    scalars = numpy_to_vtk(arr.ravel(order="C"), deep=True)
    scalars.SetName("mask")
    image.GetPointData().SetScalars(scalars)
    return image


def _surface_actor(
    volume: np.ndarray,
    spacing: VoxelSpacing,
    *,
    color: tuple[float, float, float],
    opacity: float,
) -> vtkActor:
    marching = vtkMarchingCubes()
    marching.SetInputData(_vtk_image(volume, spacing))
    marching.SetValue(0, 0.5)
    marching.ComputeNormalsOn()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(marching.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(float(opacity))
    actor.GetProperty().SetAmbient(0.24)
    actor.GetProperty().SetDiffuse(0.72)
    actor.GetProperty().SetSpecular(0.18)
    return actor


def _render_viewport(
    output_path: Path,
    *,
    solid_vessel: np.ndarray,
    vessel_spacing: VoxelSpacing,
    microglia_labels: np.ndarray,
    microglia_spacing: VoxelSpacing,
) -> Path:
    renderer = vtkRenderer()
    renderer.SetBackground(0.015, 0.020, 0.030)
    renderer.SetBackground2(0.055, 0.065, 0.090)
    renderer.SetGradientBackground(True)
    renderer.AddActor(
        _surface_actor(
            solid_vessel,
            vessel_spacing,
            color=(0.92, 0.035, 0.025),
            opacity=1.0,
        )
    )
    renderer.AddActor(
        _surface_actor(
            microglia_labels > 0,
            microglia_spacing,
            color=(0.03, 0.92, 0.12),
            opacity=0.94,
        )
    )
    key = vtkLight()
    key.SetLightTypeToCameraLight()
    key.SetIntensity(0.92)
    renderer.AddLight(key)
    fill = vtkLight()
    fill.SetLightTypeToSceneLight()
    fill.SetPosition(-2.0, -1.0, 2.0)
    fill.SetIntensity(0.35)
    renderer.AddLight(fill)

    window = vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(1200, 900)
    window.SetMultiSamples(4)
    window.AddRenderer(renderer)
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.Azimuth(-28.0)
    camera.Elevation(22.0)
    camera.Roll(-2.0)
    camera.Zoom(1.10)
    renderer.ResetCameraClippingRange()
    window.Render()

    capture = vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.ReadFrontBufferOff()
    capture.Update()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtkPNGWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()
    window.Finalize()
    return output_path


def _process_sample(
    sample: str,
    config: dict[str, Any],
    *,
    previous_root: Path,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = Path(config["source"])
    threshold_green = float(config["threshold_green"])
    threshold_red = float(config["threshold_red"])
    green_channel, red_channel, spacing = load_czi_channels(source)
    raw_dataset = DatasetVolume(
        green=green_channel,
        red=red_channel,
        shared_z_range=(1, int(green_channel.data.shape[0])),
    )
    preprocess = PreprocessConfig()
    preprocessed = preprocess_dataset(raw_dataset, preprocess)
    green = enhance_microglia_background(
        preprocessed.green.data,
        preprocess,
        method="microscopy_clean",
    )
    green = wipe_small_specks(
        green,
        threshold=threshold_green,
        min_voxels=GREEN_WIPE_MIN_VOXELS,
    )

    old_masks = np.load(previous_root / sample / f"{sample}_vascular_masks.npz")
    wall = np.asarray(old_masks["vascular_wall_mask"], dtype=np.float32)
    z_indices = list(range(1, int(green.shape[0]) + 1))
    processed = DatasetVolume(
        green=ChannelVolume("green", green, z_indices, spacing),
        red=ChannelVolume("red", wall, z_indices, spacing),
        shared_z_range=(1, int(green.shape[0])),
    )
    render = RenderConfig(
        threshold_green=threshold_green,
        threshold_red=0.5,
    )
    basic = compute_metrics(processed, render)
    vascular = analyze_vasculature(
        wall,
        threshold=0.5,
        spacing=spacing,
        render=render,
    )
    masks = build_vascular_masks(
        wall,
        threshold=0.5,
        spacing=spacing,
        render=render,
    )

    visual = prepare_dataset_for_mesh(processed, preprocess)
    spacing_zyx = (
        float(visual.green.spacing.z_um),
        float(visual.green.spacing.y_um),
        float(visual.green.spacing.x_um),
    )
    labels, order, sizes = compute_component_labels(
        visual.green.data,
        threshold=threshold_green,
        min_voxels=MICROGLIA_COMPONENT_MIN_VOXELS,
        max_components=256,
        smooth_sigma=(0.2, 0.45, 0.45),
        branch_sensitivity=BRANCH_SENSITIVITY,
        spacing=spacing_zyx,
    )
    labels, order, sizes = filter_components_by_preferred_voxel_floor(labels, order, sizes)
    microglia = analyze_microglia_cells(
        visual.green.data,
        visual.red.data,
        labels,
        order,
        spacing=visual.green.spacing,
        render=render,
        branch_sensitivity=BRANCH_SENSITIVITY,
    )
    association = summarize_neurovascular_association(microglia)

    sample_dir = output_root / sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    basic_rows = metrics_to_csv_rows(basic)
    vascular_rows = vascular_analysis_to_csv_rows(vascular)
    microglia_rows = microglia_analysis_to_csv_rows(microglia)
    neurovascular_rows = neurovascular_association_to_csv_rows(association)
    export_metrics_csv(basic_rows, sample_dir / f"{sample}_basic_metrics.csv")
    export_metrics_csv(vascular_rows, sample_dir / f"{sample}_vascular_metrics.csv")
    export_metrics_csv(
        microglia_rows,
        sample_dir / f"{sample}_microglia_cells_with_coordinates.csv",
    )
    export_metrics_csv(
        neurovascular_rows,
        sample_dir / f"{sample}_neurovascular_metrics.csv",
    )
    np.savez_compressed(
        sample_dir / f"{sample}_analysis_masks.npz",
        vascular_wall_mask=masks.wall_mask.astype(np.uint8),
        vascular_solid_mask=masks.solid_mask.astype(np.uint8),
        microglia_component_labels=labels.astype(np.uint16),
    )

    complete: list[dict[str, Any]] = []
    complete.extend(_provenance_rows(sample, source, spacing, threshold_green, threshold_red))
    complete.extend(
        _normalise_rows(sample=sample, source=source, family="basic", rows=basic_rows)
    )
    complete.extend(
        _normalise_rows(sample=sample, source=source, family="vascular", rows=vascular_rows)
    )
    complete.extend(
        _normalise_rows(
            sample=sample,
            source=source,
            family="microglia_cell",
            rows=microglia_rows,
            component_key="component_id",
        )
    )
    complete.extend(
        _normalise_rows(
            sample=sample,
            source=source,
            family="neurovascular",
            rows=neurovascular_rows,
        )
    )
    export_metrics_csv(
        complete,
        sample_dir / f"{sample}_complete_metrics.csv",
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
    viewport_path = _render_viewport(
        sample_dir / f"{sample}_viewport_final.png",
        solid_vessel=masks.solid_mask,
        vessel_spacing=spacing,
        microglia_labels=labels,
        microglia_spacing=visual.green.spacing,
    )
    summary = {
        "sample": sample,
        "source_file": str(source),
        "threshold_green": threshold_green,
        "threshold_red_source": threshold_red,
        "microglia_cells": int(microglia.analyzed_cell_count),
        "vascular_components": int(vascular.component_count),
        "vascular_length_um": float(vascular.total_length_um),
        "vascular_mean_radius_um": float(vascular.mean_radius_um),
        "vascular_mean_diameter_um": float(vascular.mean_diameter_um),
        "vascular_equivalent_diameter_um": float(
            vascular.volume_length_equivalent_diameter_um
        ),
        "vascular_lumen_fill_fraction": float(vascular.reconstructed_lumen_fill_fraction),
        "viewport_png": str(viewport_path.resolve()),
        "complete_csv": str((sample_dir / f"{sample}_complete_metrics.csv").resolve()),
        "microglia_coordinate_csv": str(
            (sample_dir / f"{sample}_microglia_cells_with_coordinates.csv").resolve()
        ),
    }
    return complete, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--previous-root",
        default="analysis_outputs/czi_latest_ppa_saline_20260720_104135",
    )
    parser.add_argument(
        "--output",
        default="analysis_outputs/final_ppa_saline_20260720",
    )
    args = parser.parse_args()
    previous_root = Path(args.previous_root).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    combined: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for sample, config in SAMPLES.items():
        rows, summary = _process_sample(
            sample,
            config,
            previous_root=previous_root,
            output_root=output_root,
        )
        combined.extend(rows)
        summaries.append(summary)
    export_metrics_csv(
        combined,
        output_root / "ppa_saline_complete_metrics.csv",
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
    export_metrics_csv(summaries, output_root / "ppa_saline_summary.csv")
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "output_root": str(output_root),
                "samples": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
