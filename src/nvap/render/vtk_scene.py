from __future__ import annotations

import colorsys
from dataclasses import dataclass
import logging
from pathlib import Path
import time

import numpy as np
from PySide6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPiecewiseFunction, vtkPolyData, vtkPolyLine
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkImagingCore import vtkImageResample
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor, vtkLegendScaleActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkColorTransferFunction,
    vtkLight,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderer,
    vtkTextActor,
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

from nvap.config.types import RenderConfig, VoxelSpacing

logger = logging.getLogger(__name__)

_RENDER_CUBIC_MAX_AXIS_UPSAMPLE = 2.0
_RENDER_CUBIC_MAX_TOTAL_UPSAMPLE = 3.0
_RENDER_CUBIC_MAX_OUTPUT_VOXELS = 240_000_000
_RENDER_SAMPLE_RATIO_DEFAULT = 0.18
_RENDER_SAMPLE_RATIO_ANISO = 0.16
_RENDER_SAMPLE_RATIO_LABELS = 0.14


def _volume_debug_summary(volume: np.ndarray) -> str:
    arr = np.asarray(volume)
    if arr.size == 0:
        return (
            f"shape={arr.shape} dtype={arr.dtype} empty "
            f"c_contiguous={arr.flags.c_contiguous} f_contiguous={arr.flags.f_contiguous}"
        )
    finite = np.asarray(arr[np.isfinite(arr)], dtype=np.float32)
    if finite.size == 0:
        stats = "all_nonfinite"
    else:
        stats = (
            f"min={float(np.min(finite)):.5f} mean={float(np.mean(finite)):.5f} "
            f"max={float(np.max(finite)):.5f} positive={100.0 * float(np.count_nonzero(finite > 0.0)) / max(1, int(finite.size)):.2f}%"
        )
    nonfinite = int(arr.size - finite.size)
    return (
        f"shape={arr.shape} dtype={arr.dtype} {stats} nonfinite={nonfinite} "
        f"c_contiguous={arr.flags.c_contiguous} f_contiguous={arr.flags.f_contiguous}"
    )


def _cubic_render_spacing(
    shape: tuple[int, int, int],
    spacing: VoxelSpacing,
) -> VoxelSpacing:
    if len(shape) != 3 or any(int(axis) <= 0 for axis in shape):
        return spacing

    spacing_xyz = np.array(
        [float(spacing.x_um), float(spacing.y_um), float(spacing.z_um)],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(spacing_xyz)):
        logger.warning("VTK cubic spacing fallback: non-finite spacing=%s", spacing)
        return spacing
    finest = float(np.min(spacing_xyz[:2]))
    if finest <= 0.0:
        logger.warning("VTK cubic spacing fallback: non-positive spacing=%s", spacing)
        return spacing

    desired_zoom = np.array(
        [
            float(spacing.z_um) / finest,
            float(spacing.y_um) / finest,
            float(spacing.x_um) / finest,
        ],
        dtype=np.float32,
    )
    zoom = np.clip(desired_zoom, 1.0, _RENDER_CUBIC_MAX_AXIS_UPSAMPLE)

    active = zoom > 1.0 + 1.0e-3
    if np.any(active):
        input_voxels = float(np.prod(np.asarray(shape, dtype=np.float64)))
        max_zoom_product = float(_RENDER_CUBIC_MAX_TOTAL_UPSAMPLE)
        if input_voxels > 0.0:
            max_zoom_product = min(
                max_zoom_product,
                float(_RENDER_CUBIC_MAX_OUTPUT_VOXELS) / input_voxels,
            )
        if max_zoom_product <= 1.0 + 1.0e-6:
            logger.info(
                "VTK cubic spacing disabled for large volume: shape=%s spacing=%s", shape, spacing
            )
            return spacing

        log_sum = float(np.sum(np.log(zoom[active])))
        if log_sum > 0.0:
            max_log = float(np.log(max_zoom_product))
            if log_sum > max_log:
                scale = max_log / log_sum
                zoom[active] = np.power(zoom[active], scale)

    if not np.any(zoom > 1.0 + 1.0e-3):
        return spacing

    return VoxelSpacing(
        x_um=float(spacing.x_um) / float(zoom[2]),
        y_um=float(spacing.y_um) / float(zoom[1]),
        z_um=float(spacing.z_um) / float(zoom[0]),
    )


def _same_spacing(left: VoxelSpacing, right: VoxelSpacing) -> bool:
    return bool(
        np.isclose(left.x_um, right.x_um)
        and np.isclose(left.y_um, right.y_um)
        and np.isclose(left.z_um, right.z_um)
    )


def _recommended_sample_distance(
    spacing: VoxelSpacing,
    *,
    label_mode: bool = False,
) -> float:
    spacing_xyz = np.asarray(
        [float(spacing.x_um), float(spacing.y_um), float(spacing.z_um)],
        dtype=np.float32,
    )
    min_sp = float(np.min(spacing_xyz))
    max_sp = float(np.max(spacing_xyz))
    if min_sp <= 0.0:
        return 0.02
    anisotropy = max_sp / max(min_sp, 1.0e-6)
    if label_mode:
        ratio = _RENDER_SAMPLE_RATIO_LABELS
    else:
        ratio = _RENDER_SAMPLE_RATIO_ANISO if anisotropy >= 1.30 else _RENDER_SAMPLE_RATIO_DEFAULT
    return float(max(0.01, ratio * min_sp))


@dataclass
class _ChannelActors:
    image: vtkImageData
    resample: vtkImageResample | None
    volume_actor: vtkVolume
    volume_mapper: object
    volume_property: vtkVolumeProperty
    iso_actor: vtkActor
    marching: vtkMarchingCubes


class VTKScene:
    def __init__(self, parent=None) -> None:
        self._widget = QVTKRenderWindowInteractor(parent)
        self._render_window: vtkRenderWindow = self._widget.GetRenderWindow()
        self._renderer = vtkRenderer()
        self._render_window.AddRenderer(self._renderer)
        self._renderer.SetBackground(0.025, 0.030, 0.042)
        self._renderer.SetBackground2(0.075, 0.082, 0.110)
        self._renderer.SetGradientBackground(True)
        # Better rendering quality
        self._render_window.SetMultiSamples(4)  # Anti-aliasing
        self._render_window.SetAlphaBitPlanes(1)
        self._widget.setFocusPolicy(Qt.StrongFocus)
        self._widget.setMouseTracking(True)

        self._interactor = self._render_window.GetInteractor()
        self._interactor_style = vtkInteractorStyleTrackballCamera()
        self._interactor.SetInteractorStyle(self._interactor_style)

        self._actors: dict[str, _ChannelActors] = {}
        self._spacing: dict[str, VoxelSpacing] = {}
        self._current = RenderConfig()
        self._component_coloring: dict[str, int] = {}
        self._debug_overlay_actors: list[vtkActor] = []
        self._last_render_time = time.perf_counter()
        self._fps_actor = self._build_fps_actor()
        self._scale_actor = vtkLegendScaleActor()
        self._scale_actor.SetLegendVisibility(False)
        self._scale_actor.SetTopAxisVisibility(False)
        self._scale_actor.SetRightAxisVisibility(False)
        self._scale_actor.GetBottomAxis().SetTitle("scale")
        self._scale_actor.GetBottomAxis().GetProperty().SetColor(0.72, 0.76, 0.82)
        self._scale_actor.GetBottomAxis().GetLabelTextProperty().SetColor(0.72, 0.76, 0.82)
        self._scale_actor.GetBottomAxis().GetTitleTextProperty().SetColor(0.86, 0.74, 0.45)
        self._renderer.AddViewProp(self._fps_actor)
        self._renderer.AddActor(self._scale_actor)
        self._configure_lighting()

        self._interactor.Initialize()
        self._interactor.Enable()
        self._orientation_marker = self._build_orientation_marker()
        self._render_window.AddObserver("EndEvent", self._update_fps_overlay)
        self._interactor.Start()
        logger.info("VTK interactor initialized with TrackballCamera style.")

    def widget(self) -> QVTKRenderWindowInteractor:
        return self._widget

    def activate_interaction(self) -> None:
        if self._interactor is not None:
            self._interactor.Enable()
        self._widget.setFocus(Qt.OtherFocusReason)
        logger.debug("VTK interaction re-activated and viewport focused.")

    def reset_camera(self) -> None:
        self._renderer.ResetCamera()
        self.render()
        self.activate_interaction()
        logger.debug("VTK camera reset.")

    def _configure_lighting(self) -> None:
        self._renderer.RemoveAllLights()
        self._renderer.SetAmbient(0.16, 0.17, 0.20)

        key = vtkLight()
        key.SetLightTypeToCameraLight()
        key.SetIntensity(0.78)
        key.SetPosition(1.0, 1.2, 1.0)
        key.SetFocalPoint(0.0, 0.0, 0.0)
        self._renderer.AddLight(key)

        fill = vtkLight()
        fill.SetLightTypeToSceneLight()
        fill.SetIntensity(0.32)
        fill.SetPosition(-2.0, -1.0, 1.4)
        fill.SetFocalPoint(0.0, 0.0, 0.0)
        fill.SetColor(0.78, 0.84, 1.0)
        self._renderer.AddLight(fill)

        rim = vtkLight()
        rim.SetLightTypeToSceneLight()
        rim.SetIntensity(0.24)
        rim.SetPosition(1.5, -2.0, 2.0)
        rim.SetFocalPoint(0.0, 0.0, 0.0)
        rim.SetColor(1.0, 0.88, 0.62)
        self._renderer.AddLight(rim)

    def _build_orientation_marker(self) -> vtkOrientationMarkerWidget:
        axes = vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.035)
        axes.SetConeRadius(0.18)
        axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.86, 0.40, 0.44)
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.38, 0.82, 0.58)
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.60, 0.70, 1.0)

        marker = vtkOrientationMarkerWidget()
        marker.SetOrientationMarker(axes)
        marker.SetInteractor(self._interactor)
        marker.SetViewport(0.012, 0.012, 0.16, 0.16)
        marker.SetEnabled(1)
        marker.InteractiveOff()
        return marker

    @staticmethod
    def _build_fps_actor() -> vtkTextActor:
        actor = vtkTextActor()
        actor.SetInput("FPS --")
        actor.SetDisplayPosition(14, 14)
        prop = actor.GetTextProperty()
        prop.SetFontFamilyToArial()
        prop.SetFontSize(12)
        prop.SetBold(False)
        prop.SetColor(0.86, 0.74, 0.45)
        prop.SetBackgroundColor(0.02, 0.025, 0.035)
        prop.SetBackgroundOpacity(0.62)
        prop.SetFrame(True)
        prop.SetFrameColor(0.18, 0.22, 0.28)
        return actor

    def _update_fps_overlay(self, *_args) -> None:
        now = time.perf_counter()
        elapsed = max(1.0e-6, now - self._last_render_time)
        self._last_render_time = now
        self._fps_actor.SetInput(f"FPS {1.0 / elapsed:4.1f}")

    def clear_debug_overlays(self) -> None:
        for actor in self._debug_overlay_actors:
            self._renderer.RemoveActor(actor)
        self._debug_overlay_actors.clear()
        self.render()

    def set_debug_overlays(
        self,
        points: list[dict[str, object]] | None = None,
        lines: list[dict[str, object]] | None = None,
        *,
        visibility: dict[str, bool] | None = None,
    ) -> None:
        self.clear_debug_overlays()
        visibility = visibility or {}
        grouped_points: dict[str, list[tuple[float, float, float]]] = {}
        for item in points or []:
            kind = str(item.get("kind", "point"))
            if not self._debug_kind_visible(kind, visibility):
                continue
            xyz = item.get("xyz_um")
            if not isinstance(xyz, list | tuple) or len(xyz) != 3:
                continue
            grouped_points.setdefault(kind, []).append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
        for kind, coords in grouped_points.items():
            actor = self._make_point_actor(coords, self._debug_color(kind), point_size=self._debug_point_size(kind))
            self._renderer.AddActor(actor)
            self._debug_overlay_actors.append(actor)

        grouped_lines: dict[str, list[list[tuple[float, float, float]]]] = {}
        for item in lines or []:
            kind = str(item.get("kind", "line"))
            if not self._debug_kind_visible(kind, visibility):
                continue
            raw_points = item.get("points_xyz_um")
            if not isinstance(raw_points, list | tuple):
                continue
            coords = []
            for raw in raw_points:
                if isinstance(raw, list | tuple) and len(raw) == 3:
                    coords.append((float(raw[0]), float(raw[1]), float(raw[2])))
            if len(coords) >= 2:
                grouped_lines.setdefault(kind, []).append(coords)
        for kind, paths in grouped_lines.items():
            actor = self._make_line_actor(paths, self._debug_color(kind))
            self._renderer.AddActor(actor)
            self._debug_overlay_actors.append(actor)
        self.render()

    @staticmethod
    def _debug_kind_visible(kind: str, visibility: dict[str, bool]) -> bool:
        mapping = {
            "soma_center": "soma",
            "branch_path": "branches",
            "tip": "tips",
            "cell_to_vessel": "connectors",
            "tip_to_vessel": "connectors",
            "nearest_vessel_point": "vessels",
            "diameter_sample": "diameter",
            "vessel_crossing": "crossings",
        }
        key = mapping.get(kind, kind)
        return bool(visibility.get(key, True))

    @staticmethod
    def _debug_color(kind: str) -> tuple[float, float, float]:
        colors = {
            "soma_center": (1.0, 0.92, 0.0),
            "branch_path": (0.0, 0.72, 1.0),
            "tip": (1.0, 0.0, 0.9),
            "cell_to_vessel": (1.0, 1.0, 1.0),
            "tip_to_vessel": (0.7, 0.7, 1.0),
            "nearest_vessel_point": (1.0, 0.25, 0.15),
            "diameter_sample": (1.0, 0.55, 0.0),
            "vessel_crossing": (1.0, 1.0, 1.0),
        }
        return colors.get(kind, (1.0, 1.0, 1.0))

    @staticmethod
    def _debug_point_size(kind: str) -> float:
        if kind == "vessel_crossing":
            return 11.0
        if kind in {"soma_center", "tip"}:
            return 8.0
        return 6.0

    @staticmethod
    def _make_point_actor(
        coords: list[tuple[float, float, float]],
        color: tuple[float, float, float],
        *,
        point_size: float,
    ) -> vtkActor:
        points = vtkPoints()
        vertices = vtkCellArray()
        for x, y, z in coords:
            point_id = points.InsertNextPoint(float(x), float(y), float(z))
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(point_id)
        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetVerts(vertices)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetPointSize(float(point_size))
        actor.GetProperty().SetRenderPointsAsSpheres(True)
        return actor

    @staticmethod
    def _make_line_actor(
        paths: list[list[tuple[float, float, float]]],
        color: tuple[float, float, float],
    ) -> vtkActor:
        points = vtkPoints()
        cells = vtkCellArray()
        for path in paths:
            polyline = vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(len(path))
            for idx, (x, y, z) in enumerate(path):
                point_id = points.InsertNextPoint(float(x), float(y), float(z))
                polyline.GetPointIds().SetId(idx, point_id)
            cells.InsertNextCell(polyline)
        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(cells)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(2.0)
        return actor

    def set_channel_component_coloring(
        self,
        channel: str,
        enabled: bool,
        *,
        label_count: int = 0,
    ) -> None:
        channel = channel.lower()
        if enabled:
            self._component_coloring[channel] = max(0, int(label_count))
        else:
            self._component_coloring.pop(channel, None)
        if channel in self._actors:
            self.apply_render_config(self._current)

    def set_channel_data(self, channel: str, volume: np.ndarray, spacing: VoxelSpacing) -> None:
        channel = channel.lower()
        if channel not in {"green", "red"}:
            raise ValueError("channel must be 'green' or 'red'.")
        if volume.ndim != 3:
            raise ValueError("volume must have shape (z, y, x).")
        render_spacing = _cubic_render_spacing(volume.shape, spacing)
        needs_resample = not _same_spacing(render_spacing, spacing)
        logger.info(
            "VTK set_channel_data: channel=%s %s input_spacing=(%.4f, %.4f, %.4f) "
            "render_spacing=(%.4f, %.4f, %.4f) resample=%s",
            channel,
            _volume_debug_summary(volume),
            float(spacing.x_um),
            float(spacing.y_um),
            float(spacing.z_um),
            render_spacing.x_um,
            render_spacing.y_um,
            render_spacing.z_um,
            needs_resample,
        )

        if channel in self._actors:
            actor = self._actors[channel]
            expected_dims = (int(volume.shape[2]), int(volume.shape[1]), int(volume.shape[0]))
            actor_has_resample = actor.resample is not None
            if tuple(actor.image.GetDimensions()) == expected_dims and actor_has_resample == needs_resample:
                # Fast path: keep actors/mappers and only replace scalar data.
                self._update_vtk_image(actor.image, volume, spacing)
                if actor.resample is not None:
                    self._configure_resample(
                        actor.resample,
                        render_spacing,
                        nearest=channel in self._component_coloring,
                    )
                unit_distance = float(max(render_spacing.x_um, render_spacing.y_um, render_spacing.z_um) * 1.5)
                actor.volume_property.SetScalarOpacityUnitDistance(max(0.08, unit_distance))
                # Re-sync sample distance to current spacing.
                actor.volume_mapper.SetSampleDistance(
                    _recommended_sample_distance(
                        render_spacing,
                        label_mode=channel in self._component_coloring,
                    )
                )
                self._spacing[channel] = render_spacing
                self.apply_render_config(self._current)
                logger.info(
                    "VTK set_channel_data fast-update: channel=%s vtk_dims=%s scalar_count=%d",
                    channel,
                    actor.image.GetDimensions(),
                    actor.image.GetPointData().GetScalars().GetNumberOfTuples(),
                )
                return

            old = self._actors[channel]
            self._renderer.RemoveVolume(old.volume_actor)
            self._renderer.RemoveActor(old.iso_actor)

        image = self._numpy_to_vtk_image(volume, spacing)
        mapper_input, resample = self._build_render_input(
            image,
            render_spacing,
            nearest=channel in self._component_coloring,
        )
        mapper = self._build_volume_mapper(
            mapper_input,
            render_spacing,
            nearest=channel in self._component_coloring,
        )
        prop = vtkVolumeProperty()
        prop.ShadeOn()
        if channel in self._component_coloring:
            prop.SetInterpolationTypeToNearest()
        else:
            prop.SetInterpolationTypeToLinear()
        prop.IndependentComponentsOn()
        # Enhanced lighting for depth perception
        prop.SetAmbient(0.15)
        prop.SetDiffuse(0.82)
        prop.SetSpecular(0.3)
        prop.SetSpecularPower(24.0)
        unit_distance = float(max(render_spacing.x_um, render_spacing.y_um, render_spacing.z_um) * 1.5)
        prop.SetScalarOpacityUnitDistance(max(0.08, unit_distance))
        volume_actor = vtkVolume()
        volume_actor.SetMapper(mapper)
        volume_actor.SetProperty(prop)

        marching = vtkMarchingCubes()
        self._set_pipeline_input(marching, mapper_input)
        marching.SetValue(0, self._current.iso_green if channel == "green" else self._current.iso_red)
        marching.ComputeNormalsOn()
        marching.ComputeGradientsOn()
        iso_mapper = vtkPolyDataMapper()
        iso_mapper.SetInputConnection(marching.GetOutputPort())
        iso_actor = vtkActor()
        iso_actor.SetMapper(iso_mapper)

        if channel == "green":
            rgb = (0.08, 0.95, 0.18)
        else:
            rgb = (0.95, 0.18, 0.18)
        iso_prop = iso_actor.GetProperty()
        iso_prop.SetColor(*rgb)
        iso_prop.SetOpacity(0.85)
        iso_prop.SetInterpolationToPhong()
        iso_prop.SetAmbient(0.12)
        iso_prop.SetDiffuse(0.85)
        iso_prop.SetSpecular(0.35)
        iso_prop.SetSpecularPower(28.0)

        self._renderer.AddVolume(volume_actor)
        self._renderer.AddActor(iso_actor)

        self._actors[channel] = _ChannelActors(
            image=image,
            resample=resample,
            volume_actor=volume_actor,
            volume_mapper=mapper,
            volume_property=prop,
            iso_actor=iso_actor,
            marching=marching,
        )
        self._spacing[channel] = render_spacing
        self.apply_render_config(self._current)
        self._renderer.ResetCamera()
        self.render()
        self.activate_interaction()

    def _build_render_input(
        self,
        image: vtkImageData,
        render_spacing: VoxelSpacing,
        *,
        nearest: bool = False,
    ):
        input_spacing = image.GetSpacing()
        source_spacing = VoxelSpacing(
            x_um=float(input_spacing[0]),
            y_um=float(input_spacing[1]),
            z_um=float(input_spacing[2]),
        )
        if _same_spacing(render_spacing, source_spacing):
            return image, None

        resample = vtkImageResample()
        resample.SetInputData(image)
        self._configure_resample(resample, render_spacing, nearest=nearest)
        return resample.GetOutputPort(), resample

    def _configure_resample(
        self,
        resample: vtkImageResample,
        render_spacing: VoxelSpacing,
        *,
        nearest: bool = False,
    ) -> None:
        if nearest:
            resample.SetInterpolationModeToNearestNeighbor()
        else:
            # Linear interpolation retains detail better than cubic for these
            # sparse channel volumes while avoiding excessive side-view blur.
            resample.SetInterpolationModeToLinear()
        resample.SetAxisOutputSpacing(0, float(render_spacing.x_um))
        resample.SetAxisOutputSpacing(1, float(render_spacing.y_um))
        resample.SetAxisOutputSpacing(2, float(render_spacing.z_um))

    def _set_pipeline_input(self, consumer, image_input) -> None:
        if isinstance(image_input, vtkImageData):
            consumer.SetInputData(image_input)
            return
        consumer.SetInputConnection(image_input)

    def _numpy_to_vtk_image(self, volume: np.ndarray, spacing: VoxelSpacing) -> vtkImageData:
        z, y, x = volume.shape
        vtk_img = vtkImageData()
        vtk_img.SetDimensions(x, y, z)
        vtk_img.SetSpacing(spacing.x_um, spacing.y_um, spacing.z_um)
        vtk_img.SetOrigin(0.0, 0.0, 0.0)

        vtk_img.GetPointData().SetScalars(self._numpy_to_vtk_scalars(volume))
        logger.debug(
            "VTK image created: dims=%s spacing=%s scalar_count=%d",
            vtk_img.GetDimensions(),
            vtk_img.GetSpacing(),
            vtk_img.GetPointData().GetScalars().GetNumberOfTuples(),
        )
        return vtk_img

    def _numpy_to_vtk_scalars(self, volume: np.ndarray):
        # vtkImageData point ids advance x fastest, then y, then z. A C-order
        # flatten of a (z, y, x) numpy volume preserves that exact order.
        arr = np.asarray(volume)
        if np.issubdtype(arr.dtype, np.integer):
            flat = np.ascontiguousarray(arr.ravel(order="C"))
        else:
            flat = np.ascontiguousarray(arr.ravel(order="C"), dtype=np.float32)
        vtk_array = numpy_to_vtk(flat, deep=False)
        vtk_array.SetName("label" if np.issubdtype(arr.dtype, np.integer) else "intensity")
        return vtk_array

    def _update_vtk_image(self, image: vtkImageData, volume: np.ndarray, spacing: VoxelSpacing) -> None:
        image.SetSpacing(spacing.x_um, spacing.y_um, spacing.z_um)
        image.GetPointData().SetScalars(self._numpy_to_vtk_scalars(volume))
        image.Modified()
        logger.debug(
            "VTK image updated: dims=%s spacing=%s scalar_count=%d %s",
            image.GetDimensions(),
            image.GetSpacing(),
            image.GetPointData().GetScalars().GetNumberOfTuples(),
            _volume_debug_summary(volume),
        )

    def _build_volume_mapper(
        self,
        image_input,
        render_spacing: VoxelSpacing,
        *,
        nearest: bool = False,
    ):
        mapper = vtkSmartVolumeMapper()
        self._set_pipeline_input(mapper, image_input)
        mapper.SetBlendModeToComposite()
        # Prevent quality degradation during camera rotation / angled views.
        mapper.SetAutoAdjustSampleDistances(False)
        mapper.SetInteractiveAdjustSampleDistances(False)
        if hasattr(mapper, "SetMinimumImageSampleDistance"):
            mapper.SetMinimumImageSampleDistance(1.0)
        if hasattr(mapper, "SetMaximumImageSampleDistance"):
            mapper.SetMaximumImageSampleDistance(1.0)
        if hasattr(mapper, "SetImageSampleDistance"):
            mapper.SetImageSampleDistance(1.0)
        # Tighter ray steps improve oblique/side-view crispness.
        mapper.SetSampleDistance(
            _recommended_sample_distance(render_spacing, label_mode=nearest)
        )
        logger.debug(
            "VTK volume mapper configured: sample_distance=%.5f render_spacing=(%.4f, %.4f, %.4f)",
            mapper.GetSampleDistance(),
            render_spacing.x_um,
            render_spacing.y_um,
            render_spacing.z_um,
        )
        return mapper

    def apply_render_config(self, config: RenderConfig) -> None:
        self._current = config
        logger.debug(
            "Applying render config: thresholds=(%.3f, %.3f) opacity=(%.3f, %.3f)",
            config.threshold_green,
            config.threshold_red,
            config.opacity_green,
            config.opacity_red,
        )
        for channel in ("green", "red"):
            if channel not in self._actors:
                continue
            self._apply_channel_properties(channel, config)

        if "green" in self._actors:
            green_shift = (config.offset_x_um, config.offset_y_um, config.offset_z_um)
            self._actors["green"].volume_actor.SetPosition(*green_shift)
            self._actors["green"].iso_actor.SetPosition(*green_shift)

        self.render()

    def _apply_channel_properties(self, channel: str, config: RenderConfig) -> None:
        actor = self._actors[channel]
        if channel == "green":
            threshold = float(config.threshold_green)
            opacity = float(config.opacity_green)
            iso = float(config.iso_green)
            visible = bool(config.show_green)
            show_iso = bool(config.show_iso_green)
            rgb = (0.1, 1.0, 0.2)
            knee = 0.022
            low_opacity_scale = 0.07
        else:
            threshold = float(config.threshold_red)
            opacity = float(config.opacity_red)
            iso = float(config.iso_red)
            visible = bool(config.show_red)
            show_iso = bool(config.show_iso_red)
            rgb = (1.0, 0.2, 0.2)
            knee = 0.018
            low_opacity_scale = 0.15

        threshold = float(np.clip(threshold, 0.0, 1.0))
        opacity = float(np.clip(opacity, 0.0, 1.0))
        iso = float(np.clip(iso, 0.0, 1.0))

        label_count = int(self._component_coloring.get(channel, 0))
        if label_count > 0:
            color_tf = self._component_color_transfer_function(label_count)
            scalar_opacity = actor.volume_property.GetScalarOpacity()
            if scalar_opacity is None:
                scalar_opacity = vtkPiecewiseFunction()
            scalar_opacity.RemoveAllPoints()
            scalar_opacity.AddPoint(0.0, 0.0)
            scalar_opacity.AddPoint(0.5, 0.0)
            scalar_opacity.AddPoint(1.0, opacity * 0.74)
            scalar_opacity.AddPoint(float(label_count) + 0.5, opacity * 0.88)

            gradient_opacity = vtkPiecewiseFunction()
            gradient_opacity.AddPoint(0.0, 1.0)
            gradient_opacity.AddPoint(float(label_count) + 0.5, 1.0)

            actor.volume_property.SetInterpolationTypeToNearest()
            actor.volume_property.SetColor(color_tf)
            actor.volume_property.SetScalarOpacity(scalar_opacity)
            actor.volume_property.SetGradientOpacity(0, gradient_opacity)
            actor.volume_actor.SetVisibility(1 if visible else 0)
            actor.marching.SetValue(0, 0.5)
            actor.iso_actor.SetVisibility(0)
            return

        color_tf = vtkColorTransferFunction()
        color_tf.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        color_tf.AddRGBPoint(max(0.0, threshold - knee), 0.0, 0.0, 0.0)
        color_tf.AddRGBPoint(min(1.0, threshold + knee), rgb[0], rgb[1], rgb[2])
        color_tf.AddRGBPoint(1.0, rgb[0], rgb[1], rgb[2])

        scalar_opacity = actor.volume_property.GetScalarOpacity()
        if scalar_opacity is None:
            scalar_opacity = vtkPiecewiseFunction()
        scalar_opacity.RemoveAllPoints()
        scalar_opacity.AddPoint(0.0, 0.0)
        scalar_opacity.AddPoint(max(0.0, threshold - knee), 0.0)
        scalar_opacity.AddPoint(min(1.0, threshold + knee), opacity * low_opacity_scale)
        scalar_opacity.AddPoint(min(1.0, threshold + (knee * 2.8)), opacity * 0.55)
        scalar_opacity.AddPoint(1.0, opacity)

        gradient_opacity = vtkPiecewiseFunction()
        gradient_opacity.AddPoint(0.0, 0.0)
        gradient_opacity.AddPoint(max(0.01, threshold * 0.3), 0.05)
        gradient_opacity.AddPoint(min(1.0, threshold + 0.06), 0.55)
        gradient_opacity.AddPoint(1.0, 1.0)

        actor.volume_property.SetInterpolationTypeToLinear()
        actor.volume_property.SetColor(color_tf)
        actor.volume_property.SetScalarOpacity(scalar_opacity)
        actor.volume_property.SetGradientOpacity(0, gradient_opacity)
        actor.volume_actor.SetVisibility(1 if visible else 0)
        actor.marching.SetValue(0, iso)
        actor.iso_actor.SetVisibility(1 if (visible and show_iso) else 0)

    @staticmethod
    def _component_color_transfer_function(label_count: int) -> vtkColorTransferFunction:
        color_tf = vtkColorTransferFunction()
        color_tf.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        for label_id in range(1, int(label_count) + 1):
            hue = (0.11 + (0.61803398875 * float(label_id))) % 1.0
            saturation = 0.74 if label_id % 3 else 0.58
            value = 0.98 if label_id % 2 else 0.86
            red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
            x = float(label_id)
            color_tf.AddRGBPoint(x - 0.49, red, green, blue)
            color_tf.AddRGBPoint(x + 0.49, red, green, blue)
        return color_tf

    def capture_snapshot(self, output_path: str | Path, *, scale: int = 2) -> Path:
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.render()
        to_image = vtkWindowToImageFilter()
        to_image.SetInput(self._render_window)
        if hasattr(to_image, "SetScale"):
            to_image.SetScale(max(1, int(scale)))
        if hasattr(to_image, "FixBoundaryOn"):
            to_image.FixBoundaryOn()
        to_image.ReadFrontBufferOff()
        to_image.Update()

        writer = vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(to_image.GetOutputPort())
        writer.Write()
        logger.info("Snapshot saved: %s", path)
        return path

    def render(self) -> None:
        self._render_window.Render()

    def cleanup(self) -> None:
        """Release VTK resources to avoid GPU/memory leaks on window close."""
        self.clear_debug_overlays()
        for actor in self._actors.values():
            self._renderer.RemoveVolume(actor.volume_actor)
            self._renderer.RemoveActor(actor.iso_actor)
        self._actors.clear()
        self._spacing.clear()
        if self._interactor is not None:
            if hasattr(self, "_orientation_marker"):
                self._orientation_marker.SetEnabled(0)
            self._interactor.TerminateApp()
        if self._render_window is not None:
            self._render_window.Finalize()
