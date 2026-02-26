from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

from nvap.config.types import RenderConfig, VoxelSpacing

logger = logging.getLogger(__name__)


@dataclass
class _ChannelActors:
    image: vtkImageData
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
        self._renderer.SetBackground(0.04, 0.04, 0.06)
        # Better rendering quality
        self._render_window.SetMultiSamples(4)  # Anti-aliasing
        self._widget.setFocusPolicy(Qt.StrongFocus)
        self._widget.setMouseTracking(True)

        self._interactor = self._render_window.GetInteractor()
        self._interactor_style = vtkInteractorStyleTrackballCamera()
        self._interactor.SetInteractorStyle(self._interactor_style)

        self._actors: dict[str, _ChannelActors] = {}
        self._spacing: dict[str, VoxelSpacing] = {}
        self._current = RenderConfig()

        self._interactor.Initialize()
        self._interactor.Enable()
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

    def set_channel_data(self, channel: str, volume: np.ndarray, spacing: VoxelSpacing) -> None:
        channel = channel.lower()
        if channel not in {"green", "red"}:
            raise ValueError("channel must be 'green' or 'red'.")
        if volume.ndim != 3:
            raise ValueError("volume must have shape (z, y, x).")
        logger.info("VTK set_channel_data: channel=%s shape=%s", channel, volume.shape)

        if channel in self._actors:
            actor = self._actors[channel]
            expected_dims = (int(volume.shape[2]), int(volume.shape[1]), int(volume.shape[0]))
            if tuple(actor.image.GetDimensions()) == expected_dims:
                # Fast path: keep actors/mappers and only replace scalar data.
                self._update_vtk_image(actor.image, volume, spacing)
                unit_distance = float(max(spacing.x_um, spacing.y_um, spacing.z_um) * 1.5)
                actor.volume_property.SetScalarOpacityUnitDistance(max(0.08, unit_distance))
                self._spacing[channel] = spacing
                self.apply_render_config(self._current)
                logger.info("VTK set_channel_data fast-update: channel=%s shape=%s", channel, volume.shape)
                return

            old = self._actors[channel]
            self._renderer.RemoveVolume(old.volume_actor)
            self._renderer.RemoveActor(old.iso_actor)

        image = self._numpy_to_vtk_image(volume, spacing)
        mapper = self._build_volume_mapper(image)
        prop = vtkVolumeProperty()
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()
        prop.IndependentComponentsOn()
        # Enhanced lighting for depth perception
        prop.SetAmbient(0.15)
        prop.SetDiffuse(0.82)
        prop.SetSpecular(0.3)
        prop.SetSpecularPower(24.0)
        unit_distance = float(max(spacing.x_um, spacing.y_um, spacing.z_um) * 1.5)
        prop.SetScalarOpacityUnitDistance(max(0.08, unit_distance))
        volume_actor = vtkVolume()
        volume_actor.SetMapper(mapper)
        volume_actor.SetProperty(prop)

        marching = vtkMarchingCubes()
        marching.SetInputData(image)
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
            volume_actor=volume_actor,
            volume_mapper=mapper,
            volume_property=prop,
            iso_actor=iso_actor,
            marching=marching,
        )
        self._spacing[channel] = spacing
        self.apply_render_config(self._current)
        self._renderer.ResetCamera()
        self.render()
        self.activate_interaction()

    def _numpy_to_vtk_image(self, volume: np.ndarray, spacing: VoxelSpacing) -> vtkImageData:
        z, y, x = volume.shape
        vtk_img = vtkImageData()
        vtk_img.SetDimensions(x, y, z)
        vtk_img.SetSpacing(spacing.x_um, spacing.y_um, spacing.z_um)
        vtk_img.SetOrigin(0.0, 0.0, 0.0)

        vtk_img.GetPointData().SetScalars(self._numpy_to_vtk_scalars(volume))
        return vtk_img

    def _numpy_to_vtk_scalars(self, volume: np.ndarray):
        flat = np.ascontiguousarray(volume.transpose(2, 1, 0).ravel(order="F"))
        vtk_array = numpy_to_vtk(flat, deep=True)
        vtk_array.SetName("intensity")
        return vtk_array

    def _update_vtk_image(self, image: vtkImageData, volume: np.ndarray, spacing: VoxelSpacing) -> None:
        image.SetSpacing(spacing.x_um, spacing.y_um, spacing.z_um)
        image.GetPointData().SetScalars(self._numpy_to_vtk_scalars(volume))
        image.Modified()

    def _build_volume_mapper(self, image: vtkImageData):
        mapper = vtkSmartVolumeMapper()
        mapper.SetInputData(image)
        mapper.SetBlendModeToComposite()
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

        actor.volume_property.SetColor(color_tf)
        actor.volume_property.SetScalarOpacity(scalar_opacity)
        actor.volume_property.SetGradientOpacity(0, gradient_opacity)
        actor.volume_actor.SetVisibility(1 if visible else 0)
        actor.marching.SetValue(0, iso)
        actor.iso_actor.SetVisibility(1 if (visible and show_iso) else 0)

    def capture_snapshot(self, output_path: str | Path) -> Path:
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        to_image = vtkWindowToImageFilter()
        to_image.SetInput(self._render_window)
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
        self.activate_interaction()
