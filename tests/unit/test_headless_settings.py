from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from nvap.config.headless_settings import (
    build_headless_settings_payload,
    load_headless_settings,
    save_headless_settings,
    settings_channel_overrides,
    settings_dataset_root,
    settings_mesh_config,
    settings_preprocess_config,
    settings_render_config,
)
from nvap.config.types import MeshExportConfig, PreprocessConfig, PSFConfig, RenderConfig, VoxelSpacing


def test_render_config_defaults_reduce_z_height_and_enable_surface_rendering() -> None:
    config = RenderConfig()

    assert config.display_z_scale == 0.5
    assert config.show_iso_green is True
    assert config.show_iso_red is True


def test_headless_settings_round_trip_allows_cli_config_rehydration(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    green = root / "green"
    red = root / "red"
    green.mkdir(parents=True)
    red.mkdir(parents=True)

    payload = build_headless_settings_payload(
        dataset_root=root,
        channel_sources={"green": green, "red": red},
        load_mode="manual",
        spacing=VoxelSpacing(x_um=0.5, y_um=0.6, z_um=0.7),
        psf_config=PSFConfig(enabled=False, iterations=0),
        preprocess_config=replace(PreprocessConfig(), green_branch_protection=0.83),
        render_config=replace(RenderConfig(), threshold_green=0.22, trim_first_slices=3),
        mesh_config=replace(MeshExportConfig(), export_format="stl", smooth_iterations=7),
        microglia_enhancement_method="imagej_rolling_ball",
    )

    path = save_headless_settings(tmp_path / "settings.json", payload)
    loaded = load_headless_settings(path)

    assert settings_dataset_root(loaded, "Input") == root.resolve()
    assert settings_channel_overrides(loaded) == {
        "green": str(green.resolve()),
        "red": str(red.resolve()),
    }
    assert settings_preprocess_config(loaded).green_branch_protection == 0.83
    assert settings_render_config(loaded).threshold_green == 0.22
    assert settings_render_config(loaded).trim_first_slices == 3
    assert settings_mesh_config(loaded).export_format == "stl"
    assert settings_mesh_config(loaded, export_format="ply").export_format == "ply"
