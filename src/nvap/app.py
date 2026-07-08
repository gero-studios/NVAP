from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        log_dir = Path(os.environ.get("NVAP_HOME") or (Path.home() / ".nvap"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "nvap.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handlers.append(file_handler)
    except OSError:
        pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)


def run_headless_smoke(input_path: str | Path) -> int:
    from nvap.analysis.metrics import compute_metrics
    from nvap.analysis.vascular_analysis import analyze_vasculature
    from nvap.config.types import DEFAULT_SPACING, PSFConfig, PreprocessConfig, RenderConfig
    from nvap.io.stack_loader import load_dataset
    from nvap.pipeline import (
        apply_psf_to_dataset,
        default_green_threshold,
        default_threshold,
        fill_and_sync_dataset,
    )
    from nvap.preprocess.enhancement import preprocess_dataset

    source = Path(input_path).resolve()
    logger.info("Headless smoke start: input=%s", source)
    dataset = load_dataset(source, spacing=DEFAULT_SPACING)
    dataset = fill_and_sync_dataset(dataset)

    # Green is pass-through; only non-green preprocessing paths remain active.
    preprocess_cfg = PreprocessConfig(enabled=True)
    dataset = preprocess_dataset(dataset, preprocess_cfg)

    # Final processing stage
    processed = apply_psf_to_dataset(dataset, PSFConfig(enabled=False, iterations=0), preprocess_config=preprocess_cfg)

    render = RenderConfig(
        threshold_green=default_green_threshold(processed.green.data),
        threshold_red=default_threshold(processed.red.data),
    )
    metrics = compute_metrics(processed, render)
    print(f"NVAP smoke run OK - source={source}")
    for item in metrics.channel_results:
        print(
            f"{item.channel}: voxels={item.voxel_count}, "
            f"volume_um3={item.volume_um3:.3f}, components={item.component_count}"
        )
    print(
        f"overlap: voxels={metrics.overlap_voxel_count}, "
        f"volume_um3={metrics.overlap_volume_um3:.3f}"
    )

    vascular = analyze_vasculature(
        processed.red.data,
        threshold=float(render.threshold_red),
        spacing=processed.red.spacing,
        render=render,
    )
    print(
        "vasculature: "
        f"vol_fraction={vascular.vessel_volume_fraction:.4f}, "
        f"length_um={vascular.total_length_um:.1f}, "
        f"length_density_mm_per_mm3={vascular.length_density_mm_per_mm3:.2f}, "
        f"mean_diameter_um={vascular.mean_diameter_um:.2f}, "
        f"junctions={vascular.junction_count}, segments={vascular.segment_count}, "
        f"tortuosity={vascular.mean_tortuosity:.3f}"
    )
    logger.info("Headless smoke complete.")
    return 0


def run_mesh_export(input_path: str | Path, output_dir: str | Path, fmt: str = "ply") -> int:
    """Full headless pipeline: load -> preprocess -> process -> mesh export."""
    from nvap.config.types import DEFAULT_SPACING, MeshExportConfig, PSFConfig, PreprocessConfig
    from nvap.export.mesh_export import export_dataset_meshes, reconstruct_combined_mesh
    from nvap.io.stack_loader import load_dataset
    from nvap.pipeline import apply_psf_to_dataset, fill_and_sync_dataset, prepare_dataset_for_mesh
    from nvap.preprocess.enhancement import preprocess_dataset

    source = Path(input_path).resolve()
    out = Path(output_dir).resolve()
    logger.info("Mesh export pipeline: input=%s output=%s format=%s", source, out, fmt)

    dataset = load_dataset(source, spacing=DEFAULT_SPACING)
    dataset = fill_and_sync_dataset(dataset)

    # Preprocess
    preprocess_cfg = PreprocessConfig(enabled=True)
    dataset = preprocess_dataset(dataset, preprocess_cfg)

    # Final processing stage
    processed = apply_psf_to_dataset(dataset, PSFConfig(enabled=False, iterations=0), preprocess_config=preprocess_cfg)

    # Prepare for mesh
    visual = prepare_dataset_for_mesh(processed, preprocess_cfg)

    # Export meshes
    mesh_cfg = MeshExportConfig(export_format=fmt)  # type: ignore
    results = export_dataset_meshes(visual, mesh_cfg, out)

    # Combined mesh
    combined = reconstruct_combined_mesh(visual, mesh_cfg, out / f"combined_mesh.{fmt}")

    print(f"NVAP mesh export complete - output={out}")
    for channel, path in results.items():
        print(f"  {channel}: {path}")
    if combined:
        print(f"  combined: {combined}")
    return 0


def run_benchmark_denoise(
    input_path: str | Path,
    output_path: str | Path,
    profile: str = "default",
) -> int:
    """Run green denoise benchmark and write JSON report."""
    from dataclasses import replace as dc_replace

    from nvap.analysis.green_benchmark import run_green_denoise_benchmark
    from nvap.config.types import PreprocessConfig

    source = Path(input_path).resolve()
    out = Path(output_path).resolve()
    logger.info("Benchmark denoise: input=%s output=%s profile=%s", source, out, profile)

    base_cfg = PreprocessConfig(enabled=True)
    profiles = {
        "default": base_cfg,
        "low_snr": dc_replace(
            base_cfg,
            green_branch_protection=0.80,
            green_denoise_multiplier=2.1,
        ),
        "high_snr": dc_replace(
            base_cfg,
            green_branch_protection=0.60,
            green_denoise_multiplier=1.5,
        ),
    }
    cfg = profiles.get(profile)
    if cfg is None:
        print(f"Unknown profile '{profile}'. Available: {', '.join(profiles)}")
        return 1

    result_path = run_green_denoise_benchmark(source, out, cfg)
    print(f"NVAP benchmark complete - report={result_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NVAP - NeuroVascular Analytics Program")
    parser.add_argument(
        "--headless-smoke",
        action="store_true",
        help="Run load/process/metrics pipeline without GUI.",
    )
    parser.add_argument(
        "--input",
        default="Input",
        help="Dataset root path (default: Input).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    parser.add_argument(
        "--compute-backend",
        default=os.environ.get("NVAP_GPU_BACKEND", "auto"),
        choices=["auto", "cpu", "cuda", "rocm", "directml", "mps"],
        help="Compute backend preference. Default: auto-detect best available backend.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=0,
        help="Override auto-selected CPU worker count. Default: auto.",
    )
    parser.add_argument(
        "--print-runtime-profile",
        action="store_true",
        help="Print backend/thread/memory optimization profile and exit.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Safely clear NVAP processed cache files in .nvap_cache.",
    )
    parser.add_argument(
        "--cache-root",
        default=".",
        help="Root folder containing .nvap_cache for --clear-cache (default: current directory).",
    )
    parser.add_argument(
        "--export-mesh",
        action="store_true",
        help="Run full pipeline and export 3D meshes (PLY/OBJ/STL).",
    )
    parser.add_argument(
        "--mesh-output",
        default="meshes",
        help="Output directory for mesh files (default: meshes).",
    )
    parser.add_argument(
        "--mesh-format",
        default="ply",
        choices=["ply", "obj", "stl"],
        help="Mesh export format (default: ply).",
    )
    parser.add_argument(
        "--benchmark-denoise",
        action="store_true",
        help="Run green denoise benchmark and write JSON report.",
    )
    parser.add_argument(
        "--output",
        default="green_denoise_report.json",
        help="Output path for benchmark report (default: green_denoise_report.json).",
    )
    parser.add_argument(
        "--green-denoise-profile",
        default="default",
        help="Denoise profile for benchmark: default, low_snr, high_snr.",
    )
    args = parser.parse_args()
    configure_logging(args.debug)
    from nvap.runtime_optimization import configure_runtime_environment

    runtime_profile = configure_runtime_environment(
        requested_backend=args.compute_backend,
        requested_cpu_workers=args.cpu_workers if args.cpu_workers > 0 else None,
    )
    logger.info(
        "NVAP startup (debug=%s backend=%s workers=%d)",
        args.debug,
        runtime_profile.selected_backend,
        runtime_profile.cpu_workers,
    )

    if args.print_runtime_profile:
        print(runtime_profile.to_json())
        return 0

    if args.clear_cache:
        from nvap.cache.processed_cache import clear_processed_cache

        removed, cache_path = clear_processed_cache(args.cache_root)
        print(f"NVAP cache clear complete - removed={removed} path={cache_path}")
        return 0

    if args.headless_smoke:
        return run_headless_smoke(args.input)

    if args.benchmark_denoise:
        return run_benchmark_denoise(args.input, args.output, args.green_denoise_profile)

    if args.export_mesh:
        return run_mesh_export(args.input, args.mesh_output, args.mesh_format)

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QSplashScreen
    from nvap.ui.loading import build_loading_pixmap
    from nvap.ui.theme import DARK_THEME_QSS

    app = QApplication([])
    app.setStyleSheet(DARK_THEME_QSS)
    splash_pixmap = build_loading_pixmap(
        detail=(
            f"Runtime: {runtime_profile.selected_backend.upper()} "
            f"/ {runtime_profile.cpu_workers} workers"
        ),
        progress_fraction=0.45,
    )
    splash = QSplashScreen(splash_pixmap, Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()

    from nvap.ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    splash.finish(window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
