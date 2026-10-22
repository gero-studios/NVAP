NVAP 0.1.5 fixes microglia structures disappearing from the colored post-segmentation All view.

### Fixed
- The colored All view now preserves the full above-threshold green signal, including voxels that segmentation has not assigned to an accepted cell. Detected cells keep their individual tints; unassigned signal stays green.
- Unassigned signal is display-only and is not counted as an extra cell or added to per-cell analysis.
- Large-volume rendering preserves thin foreground branches during downsampling instead of skipping them between sampled positions.
- Colored surfaces use the same opacity as the original intensity view.

### Verification
- On the saved cortex reference, the old view omitted 511,935 of 1,608,598 above-threshold voxels. The corrected All view preserves the complete threshold mask, with zero missing voxels in both dense and memory-saving cache modes.
- Threshold and slice trimming remain active. Source intensities, cell labels, and analysis measurements are unchanged.

### Windows downloads
- **NVAP-0.1.5-windows-cpu.exe** — standalone Windows x64 CPU build.
- **NVAP-0.1.5-windows-directml.exe** — standalone Windows x64 build with DirectML GPU support and CPU fallback.
- **SHA256SUMS.txt** — checksums for both executables.

Download the appropriate executable and run it. To view the corrected segmented result, run **Segmentation** and choose **Cell: All** in the **Microglia Viewer** with **View one microglia** enabled.
