NVAP 0.1.6 improves actual microglia cell and branch assignments, beyond the display-preservation fix in 0.1.5.

### Fixed and improved
- Unassigned signal regions are checked for credible cell bodies at the active threshold, so a brighter neighbouring cell does not prevent detection of a dimmer soma.
- Branches can recover a cell owner through fainter measured signal paths. These paths establish ownership without adding below-mask bridge voxels to measured cells. Fragments with competing owners or no supporting signal path remain unassigned.
- Smaller accepted cells are preserved when larger cells are present; the previous 15,000-voxel preference now affects ranking only.
- Region growth preserves diagonal connections between Z planes.
- The Microglia Viewer reports the percentage of visible signal assigned within the current slice trim. Remaining unassigned signal stays visible in green.

### Verification on the saved cortex reference
At unchanged thresholds, assigned above-threshold signal increased from **68.2% to 92.9%**, and accepted cell candidates increased from **11 to 13**. The correction assigns **398,481 additional above-threshold voxels** to actual cell labels. No previously assigned voxels became unassigned.

All 13 candidates produced per-cell soma and process measurements on the native stack. The test suite passed in both CPU and DirectML development environments (239 passed, 3 skipped in each).

The remaining **7.1%** is retained visibly as unassigned signal. Assignment coverage is not a claim of biological ground-truth accuracy.

### Windows downloads
- **NVAP-0.1.6-windows-cpu.exe** — standalone Windows x64 CPU build.
- **NVAP-0.1.6-windows-directml.exe** — standalone Windows x64 build with DirectML GPU support and CPU fallback.
- **SHA256SUMS.txt** — checksums for both executables.

After updating, rerun **Segmentation** and **Analysis** to refresh cell labels and measurements. Choose **Cell: All** to see the complete signal and assignment coverage.

