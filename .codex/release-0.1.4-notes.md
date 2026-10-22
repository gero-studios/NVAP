NVAP 0.1.4 makes segmented microglia easier to distinguish in 3D: each detected component has its own tint.

### Fixed and improved
- **All components:** distinct tints now appear in both volume and surface rendering.
- **Single-cell inspection:** selecting a cell preserves the tint it had in the All view.
- **Touching cells:** surface rendering preserves each component's boundary and color.
- **Large stacks:** component coloring also works with the memory-saving component cache.
- Green threshold and slice trimming still apply. Enhanced intensities and analysis data are unchanged.

### Using the new colors
Run **Segmentation**, then choose **Cell: All** in the **Microglia Viewer** with **View one microglia** enabled. Choose a cell index to inspect that component individually.

### Windows downloads
- **NVAP-0.1.4-windows-cpu.exe** — standalone Windows x64 CPU build.
- **NVAP-0.1.4-windows-directml.exe** — standalone Windows x64 build with DirectML GPU support and CPU fallback.
- **SHA256SUMS.txt** — checksums for both executables.

Download the appropriate executable and run it. Python does not need to be installed.
