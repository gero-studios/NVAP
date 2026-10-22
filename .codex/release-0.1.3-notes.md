NVAP 0.1.3 fixes red-channel cleanup, single-scan CSV export, and missing bundled projects, and includes the current vascular and microglia analysis improvements.

### Fixed and improved
- Red cleanup evaluates connections inside the retained Z slices and separates corner-touching debris. Red rendering stays on the cleaned native grid so interpolation cannot recreate detached dots.
- Analytics includes **Export Current Dataset CSV**. Exports compute missing or stale results in the background and include channel metrics, vascular morphometry, per-cell microglia measurements, neurovascular summaries, provenance, and vascular masks.
- Bundled sample projects use persistent locations in packaged builds. Obsolete temporary-folder duplicates are migrated into one entry. Moved projects can be relinked while preserving pins and history.
- First and last Z trims default to **10 slices each**.
- Updated desktop presentation, application icon, and inspector layout.
- Update checks follow newer GitHub releases and select the matching Windows build.

### Windows downloads
- **NVAP-0.1.3-windows-cpu.exe** — standalone Windows x64 CPU build.
- **NVAP-0.1.3-windows-directml.exe** — standalone Windows x64 build with DirectML GPU support and CPU fallback.
- **SHA256SUMS.txt** — checksums for both executables.

Download the appropriate executable and run it. Python does not need to be installed. Existing microscopy source files are not modified by the update.
