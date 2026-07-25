# NVAP — NeuroVascular Analytics Program

NVAP is a desktop application for loading, viewing, and quantifying two-channel
3D microscopy stacks of brain tissue:

- **Green channel — microglia**
- **Red channel — vasculature**

It loads image stacks, cleans and aligns them, renders both channels together in
interactive 3D, and computes morphometric and spatial-association metrics that
you can export as CSV. It runs on CPU everywhere and uses a GPU automatically
when a supported one is available.

For a full explanation of the processing pipeline and the meaning of every
metric, see **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)**.

---

## What NVAP does

- Loads per-slice image folders or single multi-page TIFF stacks for each channel.
- Fills missing z-slices and aligns the two channels onto a common z-range.
- Cleans the green channel with a branch-preserving masking pipeline.
- Renders each channel as a volume and/or isosurface with independent controls.
- Computes and displays:
  - basic per-channel metrics (voxel count, physical volume, connected
    components, largest component, red–green overlap);
  - vascular morphometry using separate wall and reconstructed-solid masks
    (wall coverage, anatomical volume fraction, centreline length and density,
    radius/diameter, junctions, tortuosity, surface area);
  - per-cell microglia morphometry (branches, tips, process length, Sholl
    profile, soma shape, distance to vasculature);
  - neurovascular association (perivascular fractions, cell/soma/tip-to-vessel
    distances).
- Exports metrics to CSV, a snapshot PNG, and 3D meshes (PLY/OBJ/STL).

---

## Voxel spacing

Metrics are physically calibrated using voxel spacing in micrometres. X, Y,
and Z spacing are editable in the **Voxel Spacing** section of the workbench.
For CZI files, NVAP reads these values from Zeiss metadata and fills the controls
automatically; users can still override any axis manually. Other formats begin
with these fallback values:

| Axis | Spacing |
|------|---------|
| x    | 0.331 µm |
| y    | 0.331 µm |
| z    | 0.4 µm   |

---

## Installing and running from source

Windows / PowerShell:

```powershell
# From the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
nvap --debug
```

If PowerShell blocks script activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

`nvap` launches the GUI. `nvap --debug` launches it with verbose logging shown
in the in-app **Debug Log** panel.

---

## GPU acceleration

NVAP selects a compute backend automatically at startup. It probes CUDA, ROCm,
DirectML, and Apple MPS in that order and falls back to CPU if none is usable.
It also picks CPU worker counts and numeric-library thread limits from the local
CPU/RAM profile. No environment variable is required.

The header status pill shows the active backend, e.g. `GPU DirectML` or
`CPU compute`.

Optional backends are installed as extras:

```powershell
# NVIDIA CUDA / generic Torch GPU
python -m pip install -e ".[denoise_torch]"

# AMD / Intel / other DirectX 12 GPUs on Windows (requires Python 3.12)
python -m pip install -e ".[directml]"
```

**AMD GPUs on Windows:** the DirectML extra is supported on Windows, but the
current dependency stack installs most reliably with Python 3.11. A
project-local Python 3.11 setup:

```powershell
winget install --id astral-sh.uv -e
uv python install 3.11
uv venv --python 3.11 --seed .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
uv pip install -e ".[dev,directml]"
nvap --debug
```

If `pip` is missing in a uv-created environment, recreate the venv with
`--seed` or install dependencies with `uv pip` as shown above.

Force CPU (for troubleshooting) or inspect the runtime decision:

```powershell
$env:NVAP_GPU_BACKEND = "cpu"; nvap
nvap --print-runtime-profile
nvap --compute-backend cpu --cpu-workers 4
```

---

## Loading a dataset

### Accepted input layouts

NVAP accepts either a folder of one-image-per-slice files, or a single
multi-page TIFF, for each channel.

**Per-slice folders.** Filenames must contain a z-index and a channel marker:

- Green: `..._z030c1.png` / `..._z030c1.tif`
- Red:   `..._z030c2.png` / `..._z030c2.tif`

A common auto-detected layout is:

```
Input/Segmented/Green/*.(png|tif|tiff)
Input/Segmented/Red/*.(png|tif|tiff)
```

**Single stack files.** Each `Green` or `Red` folder may instead contain one
multi-page TIFF holding all z-slices (e.g. `green_stack.tif`, `red_stack.tif`).

NVAP also accepts a two-or-more-channel Zeiss CZI file directly. It reads the
first acquisition volume, maps C0 to red/vasculature and C1 to green/microglia,
and imports the X/Y/Z physical scaling metadata. If a CZI contains more than two
channels, additional channels are ignored with a log warning.

**RGB slice folders.** A single folder of RGB PNG/TIFF slices is also accepted
when filenames include `_z###` and pixels are red/green-only (black background
is fine; blue or mixed pixels are ignored).

### Loading in the GUI

1. Click **Load Dataset** (or press **Ctrl+L**).
2. When prompted, choose the source for each channel in this order:
   1. Vasculature (Red)
   2. Microglia (Green)

   For each, pick either a single stack file or an image-sequence folder.
3. NVAP loads, then runs, with an elapsed-time and ETA dialog:
   - channel stack loading,
   - missing-slice interpolation,
   - green microglia masking,
   - initial render and metrics.
4. Adjust thresholds, opacity, and other controls in the left panel.

---

## Using the app

### Simple and advanced modes

NVAP starts in **simple mode**. The core workflow is: **Load Dataset → adjust
thresholds → export.** Toggle **Show advanced controls** to expose masking,
preprocessing, and denoising settings.

### Rendering controls

- **Threshold (green / red):** the intensity cutoff that defines each structure.
  The same threshold drives both the 3D view and every metric, so what you see
  is what is measured.
- **Opacity (green / red):** transparency of each volume.
- **Isosurface toggles:** show a solid surface at the threshold instead of a
  translucent cloud.
- **Trim first/last Z:** exclude edge slices from both the view and all metrics
  (default: 0 slices at each end).
- **Offset (x/y/z):** shift the green channel relative to red to correct
  registration; applied identically to the view and the overlap metric.
- **Z height scale:** a display-only depth scale (default `0.70`; `1.0` matches
  physical spacing). Metrics always use physical spacing regardless of this
  setting.

### Individual microglia viewer

Use the **Microglia Viewer** panel to inspect one green component at a time:

1. Enable **View one microglia**.
2. Select a **Cell** index to isolate that component in 3D.

Components are ranked by size (largest first) at the current green threshold.

### Automatic on load

Every time a dataset finishes loading — including each dataset when you page
through a multi-stack project set — NVAP cleans it up automatically:

1. **Automatic thresholds** are estimated from the processed channels: a
   branch-aware threshold for green microglia and Otsu's intensity threshold
   for red vasculature. Green 0.80 / Red 0.60 are used only as safe fallbacks
   for blank or degenerate channels.
2. **Enhance microglia (clean)** runs the selected enhancement (default
   *Microscopy clean soma/branch*) on the green channel. Skipped if the dataset
   already has a cached enhancement.
3. **Wipe specks** removes small isolated blobs from both channels.

The two cleanup passes are controlled by the **Enhance microglia (clean) on
load** and **Wipe specks on load** checkboxes in the Microglia Workbench (both on
by default). They are always available, so you can turn them off *before* loading
anything. **Speck size** sets the largest connected component (in voxels) treated
as a speck; larger structures are kept. You can also run **Enhance Microglia** and
**Wipe Specks** manually at any time. Use **Auto Thresholds** to recalculate
both channel thresholds from the currently loaded dataset.

### Update mode

Setting changes are debounced and applied automatically. Turn **Auto-apply**
off to batch several edits and apply them together with the Apply button or
**F5**.

### Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+L | Load dataset |
| Ctrl+E | Export metrics CSV |
| Ctrl+S | Export snapshot PNG |
| Ctrl+M | Export 3D mesh |
| Ctrl+A | Toggle auto-apply |
| F5 / Return | Apply pending changes |

---

## Exports

Exporting metrics writes a base CSV plus companion files alongside it. If you
save as `metrics.csv`, you get:

| File | Contents |
|------|----------|
| `metrics.csv` | Per-channel basic metrics and red–green overlap |
| `metrics_provenance.csv` | The settings that produced the numbers (thresholds, trim, offsets, spacing, backend, dataset, timestamp) |
| `metrics_vascular.csv` | Vascular morphometry (red wall mask plus reconstructed solid vessel mask) |
| `metrics_vascular_masks.npz` | `vascular_wall_mask` and `vascular_solid_mask` arrays used for the vascular export |
| `metrics_microglia.csv` | Per-cell microglia morphometry plus cell/soma/tip and nearest-vessel coordinates in voxels and µm |
| `metrics_neurovascular.csv` | Neurovascular association patterns |

Snapshot export writes a `snapshot.png`. Mesh export writes PLY/OBJ/STL files.

The meaning of every column is documented in
**[HOW_IT_WORKS.md](HOW_IT_WORKS.md)**.

---

## Processed cache

Processed volumes are cached in `.nvap_cache/`. Re-loading the same dataset with
the same processing settings reuses the cache and skips reprocessing. Clear it
with:

```powershell
nvap --clear-cache
nvap --clear-cache --cache-root "C:\path\to\root"
```

---

## Command-line reference

`nvap` also runs headless operations without opening the GUI:

| Command | Purpose |
|---------|---------|
| `nvap` | Launch the GUI |
| `nvap --debug` | Launch with verbose logging |
| `nvap --print-runtime-profile` | Print the selected backend/threads/memory profile and exit |
| `nvap --compute-backend {auto,cpu,cuda,rocm,directml,mps}` | Force a compute backend |
| `nvap --cpu-workers N` | Override the auto-selected worker count |
| `nvap --headless-smoke --input Input` | Run load/process/metrics without the GUI |
| `nvap --export-mesh --input Input --mesh-output meshes --mesh-format ply` | Export meshes headless |
| `nvap --benchmark-denoise --input Input --output report.json` | Run the green-denoise benchmark |
| `nvap --clear-cache [--cache-root PATH]` | Remove `.nvap_cache/processed_*.npz` |

Run the tests with:

```powershell
python -m pytest
```

---

## Windows packaging

Build a standalone executable with PyInstaller:

```powershell
.\scripts\build_windows.ps1
```

This produces `dist\NVAP.exe`. The build packages a CPU-safe app by default and
includes DirectML automatically when built on Windows with a compatible Python
(3.12). Launch by double-clicking `dist\NVAP.exe`; do not run anything from
`build\`, which is PyInstaller scratch space.

Force a packaging profile:

```powershell
.\scripts\build_windows.ps1 -Acceleration cpu
.\scripts\build_windows.ps1 -Acceleration directml -PythonExe .\.venv-gpu\Scripts\python.exe
.\scripts\build_windows.ps1 -PackageMode onedir
```

---

## Extending NVAP

NVAP exposes plugin discovery through the `nvap.plugins` entry-point group and a
`ChannelAnalyzerPlugin` protocol (`src/nvap/plugins/`). This is an extension
point for adding custom per-channel analyzers; no analyzers ship by default.
