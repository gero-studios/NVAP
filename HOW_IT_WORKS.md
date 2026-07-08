# How NVAP works

This document explains NVAP's internals: the processing pipeline a dataset flows
through, the software architecture, and the exact definition of every metric it
reports. It is meant to be read alongside the code in `src/nvap/`.

For installation and day-to-day usage, see [README.md](README.md).

---

## 1. Conventions

- **Array order.** All volumes are 3D NumPy arrays in `(z, y, x)` order. The VTK
  renderer maps these to world axes `(x, y, z)` explicitly; nothing is
  transposed, so analysis coordinates and rendered positions stay aligned.
- **Units.** Lengths are micrometres (µm), areas µm², volumes µm³, unless a field
  name says otherwise (e.g. length density in mm/mm³). Physical scale comes from
  `VoxelSpacing(x_um, y_um, z_um)`; anisotropic spacing is respected everywhere
  distances or volumes are computed.
- **Intensity.** Channel data is normalised to floating point. A structure is
  defined by a single threshold: a voxel belongs to the structure when
  `intensity >= threshold`. The same threshold drives the 3D view and every
  metric.

---

## 2. Data model

Defined in `src/nvap/config/types.py`:

| Type | Role |
|------|------|
| `VoxelSpacing` | Physical voxel size `(x_um, y_um, z_um)`; exposes `voxel_volume_um3`. |
| `ChannelVolume` | One channel: `(z, y, x)` data, `z_indices` (physical slice numbers), and spacing. |
| `DatasetVolume` | The `green` and `red` channels plus their `shared_z_range`. |
| `RenderConfig` | Everything that defines a view/measurement: thresholds, opacities, isosurface toggles, `trim_first_slices`/`trim_last_slices`, `offset_*_um`, `display_z_scale`. |
| `PreprocessConfig` | Preprocessing and green-masking parameters. |
| `MetricsResult` / `MetricsComputation` | Basic per-channel metrics and overlap. |

A single `RenderConfig` instance drives both the renderer and the analysis calls,
so the picture and the numbers never diverge (see §7).

---

## 3. Processing pipeline

A dataset flows through these stages (orchestrated in `src/nvap/pipeline.py` and
driven by the UI in `src/nvap/ui/main_window.py`):

1. **Load** (`io/stack_loader.py`). Read per-slice folders or multi-page TIFFs
   into a `ChannelVolume` per channel, parsing `_z###` indices and channel
   markers. Intensities are normalised to float.

2. **Fill and sync** (`pipeline.fill_and_sync_dataset`,
   `preprocess/missing_slices.py`). Interpolate missing z-slices, then align both
   channels onto one continuous z-range. The overlapping range is stored as
   `shared_z_range` and is where cross-channel overlap is measured.

3. **Preprocess and mask** (`preprocess/enhancement.py`,
   `preprocess/microglia_masking.py`). Flat-field/background correction and
   per-slice contrast normalisation, then green-channel masking with the
   branch-preserving microglia pipeline. The green channel bypasses PSF
   deconvolution by default (`pipeline._green_bypasses_psf`).

4. **Component labelling** (`analysis/microglia_components.py`). The masked green
   volume is separated into individual microglia. Soma-like blobs are detected
   from the Euclidean distance transform (thick interior regions), used as seeds,
   and grown outward so each connected cell — including touching cells — gets its
   own label. Components are ranked largest-first; that ranking is the `Cell`
   index in the viewer.

5. **Render and analyse.** The renderer (`render/vtk_scene.py`) and the analysis
   modules (`analysis/`) both consume the processed dataset and the active
   `RenderConfig`. For mesh rendering only, the volume is resampled toward
   isotropic z-spacing (`preprocess/resample.py`); metrics are always computed on
   the non-resampled data.

6. **Export** (`export/exporters.py`, `export/mesh_export.py`). Metrics to CSV,
   the current view to PNG, and isosurfaces to PLY/OBJ/STL.

Processed volumes are cached in `.nvap_cache/` (`cache/processed_cache.py`) and
reused when the dataset and processing settings are unchanged.

---

## 4. Analytics

All analysis lives in `src/nvap/analysis/`. Every metric below is computed after
applying the active threshold, slice trim, and (for overlap) the registration
offset, so the numbers describe exactly the trimmed structure shown in 3D.

Exports are self-describing: alongside the metric CSVs, `metrics_provenance.csv`
records the settings that produced them (thresholds, trim, offsets, spacing,
compute backend, dataset identity, timestamp, and the on-load cleanup used).

**Automatic on load.** When a dataset finishes loading — including each dataset in
a multi-stack project set — NVAP applies the default thresholds and, unless turned
off, runs the microglia "clean" enhancement (skipped if a cached enhancement
exists) and the speck wipe, so every dataset in a project is cleaned consistently
before it is measured.

### 4.1 Basic per-channel metrics — `metrics.py`

Exported as `metrics.csv`, one row per channel plus an overlap row.

| Column | Definition |
|--------|-----------|
| `voxel_count` | Number of voxels with `intensity >= threshold` (after trim). |
| `volume_um3` | `voxel_count × voxel_volume_um3`. |
| `component_count` | Connected components of the mask (26-connectivity). |
| `largest_component_voxels` | Voxel count of the biggest component. |
| `overlap_voxel_count` | Voxels where green and red masks coincide within the shared z-range, after shifting green by the offset. |
| `overlap_volume_um3` | `overlap_voxel_count × voxel_volume_um3`. |

**Overlap and offset.** The registration offset is converted to a whole-voxel
shift, `round(offset_um / spacing_um)` per axis, and the green mask is shifted by
that amount before intersecting with red. The shift is resolved in physical
z-slice space, so green voxels crossing the shared-range boundary are handled
correctly rather than dropped (`_shifted_overlap_voxels`).

### 4.2 Vascular morphometry — `vascular_analysis.py`

Computed on the red channel; exported as `metrics_vascular.csv` (long format:
`metric,value`).

Method: threshold red → fill enclosed background pockets smaller than 64 voxels
(removes lumen speckle) → label components → build a physical Euclidean distance
field → skeletonise (Lee 3D) → measure radius, topology, tortuosity, and surface.

| Metric | Definition |
|--------|-----------|
| `vessel_volume_fraction` | Vessel voxels ÷ **retained** tissue voxels. Trimmed slices are excluded from both numerator and denominator. |
| `total_length_um` | Sum of skeleton branch lengths (spacing-correct, via `skan`). |
| `length_density_mm_per_mm3` | Centreline length per unit tissue volume (mm/mm³), over the retained tissue. |
| `mean_radius_um` / `median_radius_um` / `max_radius_um` | Vessel radius sampled along the medial axis. The distance field is ridge-corrected (local maximum in a 1-voxel neighbourhood, so an off-ridge skeleton voxel still reads the true radius) and offset by half a voxel to correct the systematic EDT boundary underestimate. |
| `mean_diameter_um` | `2 × mean_radius_um`. |
| `junction_count` | Branch points: skeleton nodes of degree ≥ 3, clustered by connectivity so adjacent junction voxels count once. |
| `junction_density_per_mm3` | Junctions per mm³ of retained tissue. |
| `endpoint_count` | Skeleton nodes of degree 1 (free ends). |
| `segment_count` / `mean_segment_length_um` | Number of skeleton segments and their mean length. |
| `mean_tortuosity` | Mean of geodesic ÷ straight-line length per segment, clamped to `[1, 50)`. |
| `surface_area_um2` | Sum of vessel↔background voxel-face areas. Faces on the volume boundary are excluded, so vessels clipped by the field of view are not counted as real surface. |
| `surface_to_volume_ratio_per_um` | `surface_area_um2 ÷ vessel_volume_um3`. |
| `decussation_candidate_count` / `mean_decussation_z_separation_um` | Conservative crossing detector: `(y, x)` centreline columns containing two or more z-separated runs, and their mean z-separation. |

Topology uses `skan` when available; a dependency-free fallback keeps the
density and radius metrics available otherwise.

### 4.3 Per-cell microglia morphometry — `microglia_analysis.py`

One row per cell in `metrics_microglia.csv`. Each cell is processed inside its
own bounding box for speed.

**Soma segmentation.** The soma is the thick core of the cell: within a component,
the distance-transform peak marks the centre, and voxels above a fraction of that
peak are grown into the soma body (`_segment_soma_body`).

**Skeleton and branches.** The cell is skeletonised, the soma is carved out, and
the remaining skeleton is decomposed with `skan` into branches, junctions, and
lengths. Short terminal twigs are pruned with a thickness-aware rule (a spur is
removed when it is shorter than both a length floor and a multiple of the
structure radius at its root).

**Process tips.** Tip *placement* comes from the visible voxel cloud, not the
skeleton: a geodesic distance field is grown outward from the soma through the
thresholded voxels, and each process terminus is a regional maximum of that field
(`h_maxima`). Maxima from one lamellar "fan" within a merge radius collapse to
their most distal voxel. This is robust to the fragmented endpoints that skeletons
produce at flat process tips. Where `skimage.graph` is unavailable, a
skeleton-endpoint fallback with thickness/visibility gating is used.

**Sholl analysis.** Concentric spherical shells are centred on the soma centroid.
Each shell is one voxel-diagonal thick — thin enough not to merge distinct
processes, thick enough that a radially-travelling process is never skipped.
Intersections per shell are the connected components of skeleton voxels in that
shell; the peak over all shells is reported.

**Distances to vasculature.** A physical distance field to the nearest vessel is
built once. Each cell's distance is the minimum over its voxels, evaluated at the
offset-shifted position. Voxels sitting inside a vessel (distance 0, e.g. from
spectral bleed-through) are excluded so the reported distance is to the nearest
*non-overlapping* cell voxel.

| Column | Definition |
|--------|-----------|
| `voxel_count`, `volume_um3` | Cell size in voxels and µm³. |
| `soma_voxel_count`, `soma_volume_um3` | Soma size. |
| `branch_count` | Distinct terminal process segments. |
| `tip_count` | Process terminals (geodesic maxima). |
| `branch_point_count` | Skeleton junctions (clustered). |
| `total_process_length_um`, `mean_branch_length_um` | Process length totals. |
| `mean_branch_tortuosity` | Mean geodesic/straight-line ratio per branch. |
| `sholl_max_intersections` | Peak shell intersection count. |
| `sholl_critical_radius_um` | Radius at which the peak occurs. |
| `sholl_enclosing_radius_um` | Farthest process distance from the soma centroid. |
| `soma_equivalent_diameter_um` | Diameter of a sphere with the soma's volume. |
| `soma_roundness`, `soma_elongation` | From the soma's covariance eigenvalues (min/max axis ratio, and its inverse). |
| `nearest_cell_to_vessel_um` | Min distance from any cell voxel to a vessel. |
| `soma_to_vessel_um`, `soma_centroid_to_vessel_um` | Soma-body and soma-centre distances to a vessel. |
| `nearest_tip_to_vessel_um` | Min distance from a process tip to a vessel. |
| `tip_near_vessel_component_count`, `tips_near_multiple_vessels` | Distinct vessel components within ~5 µm of the cell's tips, and whether that is ≥ 2. |

### 4.4 Neurovascular association — `neurovascular.py`

Population-level patterns aggregated from the per-cell distances; exported as
`metrics_neurovascular.csv`.

| Metric | Definition |
|--------|-----------|
| `cell_count`, `cells_with_vessel` | Total cells, and cells with a measurable vessel distance. |
| `perivascular_fraction_within_{5,10,20,50}um` | Fraction of cells whose nearest voxel is within that radius of a vessel. |
| `mean_/median_cell_to_vessel_um`, `min_cell_to_vessel_um` | Cell-to-vessel distance summaries. |
| `mean_/median_soma_to_vessel_um`, `..._soma_centroid_to_vessel_um` | Soma distance summaries. |
| `mean_/median_tip_to_vessel_um` | Process-tip distance summaries. |
| `tip_leading_fraction` | Fraction of cells whose nearest process tip reaches a vessel at least as close as the **soma** does — i.e. processes extend ahead of the cell body toward vasculature. (Compared against the soma, not the whole-cell minimum, since the tips are a subset of the cell's voxels.) |

---

## 5. Rendering

`render/vtk_scene.py` builds a VTK scene with one volume actor and one isosurface
actor per channel.

- **Isosurface** (default on): marching cubes at exactly the channel threshold, so
  the rendered surface is the boundary of precisely the voxels the metrics count.
- **Volume cloud:** a translucent rendering whose opacity ramps up around the
  threshold. The ramp is a cosmetic softening for the cloud view only; it does not
  change any measurement.
- **Offset:** the green actor is translated by the same whole-voxel shift the
  overlap metric uses, so the visual overlap matches the reported overlap.
- **Z height scale:** a display-only vertical scale; metrics ignore it.

---

## 6. Runtime and acceleration

`runtime_optimization.py` chooses, at startup, the compute backend, CPU worker
count, and numeric-library thread caps from the local machine profile, and
records them in a `RuntimeOptimization` profile (printable via
`nvap --print-runtime-profile`).

`accelerate.py` implements the GPU filters NVAP uses (Gaussian, uniform, maximum,
and a Frangi-like tubeness response) as Torch convolutions. Backend selection is
defensive in three layers:

1. **Selection** — probe CUDA → ROCm → DirectML → MPS; fall back to CPU if none is
   usable.
2. **Capability probe** — a candidate backend must actually run NVAP's
   convolutions on the device before it is chosen, so an incompatible GPU is
   rejected rather than trusted. DirectML cannot run native 3D convolution, so
   NVAP uses a separable path built from 2D convolutions there.
3. **Per-operation fallback** — any GPU filter that raises falls back to the SciPy
   CPU implementation, so a mid-run GPU failure degrades instead of crashing.

The result: one build runs on GPU where possible and on CPU everywhere else,
producing the same results.

---

## 7. View ↔ measurement consistency

Because a single `RenderConfig` feeds both paths, the three settings that change
what is measured are applied identically to the view and the analysis:

| Setting | View | Analysis |
|---------|------|----------|
| Threshold | Isosurface at `threshold` | Mask `intensity >= threshold` |
| Trim first/last Z | Trimmed slices zeroed before upload | Same slices zeroed before metrics |
| Offset (x/y/z) | Green actor shifted by the rounded voxel offset | Green mask shifted by the same rounded voxel offset |

Voxel spacing and axis order are shared by both, so a structure appears where it is
measured and is measured where it appears.

---

## 8. Module map

| Path | Responsibility |
|------|----------------|
| `app.py` | CLI parsing and GUI bootstrap. |
| `pipeline.py` | Stage orchestration: fill/sync, PSF, preprocessing, mesh prep, thresholds. |
| `config/types.py` | Core dataclasses and defaults. |
| `io/stack_loader.py` | Load image stacks/folders into `ChannelVolume`. |
| `preprocess/` | Missing-slice fill, enhancement, microglia masking, denoisers, PSF, mesh resample. |
| `analysis/metrics.py` | Basic per-channel metrics and overlap. |
| `analysis/microglia_components.py` | Green component labelling / cell separation. |
| `analysis/microglia_analysis.py` | Per-cell microglia morphometry and vessel distances. |
| `analysis/vascular_analysis.py` | Vascular morphometry. |
| `analysis/neurovascular.py` | Population neurovascular association. |
| `render/vtk_scene.py` | VTK 3D scene. |
| `ui/` | Qt/PySide6 interface: main window, control panel, panels, charts, services. |
| `export/` | CSV and mesh export. |
| `cache/processed_cache.py` | Processed-volume cache. |
| `runtime_optimization.py`, `accelerate.py` | Backend/worker selection and GPU filters. |
| `plugins/` | Plugin discovery and the analyzer protocol. |
