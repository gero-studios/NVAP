The segmented microglia view now shows only accepted cell labels used by analysis. Unassigned fluorescence no longer appears as a green overlay, including when no cells are accepted. Each accepted cell keeps its individual tint in All and isolated views.

The assignment rules are unchanged: an orphan branch network needs a credible cell body or an unambiguous measured connection to an existing cell before it is included in cell metrics.

Validated on the saved native cortex stack: all 13 accepted cells are preserved in both dense and sparse label caches, while 113,454 unassigned above-threshold voxels are hidden. The active threshold and slice trim still apply.

Download the CPU executable for standard use, or DirectML for Windows GPU acceleration. SHA256SUMS.txt contains checksums for both downloads.
