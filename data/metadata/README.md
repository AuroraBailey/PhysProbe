# Benchmark Metadata

Metadata describe sampling and evaluation groupings; they are not model inputs.

- [stress_assignments/](stress_assignments/README.md): OpenRooms-FF / InteriorVerse
  scene-level stress labels and PBR-Rooms renderer-defined lighting assignments.

- [segmentation/](segmentation/README.md): roughness SAM3 path conventions and generation methods.

The sample, ground-truth, and mask paths remain in `../manifests/`. These metadata
supplement those manifests rather than replacing them. Normal calibration pairs
are documented under [../manifests/calibration/](../manifests/calibration/README.md).
The bundled [scene-audit example](../../examples/annotations/openroomff_mainaxis/README.md)
is separate from these sampling tables. The metadata do not automatically
reclassify audit records or select review candidates.
