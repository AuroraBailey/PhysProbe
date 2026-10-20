# Benchmark Data Index

This directory contains the retained sample manifests and stress metadata for
reproducing the benchmark. Full source images, dense maps, and predictions are
not bundled here. Obtain source datasets from their official providers and
observe their original licenses; the code license does not relicense source data.

See [Data Sources and Attribution](sources.md) for official download/request
entries, target coverage, citations, and source-specific licensing terms,
including the PBR-Rooms Hugging Face release.

```text
data/
├── sources.md             # Official sources, access terms, and citation guide
├── references.bib         # Source-paper and dataset references
├── manifests/
│   ├── mainaxis/       # Five target-specific official sample lists
│   ├── stresstest/     # Five target-specific stress sample lists
│   ├── ablation/       # Five target-specific diagnostic sample lists
│   └── calibration/    # Independent normal coordinate-convention samples
└── metadata/
    ├── stress_assignments/  # Scene slices, renderer lighting, paired cameras
    └── segmentation/       # Roughness region-map conventions
```

## Sample manifests

Each manifest records `datasource`, `split`, `room_category`, `scene_id`,
`rgb_path`, the target ground-truth path, and `mask_path`. Metallic and roughness manifests also
record `segmentation_path`. Paths are relative to the corresponding local source
and split root, not this Git repository. Ablation paths retain their source/split
prefixes; configure roots accordingly. In ablation manifests, `split` records the
original main/stress membership, not a training split.

Some manifests reference raw EXR/RGBE channels. They are sample indexes, not a
claim that every path can be passed directly to a PNG-based evaluator. Follow the
target's required encoding and mask contract.

## Stress assignments

See [metadata/stress_assignments/README.md](metadata/stress_assignments/README.md)
for source-specific scene labels, PBR-Rooms L0–L5 definitions, camera pairing,
join keys, and slice counts. These are sampling/evaluation metadata, not model
inputs. Keep one canonical copy of each assignment table in that directory.

Local datasets can remain outside the repository (for example, on NAS).
`data/raw/`, `data/intermediate/`, and `data/private/` remain ignored by Git if
used locally. No train/validation/test demo split is part of the benchmark.

## Derived segmentation

Roughness SAM3 region-map paths and conventions are described in
[metadata/segmentation/](metadata/segmentation/README.md). Region maps are separate
from mandatory valid masks. Obtain or generate the matching region maps before
evaluating a retained split.

## Normal calibration

[manifests/calibration/](manifests/calibration/README.md) contains 50 OpenRooms-FF
and 50 InteriorVerse RGB/normal pairs for the shared Lotus/StableNormal x-axis
convention adapter. These are separate from official scoring samples and normal
editor exemplars. Paths are relative to source-specific exported calibration
roots, with source/archive references retained in separate columns.

## Retained sample counts

| Target | MainAxis images | Stress images | Diagnostic manifest images |
| --- | ---: | ---: | ---: |
| Depth | 3,234 | 1,960 | 96 |
| Normal | 3,234 | 1,960 | 96 |
| Albedo | 3,234 | 1,960 | 96 |
| Roughness | 1,125 | 709 | 96 |
| Metallic | 960 | 450 | 120 |

Diagnostic manifests preserve original main/stress membership and overlap those
partitions by design. The metallic diagnostic manifest is a retained selection;
it has no corresponding paper access-ablation runner. Calibration has 50 pairs
per source, separate from these counts.

For the runnable two-sample walkthrough, use [examples/](../examples/README.md).
For a full run, follow [dataset layout](../docs/dataset.md) and prepare only the
retained queries before invoking directory-scanning main/stress launchers.
