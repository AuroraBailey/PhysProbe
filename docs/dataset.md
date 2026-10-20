# Dataset layout and input requirements

The [data index](../data/README.md) records retained samples; [source attribution](../data/sources.md)
provides acquisition links and terms. Full datasets must be obtained separately.
The [bundled examples](../examples/README.md) can run without downloading them.

## Manifests and roots

| Directory | Purpose | Relative-path base |
| --- | --- | --- |
| `data/manifests/mainaxis/` | Official retained images for five targets | Corresponding local source/split root |
| `data/manifests/stresstest/` | Stress images for five targets | Corresponding local source/split root |
| `data/manifests/ablation/` | Diagnostic selections with original partition membership | Processed root containing source/split prefixes |
| `data/manifests/calibration/` | Independent normal convention calibration pairs | Source-specific exported calibration root |

A manifest is a sample index. Raw `.rgbe`/`.exr` paths do not imply that those
files can be passed directly to PNG generation runners. Supply prepared PNG
inputs and aligned GT/masks, preserving the scene/render/view identifiers.

The prepared launcher input root for depth, normal, and albedo has this form:

```text
prepared_data/
├── openroomff_mainaxis/<scene>/<variant>/N_im.png
├── openroomff_stresstest/<scene>/<variant>/N_im.png
├── interiorverse_mainaxis/<scene>/<view>/<frame>_im.png
└── interiorverse_stresstest/<scene>/<view>/<frame>_im.png
```

Roughness uses the OpenRooms-FF folders only. Preserve source-relative GT/mask
paths in the corresponding processed trees. The exact InteriorVerse scene/view
structure follows the selected manifest/export; do not flatten it.

Metallic uses a separate PBR-Rooms root:

```text
pbr_rooms/
├── mainaxis/<category>/<seed>/<variant>/Image/<camera>/Image_*.png
└── stresstest/<category>/<seed>/<variant>/Image/<camera>/Image_*.png
```

Sibling `Metallic/`, `MetallicEvalMask/`, and `ObjectSegmentation/` trees retain
aligned GT, dedicated valid masks, and renderer regions. Preserve metallic GT
precision; the bundled example uses an existing 16-bit PNG export of the raw GT.

## Retained sample selection

Main/stress launchers scan the prepared RGB trees. Before a full run, select the
intended manifest images into those trees and check for extra queries. The
launchers do not materialize manifest-selected datasets for you.

Ablation launchers read their manifest directly. They map OpenRooms
`N_im_5.rgbe` to `N_im.png`, and EXR RGB references to PNG. The input root includes
manifest source/split prefixes. Missing prepared images or required segmentation
maps fail before generation, including during `--dry-run`.

Valid masks are distinct from SAM3 region maps. See
[stress metadata](../data/metadata/stress_assignments/README.md),
[roughness regions](../data/metadata/segmentation/README.md), and
[normal calibration](../data/manifests/calibration/README.md).

## Inspect retained selections

This command reads manifests only; it does not check image files:

```bash
python3 - <<'PY_COUNTS'
import csv
from pathlib import Path
for path in sorted(Path("data/manifests").glob("*/*.csv")):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scenes = {(row["datasource"], row["scene_id"]) for row in rows}
    print(path, "images=", len(rows), "scenes=", len(scenes))
PY_COUNTS
```

The code license does not replace dataset terms. Local `data/raw/`,
`data/intermediate/`, and `data/private/` are ignored by Git.
