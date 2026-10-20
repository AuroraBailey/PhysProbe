# Surface-Normal Calibration Set

These manifests index the fixed RGB/normal pairs supplied for the shared normal
coordinate-convention adapter used by Lotus and StableNormal. They are auxiliary
calibration data, not official scoring images or editor convention exemplars.
No images, normals, model outputs, or checkpoints are bundled here.

| Manifest | Source | Images | Scenes |
| --- | --- | ---: | ---: |
| `normal_manifest_openroomff.csv` | OpenRooms-FF | 50 | 50 |
| `normal_manifest_interiorverse.csv` | InteriorVerse | 50 | 50 |

Scene IDs have been checked against the current normal main-axis, stress, and
ablation manifests; there is no intersection for either source. This checks the
released identifiers, not pixel content or any specialist's training data.

## Fixed adapter

Both Lotus and StableNormal use the same prediction-vector conversion:

```text
(x, y, z) -> (-x, y, z)
```

The conversion is fixed; it is not selected independently for each query.
Editors receive no such axis conversion. The existing implementation is
`fixed_pred_flips_for_path()` in
[normal_eval.py](../../../tools/evaluation/normal_eval.py): it recognizes the
`Lotus` and `StableNormal` directory components in prediction paths. Preserve
those model directory names when invoking that evaluator. Do not flip input RGB
images or apply this adapter to ground truth.

These manifests identify the supplied calibration pairs. They do not themselves
contain prediction comparisons, the reasoning used to choose the adapter, or a
model inference step. Using this set does not add a fitting operation to the
official evaluator.

## Fields

Both files use UTF-8 without BOM, forward-slash relative paths, and the same
columns. Source-specific fields may be empty.

| Field | Meaning |
| --- | --- |
| `datasource` | `openroomff` or `interiorverse`, consistent with evaluation manifests. |
| `split` | Always `calibration`; not a training or official scoring split. |
| `sample_id` | Source-native scene/view identity, including source part or render/camera fields where applicable. |
| `sample_index` | Original selection/export order, 1–50 within each source. |
| `scene_id`, `view_id` | Original scene and view identifiers; retain strings such as InteriorVerse `003`. |
| `rgb_path`, `normal_path` | Exported calibration PNG names, relative to that source's exported calibration directory. |
| `source_part` | InteriorVerse partition, such as `part_35`; empty for OpenRooms-FF. |
| `render_variant` | OpenRooms-FF render subset, such as `mainDiffMat_xml`; empty for InteriorVerse. |
| `camera_id` | Original OpenRooms-FF camera/reference-image field; together with `view_id` it identifies the image. |
| `rgb_source_path` | Original processed RGB path relative to the source collection root. |
| `normal_source_path` | Existing normal PNG relative to the source collection root; empty when the source is inside an archive. |
| `normal_archive_path` | Archive filename relative to the configured source archive root. |
| `normal_archive_entry` | Exact normal file path within that archive, including any archive prefix. |
| `normal_export_source_type` | `archive_png`, `existing_png`, or `signed_exr_reexport`. |
| `seed` | Original explicit seed field (42 for OpenRooms-FF). Empty for InteriorVerse because the supplied CSV did not contain this field. |

Do not parse view identifiers as numbers if doing so would remove leading zeros.
Paths do not embed a machine-specific drive or dataset installation directory.

## Resolving paths

Use a separate exported calibration root for each source:

```text
calibration/
├── openroomff/
│   ├── <rgb_path>
│   └── <normal_path>
└── interiorverse/
    ├── <rgb_path>
    └── <normal_path>
```

The original export filenames and row order are retained, so an existing exported
set can be referenced directly without renaming its images.

For source recovery:

- OpenRooms-FF `rgb_source_path` includes `OpenRooms_FF_png/` and is relative to
  the OpenRooms-FF collection directory. Normals are PNG entries inside
  `OpenRooms_FF_geometry.zip`; preserve the `OpenRooms_FF/` entry prefix.
- InteriorVerse `rgb_source_path` and existing `normal_source_path` include
  `part_N_png/`, relative to the InteriorVerse collection directory.
- Six InteriorVerse normals are recorded as `signed_exr_reexport`. Their
  `normal_archive_path` and `normal_archive_entry` locate the original signed EXR;
  `normal_path` names the exported calibration PNG. Reuse the corresponding
  exported PNGs or reproduce their original export procedure before comparing
  predictions. This manifest does not define an EXR-to-PNG conversion formula.

OpenRooms-FF calibration samples can use noncentral local views. The central-view
restriction on official evaluation samples does not apply to calibration.

Source images and annotations retain their respective dataset licenses. The
project code license does not relicense these assets.

## Using the adapter

The manifests describe calibration provenance, not an additional API inference
job. Preserve model directory names when evaluating Lotus/StableNormal outputs:

```bash
python3 tools/evaluation/normal_eval.py --help
```

Choose the matching GT/prediction/mask arguments from the
[evaluation guide](../../../tools/evaluation/README.md). Source calibration
roots and exported filename columns are defined above; those names must not be
substituted with the normal editor exemplars.
