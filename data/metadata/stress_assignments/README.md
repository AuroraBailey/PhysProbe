# Stress Lighting Assignments

## Files and provenance

| File | Content |
| --- | --- |
| `scene_stress_slice.csv` | Canonical OpenRooms-FF / InteriorVerse scene assignments; UTF-8 without BOM. |
| `pbr_rooms_lighting_conditions.csv` | L0–L5 renderer lighting definitions. |
| `pbr_rooms_stress_assignments.csv` | Per-image camera and lighting assignments derived from the existing metallic stress manifest using that mapping. |
| `slice_counts.csv` | Scene/image counts per source and slice for each stress manifest. |

These files define the retained scene/image assignments and lighting labels.
Use the join keys below to associate them with target-specific manifests.

## OpenRooms-FF and InteriorVerse

Join `scene_stress_slice.csv` to stress manifests using **`datasource` +
`scene_id`**. The table has 271 unique keys: 166 OpenRooms-FF scenes and 105
InteriorVerse scenes. `split` is `stresstest`; `room_category` is the category
recorded in the assignment table. A semicolon-separated value is a
scene-level list of categories; manifests retain individual image categories.
Category agreement is checked by membership, not equality with the whole list.

The machine-readable labels are `dark_heavy`, `hdr`, `highlight_heavy`, and
`low_light`. Low-light occurs only in OpenRooms-FF. These labels describe slices
based on image appearance, not controlled changes of renderer lighting.

### Assignment fields and current validation

Each `datasource` + `scene_id` has exactly one `stress_slice`. All D/N/A/R
stress samples match a scene assignment, with no multi-slice scene labels.

- `sampling_part`: source partition or render-subset provenance, preserved from
  the assignment table. Some OpenRooms-FF values contain multiple parts separated
  by `;`; these do not indicate multiple slice labels. They are not new path roots.
- `image_count`: retained image count for that source/scene, not its full source
  dataset population. All values match the current stress manifests. D/N/A share
  the same 1,960 images; roughness uses their 709 OpenRooms-FF images.
- `slice_counts.csv`: source-specific scene and image counts by slice and manifest.

Image categories are checked for membership in the scene category list.
Join using source and scene ID, not room category. Scene labels do not by
themselves define RGB thresholds or slice-priority rules.

## PBR-Rooms

| Lighting ID | Machine label (`stress_slice`) | Lighting condition |
| --- | --- | --- |
| L0 | `normal_light` | Normal |
| L1 | `low_light` | Low-light |
| L2 | `hdr` | HDR |
| L3 | `dark_heavy` | Dark-Heavy |
| L4 | `highlight_heavy` | Highlight-heavy |
| L5 | `mixed_temperature` | Mixed-temperature |

Join `pbr_rooms_stress_assignments.csv` to
`../../manifests/stresstest/metallic_manifest_stresstest.csv` using
**`datasource` + `split` + `rgb_path`**. Paths retain the manifest's relative-path
base; no files are moved or converted. `camera_id` is the `camera_N` directory;
`view_id` is the `Vx_Py_Lz` render directory; `lighting_id` is its final `Lz`
component. `pair_id` is `room_category/scene_id/camera_id` and identifies the
matched camera group across lighting conditions.

The table contains 450 unique image assignments from 25 scenes. All 75 camera
groups contain exactly one record for each of L0–L5. The lighting mapping is for
the stress renders; it does not reclassify main-axis L0 renders as a stress split.

## Validation scope

Checks use CSV identifiers only: unique scene/image keys, scene coverage,
category membership, and complete PBR-Rooms camera/light groups. They do not verify
file existence, image content, mask validity, or unchanged material fields across
renders. Recheck these properties when manifests or assignments change.

## Use during aggregation

The aggregation tool supports recorded per-image labels and released scene
metadata. Select `--stress-assignment-mode current` to label unsliced exports
using this table; explicitly labelled slice directories retain their labels.
Check exact sample support separately. See the
[aggregation guide](../../../tools/aggregation/README.md).
