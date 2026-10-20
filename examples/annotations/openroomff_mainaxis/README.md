# OpenRooms-FF Scene Annotation Example

These records contain the three initial model audits and their merged annotation
for `scene0586_02`.

```text
openroomff_mainaxis/
├── doubao/scene0586_02.json
├── glm/scene0586_02.json
├── qwen/scene0586_02.json
└── merged/scene0586_02.json
```

The model directories retain each audit's `raw` response, `parsed` annotation,
`parse_error`, and `computed` lighting statistics. The merged record retains
annotation fields at the top level and model evidence in Doubao, GLM, Qwen order.
The merged record includes labels, confidence values, and lighting statistics.

No second-stage review was performed because this scene did not meet the
review-selection criteria. `merged/` contains the three-model merged annotation;
no reviewed annotation is supplied.

## Language

Chinese free text in `raw`, `parsed`, and merged evidence fields has been
translated to English for release. Consequently, `raw` preserves the response
structure and labels but is not a byte-for-byte copy of the original response.
Field names, categorical labels, numerical values, and original view identifiers
are unchanged. Differences between model responses and programmatically
completed parsed lighting fields are retained.

## Scene views and image assets

The audits used four scene views: `0_im_5.png`, `1_im_5.png`, `2_im_5.png`, and
`4_im_5.png`. Their original identifiers remain in each record's `images` field;
these are audit view identifiers, not paths relative to this directory.

The [runnable physical-map example](../../README.md) bundles only the processed
central-view-5 image for view `0`, at
[scene0586_02/main_xml/0_im.png](../../assets/openroomff_mainaxis/scene0586_02/main_xml/0_im.png),
alongside its GT maps and evaluation auxiliaries. The other three audit views
are not included. The records illustrate a scene-level audit, whereas the
runnable example generates and evaluates physical maps for a single view.

## Reproduce the merge

From the repository root, the offline merger can read this directory layout.
Write to a separate output directory to preserve the supplied merged record:

```bash
python3 annotation/scripts/merge_scene_audits.py \
  --root examples/annotations/openroomff_mainaxis \
  --doubao-dir doubao --glm-dir glm --qwen-dir qwen \
  --output "$PWD/outputs/examples/annotation/merged" --seed 42
```

This command requires no API calls or image files. See the
[annotation workflow](../../../annotation/README.md) for the merge rules.
