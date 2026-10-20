# Scene audits and annotation utilities

Scene audits record category, lighting descriptors, and plausibility judgments
used for sampling. The scripts are separate from physical-map generation and
evaluation. Run commands from the repository root after installing dependencies
and configuring [provider credentials](../docs/api_configuration.md).

## Tools and records

| Script | Role |
| --- | --- |
| `audit_scenes_doubao.py` | Initial Doubao scene audit |
| `audit_scenes_glm.py` | Initial GLM scene audit |
| `audit_scenes_qwen.py` | Initial Qwen scene audit |
| `merge_scene_audits.py` | Offline three-model voting and evidence merge |
| `review_scene_audits_doubao.py` | Second-stage review of supplied scenes using merged evidence |
| `scene_audit_utils.py` | Scene/image listing, response parsing, lighting statistics |
| `scene_audit_prompts.py` | Audit/review prompt templates |

Initial records contain `scene`, `images`, `raw`, `parsed`, `parse_error`, and
`computed`. Merged records place annotation labels at the top level and preserve
three-model evidence. The fields and merge behavior are described below.

## Offline example

The [bundled scene audit](../examples/annotations/openroomff_mainaxis/README.md)
contains three initial records and a merge for `scene0586_02`. It was not selected
for second-stage review. Reproduce its merge without images or API credentials:

```bash
python3 annotation/scripts/merge_scene_audits.py \
  --root examples/annotations/openroomff_mainaxis \
  --doubao-dir doubao --glm-dir glm --qwen-dir qwen \
  --output "$PWD/outputs/examples/annotation/merged" --seed 42
```

The example's prose is translated to English, including text inside `raw`.
Its original view identifiers and numerical/categorical fields are preserved.

## Audit your scenes

Prepare one directory per scene, with the intended processed views directly
inside it. Current listing selects RGB filenames ending in `_im.png`.
The example audit records retain earlier view identifiers such as `0_im_5.png`;
those identifiers are not a ready-to-rerun four-view image directory.

```text
scene_images/
├── <scene_a>/0_im.png
├── <scene_a>/1_im.png
└── <scene_b>/0_im.png
```

The following three commands invoke APIs. Always provide explicit input/output
paths; use `--limit 1` for an initial scene-level smoke run.

```bash
python3 annotation/scripts/audit_scenes_doubao.py \
  --input /path/to/scene_images --output outputs/annotation/annotation_doubao --limit 1
python3 annotation/scripts/audit_scenes_glm.py \
  --input /path/to/scene_images --output outputs/annotation/annotation_glm --limit 1
python3 annotation/scripts/audit_scenes_qwen.py \
  --input /path/to/scene_images --output outputs/annotation/annotation_qwen --limit 1

python3 annotation/scripts/merge_scene_audits.py \
  --root outputs/annotation --output merged --seed 42
```

Omit `--limit` for the intended full scene set. Confirm model IDs and source
views before a full audit. Merging requires only the standard library and makes
no provider requests. Model order is Doubao, GLM, Qwen.

`--output` is relative to `--root` unless absolute. Custom input names use
`--doubao-dir`, `--glm-dir`, and `--qwen-dir`. Inputs are matched by relative JSON
path; the output must not overlap them. Matching output files are overwritten;
unrelated old outputs remain, so use a fresh output root for a changed scene set.

## Review selected scenes

The current reviewer processes every matching scene supplied to it. Perform
review selection separately and provide a scene directory containing only the
selected scene folders, or use `--scene` for one scene. There is no automatic
threshold-based review-selection stage in this script.

```bash
python3 annotation/scripts/review_scene_audits_doubao.py \
  --input /path/to/selected_review_scenes --merged outputs/annotation/merged \
  --output outputs/annotation/reviewed
```

This command invokes Doubao. The reviewer expects flat `<scene>.json` merged
files. Retain unrevised merged records separately from reviewed records.

## Merge rules

- Model order is always **Doubao, GLM, Qwen**.
- A scene file is merged only when its relative path exists in all three inputs.
  Missing files are skipped and listed in the JSON report printed to stdout.
  Invalid JSON or a non-object record stops the merge with the offending path.
- `scene_category` and the 11 fields listed in `VOTE_FIELDS` use a two-vote
  majority. Without a majority, the three original values remain in a list.
  Missing fields contribute `null`, including the original behavior that two
  nulls can win. String votes are case-insensitive, like PowerShell hashtables.
- `confidence` converts usable values to numbers: repeated values win; otherwise
  it averages the usable values. No usable value produces `null`. Non-finite
  values are excluded so that outputs remain interoperable JSON.
- `lighting_challenge_type`, `reason_short`, `raw`, and `parse_error` preserve the
  three model values in order. `issues` preserves one list per model, without
  flattening or deduplication.
- `images` comes from the record containing the most image names, with ties
  resolved by model order. `computed` comes from a record with that same maximum
  image count, with ties selected using the seeded .NET `System.Random`
  sequence. These two fields can therefore originate from different models.
- The output keeps audit fields at the **top level**, as expected by
  `review_scene_audits_doubao.py`; there is no enclosing `parsed` field.

This reproduces the retained merge policy rather than adding strict scene-label
validation. A record with `parsed: null` is retained as evidence when all three
files exist. The merger does not recompute lighting labels, normalize category
vocabularies, or select retained benchmark scenes. Python sorts relative paths
lexicographically; keep the scene set and seed fixed for reproducible selections.
