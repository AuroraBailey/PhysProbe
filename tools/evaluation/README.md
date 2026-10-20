# Evaluate saved predictions

Evaluators read saved predictions, aligned ground truth, and valid masks. They
make no generation API calls. Install the `evaluation` extra and run commands
from the repository root. The easiest complete walkthrough is:

```bash
python3 examples/run.py --stage evaluate \
  --output-root outputs/examples/gpt-image-1.5
```

Generate predictions in that output root first using the
[example runner](../../examples/README.md). Evaluation-only mode uses existing
predictions; it does not generate missing images.

## Evaluators and official defaults

| Target | Entry | Prediction interpolation | Main scoring convention |
| --- | --- | --- | --- |
| Depth | `depth_eval.py` | Bilinear | Relative depth; polarity selection and affine alignment |
| Normal | `normal_eval.py` | Nearest-neighbor | RGB-encoded vectors; fixed convention; angular metrics |
| Albedo | `albedo_eval.py` | Bicubic | RGB reflectance fidelity |
| Roughness | `roughness_eval.py` | Bilinear | Image-macro RMSE/SSIM; balanced bin and region errors |
| Metallic | `metallic_eval.py` | Bilinear | Scene-pooled metrics, then equal scene averaging; positives `>= 0.5` |

Masks and color-ID region maps are resized with nearest-neighbor.
Mandatory valid masks are enabled by default. `--disable_mask` is an explicit
opt-out for the four applicable evaluators, not the official benchmark setting.
Use `--mask_fallback none` with the metallic evaluator to require MetallicEvalMask.

## Single-example commands

The following commands match the paths produced by `examples/run.py` with
`--output-root outputs/examples/gpt-image-1.5`. They explicitly set interpolation.
They do not match the different task-dispatcher output layout automatically.

```bash
SAMPLE_ROOT=examples/assets/openroomff_mainaxis/scene0586_02/main_xml
EXAMPLE_RUN=outputs/examples/gpt-image-1.5

python3 tools/evaluation/depth_eval.py \
  --gt_root "$SAMPLE_ROOT" --mask_root "$SAMPLE_ROOT" \
  --pred_root "$EXAMPLE_RUN/predictions/depth/relative_depth" \
  --output_dir "$EXAMPLE_RUN/metrics/depth" --require_mask --resize_interp bilinear

python3 tools/evaluation/normal_eval.py \
  --gt_path "$SAMPLE_ROOT/0_normal.png" \
  --pred_path "$EXAMPLE_RUN/predictions/normal/normal/0_im_normal.png" \
  --mask_path "$SAMPLE_ROOT/0_mask.png" \
  --output_dir "$EXAMPLE_RUN/metrics/normal" --resize_interp nearest

python3 tools/evaluation/albedo_eval.py \
  --gt_path "$SAMPLE_ROOT/0_albedo.png" \
  --pred_path "$EXAMPLE_RUN/predictions/albedo/albedo/0_im_albedo.png" \
  --mask_path "$SAMPLE_ROOT/0_mask.png" \
  --output_dir "$EXAMPLE_RUN/metrics/albedo" --resize_interp bicubic --no_compute_lpips

python3 tools/evaluation/roughness_eval.py \
  --gt_path "$SAMPLE_ROOT/0_roughness.png" \
  --pred_path "$EXAMPLE_RUN/predictions/roughness/roughness/0_im_roughness.png" \
  --mask_path "$SAMPLE_ROOT/0_mask.png" --region_path "$SAMPLE_ROOT/0_seg.png" \
  --output_dir "$EXAMPLE_RUN/metrics/roughness" --resize_interp bilinear --no_compute_lpips

python3 tools/evaluation/metallic_eval.py \
  --gt_root examples/assets/pbr_rooms \
  --pred_root "$EXAMPLE_RUN/predictions/metallic" \
  --output_dir "$EXAMPLE_RUN/metrics/metallic" \
  --resize_interp bilinear --mask_fallback none --gt_threshold 0.5 --pred_threshold 0.5
```

Albedo and roughness commands disable LPIPS for a lightweight example. Enable
`--compute_lpips` with the required package and weights to include that metric.

## Full datasets and diagnostic subsets

Run each evaluator with `--help` for directory pairing options. Normal, albedo,
and roughness support `--gt_dir`, `--pred_dir`, and suffix/path matching controls.
Prepare target-specific GT trees or check the selected pairs carefully: simply
pointing a generic directory evaluator at a mixed-channel directory can select
unintended files. Depth and metallic use their source-specific root conventions.

Match every retained manifest query to its GT, prediction, and mask before
scoring. Preserve source/scene/view identifiers in the output CSVs. Validate both
missing and extra predictions; a process returning successfully is not a sample
identity audit. Evaluate normal E1/E2/E3 separately. Keep main/stress and access
ablation outputs in separate trees.

For roughness batch regions, use `--region_source dir --region_dir ...` and the
matching `--region_suffix _seg`. The evaluator does not read manifest CSV columns
automatically. See [segmentation conventions](../segmentation/README.md).
A2 region filling during generation is distinct from the evaluator's color-ID
region grouping.

## Outputs and aggregation

Evaluators write per-image CSVs and JSON summaries; metallic also writes
`metallic_scene_metrics.csv`. Official metallic aggregation reads that scene
file, because image-wise AP/IoU averages cannot reconstruct pooled scene AP/IoU.
Roughness `main_table_recommendation` uses image-macro RMSE/SSIM.

Use [aggregation](../aggregation/README.md) for source balancing, normal exemplar
averaging, diagnostic estimators, and scene-cluster confidence intervals.
See the [reproducibility guide](../../docs/reproducibility.md) for experiment
recording requirements.
