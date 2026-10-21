# Paper result aggregation

This directory provides paper result aggregation. The tool reads
completed evaluator CSVs; it neither calls models nor re-evaluates image pixels.

```bash
python3 tools/aggregation/aggregate_results.py \
  --evaluation-root /path/to/Experiment_Evaluation2 \
  --output-dir outputs/aggregation
```

The input root contains `1_mainaxis`, `2_stresstest`, `3_comparison`.
NumPy is required. Default bootstrap: 1,000 scene-cluster draws, seed 42,
PCG64, linear percentile quantiles. Use `--resamples 0` for point estimates only.
All input trees are read-only. Outputs are written only under `--output-dir`.

## Input layout

This tool consumes evaluator exports, not generation output folders. Preserve
source/scene/view identifiers and the required per-target CSV columns.

```text
evaluation_exports/
├── 1_mainaxis/<target>/<model>/<source>_mainaxis/<target>_metrics.csv
├── 1_mainaxis/normal/<model>/E1/<source>_mainaxis/normal_metrics.csv
├── 2_stresstest/<target>/<model>/<slice>/<source>_stresstest/<target>_metrics.csv
└── 3_comparison/<target>/<model>/<setting>/<source>_<partition>/<target>_metrics.csv
```

Normal uses separate `E1`, `E2`, and `E3` directories. Metallic uses
`metallic_scene_metrics.csv` rather than the image-metric CSV. All tracks use
canonical model directory names, including `GPT-Image-1.5`, `GPT-Image-2.0`,
`Doubao_Seedream_5.0`, `Gemini_3_Pro_Image`, `Qwen-Image-2.0`, and `WAN2.7-Image`.
FireRed normal diagnostics use `FireRed-Image-Edit`.

Diagnostic setting directories are named `A0`, `A1`, `A2`, `A3`, or `B0` where
defined for that target. The ID is read directly; for example, Albedo `A0` is
always the full RGB-only setting. No directory-name conversion, setting override,
or duplicate-export exclusion mapping is supported. Unsupported setting IDs are
skipped and recorded in `audit.json`. Use only the intended evaluation export
for each sample, model, setting, and exemplar.

Normal diagnostic A0/A2 additionally require an `E1`, `E2`, or `E3` directory
beneath the setting. Normal A1/A3 are RGB-only and have no exemplar directory.
Store an all-zero metallic baseline under `metallic/all_black/` using the same
source/slice conventions; duplicate scoring rows are errors. Source directories
contain `openroomff` or `interiorverse`; their spelling is used for source detection.

Select the Stress label source directly with `--stress-assignment-mode`:

```bash
python3 tools/aggregation/aggregate_results.py \
  --evaluation-root /path/to/evaluation_exports \
  --stress-assignment-mode current \
  --output-dir outputs/aggregation/current --resamples 1000 --seed 42
```

`current` uses the released scene-level metadata to label unsliced exports.
Explicit slice directories retain their labels. Check sample support against
the retained manifests before reporting results.

## Estimators

- Depth and albedo official results: average images within each source, then
  give OpenRooms-FF and InteriorVerse equal weight. Low-light is OpenRooms-only.
- Normal: retain the evaluator's `fixed` strategy; average image metrics within
  source, source-balance, then equally average the three independent E1/E2/E3
  runs. Both per-exemplar and `Avg` results are retained. No prediction fusion.
- Roughness official main-axis: image-macro RMSE and SSIM; pool valid pixels
  within each GT bin before averaging nonempty bins; weight each color-ID region
  equally across images. Stress RMSE/SSIM retain image means; bin and region metrics retain their corresponding pooled weights.
- Metallic: use evaluator scene CSVs, whose pixels have already been pooled
  across the views in each scene. Average scene scores equally. AP and IoU must
  not be reconstructed by averaging image AP/IoU. Stress lighting groups remain
  separate; no new pooled AP is inferred from summary numbers.
- Albedo diagnostics: equal source/partition cells. Depth diagnostics: image
  means. Normal A0/A2: three-exemplar score averages. Roughness diagnostics:
  compute valid-pixel-pooled RMSE in each of main/stress and average the two;
  Bin-BMAE pools bins, Region-Macro MAE weights regions, SSIM averages images.

Bootstrap retains complete scenes including all image rows, and shares draws
across methods, metrics and normal exemplars with identical scene support. Its
primary metrics are AbsRel-AI, Acc@22.5, MAE, and RMSE for depth, normal,
albedo, and roughness respectively; all four metallic metrics get CIs.

## Outputs and checks

- `tables.csv`, `tables.json`: source-specific, source-balanced, exemplar,
  lighting-slice and diagnostic scores. `entries` counts scoring rows (scene
  rows for Metallic). `support_complete_by_manifest_counts` compares image counts
  against the current released manifests; it is a count audit, not a pixel-level
  identity check. Incomplete groups are flagged rather than silently certified.
- `audit.json`: input checksums, unsupported setting/exemplar directories,
  sample-support flags and runtime/configuration information.
- `scene_resampling_indices.npz`, `bootstrap_scene_ids.json`: frozen draws and
  scene order needed to reproduce the confidence intervals.

Duplicate scoring entries fail early. NaN/undefined metrics are not zero-filled.
See [diagnostic protocol](../../experiments/ablation/PROTOCOL.md) for setting definitions.

Changing thresholds or image I/O requires new evaluator outputs, not aggregation.

## Frozen Stress assignments

The default `--stress-assignment-mode` is `recorded`. It reads per-image slice
labels from explicitly labelled evaluator exports, keyed by source, scene,
render variant and view. Conflicting labels for the same image fail early.
The recovered labels are written to `recorded_stress_assignments.csv` without
modifying release metadata. Unsliced exports use these labels when available,
then fall back to released scene metadata.

With `--stress-assignment-mode current`, unsliced exports use the released
scene metadata directly. Explicit slice directories retain their labels.
Manifest and recorded-export support counts are reported separately.

Audit file paths are relative to `--evaluation-root`; configurable roots and
machine-specific absolute paths are not persisted in release reports.

## Confidence intervals

The command above is the single entry point for point estimates and confidence
intervals. Configure
`--evaluation-root`, `--output-dir`, `--resamples`, and `--seed` explicitly when
needed; inputs are never modified. Main-axis confidence intervals are included
in `tables.csv`/`tables.json`, and the scene IDs and resampling indices are saved
alongside them.

## Inspect results

```bash
python3 - <<'PY_CHECK'
import csv
from pathlib import Path
path = Path("outputs/aggregation/tables.csv")
with path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
incomplete = sum(row.get("support_complete_by_manifest_counts") == "False" for row in rows)
print(path.name, "groups=", len(rows), "incomplete_by_counts=", incomplete)
PY_CHECK
```

Check `audit.json` and the support flags. Counts
do not establish exact sample membership.
