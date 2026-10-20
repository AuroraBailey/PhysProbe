# Access-Protocol Ablations

The setting identifiers in this directory follow the paper's **Diagnostic Access
Variants** table and **Access-Protocol Ablations** appendix. They are target-local:
Albedo-A1 adds illumination analysis; Depth-A1 adds an exemplar; Normal-A1 removes
exemplars.

| Target | Paper settings | Subset | Run guide |
| --- | --- | --- | --- |
| Albedo | A0 full, A1 shared analysis, A2 weakened, A3 minimal, B0 fixed exemplar | 96 images, 24 per source/partition cell | [Albedo](albedo/README.md) |
| Depth | A0 RGB-only, A1 exemplar, A2 segmentation | 96 images, 24 per source/partition cell | [Depth](depth/README.md) |
| Normal | A0 full+exemplars, A1 full only, A2 minimal+exemplars, A3 minimal only | 96 images, 24 per source/partition cell | [Normal](normal/README.md) |
| Roughness | A0 RGB-only, A1 soft segmentation, A2 region fill, A3 exemplar | 96 OpenRooms-FF images, 48 per partition | [Roughness](roughness/README.md) |

The current paper does not define a metallic access-ablation matrix. The metallic
ablation manifest in `data/` does not by itself establish an additional paper
experiment; there is no metallic access-ablation runner.

## Shared launchers

Each task has `scripts/run.sh`, `scripts/start.sh` and `scripts/progress.sh`,
backed by [tasks.py](tasks.py). Use canonical model names: `doubao`,
`gpt-image-1.5`, `gpt-image-2.0`, `qwen2`, `wan`, `gemini` where available.
Configure the provider in root `.env`; GPT uses shared Azure OpenAI settings.
`qwen2` selects `qwen-image-2.0`; `wan` selects `wan2.7-image`.
Use the paper setting IDs A0/A1/A2/A3/B0 when scheduling diagnostic runs.

```bash
bash experiments/ablation/depth/scripts/run.sh --model gpt-image-1.5 \
  --input-root /path/to/processed_data --settings A0 A1 A2 --dry-run
```

Inputs are selected from `data/manifests/ablation/<target>_manifest_ablation.csv`.
The input root must contain the processed source/partition tree. RGB references
ending `_im_5.rgbe` resolve to `_im.png`, and `.exr` references resolve to `.png`.
No conversion or resampling occurs in this launcher. Missing prepared files fail
early. A dispatcher-owned `inputs/` tree contains symlinks to precisely the
selected queries and required segmentation maps; other GT channels are not staged.
Segmentation defaults to the input root; `--seg-root` accepts a separate mirror.
For depth manifests without a segmentation column, `N_im.png` pairs with `N_seg.png`.

`--settings` selects paper settings; without it, all settings for the target run.
`--max-generate N` uses the first N manifest records for a smoke test; this subset
is not the complete paper result. `--manifest` accepts an explicit alternative
selection. All variants share the same selected records. Use `--seed` to set the
recorded generation seed and `--deployment` to override the model deployment.

Default outputs are `outputs/ablation/<target>/<model>/<setting>/`.
Normal A0/A2 add an `E1/`, `E2/`, or `E3/` directory. Supply a different
`--output-root` for a separate run. Selection metadata records manifest SHA256,
source/partition counts, query paths and seed; provider runners save actual
prompts and applicable cues in per-image records. `--overwrite` regenerates.

Replace `run.sh` with `start.sh` for detached execution, or `progress.sh` for
expected-image counts (`--watch 10` refreshes). Keep all selection/path/exemplar
arguments consistent. Logs are in `logs/`; `run.pid`, `.run.lock` and
`run_status.json` track execution. Policy skips can leave missing predictions;
process completion does not imply a complete scoreable result.

## Prompt provenance and interpretation

Each task has one `prompts.py`. Albedo, depth, roughness and normal A0 use
the corresponding current main/stress full instruction. Normal A1 removes
exemplar-dependent clauses; A2/A3 use minimal instructions. Normal three-image
inputs are always query RGB, example RGB, example normal. See [protocol](PROTOCOL.md).

## Evaluation contracts

These diagnostic subsets overlap retained main/stress evaluation images by design.
They are excluded from official aggregation. Keep the same evaluator and masks:

- Albedo: MAE balanced across source and partition cells; PSNR/SSIM/LPIPS average images.
- Depth: existing polarity/affine alignment; rank correlation before affine fitting.
- Normal: average diagnostic image metrics; A0/A2 additionally average E1-E3 scores.
  Do not fuse or select exemplar-conditioned predictions.
- Roughness: compute pooled valid-pixel RMSE separately for main and stress, then
  average those two values. This differs from official image-macro RMSE.
  Bin-BMAE and Region-Macro MAE retain their bin/region balancing rules.

These launchers handle generation; they do not silently implement a new metric
aggregation. This directory covers API-backed editor runs. Open-source model inference
(FireRed-Image-Edit, FLUX and LongCat) runs in upstream environments. The normal API launcher exposes Doubao and GPT Image 1.5.

```bash
python3 -m unittest discover -s tests -p 'test_ablation_protocols.py'
```

The offline tests verify selection balance, paper IDs, per-setting image evidence,
shared analysis reuse, exemplar scheduling, output paths, provider identity and
connected-component region fill, with network access blocked.

Current image handling: [shared image I/O policy](../IMAGE_IO.md). See also [diagnostic protocol](PROTOCOL.md).

## Background example

After preparing the required queries and segmentation, use matching selection
options for detached execution and progress:

```bash
bash experiments/ablation/depth/scripts/start.sh --model gpt-image-1.5 \
  --input-root /path/to/processed_data --output-root outputs/ablation/depth/gpt-image-1.5 \
  --settings A0 A1 A2
bash experiments/ablation/depth/scripts/progress.sh --model gpt-image-1.5 \
  --input-root /path/to/processed_data --output-root outputs/ablation/depth/gpt-image-1.5 \
  --settings A0 A1 A2 --watch 10
```

Evaluate saved predictions with the [target tools](../../tools/evaluation/README.md),
then use [aggregation](../../tools/aggregation/README.md) with canonical model
directories and the paper setting IDs as directory names. Generation folders
are not evaluator exports.
