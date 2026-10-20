# Roughness main-axis and stress runs

Generation receives query RGB only and the shared ablation A0 instruction via
[prompts.py](prompts.py). The frozen SAM3 region map is used for Region-Macro
evaluation, not as an official generation input.

## Models and entries

| Launcher model ID | Python entry |
| --- | --- |
| `doubao` | [roughness_doubao.py](<Doubao_Seedream_5.0/roughness_doubao.py>) |
| `gpt-image-1.5` | [roughness_gpt_image_15.py](<GPT-Image-1.5/roughness_gpt_image_15.py>) |
| `gpt-image-2.0` | [roughness_gpt_image_2.py](<GPT-Image-2.0/roughness_gpt_image_2.py>) |
| `qwen2` | [roughness_qwen2.py](<Qwen-Image-2.0/roughness_qwen2.py>) |
| `gemini` | [roughness_gemini.py](<Gemini_3_Pro_Image/roughness_gemini.py>) |

Use the shared shell launchers for dataset scheduling. Direct Python entries
have their own arguments; run an entry with `--help` before using it directly.
Configure the selected provider using [API settings](../../../docs/api_configuration.md).

## Run the bundled input

From the repository root, with the installed environment active:

```bash
# Preview only; no provider requests.
bash experiments/main_stress/roughness/scripts/run.sh --model gpt-image-1.5 \
  --dataset openroomff_mainaxis --input-root examples/assets \
  --output-root outputs/examples/roughness --max-generate 1 --dry-run

# Generate one prediction for the bundled input; invokes the selected API.
bash experiments/main_stress/roughness/scripts/run.sh --model gpt-image-1.5 \
  --dataset openroomff_mainaxis --input-root examples/assets \
  --output-root outputs/examples/roughness --max-generate 1
```

These commands generate predictions. To run the bundled generation/evaluation
walkthrough together, use [examples/run.py](../../../examples/README.md), whose
output layout is separate from these task-launcher outputs.

## Background and progress

```bash
bash experiments/main_stress/roughness/scripts/start.sh --model gpt-image-1.5 \
  --dataset openroomff_mainaxis --input-root examples/assets \
  --output-root outputs/examples/roughness --max-generate 1
bash experiments/main_stress/roughness/scripts/progress.sh --model gpt-image-1.5 \
  --dataset openroomff_mainaxis --input-root examples/assets \
  --output-root outputs/examples/roughness --max-generate 1 --watch 10
```

Keep the same model, input/output roots, dataset selection, and sample limit.
`start.sh` runs detached; `progress.sh` checks readable expected image files.
Logs, `run.pid`, and `run_status.json` are stored under the output root. A finished
process may still have provider-skipped queries; inspect logs and remaining counts.

## Full retained splits

`--input-root` contains `openroomff_mainaxis` and `openroomff_stresstest`.
All supported folders are selected by default. `--dataset` narrows the list;
`--split mainaxis` or `--split stresstest` filters it. Prepared queries end in
`*_im.png`. Launchers scan directories rather than manifest CSVs: retain exactly
the intended benchmark queries in those trees.

```text
<output-root>/<dataset>/roughness/<relative-parent>/<frame>_im_roughness.png
```

For a full run, replace the example root with your prepared retained dataset,
choose `--split`, and omit `--max-generate`. `GT_ROOT` and `BASE_OUTPUT_DIR` supply
root defaults; explicit CLI paths take precedence. Relative task-launcher paths
resolve from the repository root. Use separate output roots per target/model.
`--deployment` overrides the generation model/deployment; `--overwrite` requests
regeneration. `--max-generate N` selects the first N queries per dataset.

See [shared launcher options](../README.md), [image I/O](../../IMAGE_IO.md), and
[target evaluation](../../../tools/evaluation/README.md). Main/stress generation
and pixel evaluation are separate stages.
