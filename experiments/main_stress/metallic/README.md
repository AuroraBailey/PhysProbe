# Metallic main-axis and stress runs

Generation receives query RGB only. [prompts.py](prompts.py) contains the shared
full instruction and the shorter Qwen/Wan instruction. GT, valid masks, and object
segmentation remain evaluator inputs.

## Models and entries

| Launcher model ID | Python entry |
| --- | --- |
| `doubao` | [metallic_doubao.py](<Doubao_Seedream_5.0/metallic_doubao.py>) |
| `gpt-image-1.5` | [metallic_gpt_image_15.py](<GPT-Image-1.5/metallic_gpt_image_15.py>) |
| `gpt-image-2.0` | [metallic_gpt_image_2.py](<GPT-Image-2.0/metallic_gpt_image_2.py>) |
| `qwen2` | [metallic_qwen2.py](<Qwen-Image-2.0/metallic_qwen2.py>) |
| `wan` | [metallic_wan.py](<Wan2.7-Image/metallic_wan.py>) |
| `gemini` | [metallic_gemini.py](<Gemini_3_Pro_Image/metallic_gemini.py>) |

Use the shared shell launchers for dataset scheduling. Direct Python entries
have their own arguments; run an entry with `--help` before using it directly.
Configure the selected provider using [API settings](../../../docs/api_configuration.md).

## Run the bundled input

From the repository root, with the installed environment active:

```bash
# Preview only; no provider requests.
bash experiments/main_stress/metallic/scripts/run.sh --model gpt-image-1.5 \
  --dataset mainaxis --input-root examples/assets/pbr_rooms \
  --output-root outputs/examples/metallic --max-generate 1 --dry-run

# Generate one prediction for the bundled input; invokes the selected API.
bash experiments/main_stress/metallic/scripts/run.sh --model gpt-image-1.5 \
  --dataset mainaxis --input-root examples/assets/pbr_rooms \
  --output-root outputs/examples/metallic --max-generate 1
```

These commands generate predictions. To run the bundled generation/evaluation
walkthrough together, use [examples/run.py](../../../examples/README.md), whose
output layout is separate from these task-launcher outputs.

## Background and progress

```bash
bash experiments/main_stress/metallic/scripts/start.sh --model gpt-image-1.5 \
  --dataset mainaxis --input-root examples/assets/pbr_rooms \
  --output-root outputs/examples/metallic --max-generate 1
bash experiments/main_stress/metallic/scripts/progress.sh --model gpt-image-1.5 \
  --dataset mainaxis --input-root examples/assets/pbr_rooms \
  --output-root outputs/examples/metallic --max-generate 1 --watch 10
```

Keep the same model, input/output roots, dataset selection, and sample limit.
`start.sh` runs detached; `progress.sh` checks readable expected image files.
Logs, `run.pid`, and `run_status.json` are stored under the output root. A finished
process may still have provider-skipped queries; inspect logs and remaining counts.

## Full retained splits

`--input-root` must contain PBR-Rooms `mainaxis/` and `stresstest/` trees.
`--dataset` accepts those two names; both are selected by default. Queries are
`Image_*.png` files under `Image/camera_*`. Camera-directory groups run sequentially.

```text
<output-root>/<mainaxis|stresstest>/<category>/<seed>/<variant>/Image/<camera>/metallic/<frame>_metallic.png
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
