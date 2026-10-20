# Normal main-axis and stress runs

Each query is processed independently with one fixed RGB/normal exemplar pair.
E1, E2, and E3 run separately. [prompts.py](prompts.py) specifies the input order:
query RGB, example RGB, example normal. Average scores after evaluation; do not
fuse the predictions.

## Models and entries

| Launcher model ID | Python entry |
| --- | --- |
| `doubao` | [normal_doubao.py](<Doubao_Seedream_5.0/normal_doubao.py>) |
| `gpt-image-1.5` | [normal_gpt_image_15.py](<GPT-Image-1.5/normal_gpt_image_15.py>) |

Use the shared shell launchers for dataset scheduling. Direct Python entries
have their own arguments; run an entry with `--help` before using it directly.
Configure the selected provider using [API settings](../../../docs/api_configuration.md).

## Run the bundled input

From the repository root, with the installed environment active:

```bash
# Preview only; no provider requests.
bash experiments/main_stress/normal/scripts/run.sh --model gpt-image-1.5 \
  --dataset openroomff_mainaxis --input-root examples/assets \
  --output-root outputs/examples/normal --max-generate 1 --exemplar E1 --dry-run

# Generate one prediction for the bundled input; invokes the selected API.
bash experiments/main_stress/normal/scripts/run.sh --model gpt-image-1.5 \
  --dataset openroomff_mainaxis --input-root examples/assets \
  --output-root outputs/examples/normal --max-generate 1 --exemplar E1
```

These commands generate predictions. To run the bundled generation/evaluation
walkthrough together, use [examples/run.py](../../../examples/README.md), whose
output layout is separate from these task-launcher outputs.

## Background and progress

```bash
bash experiments/main_stress/normal/scripts/start.sh --model gpt-image-1.5 \
  --dataset openroomff_mainaxis --input-root examples/assets \
  --output-root outputs/examples/normal --max-generate 1 --exemplar E1
bash experiments/main_stress/normal/scripts/progress.sh --model gpt-image-1.5 \
  --dataset openroomff_mainaxis --input-root examples/assets \
  --output-root outputs/examples/normal --max-generate 1 --exemplar E1 --watch 10
```

Keep the same model, input/output roots, dataset selection, and sample limit.
`start.sh` runs detached; `progress.sh` checks readable expected image files.
Logs, `run.pid`, and `run_status.json` are stored under the output root. A finished
process may still have provider-skipped queries; inspect logs and remaining counts.

## Full retained splits

`--input-root` contains `openroomff_mainaxis`, `openroomff_stresstest`, `interiorverse_mainaxis`, and `interiorverse_stresstest`.
All supported folders are selected by default. `--dataset` narrows the list;
`--split mainaxis` or `--split stresstest` filters it. Prepared queries end in
`*_im.png`. Launchers scan directories rather than manifest CSVs: retain exactly
the intended benchmark queries in those trees.

```text
<output-root>/<E1|E2|E3>/<dataset>/normal/<relative-parent>/<frame>_im_normal.png
```

The full default runs E1/E2/E3. `--exemplar E1` selects a single-reference smoke
run. Shared pairs are `examples/image.png`/`normal.png`, `image2.png`/`normal2.png`,
and `image3.png`/`normal3.png`. Direct runners accept `--example_rgb` and
`--example_normal` for an explicit pair.

For a full run, replace the example root with your prepared retained dataset,
choose `--split`, and omit `--max-generate`. `GT_ROOT` and `BASE_OUTPUT_DIR` supply
root defaults; explicit CLI paths take precedence. Relative task-launcher paths
resolve from the repository root. Use separate output roots per target/model.
`--deployment` overrides the generation model/deployment; `--overwrite` requests
regeneration. `--max-generate N` selects the first N queries per dataset.

See [shared launcher options](../README.md), [image I/O](../../IMAGE_IO.md), and
[target evaluation](../../../tools/evaluation/README.md). Main/stress generation
and pixel evaluation are separate stages.
