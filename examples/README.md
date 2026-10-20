# Runnable Examples

Two retained main-axis samples cover the five benchmark targets. This is a small
generation/evaluation walkthrough, not a benchmark score or an additional split.

| Sample | Targets | RGB | Segmentation |
| --- | --- | --- | --- |
| OpenRooms-FF `scene0586_02/main_xml/0`, central view 5 | Depth, normal, albedo, roughness | [RGB](assets/openroomff_mainaxis/scene0586_02/main_xml/0_im.png) | [SAM3 regions](assets/openroomff_mainaxis/scene0586_02/main_xml/0_seg.png) |
| PBR-Rooms `Bathroom/seed_10007/V2_P1_L0`, `camera_12` | Metallic | [RGB](assets/pbr_rooms/mainaxis/Bathroom/seed_10007/V2_P1_L0/Image/camera_12/Image_0_0_0001_12.png) | [Object regions](assets/pbr_rooms/mainaxis/Bathroom/seed_10007/V2_P1_L0/ObjectSegmentation/camera_12/ObjectSegmentation_0_0_0001_12.png) |

## Run from the repository root

From the repository root, install dependencies and configure the shared Azure
credentials in `.env` using [.env.example](../.env.example):

```bash
python3 -m pip install -e ".[full]"
cp -n .env.example .env  # Preserve an existing .env.
# Fill AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_VERSION.

# Check assets and preview all generation/evaluation commands, without API calls.
python3 examples/run.py --dry-run

# Generate one prediction per target and evaluate it. This invokes the Azure API.
python3 examples/run.py --output-root outputs/examples/gpt-image-1.5

# Run one task, or evaluate predictions that already exist.
python3 examples/run.py --target roughness --output-root outputs/examples/gpt-image-1.5
python3 examples/run.py --stage evaluate --output-root outputs/examples/gpt-image-1.5
```

The runner uses existing GPT Image generation scripts, sequentially. Its default
Azure deployment is `gpt-image-1.5`; use `--deployment YOUR_DEPLOYMENT` to select
another image deployment. When changing deployments, also use a separate
`--output-root outputs/examples/YOUR_RUN` to keep predictions separate.
Default outputs are `outputs/examples/gpt-image-1.5/predictions/` and `metrics/`.
Existing outputs follow each generation script's skip behavior.

The root `.env` can configure example paths with `EXAMPLE_ASSET_ROOT=examples/assets`
and `EXAMPLE_OUTPUT_ROOT=outputs/examples/gpt-image-1.5`. Relative paths are
resolved against the repository root, including when launching from another
working directory. `--asset-root` and `--output-root` override these variables.
When relocating assets, keep their directory structure and original contents;
the sample index and checksums remain in `examples/samples.json`.

Depth, albedo, roughness, and metallic use RGB-only inputs. Roughness segmentation
is supplied to the evaluator for Region-Macro metrics, not to generation.
Normal uses the existing fixed reference pair in
`experiments/main_stress/normal/examples/`; the query's normal GT is used only
for evaluation. This is the existing normal one-shot access setting.

LPIPS is disabled by default for a lighter walkthrough. Add `--lpips` to request
it with the evaluation runtime and weights installed. Metrics without LPIPS do
not cover every metric reported in the paper. One sample per target also cannot
reproduce dataset-level or scene-bootstrap results.

## Run stages separately

The two samples, masks, GT maps, and roughness region map are bundled. No full
dataset, SAM3 SDK, or model checkpoint is needed for this walkthrough.
Generation calls Azure; evaluation reads only local files.

```bash
# Generate all five targets without scoring yet.
python3 examples/run.py --stage generate --deployment gpt-image-1.5 \
  --output-root outputs/examples/gpt-image-1.5

# Score the predictions saved by that command.
python3 examples/run.py --stage evaluate \
  --output-root outputs/examples/gpt-image-1.5

# Preview only the roughness task and both of its commands.
python3 examples/run.py --target roughness --dry-run
```

The walkthrough uses the GPT Image 1.5 runner family with the supplied deployment
name. To exercise another model family's entry, use its
[main/stress launcher](../experiments/main_stress/README.md) on the same asset
root. Those task launchers have a different output layout and do not automatically
feed their outputs to `examples/run.py --stage evaluate`.

## Output files

With the explicit output root above:

| Target | Prediction under `outputs/examples/gpt-image-1.5/` |
| --- | --- |
| Depth | `predictions/depth/relative_depth/0_im_relative_depth.png` |
| Normal | `predictions/normal/normal/0_im_normal.png` |
| Albedo | `predictions/albedo/albedo/0_im_albedo.png` |
| Roughness | `predictions/roughness/roughness/0_im_roughness.png` |
| Metallic | `predictions/metallic/mainaxis/Bathroom/seed_10007/V2_P1_L0/Image/camera_12/metallic/Image_0_0_0001_12_metallic.png` |

Each target's scoring CSVs and summary JSON are in `metrics/<target>/`.
Normal uses E1 only in this walkthrough; the full benchmark runs E1/E2/E3.
Use [evaluation commands](../tools/evaluation/README.md) to inspect or repeat
individual scores. A single example cannot reproduce source-balanced tables,
scene-cluster intervals, or the full normal exemplar average.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing Azure settings | Fill the three shared values in `.env`; run `python3 tools/check_env.py --provider azure`. |
| Deployment not found | Pass your actual Azure deployment using `--deployment` and verify account access. |
| Missing dependencies | Install `.[full]` in the interpreter used to run the example. |
| Asset hash mismatch | Restore the bundled asset or supply the unchanged asset tree through `--asset-root`. |
| Evaluation cannot find a prediction | Run the generation stage first and use the same explicit output root. |
| Existing prediction skipped | Choose a fresh output root for a new deployment or protocol. |
| Missing LPIPS weights | Use the default lightweight run or provision the weights before adding `--lpips`. |

## Scene annotations

The [OpenRooms-FF scene annotation example](annotations/openroomff_mainaxis/README.md)
contains the initial Doubao, GLM, and Qwen audits and their merged annotation for
`scene0586_02`. It was not selected for second-stage review. These four-view
scene-audit records are separate from the single-view physical-map walkthrough.

## Files and provenance

[samples.json](samples.json) records original sample IDs, repository-relative
asset paths, and SHA-256 hashes. Files preserve their source directory layout so
that existing evaluators can resolve their inputs.

- `0_im.png`: processed query RGB, 640 × 480.
- `0_depth.dat`: raw depth GT. Use this for scoring, rather than a depth
  visualization. Its header stores height and width as 32-bit integers, followed
  by float32 depth values in meters.
- `0_normal.png`, `0_albedo.png`, `0_roughness.png`: aligned GT maps.
- `0_mask.png`: valid-pixel mask.
- `0_seg.png`: frozen SAM3 RGB color-ID regions used for roughness evaluation.
- PBR-Rooms `Metallic_0001.png`: 16-bit metallic GT; preserve its bit depth.
- PBR-Rooms `MetallicEvalMask_0001.png`: dedicated valid evaluation mask.
- PBR-Rooms `ObjectSegmentation_*.png`: renderer object regions, separate from
  the evaluation mask and SAM3 regions.

The raw OpenRooms-FF manifest reference is `scene0586_02/main_xml/0_im_5.rgbe`;
the example uses its existing processed PNG export. The metallic manifest
references EXR GT; the example includes the existing 16-bit PNG evaluation
export. No RGB/GT values or segmentation labels were modified when bundling.

## Sources and licenses

OpenRooms-FF comes from the [MAIR dataset release](https://github.com/bring728/OpenRooms_FF).
Its [provider license](licenses/OpenRooms-FF.txt) is retained with the example.
PBR-Rooms is the PhysProbe companion source available through its
[Hugging Face release](https://huggingface.co/datasets/pbr-rooms-benchmark/PBR-Rooms/tree/main);
the benchmark manuscript states CC BY-NC 4.0 for that data. SAM3 regions are
derived auxiliary assets, not provider ground truth.

See the [citation guide](../data/sources.md#citation-guide) and
[BibTeX entries](../data/references.bib) for attribution of these samples.
See [Data Sources and Attribution](../data/sources.md) for references and source
terms. The code's MIT license does not relicense these dataset assets.
