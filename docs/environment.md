# Runtime environment

Start with [installation](install.md) and [API configuration](api_configuration.md).
All Python commands must use the environment where the repository dependencies
were installed. Activate that environment before invoking shell launchers.

## Components

| Component | Runtime requirements |
| --- | --- |
| API generation | `main` extra; provider credentials |
| Access ablations | `ablation` extra; prepared inputs and setting-specific cues |
| Evaluation | `evaluation` extra; saved predictions, GT, valid masks |
| Aggregation | NumPy and completed evaluator CSVs |
| Scene-audit merging | Python standard library |
| Scene-audit API calls | `requests`, `Pillow`, `python-dotenv`; audit credentials |
| SAM3 generation | External SAM3 SDK/checkpoint, PyTorch, NumPy, Pillow; SciPy for contextual repair |

LPIPS may download pretrained weights on first use. Prepare those weights in
advance for offline evaluation. The example walkthrough disables LPIPS by
default; add `--lpips` to enable it. EXR GT requires the EXR dependencies;
normal/albedo/roughness PNG workflows do not require EXR decoding.

## Interpreter and path configuration

Python runners load repository-root `.env`; the current working directory does
not determine which `.env` is read. Documentation commands assume the repository
root so that relative CLI paths and shell commands remain easy to follow.

Shell launchers use `PYTHON_BIN` from the launching environment, or `python3`,
to load configuration. That interpreter needs `python-dotenv`. `--python` chooses
the child generation interpreter in the task dispatcher.

```bash
export PYTHON_BIN="$(pwd)/.venv/bin/python"
python3 tools/check_env.py --provider azure
bash experiments/main_stress/albedo/scripts/run.sh --model gpt-image-1.5 \
  --dataset openroomff_mainaxis --input-root examples/assets \
  --output-root outputs/examples/albedo --max-generate 1 --dry-run
```

Optional `GT_ROOT` and `BASE_OUTPUT_DIR` provide launcher path defaults; explicit
`--input-root` and `--output-root` override them. Keep outputs outside input data
roots and use a separate output root for each target/model/run.

## Common setup issues

| Symptom | Check |
| --- | --- |
| `ModuleNotFoundError` | Activate the intended environment and install the required extra there. |
| Missing provider configuration | Edit `.env`; run `tools/check_env.py --provider ...`. |
| Prepared PNG not found | Resolve the manifest against the correct data root and check the prepared PNG input tree. |
| Missing mask/region map | Supply the aligned valid mask and frozen segmentation required by the metric. |
| LPIPS/EXR unavailable | Install evaluation dependencies and required weights/file-format support. |
| Empty progress or wrong output lookup | Reuse the generation command's model, dataset, paths, limit, and exemplar selection. |

Provider presence checks and local dry-runs do not verify live account access or
model availability. Record actual request settings following the
[reproducibility guide](reproducibility.md).
