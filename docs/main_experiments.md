# Main-axis and stress workflow

Use the [bundled examples](../examples/README.md) first to check generation and
evaluation on two samples. For full experiments, obtain the source data, prepare
the retained sample trees, and configure the required providers.

## 1. Prepare data

Follow the [dataset layout and input requirements](dataset.md).
Main/stress launchers scan processed directories; they do not read manifest CSVs.
Each directory must contain exactly the retained queries intended for that run.
Use source/split-specific roots and keep material maps, masks, and region maps
aligned with the query RGBs.

## 2. Preview and run generation

The shared launchers use model IDs and options documented in the
[main/stress guide](../experiments/main_stress/README.md).
This bundled-input command makes no API calls:

```bash
bash experiments/main_stress/albedo/scripts/run.sh --model gpt-image-1.5 \
  --dataset openroomff_mainaxis --input-root examples/assets \
  --output-root outputs/examples/albedo --max-generate 1 --dry-run
```

For a prepared full dataset, replace the root and choose one split:

```bash
bash experiments/main_stress/albedo/scripts/run.sh --model gpt-image-1.5 \
  --split mainaxis --input-root /path/to/prepared_data \
  --output-root outputs/main_stress/albedo/gpt-image-1.5
```

Removing `--dry-run` invokes the selected provider. `--max-generate N` limits
queries per dataset and is for smoke runs; omit it for the full retained split.
`--deployment` overrides the selected generation model/deployment name.

## 3. Background execution and progress

Use the same model, paths, dataset selection, limits, and exemplar options:

```bash
bash experiments/main_stress/albedo/scripts/start.sh --model gpt-image-1.5 \
  --split mainaxis --input-root /path/to/prepared_data \
  --output-root outputs/main_stress/albedo/gpt-image-1.5

bash experiments/main_stress/albedo/scripts/progress.sh --model gpt-image-1.5 \
  --split mainaxis --input-root /path/to/prepared_data \
  --output-root outputs/main_stress/albedo/gpt-image-1.5 --watch 10
```

Logs, `run.pid`, and `run_status.json` are stored under the output root. Detached
jobs continue after terminal closure. Inspect logs and remaining-image counts:
a successful process exit does not establish complete prediction coverage.
Use the recorded process-group PID to stop a detached job, for example
`kill -- -12345`, substituting the actual PID.

## 4. Evaluate predictions

Run the [target evaluators](../tools/evaluation/README.md) against matching GT
and masks. Keep source-specific outputs separate. Normal requires separate
E1/E2/E3 evaluations; scores are averaged after source balancing. Roughness
Region-Macro metrics need the corresponding frozen SAM3 color-ID maps.

For the bundled example's existing predictions:

```bash
python3 examples/run.py --stage evaluate \
  --output-root outputs/examples/gpt-image-1.5
```

## 5. Aggregate and compare

Place evaluator CSVs in the documented
[aggregation input layout](../tools/aggregation/README.md), then run:

```bash
python3 tools/aggregation/aggregate_results.py \
  --evaluation-root /path/to/evaluation_exports \
  --output-dir outputs/aggregation --resamples 1000 --seed 42
```

Generation output trees are not aggregation input trees. Aggregate completed
pixel-evaluation CSVs, rather than prediction PNGs or launcher status files.
Review sample coverage and follow the [reproducibility guide](reproducibility.md)
before reporting official results.
