# External comparison methods

Run open-source comparison methods in their upstream inference environments,
using the checkpoint/version and input convention appropriate to the paper.
This repository supplies target evaluators and table aggregation, not local
inference implementations for FireRed, FLUX, LongCat, or specialist models.

## Evaluate upstream predictions

1. Generate predictions for the same retained samples listed in the target
   manifest. Record the checkpoint, code revision, encoding, and sample IDs.
2. Use the [target evaluator](../tools/evaluation/README.md) with the same GT,
   valid mask, interpolation, and metric settings as the API model evaluations.
3. Export the evaluator CSVs under the corresponding model/source directories
   in `1_mainaxis` or `2_stresstest`.
4. Aggregate them using the [table tool](../tools/aggregation/README.md).

Lotus and StableNormal normal predictions use the shared fixed X-axis adapter
in `normal_eval.py`. It is selected by model directory components in prediction
paths; preserve the `Lotus` and `StableNormal` directory names. See the
[calibration manifest documentation](../data/manifests/calibration/README.md).
Do not apply the specialist adapter to editor predictions or GT.

Example aggregation after pixel evaluation:

```bash
python3 tools/aggregation/aggregate_results.py \
  --evaluation-root /path/to/evaluation_exports \
  --output-dir outputs/aggregation --resamples 1000 --seed 42
```

Diagnostic access settings remain in `3_comparison`; they are not alternative
main-axis results. Follow the [reproducibility guide](reproducibility.md) when
recording inference and evaluation settings.
