# Reproducibility guide

Use the retained manifests, fixed prompts and reference assets, and target-specific
evaluation settings for each experiment. Run commands are provided in the
[main experiment guide](main_experiments.md), [ablation guide](ablations.md), and
[runnable examples](../examples/README.md).

## Execution protocol

- Provider inputs and saved responses follow the [image I/O policy](../experiments/IMAGE_IO.md).
- Valid masks are enabled by default. Prediction interpolation is nearest for
  normal, bicubic for albedo, and bilinear for depth, roughness, and metallic.
- Roughness official RMSE/SSIM use image means. Diagnostic Balanced RMSE pools
  squared errors within each partition before balancing main and stress.
- Main/stress launchers scan prepared directories. Ablations read manifests.
  Aggregation count checks do not establish exact sample identity.
- Normal E1/E2/E3 are separate inference runs, combined only at score level.

## Experiment records

Retain model/backend and deployment IDs, exact prompts, input order, seed and
whether it was sent, exemplar IDs, actual request parameters, timestamps,
per-sample completion/failure status, prediction dimensions and hashes, and the
evaluator configuration. Use separate output directories for each run.

Use the same frozen valid masks and segmentation maps across evaluated methods.
SAM3 provides `standard` and `contextual` workflows; record the selected method,
checkpoint, and configuration when generating region maps. The bundled roughness
example includes its region map.

For scene annotation, retain the selected-scene list, model annotations, merged
labels, and any review decisions. Review selection is performed separately from
the reviewer command.

## Release scope

The repository includes sample and calibration manifests, prompts, reference
assets, scene-audit utilities, runnable examples, evaluators, and aggregation code.
Full datasets, provider credentials, model weights, complete proprietary outputs,
and a complete frozen run-record collection are not bundled. Obtain external
assets through the [data sources](../data/sources.md) and model providers.
