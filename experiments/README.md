# Experiment entry points

| Track | Selection | Guide |
| --- | --- | --- |
| Main-axis and stress | Scan prepared source/split query directories | [main_stress/](main_stress/README.md) |
| Access diagnostics | Read target-specific ablation manifests | [ablation/](ablation/README.md) |
| Bundled walkthrough | Two fixed samples with asset hashes | [examples/](../examples/README.md) |
| External model comparisons | Upstream inference, shared target evaluation | [comparison guide](../docs/comparisons.md) |

Install the needed [dependency group](../docs/install.md) and configure root
[.env](../docs/api_configuration.md). Commands in these guides run from the
repository root. Shared shell entry points are `run.sh`, `start.sh`, and
`progress.sh`; individual Python runners have target/provider-specific options.

## First command

```bash
python3 examples/run.py --dry-run
```

This verifies bundled assets and prints all five generation/evaluation commands
without provider calls. Remove `--dry-run` after configuring Azure to run the
walkthrough. For other models, use the task launchers with the bundled input
roots shown in each target guide.

## Shared contracts

- [Image inputs and outputs](IMAGE_IO.md)
- [Main/stress prompts](main_stress/PROMPTS.md)
- [Diagnostic settings and estimators](ablation/PROTOCOL.md)
- [Pixel evaluation](../tools/evaluation/README.md)
- [Table aggregation](../tools/aggregation/README.md)
- [Reproducibility guide](../docs/reproducibility.md)

Generation saves predictions and run records. Pixel evaluation saves metric
CSVs. Aggregation consumes those CSVs. Use separate roots for each stage and
model. Record the prompt, sampling list, and evaluator settings for every run.
