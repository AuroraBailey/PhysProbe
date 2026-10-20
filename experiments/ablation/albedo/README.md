# Albedo Access Diagnostics

Settings below follow the paper. Commands run from the repository root with
the [ablation dependencies](../../../docs/install.md) and provider configured.

| Setting | Meaning | Python entry |
| --- | --- | --- |
| A0 | RGB + fixed full instruction | `run_albedo_a0.py` |
| A1 | RGB + fixed full instruction + cached query-only illumination analysis | `run_albedo_a1.py` |
| A2 | RGB + weakened instruction | `run_albedo_a2.py` |
| A3 | RGB + minimal instruction | `run_albedo_a3.py` |
| B0 | RGB + minimal exemplar instruction + fixed RGB/albedo pair | `run_albedo_b0.py` |

Generation instructions are in [prompts.py](prompts.py). See the [shared guide](../README.md) for environment setup, manifest selection, output conventions, prompt provenance and diagnostic aggregation.

A1 must read one shared cache produced by Doubao Seed 2.0 Pro from the query
RGB alone. It never invokes analysis during a model's generation run. Generate
this cache once and use the same `--analysis-dir` for all six editors:

```bash
python3 experiments/ablation/albedo/cache_illumination_analysis.py \
  --input-root /path/to/processed_data --output-root outputs/ablation/shared_albedo_analysis --dry-run

# Create the shared cache; this command invokes the Doubao API.
python3 experiments/ablation/albedo/cache_illumination_analysis.py \
  --input-root /path/to/processed_data --output-root outputs/ablation/shared_albedo_analysis

# Preview all variants after the cache exists.
bash experiments/ablation/albedo/scripts/run.sh --model gpt-image-1.5 \
  --input-root /path/to/processed_data --analysis-dir outputs/ablation/shared_albedo_analysis --dry-run
```

The A1 preview requires the cached files to exist. The generation command reads
this same cache for every selected editor.

Cache records mirror processed RGB paths with `.json` instead of `.png`.
The cache generator saves model ID, manifest hash, input hashes and analysis hashes.
The generation runner copies the per-query cache JSON unchanged into its records.
B0 uses `examples/image.png` and `examples/albedo.png` in the order
reference RGB, reference albedo, query RGB. No GT for the query is sent.

To preview settings that need no analysis cache:

```bash
bash experiments/ablation/albedo/scripts/run.sh --model gpt-image-1.5 \
  --input-root /path/to/processed_data --settings A0 A2 A3 B0 --dry-run
```
