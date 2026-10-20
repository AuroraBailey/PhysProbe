# Normal Access Diagnostics

Settings below follow the paper. Commands run from the repository root with
the [ablation dependencies](../../../docs/install.md) and provider configured.

| Setting | Meaning | Python entry |
| --- | --- | --- |
| A0 | Full instruction + E1/E2/E3 independent exemplar runs | `run_normal_a0.py` |
| A1 | Full instruction, query RGB only | `run_normal_a1.py` |
| A2 | Minimal instruction + E1/E2/E3 independent exemplar runs | `run_normal_a2.py` |
| A3 | Minimal instruction, query RGB only | `run_normal_a3.py` |

Generation instructions are in [prompts.py](prompts.py). See the [shared guide](../README.md) for environment setup, manifest selection, output conventions, prompt provenance and diagnostic aggregation.

A0 and A2 each run the same queries separately under E1, E2 and E3. They use
`experiments/main_stress/normal/examples/image{,2,3}.png` and the corresponding
`normal{,2,3}.png`. Outputs are separated by setting and exemplar.
A0 shares the exact main/stress full prompt. All three-image inputs use query
RGB, example RGB, example normal, matching their prompt numbering. A1 removes
reference-dependent clauses while retaining the full geometry restrictions.

```bash
bash experiments/ablation/normal/scripts/run.sh --model doubao \
  --input-root /path/to/processed_data --settings A0 A1 A2 A3 --dry-run
# For a smaller smoke check, add --exemplar E1 --max-generate 1.
```

Current API models: Doubao and GPT Image 1.5. FireRed (and the other open-source
models FLUX/LongCat) runs through separate local inference code and is outside
this API-ablation directory's scope. The launcher uses the shared E1-E3 paths above.
