# Depth Access Diagnostics

Settings below follow the paper. Commands run from the repository root with
the [ablation dependencies](../../../docs/install.md) and provider configured.

| Setting | Meaning | Python entry |
| --- | --- | --- |
| A0 | RGB only | `run_depth_a0.py` |
| A1 | Fixed reference RGB/depth pair + query RGB | `run_depth_a1.py` |
| A2 | Query RGB + aligned segmentation | `run_depth_a2.py` |

Generation instructions are in [prompts.py](prompts.py). See the [shared guide](../README.md) for environment setup, manifest selection, output conventions, prompt provenance and diagnostic aggregation.

A1 uses the fixed pair in `examples/image.png` and `examples/depth.png`, ordered
reference RGB, reference depth, query RGB. A2 sends query RGB and its paired
`N_seg.png`. All outputs are relative depth,
requesting white-near/black-far; evaluator polarity selection and affine fitting
remain evaluation operations. Default exemplar paths point to `examples/`.

```bash
bash experiments/ablation/depth/scripts/run.sh --model qwen2 \
  --input-root /path/to/processed_data --settings A0 A1 A2 --dry-run
```

Depth A0 matches the current main/stress depth instruction. A1
inherits the same core instruction, and A2 uses the same brighter-near/darker-far
wording. Prompt version IDs were updated for this wording change.
