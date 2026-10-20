# Access-Protocol Ablations

See the [ablation run guide](../experiments/ablation/README.md) for the paper
setting matrix, model coverage, manifest selection and metric interpretation.

| Task | Settings | Guide |
| --- | --- | --- |
| Albedo | A0 full, A1 cached query analysis, A2 weakened, A3 minimal, B0 exemplar | [Albedo](../experiments/ablation/albedo/README.md) |
| Depth | A0 RGB-only, A1 exemplar, A2 segmentation | [Depth](../experiments/ablation/depth/README.md) |
| Normal | A0 full+exemplars, A1 full-only, A2 minimal+exemplars, A3 minimal-only | [Normal](../experiments/ablation/normal/README.md) |
| Roughness | A0 RGB-only, A1 soft segmentation, A2 region fill, A3 exemplar | [Roughness](../experiments/ablation/roughness/README.md) |

Setting IDs are defined separately for each target.
The paper does not define a metallic access-ablation matrix. Open-source model inference
(FireRed, FLUX and LongCat) is outside this API-ablation directory's scope.

```bash
bash experiments/ablation/depth/scripts/run.sh --model doubao \
  --input-root /path/to/processed_data --settings A0 A1 A2 --dry-run
```

Use `start.sh` for detached execution and `progress.sh` for progress. All three
read root `.env`; input paths and selection options must agree. The default
manifest contains 96 queries per target. These are diagnostic subsets of the
retained main/stress data and do not contribute alternative official scores.
Normal A0/A2 run E1-E3 separately and average scores, not predictions.
