# Paper access-ablation protocol

| Target | Settings | API models |
| --- | --- | --- |
| Albedo | A0 full RGB-only; A1 shared illumination analysis; A2 weak; A3 minimal; B0 exemplar | Doubao, GPT 1.5, GPT 2, Gemini, Wan 2.7, Qwen Image 2.0 |
| Depth | A0 RGB-only; A1 RGB/depth exemplar; A2 segmentation | Same six |
| Normal | A0 full + E1/E2/E3; A1 full only; A2 minimal + E1/E2/E3; A3 minimal only | Doubao, GPT 1.5 |
| Roughness | A0 RGB-only; A1 soft segmentation; A2 region fill; A3 exemplar | Doubao, GPT 1.5, GPT 2, Gemini |

Normal A0 uses the exact main/stress full prompt. A1 retains its full geometry
restrictions while removing instructions that require exemplar access. A2/A3
use minimal instructions. Whenever exemplars are supplied, the actual upload
order and prompt are query RGB, example RGB, example normal. Each E1/E2/E3
condition is generated independently; aggregation averages scores, not images.

FireRed normal results can be aggregated, while local open-source inference stays
in the upstream model repositories. No metallic access-ablation table is defined.

All public settings and output tables use the paper's A0/A1/A2/A3/B0 identifiers.

Official Roughness RMSE and SSIM use image means. Bin-BMAE pools valid pixels
within GT bins before balancing bins; Region-Macro MAE balances parsed regions.
Roughness ablation Balanced RMSE instead pools valid-pixel squared errors within
each main/stress partition, then equally averages the two partition RMSE values.
Only the paper-defined estimators are exported.

Input/export commands and saved provenance use configurable roots and portable
relative paths. Generation I/O is specified in the [shared policy](../IMAGE_IO.md).

See the [run guide](README.md) for commands, manifest selection, and outputs;
see the [reproducibility guide](../../docs/reproducibility.md) for experiment
recording requirements. Generation and evaluation exports use the paper setting IDs directly.
