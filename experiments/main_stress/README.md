# Main / Stress Experiments

Each task has one `prompts.py`, model entry files named `<target>_<model>.py`,
and three launchers in `scripts/`: `run.sh`, `start.sh`, and `progress.sh`.
All launchers use [task_dispatch.py](task_dispatch.py) for environment loading,
sequential scheduling, logs, background execution and progress reporting.

| Task | Models | Generation input | Run guide |
| --- | --- | --- | --- |
| Albedo | Doubao, GPT Image 1.5/2.0, Qwen, Wan, Gemini | Query RGB | [Albedo](albedo/README.md) |
| Depth | Doubao, GPT Image 1.5/2.0, Qwen, Wan, Gemini | Query RGB | [Depth](depth/README.md) |
| Normal | Doubao, GPT Image 1.5 | Query RGB + one fixed reference RGB/normal pair | [Normal](normal/README.md) |
| Roughness | Doubao, GPT Image 1.5/2.0, Qwen, Gemini | Query RGB | [Roughness](roughness/README.md) |
| Metallic | Doubao, GPT Image 1.5/2.0, Qwen, Wan, Gemini | Query RGB | [Metallic](metallic/README.md) |

Copy root `.env.example` to `.env` and configure the provider you use.
GPT Image entries use the shared Azure OpenAI credentials and endpoint.
See [environment setup](../../docs/environment.md) for dependencies and provider settings.

```bash
bash experiments/main_stress/roughness/scripts/run.sh --model gpt-image-1.5 \
  --dataset openroomff_mainaxis --input-root examples/assets \
  --output-root outputs/examples/roughness --max-generate 1 --dry-run
```

Remove `--dry-run` to generate. Replace `run.sh` with `start.sh` for a detached
job or `progress.sh` for progress, retaining the same options. Use separate
output roots for each target/model. Input roots and output layouts differ by
task; refer to the individual run guide. Launchers scan prepared image directories,
not manifest CSVs. Normal runs E1/E2/E3 separately by default.

[Prompt Organization](PROMPTS.md) describes retained model instructions and offline checks.
The runnable small examples are documented in [examples/README.md](../../examples/README.md).
Diagnostic input modes belong to `experiments/ablation/`.

Current image handling: [shared image I/O policy](../IMAGE_IO.md).
