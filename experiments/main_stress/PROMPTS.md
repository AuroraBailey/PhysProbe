# Prompt Organization

Each target exposes generation instructions through a single `prompts.py`.
Albedo and roughness import their canonical text from the corresponding ablation
A0 prompt, keeping both execution paths identical.
Model runners import these instructions; API settings and request code remain
in the corresponding model directories.

| Target | Prompt selection |
| --- | --- |
| [Albedo](albedo/prompts.py) | All six models import the ablation A0 instruction. |
| [Depth](depth/prompts.py) | Shared Gemini `You are ...` input definition + common depth body; RGB-only. |
| [Normal](normal/prompts.py) | One complete shared instruction; inputs are query RGB, reference RGB, reference normal. |
| [Roughness](roughness/prompts.py) | All five models import the ablation A0 instruction. |
| [Metallic](metallic/prompts.py) | Qwen/Wan share the shorter RGB-only instruction; other models share the full version with texture guidance. |

Normal uses a selected fixed RGB/normal reference pair; E1/E2/E3 are separate
runs with the same prompt. Main/stress metallic and roughness do not accept
external scene-prior text, segmentation inputs or example pairs. Roughness uses one shared instruction; metallic selects the Qwen/Wan variant or the full variant by model.

For example:

```python
from experiments.main_stress.roughness.prompts import build_roughness_prompt
prompt = build_roughness_prompt(model="gpt_image_15")
```

`PROMPTS` maps model suffixes to fixed text, and `MODEL_CONFIGS` retains the
corresponding version/preset metadata. Intentionally changing text requires
updating the recorded prompt contract and applicable version metadata.
Prompt modules use only the standard library.

```bash
python3 -m unittest discover -s tests -p 'test_main_stress_prompts.py'
python3 -m unittest discover -s tests -p 'test_remaining_task_wiring.py'
```

Prompt hashes check retained instruction text. Wiring tests run the actual
model entry points with fake API responses, verify request prompts and saved
prediction paths, and block network access. They require the main runtime
dependencies; they do not test live provider availability or output quality.
