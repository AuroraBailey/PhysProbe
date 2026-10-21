# API configuration

Copy the template once and fill the providers needed for your experiment:

```bash
cp -n .env.example .env
python3 tools/check_env.py
python3 tools/check_env.py --provider azure
```

These commands assume an [installed environment](install.md). The checker reports
presence only; it does not print credentials or call an API. With `--provider`,
missing required values produce a nonzero exit. Supported checker names are
`azure`, `doubao`, `gemini`, `qwen`, `wan`, and `glm`.
`qwen` is the checker's credential-group name; the image-generation launcher ID
is `qwen2`.

## Providers

| Interface | Required credential | Connection values |
| --- | --- | --- |
| Doubao / Ark | `ARK_API_KEY` | `ARK_BASE_URL` |
| Qwen Image 2.0 / Wan 2.7 image API | `DASHSCOPE_API_KEY` | `DASHSCOPE_BASE_URL` |
| Qwen scene-audit chat | `DASHSCOPE_API_KEY` | `QWEN_BASE_URL` |
| GLM scene-audit chat | `GLM_API_KEY` | `GLM_BASE_URL` |
| Gemini through Aihubmix | `AIHUBMIX_API_KEY` | `AIHUBMIX_BASE_URL` |
| GPT through Azure OpenAI | `AZURE_OPENAI_API_KEY` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` |

GPT Image 1.5 and 2.0 share the same Azure values across all targets. Select a
deployment using `--deployment` in shared launchers, or the model/deployment
option of the direct runner. These scripts use Azure for GPT calls. Gemini uses
its separate Aihubmix interface. Native image and chat-compatible DashScope URLs
serve different APIs and should not be interchanged.

Albedo A1 needs a shared Doubao-generated analysis cache plus the credentials
for the selected image editor. Its generation stage reads the existing cache;
see the [Albedo diagnostic guide](../experiments/ablation/albedo/README.md).

## Loading and precedence

Annotation scripts, main/stress runners, diagnostic runners, and shell launchers
load repository-root `.env`. Priority is explicit CLI values, existing process
environment, nonempty `.env` entries, then script defaults. Existing environment
variables, including explicitly empty ones, are preserved.

Dotenv interpolation is disabled, so `$` in a value remains literal. Values are
parsed as configuration data; do not put shell commands in `.env`. Shell
launchers use the same parser and quote exports rather than sourcing the file.
Credentials remain in the environment instead of task command arguments.

Optional `GT_ROOT` and `BASE_OUTPUT_DIR` supply task-launcher input/output roots.
`PYTHON_BIN` selects the shell bootstrap interpreter and must have `python-dotenv`.
The example runner uses `EXAMPLE_ASSET_ROOT` and `EXAMPLE_OUTPUT_ROOT`, with CLI
`--asset-root` and `--output-root` taking precedence. See [.env.example](../.env.example).

## Gemini requests

Gemini requests send model, original image inputs, prompt, the runner's fixed
aspect-ratio system message, and `modalities=["text", "image"]`. Main/stress
metallic uses a 16:9 aspect-ratio message; the other current main/stress targets
and the shared diagnostic paths use 4:3. These are request messages; returned
image dimensions are preserved. The timeout is set on the HTTP client.

Each query requests one output image (`n=1`).
