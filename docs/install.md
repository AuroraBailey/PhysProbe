# Installation

Use Python 3.10 or newer. Bash launchers and detached jobs target Linux; the
shared task dispatchers use POSIX process and file-lock facilities.
Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[full]"
cp -n .env.example .env
```

`cp -n` preserves an existing `.env`. Fill only the providers needed for your run;
see [API configuration](api_configuration.md). Select the appropriate PyTorch
build for your hardware if the default installation is unsuitable.

## Dependency groups

| Installation | Use |
| --- | --- |
| `python3 -m pip install -e .` | Shared dotenv/configuration utilities |
| `python3 -m pip install -e ".[main]"` | API generation runners |
| `python3 -m pip install -e ".[ablation]"` | API diagnostics, including NumPy-based region fill |
| `python3 -m pip install -e ".[evaluation]"` | Pixel evaluators, perceptual metrics, EXR reading, aggregation |
| `python3 -m pip install -e ".[full]"` | All listed groups and the full offline test environment |

SAM3 is a separate dependency for the
[segmentation workflow](../tools/segmentation/README.md). The bundled examples
already include processed images and a roughness region map.

## Verify setup

```bash
python3 tools/check_env.py
python3 -m unittest discover -s tests -v
python3 examples/run.py --dry-run
```

The environment checker reports whether values are present, without exposing
credentials or calling providers. Use `--provider azure` or another supported
provider to make missing settings an error. The test suite uses mocked APIs.
The base installation alone does not supply all dependencies used by these tests.

Continue with the [runnable examples](../examples/README.md) or the
[full experiment workflow](main_experiments.md).
