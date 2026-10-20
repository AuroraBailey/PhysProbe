# Contributing

Changes should preserve the benchmark's sample identities, access settings, and
metric definitions unless an explicit protocol change is intended.

## Development setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[full]"
python3 -m unittest discover -s tests -v
python3 examples/run.py --dry-run
```

The tests use mocked API responses. Live generation is separate and requires
provider credentials; it is not a prerequisite for documentation changes.

## Changes and validation

- Keep patches focused. Explain the problem, resulting behavior, and relevant checks.
- Update runnable commands and local links when renaming public files or options.
- Keep prompts in the target's `prompts.py`; update version/hash contracts when
  intentionally changing text, and disclose its effect on existing results.
- Verify estimator, masking, interpolation, and sample-support behavior when
  changing evaluators or aggregation. Do not silently replace missing metrics.
- Use configurable paths and dataset-relative identifiers. Keep credentials,
  local datasets, checkpoints, logs, and restricted outputs out of the release.
- Preserve provider/dataset notices for third-party assets. See [data sources](data/sources.md).

Report security issues privately as described in [SECURITY.md](SECURITY.md).
