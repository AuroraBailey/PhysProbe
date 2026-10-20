#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

# Support both direct script execution and editable package installation.
import sys as _env_sys
from pathlib import Path as _EnvPath
for _env_root in _EnvPath(__file__).resolve().parents:
    if (_env_root / "src/common/utils/env.py").is_file():
        _env_sys.path.insert(0, str(_env_root / "src"))
        _env_sys.path.insert(0, str(_env_root))
        break
from common.utils.env import load_project_env as _load_project_env
_load_project_env()


import sys

from experiments.ablation.normal.normal_ablation_runner import MINIMAL_RGB_ONLY_PROMPT, VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="A3",
    variant_name="Minimal Prompt",
    prompt_version="normal_a3_minimal_prompt_query_first_v3",
    description="A3 ablation: RGB + minimal prompt, without example pair.",
    prompt_text=MINIMAL_RGB_ONLY_PROMPT,
    use_example_pair=False,
    prompt_level="minimal",
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
