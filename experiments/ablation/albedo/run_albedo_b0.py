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

from experiments.ablation.albedo.albedo_ablation_runner import MINIMAL_FIXED_EXEMPLAR_PROMPT_TEXT, VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="B0",
    variant_name="Minimal Prompt + Fixed Exemplar Pair",
    prompt_version="albedo_b0_minimal_fixed_exemplar_v1",
    description="B0 diagnostic ablation: minimal prompt with a fixed RGB/albedo exemplar pair.",
    prompt_text=MINIMAL_FIXED_EXEMPLAR_PROMPT_TEXT,
    use_example_pair=True,
    prompt_level="minimal+fixed_exemplar",
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
