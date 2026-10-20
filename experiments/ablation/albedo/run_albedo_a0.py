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

from experiments.ablation.albedo.albedo_ablation_runner import FULL_PROMPT_TEXT, VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="A0",
    variant_name="Full Prompt",
    prompt_version="albedo_a0_full_prompt_v1",
    description="A0 ablation: RGB + Full Prompt, without analysis notes.",
    prompt_text=FULL_PROMPT_TEXT,
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
