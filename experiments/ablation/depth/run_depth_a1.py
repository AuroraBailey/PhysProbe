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

from experiments.ablation.depth.depth_ablation_runner import PROMPT_VERSION_A1, VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="A1",
    variant_name="RGB + Fixed Example Pair",
    description="Generate relative depth maps from RGB inputs plus one fixed RGB/depth example pair.",
    prompt_version=PROMPT_VERSION_A1,
    prompt_kind="example",
    input_mode="rgb_plus_example",
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
