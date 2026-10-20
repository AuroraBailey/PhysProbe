#!/usr/bin/env python3
"""Paper Albedo-A1: fixed prompt plus shared cached query-image analysis."""
# Support both direct script execution and editable package installation.
import sys as _env_sys
from pathlib import Path as _EnvPath
for _env_root in _EnvPath(__file__).resolve().parents:
    if (_env_root / "src/common/utils/env.py").is_file():
        _env_sys.path.insert(0, str(_env_root / "src"))
        _env_sys.path.insert(0, str(_env_root))
        break
from common.utils.env import load_project_env
load_project_env()

from experiments.ablation.albedo.albedo_analysis_runner import main

if __name__ == "__main__":
    main()
