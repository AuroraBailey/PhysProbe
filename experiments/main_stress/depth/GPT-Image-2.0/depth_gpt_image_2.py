#!/usr/bin/env python3

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



import os
import runpy
import sys
from pathlib import Path


def ensure_arg(flag: str, value: str) -> None:
    if flag not in sys.argv[1:]:
        sys.argv.extend([flag, value])


def main() -> None:
    ensure_arg("--image_model", "gpt-image-2")
    target = Path(__file__).resolve().parents[1] / "GPT-Image-1.5" / "depth_gpt_image_15.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
