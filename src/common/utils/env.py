"""Load repository-local credentials without overriding the process environment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[3]


def load_project_env(path: str | Path | None = None) -> None:
    """Load nonempty values; existing environment variables always take priority."""
    env_path = Path(path) if path is not None else ROOT / ".env"
    # Disable interpolation so literal dollars in credentials remain unchanged.
    for key, value in dotenv_values(env_path, interpolate=False).items():
        if value:
            os.environ.setdefault(key, value)

    aliases = {
        "ARK_API_KEY": ("DOUBAO_API_KEY", "VOLCENGINE_API_KEY"),
        "DASHSCOPE_API_KEY": ("QWEN_API_KEY",),
        "GLM_API_KEY": ("ZHIPU_API_KEY",),
        "ARK_BASE_URL": ("DOUBAO_BASE_URL", "VOLCENGINE_BASE_URL"),
        "DOUBAO_BASE_URL": ("ARK_BASE_URL", "VOLCENGINE_BASE_URL"),
        "QWEN_BASE_URL": ("DASHSCOPE_COMPATIBLE_BASE_URL",),
        "DASHSCOPE_COMPATIBLE_BASE_URL": ("QWEN_BASE_URL",),
    }
    for name, sources in aliases.items():
        if name not in os.environ:
            value = next((os.environ[s] for s in sources if os.environ.get(s)), None)
            if value:
                os.environ[name] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Load project environment for shell launchers.")
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    before = dict(os.environ)
    load_project_env()
    if args.shell:
        for key, value in os.environ.items():
            if before.get(key) != value:
                print(f"export {key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
