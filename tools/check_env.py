from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.utils.env import load_project_env

# Each tuple lists interchangeable credentials for the requested interface.
REQUIREMENTS = {
    "doubao": [("ARK_API_KEY",)],
    "qwen": [("DASHSCOPE_API_KEY",)],
    "wan": [("DASHSCOPE_API_KEY",)],
    "glm": [("GLM_API_KEY",)],
    "gemini": [("AIHUBMIX_API_KEY",)],
    "azure": [("AZURE_OPENAI_API_KEY",), ("AZURE_OPENAI_ENDPOINT",),
              ("AZURE_OPENAI_API_VERSION",)],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check configuration presence without printing credentials or calling APIs.")
    parser.add_argument("--provider", choices=sorted(REQUIREMENTS))
    args = parser.parse_args()
    load_project_env()
    print(f"repo_root={ROOT}")
    print(f"python_version={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"has_runnable_examples={(ROOT / 'examples/samples.json').is_file()}")
    print(f"has_env_file={(ROOT / '.env').is_file()}")
    providers = [args.provider] if args.provider else list(REQUIREMENTS)
    missing = False
    for provider in providers:
        for alternatives in REQUIREMENTS[provider]:
            present = any(os.environ.get(key) for key in alternatives)
            print(f"{provider}:{'/'.join(alternatives)}={'set' if present else 'missing'}")
            missing |= not present
    if args.provider and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
