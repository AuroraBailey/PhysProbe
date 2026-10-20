#!/usr/bin/env python3
"""Metallic entry point for the shared main/stress task dispatcher."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.main_stress.task_dispatch import main

if __name__ == '__main__':
    raise SystemExit(main(target="metallic"))
