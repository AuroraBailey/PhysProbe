#!/usr/bin/env python3
"""Normal diagnostic launcher."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.ablation.tasks import main
if __name__ == '__main__':
    raise SystemExit(main(target='normal'))
