#!/usr/bin/env python3
"""Generate the shared query-only Doubao Seed 2.0 Pro cache once, before Albedo-A1."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from common.utils.env import load_project_env
from experiments.ablation.tasks import records, repo_path
from experiments.ablation.albedo.albedo_analysis_runner import analyze_single_image
from experiments.ablation.albedo.albedo_provider_utils import DOUBAO_ANALYSIS_MODEL, DOUBAO_BASE_URL, build_openai_client


def main(argv=None):
    load_project_env()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--manifest', default='data/manifests/ablation/albedo_manifest_ablation.csv')
    p.add_argument('--max-generate', type=int, default=0)
    p.add_argument('--timeout', type=int, default=200)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    a.input_root, a.output_root, a.manifest = map(repo_path, (a.input_root, a.output_root, a.manifest))
    if a.max_generate < 0:
        p.error('--max-generate must be nonnegative')
    if a.input_root == a.output_root or a.input_root in a.output_root.parents:
        p.error('Cache output must be outside the input data root')
    a.target, a.settings = 'albedo', ['A0']
    selected = records(a)
    if a.dry_run:
        print(f'{len(selected)} query-only analyses using {DOUBAO_ANALYSIS_MODEL}; output: {a.output_root}')
        return 0
    key = os.environ.get('ARK_API_KEY')
    if not key:
        raise ValueError('Set ARK_API_KEY in root .env')
    runtime = SimpleNamespace(analysis_model=DOUBAO_ANALYSIS_MODEL,
        analysis_client=build_openai_client(os.environ.get('ARK_BASE_URL') or DOUBAO_BASE_URL, key, a.timeout))
    a.output_root.mkdir(parents=True, exist_ok=True)
    index = []
    for record in selected:
        path = a.output_root / record['relative'].with_suffix('.json')
        if not path.exists() or a.overwrite:
            analysis = analyze_single_image(runtime, record['source'], 'high', 0)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + '\n')
            temp.replace(path)
        if not isinstance(json.loads(path.read_text()), dict):
            raise ValueError(f'Invalid cache object: {path}')
        index.append({'query': str(record['relative']), 'source_sha256': hashlib.sha256(record['source'].read_bytes()).hexdigest(),
                      'analysis_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (a.output_root / 'cache_provenance.json').write_text(json.dumps({'analysis_model': DOUBAO_ANALYSIS_MODEL,
        'manifest_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest(), 'queries': index}, indent=2) + '\n')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
