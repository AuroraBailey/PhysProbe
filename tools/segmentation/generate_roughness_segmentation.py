#!/usr/bin/env python3
"""Generate manifest-matched SAM3 color-ID maps; with an explicit segmentation method."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent


def below(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Expected a relative manifest path: {relative}')
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes configured root: {relative}')
    return target


def plan_jobs(manifest: Path, data_root: Path, output_root: Path) -> list[tuple[Path, Path]]:
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Empty manifest')
    jobs = []
    seen = set()
    for row in rows:
        if row.get('datasource') != 'openroomff':
            raise ValueError('This workflow expects OpenRooms-FF roughness samples')
        rgb = Path(row['rgb_path'])
        if rgb.name.endswith('_im_5.rgbe'):
            rgb = rgb.with_name(rgb.name[:-len('_im_5.rgbe')] + '_im.png')
        elif not rgb.name.endswith('_im.png'):
            raise ValueError(f'Unsupported RGB naming: {rgb}')
        segmentation = row['segmentation_path']
        destination = below(output_root, segmentation)
        if destination in seen:
            raise ValueError(f'Duplicate segmentation destination: {segmentation}')
        seen.add(destination)
        jobs.append((below(data_root, rgb.as_posix()), destination))
    return jobs


def load_module(name: str):
    path = HERE / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_pipeline(method: str, checkpoint: Path):
    base = load_module('scene_inference_contextual' if method == 'contextual' else 'scene_inference')
    original_builder = base.build_sam3_image_model

    def build_local_model(**kwargs):
        return original_builder(**kwargs, checkpoint_path=str(checkpoint), load_from_HF=False)

    base.build_sam3_image_model = build_local_model
    if method == 'standard':
        return base, base.merge_scene
    merging = load_module('region_merging')
    return base, merging.merge_scene


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True, help='Root containing processed N_im.png images.')
    parser.add_argument('--output-root', type=Path, required=True, help='Root to resolve segmentation_path; may equal data root.')
    parser.add_argument('--method', choices=['standard', 'contextual'], required=True)
    parser.add_argument('--checkpoint', type=Path, help='Explicit local SAM3 image-model checkpoint; required for generation.')
    parser.add_argument('--sam3-repo', type=Path, help='Optional externally installed/downloaded SAM3 checkout.')
    parser.add_argument('--cuda-device', type=int, default=0)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error('--max-images must be nonnegative')
    jobs = plan_jobs(args.manifest, args.data_root, args.output_root)
    if args.max_images:
        jobs = jobs[:args.max_images]
    missing = [str(rgb) for rgb, _ in jobs if not rgb.is_file()]
    print(json.dumps({'samples': len(jobs), 'existing_outputs': sum(seg.is_file() for _, seg in jobs),
                      'missing_inputs': missing, 'method': args.method}, indent=2))
    if missing:
        return 1
    if args.dry_run:
        return 0
    if not args.checkpoint or not args.checkpoint.is_file():
        parser.error('--checkpoint must point to an existing local SAM3 checkpoint')
    if args.sam3_repo:
        sys.path.insert(0, str(args.sam3_repo.resolve()))
    os.environ['SAM3_CUDA_DEVICE'] = str(args.cuda_device)
    base, merge = load_pipeline(args.method, args.checkpoint.resolve())
    failures = 0
    for rgb, seg in jobs:
        if seg.exists() and not args.overwrite:
            print(f'SKIP {seg}')
            continue
        relative = seg.relative_to(args.output_root.resolve())
        details = args.output_root.resolve() / '_sam3_details' / relative.parent / relative.stem
        details.mkdir(parents=True, exist_ok=True)
        try:
            summary, _ = base.run_segmentation(str(rgb), str(details))
            _, color, _ = merge(summary, str(details))
            seg.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(color, seg)
            record = {'method': args.method, 'checkpoint': str(args.checkpoint.resolve()),
                      'input': str(rgb), 'output': str(seg), 'summary': str(summary),
                      'output_sha256': hashlib.sha256(seg.read_bytes()).hexdigest(),
                      'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                      'sam3_environment': {k: v for k, v in os.environ.items() if k.startswith('SAM3_')}}
            (details / 'generation_record.json').write_text(json.dumps(record, indent=2) + '\n')
            print(f'DONE {seg}')
        except Exception as exc:
            failures += 1
            print(f'FAIL {rgb}: {exc}', file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
