#!/usr/bin/env python3
"""Manifest-based execution of the paper's diagnostic access settings."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from common.utils.env import load_project_env

SETTINGS = {'albedo': ('A0', 'A1', 'A2', 'A3', 'B0'), 'depth': ('A0', 'A1', 'A2'),
            'normal': ('A0', 'A1', 'A2', 'A3'), 'roughness': ('A0', 'A1', 'A2', 'A3')}
MODELS = {'doubao': 'doubao', 'gpt-image-1.5': 'gpt', 'gpt-image-2.0': 'gpt2',
          'qwen2': 'qwen2', 'wan': 'wan', 'gemini': 'gemini'}
PAPER_MODELS = {'albedo': tuple(MODELS), 'depth': tuple(MODELS),
                'normal': ('doubao', 'gpt-image-1.5'),
                'roughness': ('doubao', 'gpt-image-1.5', 'gpt-image-2.0', 'gemini')}


def repo_path(value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def relative_path(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Manifest paths must be relative to the data root: {value}')
    return p


def processed_rgb(value):
    p = relative_path(value)
    if p.suffix == '.rgbe' and p.stem.endswith('_im_5'):
        return p.with_name(p.stem[:-5] + '_im.png')
    if p.suffix == '.exr':
        return p.with_suffix('.png')
    if p.suffix != '.png' or not p.name.endswith('_im.png'):
        raise ValueError(f'Unsupported processed RGB mapping: {value}')
    return p


def parse_args(argv=None, target=None):
    load_project_env()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('run', 'start', 'progress'))
    p.add_argument('--target', choices=SETTINGS, default=target, required=target is None)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--settings', nargs='+', type=str.upper)
    p.add_argument('--manifest')
    p.add_argument('--input-root', default=os.environ.get('GT_ROOT') or ROOT / 'data/raw')
    p.add_argument('--output-root', default=os.environ.get('BASE_OUTPUT_DIR'))
    p.add_argument('--analysis-dir', help='Shared Albedo-A1 cache; paths mirror the processed RGB tree.')
    p.add_argument('--seg-root', help='Root mirroring input-root; default uses input-root.')
    p.add_argument('--exemplar', nargs='+', choices=('E1', 'E2', 'E3'), default=['E1', 'E2', 'E3'])
    p.add_argument('--max-generate', type=int, default=0, help='First N manifest records; 0 selects the complete subset.')
    p.add_argument('--deployment')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--sleep', type=float, default=0.2, help='Delay between requests in each child job.')
    p.add_argument('--python', default=os.environ.get('PYTHON_BIN') or sys.executable)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--watch', type=float, default=0)
    a = p.parse_args(argv)
    if a.model not in PAPER_MODELS[a.target]:
        p.error(f'{a.target} available paper model entries: {PAPER_MODELS[a.target]}; open-source local inference is outside this API launcher')
    a.settings = list(dict.fromkeys(a.settings or SETTINGS[a.target]))
    if any(s not in SETTINGS[a.target] for s in a.settings):
        p.error(f'{a.target} settings: {SETTINGS[a.target]}')
    if a.max_generate < 0 or a.watch < 0 or a.sleep < 0:
        p.error('Limits and refresh intervals must be nonnegative')
    a.input_root = repo_path(a.input_root)
    a.output_root = repo_path(a.output_root or ROOT / 'outputs/ablation' / a.target / a.model)
    a.manifest = repo_path(a.manifest or ROOT / f'data/manifests/ablation/{a.target}_manifest_ablation.csv')
    a.seg_root = repo_path(a.seg_root) if a.seg_root else a.input_root
    a.analysis_dir = repo_path(a.analysis_dir) if a.analysis_dir else None
    a.exemplar = list(dict.fromkeys(a.exemplar))
    if a.input_root == a.output_root or a.input_root in a.output_root.parents:
        p.error('Output root must be outside the input data root')
    if a.target == 'albedo' and 'A1' in a.settings and a.analysis_dir is None:
        p.error('Albedo-A1 requires --analysis-dir; first run albedo/cache_illumination_analysis.py once')
    return a


def records(args):
    with args.manifest.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Empty manifest: {args.manifest}')
    rows = rows[:args.max_generate] if args.max_generate else rows
    selected = []
    seen = set()
    needs_seg = (args.target == 'depth' and 'A2' in args.settings or
                 args.target == 'roughness' and bool({'A1', 'A2'} & set(args.settings)))
    for row in rows:
        rel = processed_rgb(row['rgb_path'])
        if rel in seen:
            raise ValueError(f'Duplicate processed query in manifest: {rel}')
        seen.add(rel)
        source = args.input_root / rel
        if not source.is_file():
            raise FileNotFoundError(f'Prepared PNG input missing: {source}; convert the raw data first')
        record = {'relative': rel, 'source': source, 'row': row}
        if needs_seg:
            seg_rel = relative_path(row['segmentation_path']) if row.get('segmentation_path') else rel.with_name(rel.stem[:-3] + '_seg.png')
            seg = args.seg_root / seg_rel
            if not seg.is_file():
                raise FileNotFoundError(f'Frozen segmentation missing: {seg}')
            record['seg'] = seg
        if args.target == 'albedo' and 'A1' in args.settings:
            analysis = args.analysis_dir / rel.with_suffix('.json')
            if not analysis.is_file():
                raise FileNotFoundError(f'Shared query analysis missing: {analysis}')
            if not isinstance(json.loads(analysis.read_text()), dict):
                raise ValueError(f'Analysis must be a JSON object: {analysis}')
        selected.append(record)
    return selected


def jobs(args):
    result = []
    for setting in args.settings:
        exemplars = args.exemplar if args.target == 'normal' and setting in {'A0', 'A2'} else [None]
        for exemplar in exemplars:
            out = args.output_root / setting
            if exemplar:
                out = out / exemplar
            cmd = [str(args.python)]
            script = HERE / args.target / f'run_{args.target}_{setting.lower()}.py'
            if args.target == 'roughness':
                script = HERE / 'roughness/roughness_ablation_runner.py'
            cmd += [str(script), '--input_dir', str(args.output_root / 'inputs'),
                    '--output_dir', str(out), '--provider', MODELS[args.model], '--sleep', str(args.sleep), '--seed', str(args.seed)]
            if args.target in {'albedo', 'normal'}:
                cmd += ['--recursive', '--preserve_relative_dirs', '--generation_mode', 'edit']
                if args.overwrite:
                    cmd.append('--overwrite')
            elif not args.overwrite:
                cmd.append('--skip_existing')
            if args.target == 'roughness':
                # This runner adds its own setting directory.
                cmd[cmd.index('--output_dir') + 1] = str(args.output_root)
                cmd += ['--variant', setting.lower()]
                out = args.output_root / setting
            if args.target == 'albedo' and setting == 'A1':
                cmd += ['--analysis_dir', str(args.analysis_dir)]
            if args.target == 'albedo' and setting == 'B0':
                refs = HERE / 'albedo/examples'
                cmd += ['--example_rgb', str(refs / 'image.png'), '--example_albedo', str(refs / 'albedo.png')]
            if exemplar:
                number = '' if exemplar == 'E1' else exemplar[1:]
                refs = ROOT / 'experiments/main_stress/normal/examples'
                cmd += ['--example_rgb', str(refs / f'image{number}.png'), '--example_normal', str(refs / f'normal{number}.png')]
            if args.deployment:
                cmd += ['--albedo_model' if args.target == 'albedo' else '--normal_model' if args.target == 'normal' else '--image_model', args.deployment]
            result.append({'setting': setting, 'exemplar': exemplar, 'output': out, 'command': cmd})
    return result


def stage(args, selected):
    folder = args.output_root / 'inputs'
    folder.mkdir(parents=True, exist_ok=True)
    # This directory is dispatcher-owned and contains symlinks only.
    for path in folder.rglob('*'):
        if path.is_file() and not path.is_symlink():
            raise ValueError(f'Refusing to replace a real input file: {path}')
    for path in folder.rglob('*'):
        if path.is_symlink():
            path.unlink()
    for record in selected:
        destination = folder / record['relative']
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(record['source'])
        if 'seg' in record:
            destination.with_name(destination.stem[:-3] + '_seg.png').symlink_to(record['seg'])


def prediction(job, target, relative):
    suffix = 'relative_depth' if target == 'depth' else target
    return job['output'] / suffix / relative.with_name(relative.stem + '_' + suffix + '.png')


def save_status(args, state):
    temp = args.output_root / 'run_status.tmp'
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temp.replace(args.output_root / 'run_status.json')


def run(args):
    selected = records(args)
    plan = jobs(args)
    if args.dry_run:
        print(f'{args.target}: {len(selected)} manifest queries; {len(plan)} jobs')
        for job in plan:
            print(shlex.join(job['command']))
        return 0
    from experiments.main_stress.task_dispatch import check_credentials
    check_credentials(args.model)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stage(args, selected)
        provenance = {'target': args.target, 'model': args.model, 'manifest': str(args.manifest),
                      'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                      'query_count': len(selected), 'seed': args.seed, 'settings': args.settings, 'exemplars': args.exemplar,
                      'source_partition_counts': dict(Counter(r['row']['datasource'] + '/' + r['row']['split'] for r in selected)),
                      'input_root': str(args.input_root), 'analysis_dir': str(args.analysis_dir) if args.analysis_dir else None,
                      'queries': [str(r['relative']) for r in selected]}
        (args.output_root / 'selection.json').write_text(json.dumps(provenance, indent=2) + '\n')
        state = {'target': args.target, 'model': args.model, 'status': 'running', 'pid': os.getpid(), 'jobs': []}
        save_status(args, state)
        try:
            for job in plan:
                label = job['setting'] + ('_' + job['exemplar'] if job['exemplar'] else '')
                log = args.output_root / 'logs' / (label + '.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                print(f'Running {label}; log: {log}', flush=True)
                with log.open('a') as handle:
                    result = subprocess.run(job['command'], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                state['jobs'].append({'setting': job['setting'], 'exemplar': job['exemplar'], 'returncode': result.returncode})
                save_status(args, state)
            state['status'] = 'failed' if any(j['returncode'] for j in state['jobs']) else 'finished'
        except BaseException:
            state['status'] = 'interrupted'
            raise
        finally:
            save_status(args, state)
        return 0 if state['status'] == 'finished' else 1


def start(args, options):
    if args.dry_run:
        return run(args)
    records(args)
    from experiments.main_stress.task_dispatch import check_credentials
    check_credentials(args.model)
    args.output_root.mkdir(parents=True, exist_ok=True)
    pidfile = args.output_root / 'run.pid'
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('A background job still owns this output root')
    log = args.output_root / 'logs/background.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), 'run', *options,
           '--target', args.target, '--input-root', str(args.input_root), '--output-root', str(args.output_root),
           '--manifest', str(args.manifest), '--python', str(args.python)]
    with log.open('a') as handle:
        child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    pidfile.write_text(str(child.pid) + '\n')
    print(f'PID: {child.pid}; log: {log}')
    return 0


def progress(args):
    from experiments.main_stress.task_dispatch import complete_image
    selected = records(args)
    while True:
        for job in jobs(args):
            done = sum(complete_image(prediction(job, args.target, r['relative'])) for r in selected)
            print(f"{job['setting']} {job['exemplar'] or ''}: {done}/{len(selected)}; remain={len(selected)-done}")
        if not args.watch:
            return 0
        time.sleep(args.watch)


def main(argv=None, target=None):
    options = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(options, target)
    try:
        return run(args) if args.action == 'run' else start(args, options[1:]) if args.action == 'start' else progress(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
