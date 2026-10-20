#!/usr/bin/env python3
"""Shared run, background, and progress handling for physical-map generation tasks."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from common.utils.env import load_project_env

MODELS = {
    'doubao': 'Doubao_Seedream_5.0', 'gpt-image-1.5': 'GPT-Image-1.5',
    'gpt-image-2.0': 'GPT-Image-2.0', 'qwen2': 'Qwen-Image-2.0',
    'wan': 'Wan2.7-Image', 'gemini': 'Gemini_3_Pro_Image',
}
MODEL_SUFFIXES = {'doubao': 'doubao', 'gpt-image-1.5': 'gpt_image_15',
                  'gpt-image-2.0': 'gpt_image_2', 'qwen2': 'qwen2', 'wan': 'wan', 'gemini': 'gemini'}
DATASETS = ('interiorverse_mainaxis', 'interiorverse_stresstest',
            'openroomff_mainaxis', 'openroomff_stresstest')

TARGET_MODELS = {
    'albedo': tuple(MODELS), 'depth': tuple(MODELS),
    'normal': ('doubao', 'gpt-image-1.5'),
    'roughness': ('doubao', 'gpt-image-1.5', 'gpt-image-2.0', 'qwen2', 'gemini'),
    'metallic': tuple(MODELS),
}
TARGET_DATASETS = {
    'albedo': DATASETS, 'depth': DATASETS, 'normal': DATASETS,
    'roughness': ('openroomff_mainaxis', 'openroomff_stresstest'),
    'metallic': ('mainaxis', 'stresstest'),
}


def repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def parse_args(argv=None, target='albedo'):
    load_project_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'start', 'progress'))
    parser.add_argument('--target', choices=TARGET_MODELS, default=target)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--dataset', nargs='+', choices=(*DATASETS, 'mainaxis', 'stresstest'))
    parser.add_argument('--split', choices=('all', 'mainaxis', 'stresstest'), default='all')
    parser.add_argument('--input-root', default=os.environ.get('GT_ROOT') or ROOT / 'data/raw')
    parser.add_argument('--output-root', default=os.environ.get('BASE_OUTPUT_DIR'))
    parser.add_argument('--max-generate', type=int, default=0, help='Limit RGB queries per dataset; 0 means all.')
    parser.add_argument('--deployment', help='Azure image deployment, or generation model name for other providers.')
    parser.add_argument('--python', default=os.environ.get('PYTHON_BIN') or sys.executable)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without API calls or output files.')
    parser.add_argument('--watch', type=float, default=0, help='Progress refresh interval in seconds; 0 prints once.')
    parser.add_argument('--exemplar', nargs='+', choices=('E1', 'E2', 'E3'),
                        help='Normal reference pairs; default runs E1, E2 and E3 separately.')
    args = parser.parse_args(argv)
    if args.model not in TARGET_MODELS[args.target]:
        parser.error(f'{args.target} supports models: {TARGET_MODELS[args.target]}')
    if args.exemplar and args.target != 'normal':
        parser.error('--exemplar applies only to normal')
    args.exemplars = list(dict.fromkeys(args.exemplar or ['E1', 'E2', 'E3']))
    if args.max_generate < 0 or args.watch < 0:
        parser.error('--max-generate and --watch must be nonnegative')
    args.input_root = repository_path(args.input_root)
    args.output_root = repository_path(args.output_root or ROOT / 'outputs/main_stress' / args.target / args.model)
    if args.input_root == args.output_root or args.input_root in args.output_root.parents:
        parser.error('Output root must be outside the input data root')
    args.datasets = list(dict.fromkeys(args.dataset or TARGET_DATASETS[args.target]))
    if any(d not in TARGET_DATASETS[args.target] for d in args.datasets):
        parser.error(f'{args.target} requires datasets: {TARGET_DATASETS[args.target]}')
    if args.split != 'all':
        args.datasets = [d for d in args.datasets if d == args.split or d.endswith('_' + args.split)]
    if not args.datasets:
        parser.error('No datasets match --dataset and --split')
    return args


def queries(args, dataset):
    folder = args.input_root / dataset
    if not folder.is_dir():
        raise FileNotFoundError(f'Input dataset does not exist: {folder}')
    pattern = 'Image_*.png' if args.target == 'metallic' else '*_im.png'
    paths = sorted(p for p in folder.rglob(pattern) if p.is_file() and
                   (args.target != 'metallic' or p.parent.parent.name == 'Image'))
    return paths[:args.max_generate] if args.max_generate else paths


def command(args, dataset, folder=None, limit=None, exemplar=None):
    folder = folder or args.input_root / dataset
    output = args.output_root / dataset
    model_folder = ('Doubao_Seedream 5.0' if args.target == 'depth' and args.model == 'doubao'
                    else MODELS[args.model])
    script = HERE / args.target / model_folder / f'{args.target}_{MODEL_SUFFIXES[args.model]}.py'
    if args.target in {'metallic', 'roughness', 'normal'}:
        task = args.target
        if task == 'normal':
            exemplar = exemplar or args.exemplars[0]
            output = args.output_root / exemplar / dataset
        elif task == 'metallic':
            output = output / folder.relative_to(args.input_root / dataset)
        cmd = [str(args.python), str(script), '--input_dir', str(folder),
               '--output_dir', str(output / task if args.model == 'gemini' else output),
               '--max_generate', str(limit if limit is not None else args.max_generate)]
        if task == 'normal':
            number = '' if exemplar == 'E1' else exemplar[1:]
            references = HERE / 'normal/examples'
            cmd += ['--example_rgb', str(references / f'image{number}.png'),
                    '--example_normal', str(references / f'normal{number}.png'),
                    '--filename_suffix', '_im.png', '--recursive', '--preserve_relative_dirs',
                    '--independent_images']
            if args.model != 'doubao':
                cmd += ['--gt_root', str(args.input_root)]
        else:
            cmd += ['--input_mode', 'rgb_only']
            if task == 'roughness':
                cmd += ['--filename_suffix', '_im.png', '--recursive', '--preserve_relative_dirs']
        if task != 'normal' and (task == 'metallic' or args.model.startswith('gpt-image-')):
            if not args.overwrite:
                cmd.append('--skip_existing')
        elif args.overwrite:
            cmd.append('--overwrite')
        if args.model == 'gemini':
            cmd += ['--dataset_root', str(folder), '--scene_list', str(args.output_root / 'inputs' / f'{dataset}.txt')]
        if args.deployment:
            flag = '--normal_deployment' if task == 'normal' and args.model.startswith('gpt-image-') else '--normal_model' if task == 'normal' else '--image_deployment' if task == 'metallic' and args.model.startswith('gpt-image-') else '--image_model'
            cmd += [flag, args.deployment]
        return cmd
    if args.target == 'depth':
        output = output / folder.relative_to(args.input_root / dataset)
        cmd = [str(args.python), str(script), '--input_dir', str(folder),
               '--output_dir', str(output / 'relative_depth' if args.model == 'gemini' else output),
               '--input_mode', 'rgb_only', '--depth_polarity', 'near_white',
               '--max_generate', str(limit if limit is not None else args.max_generate)]
        if not args.overwrite:
            cmd.append('--skip_existing')
        if args.model == 'gemini':
            cmd += ['--dataset_root', str(folder), '--scene_list', str(args.output_root / 'inputs' / f'{dataset}.txt')]
        if args.deployment:
            cmd += ['--image_model', args.deployment]
        return cmd
    cmd = [str(args.python), str(script),
           '--input_dir', str(folder), '--output_dir', str(output / 'albedo' if args.model == 'gemini' else output),
           '--filename_suffix', '_im.png', '--recursive', '--preserve_relative_dirs',
           '--max_generate', str(args.max_generate)]
    if args.overwrite:
        cmd.append('--overwrite')
    if args.model == 'doubao':
        cmd.append('--independent_images')
    else:
        cmd += ['--gt_root', str(args.input_root)]
    if args.model == 'gemini':
        cmd += ['--dataset_root', str(folder), '--scene_list', str(args.output_root / 'inputs' / f'{dataset}.txt')]
    if args.deployment:
        cmd += ['--albedo_deployment' if args.model.startswith('gpt-image-') else '--albedo_model', args.deployment]
    return cmd


def commands(args, dataset):
    if args.target in {'albedo', 'roughness'}:
        return [command(args, dataset)]
    if args.target == 'normal':
        return [command(args, dataset, exemplar=e) for e in args.exemplars]
    # Depth and metallic runners consume a single RGB directory. Keep limits at dataset level
    # by assigning each selected directory only its share of the selected queries.
    groups = {}
    for path in queries(args, dataset):
        groups[path.parent] = groups.get(path.parent, 0) + 1
    return [command(args, dataset, folder, count) for folder, count in groups.items()]


def prediction_path(args, dataset, image, exemplar=None):
    relative = image.relative_to(args.input_root / dataset)
    output = args.output_root / dataset
    if args.target == 'depth':
        return output / relative.parent / 'relative_depth' / f'{image.stem}_relative_depth.png'
    if args.target == 'metallic':
        return output / relative.parent / 'metallic' / f'{image.stem}_metallic.png'
    if args.target == 'normal':
        output = args.output_root / (exemplar or args.exemplars[0]) / dataset
    return output / args.target / relative.with_name(image.stem + f'_{args.target}.png')


def check_credentials(model):
    if model.startswith('gpt-image-'):
        names = ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT']
    elif model == 'doubao':
        names = ['ARK_API_KEY']
    elif model in {'qwen2', 'wan'}:
        names = ['DASHSCOPE_API_KEY']
    else:
        names = ['AIHUBMIX_API_KEY']
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise ValueError('Set the required variables in root .env: ' + ', '.join(missing))


def save_status(args, state):
    path = args.output_root / 'run_status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def run(args):
    for dataset in args.datasets:
        if not queries(args, dataset):
            raise ValueError(f'No RGB queries in {args.input_root / dataset}')
    if args.dry_run:
        for dataset in args.datasets:
            for cmd in commands(args, dataset):
                print(shlex.join(cmd), flush=True)
        return 0
    check_credentials(args.model)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'A task already owns this output root: {args.output_root}')
        state = {'target': args.target, 'model': args.model, 'pid': os.getpid(), 'status': 'running', 'datasets': {}}
        save_status(args, state)
        try:
            for dataset in args.datasets:
                if args.model == 'gemini':
                    scene_list = args.output_root / 'inputs' / f'{dataset}.txt'
                    scene_list.parent.mkdir(parents=True, exist_ok=True)
                    scene_list.write_text('.\n')
                logfile = args.output_root / 'logs' / f'{dataset}.log'
                logfile.parent.mkdir(parents=True, exist_ok=True)
                print(f'[{dataset}] Running; log: {logfile}', flush=True)
                returncode = 0
                with logfile.open('a') as log:
                    for cmd in commands(args, dataset):
                        result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                        returncode = returncode or result.returncode
                state['datasets'][dataset] = {'returncode': returncode, 'log': str(logfile)}
                save_status(args, state)
            state['status'] = 'failed' if any(v['returncode'] for v in state['datasets'].values()) else 'finished'
        except BaseException:
            state['status'] = 'interrupted'
            raise
        finally:
            save_status(args, state)
        print(f"Run {state['status']}: {args.output_root}", flush=True)
        return 0 if state['status'] == 'finished' else 1


def start(args, options):
    if args.dry_run:
        return run(args)
    check_credentials(args.model)
    for dataset in args.datasets:
        if not queries(args, dataset):
            raise ValueError(f'No RGB queries in {dataset}')
    args.output_root.mkdir(parents=True, exist_ok=True)
    pidfile = args.output_root / 'run.pid'
    statusfile = args.output_root / 'run_status.json'
    if statusfile.is_file():
        state = json.loads(statusfile.read_text())
        if state.get('status') == 'running':
            try:
                os.kill(int(state['pid']), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"Task is still running with PID {state['pid']}")
    logfile = args.output_root / 'logs' / 'background.log'
    logfile.parent.mkdir(parents=True, exist_ok=True)
    # Resolve configuration now so the detached process uses precisely this plan.
    cmd = [sys.executable, str(Path(__file__).resolve()), 'run', *options,
           '--target', args.target, '--input-root', str(args.input_root),
           '--output-root', str(args.output_root), '--python', str(args.python)]
    with logfile.open('a') as log:
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    pidfile.write_text(str(child.pid) + '\n')
    print(f'Background PID: {child.pid}\nPID file: {pidfile}\nLog: {logfile}')
    return 0


def complete_image(path):
    from PIL import Image
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def progress(args):
    while True:
        print(f'Target: {args.target}; model: {args.model}; output: {args.output_root}')
        print(f"{'dataset':28s} {'input':>8s} {'done':>8s} {'remain':>8s} {'percent':>9s}")
        total = done = 0
        for dataset in args.datasets:
            try:
                inputs = queries(args, dataset)
            except FileNotFoundError:
                print(f'{dataset:28s} INPUT MISSING')
                continue
            exemplars = args.exemplars if args.target == 'normal' else [None]
            count = sum(complete_image(prediction_path(args, dataset, p, e)) for e in exemplars for p in inputs)
            n = len(inputs) * len(exemplars)
            print(f'{dataset:28s} {n:8d} {count:8d} {n-count:8d} {100*count/n if n else 0:8.2f}%')
            total += n
            done += count
        print(f"{'TOTAL':28s} {total:8d} {done:8d} {total-done:8d} {100*done/total if total else 0:8.2f}%")
        status = args.output_root / 'run_status.json'
        if status.is_file():
            state = json.loads(status.read_text())
            status_name = state['status']
            if status_name == 'running':
                try:
                    os.kill(int(state['pid']), 0)
                except ProcessLookupError:
                    status_name = 'stopped'
            print(f"Last run: {status_name}; PID: {state['pid']}")
        if not args.watch:
            return 0
        time.sleep(args.watch)


def main(argv=None, target='albedo'):
    options = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(options, target=target)
    try:
        if args.action == 'run':
            return run(args)
        if args.action == 'start':
            return start(args, options[1:])
        return progress(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
