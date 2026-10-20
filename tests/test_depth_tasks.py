"""Check directory dispatch, dataset-level limits, and depth progress paths."""
import contextlib
import importlib
import io
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

try:
    import dotenv
    import requests
    from PIL import Image
except ImportError:
    HAS_RUNTIME = False
else:
    HAS_RUNTIME = True


@unittest.skipUnless(HAS_RUNTIME, 'Requires python-dotenv, requests and Pillow.')
class DepthTasksTests(unittest.TestCase):
    def test_depth_arguments_limits_and_progress(self):
        tasks = importlib.import_module('experiments.main_stress.task_dispatch')
        sample = ROOT / 'examples/assets/openroomff_mainaxis/scene0586_02/main_xml/0_im.png'
        with tempfile.TemporaryDirectory(prefix='depth-tasks-test-') as tmp:
            root = Path(tmp)
            input_root = root / 'data'
            for scene in ('scene_a', 'scene_b'):
                parent = input_root / 'openroomff_mainaxis' / scene / 'main_xml'
                parent.mkdir(parents=True)
                for frame in (0, 1):
                    shutil.copyfile(sample, parent / f'{frame}_im.png')
            for model in tasks.MODELS:
                with self.subTest(model=model), patch.object(tasks, 'load_project_env'):
                    args = tasks.parse_args(['run', '--target', 'depth', '--model', model,
                        '--dataset', 'openroomff_mainaxis', '--input-root', str(input_root),
                        '--output-root', str(root / 'output' / model), '--max-generate', '3',
                        '--deployment', 'test-model'])
                    cmds = tasks.commands(args, 'openroomff_mainaxis')
                    self.assertEqual([c[c.index('--max_generate') + 1] for c in cmds], ['2', '1'])
                    for cmd in cmds:
                        path = Path(cmd[1])
                        self.assertTrue(path.is_file())
                        if model == 'gpt-image-2.0':
                            # The wrapper forwards argv to this shared implementation.
                            path = path.parent.parent / 'GPT-Image-1.5/depth_gpt_image_15.py'
                        with patch('common.utils.env.load_project_env'), patch.object(sys, 'argv', [str(path), *cmd[2:]]):
                            module = runpy.run_path(str(path), run_name='depth_argument_test')
                            parsed = module['parse_args']()
                        self.assertEqual(parsed.input_mode, 'rgb_only')
                        self.assertEqual(parsed.image_model, 'test-model')
                    first = tasks.queries(args, 'openroomff_mainaxis')[0]
                    pred = tasks.prediction_path(args, 'openroomff_mainaxis', first)
                    pred.parent.mkdir(parents=True)
                    shutil.copyfile(sample, pred)
                    self.assertEqual(pred.parent.name, 'relative_depth')
                    stream = io.StringIO()
                    with contextlib.redirect_stdout(stream):
                        tasks.progress(args)
                    row = next(line for line in stream.getvalue().splitlines() if line.startswith('TOTAL'))
                    self.assertEqual(row.split()[1:4], ['3', '1', '2'])
