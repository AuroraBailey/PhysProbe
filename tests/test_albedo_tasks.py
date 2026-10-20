"""Check shared launch commands, detached execution, and query-matched progress."""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

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
class AlbedoTasksTests(unittest.TestCase):
    def setUp(self):
        self.tasks = importlib.import_module('experiments.main_stress.task_dispatch')
        self.tmp = tempfile.TemporaryDirectory(prefix='albedo-tasks-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / 'input'
        self.output = self.root / 'output'
        self.dataset = 'openroomff_mainaxis'
        self.folder = self.input / self.dataset / 'scene/main_xml'
        self.folder.mkdir(parents=True)
        image = ROOT / 'examples/assets/openroomff_mainaxis/scene0586_02/main_xml/0_im.png'
        for frame in (0, 1):
            shutil.copyfile(image, self.folder / f'{frame}_im.png')

    def options(self, action='run', model='qwen2'):
        return [action, '--model', model, '--dataset', self.dataset,
                '--input-root', str(self.input), '--output-root', str(self.output)]

    def parse(self, options):
        with patch.object(self.tasks, 'load_project_env'):
            return self.tasks.parse_args(options)

    def test_all_six_dispatch_commands_match_real_argument_parsers(self):
        for model in self.tasks.MODELS:
            with self.subTest(model=model):
                args = self.parse(self.options(model=model) + ['--deployment', 'selected-model'])
                cmd = self.tasks.command(args, self.dataset)
                self.assertTrue(Path(cmd[1]).name.startswith('albedo_'))
                with patch('common.utils.env.load_project_env'), patch.object(sys, 'argv', cmd[1:]):
                    runner = runpy.run_path(cmd[1], run_name='albedo_argument_test')
                    parsed = runner['parse_args']()
                self.assertEqual(parsed.input_dir, str(self.input / self.dataset))
                self.assertTrue(parsed.preserve_relative_dirs)
                self.assertNotIn('--api_key', cmd)
                self.assertNotIn('--analysis_model', cmd)

    def test_progress_ignores_orphans_and_corrupt_images(self):
        args = self.parse(self.options('progress'))
        pred = self.output / self.dataset / 'albedo/scene/main_xml'
        pred.mkdir(parents=True)
        shutil.copyfile(self.folder / '0_im.png', pred / '0_im_albedo.png')
        (pred / '1_im_albedo.png').write_text('corrupt')
        shutil.copyfile(self.folder / '0_im.png', pred / 'orphan_im_albedo.png')
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.tasks.progress(args)
        row = next(line for line in stream.getvalue().splitlines() if line.startswith('TOTAL'))
        self.assertEqual(row.split()[1:4], ['2', '1', '1'])

    def test_run_records_failure_and_writes_gemini_scene_list(self):
        args = self.parse(self.options(model='gemini'))
        with patch.object(self.tasks, 'check_credentials'), \
                patch.object(self.tasks.subprocess, 'run', return_value=Mock(returncode=7)) as execute, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.tasks.run(args), 1)
        self.assertEqual((self.output / 'inputs' / f'{self.dataset}.txt').read_text(), '.\n')
        self.assertTrue(execute.call_args.kwargs['stdout'].closed)
        state = json.loads((self.output / 'run_status.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['datasets'][self.dataset]['returncode'], 7)

    def test_background_is_detached_with_resolved_paths_and_pid(self):
        options = self.options('start')
        args = self.parse(options)
        with patch.object(self.tasks, 'check_credentials'), \
                patch.object(self.tasks.subprocess, 'Popen', return_value=Mock(pid=12345)) as launch, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.tasks.start(args, options[1:]), 0)
        self.assertTrue(launch.call_args.kwargs['start_new_session'])
        self.assertEqual((self.output / 'run.pid').read_text(), '12345\n')
        self.assertEqual(launch.call_args.args[0][2], 'run')
        self.assertIn(str(self.input), launch.call_args.args[0])
