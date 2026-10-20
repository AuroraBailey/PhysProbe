"""Exercise task commands, prompt wiring and saved outputs with fake API responses."""
import base64
import contextlib
import importlib
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
try:
    import dotenv
    import requests
    from PIL import Image
    from experiments.main_stress import task_dispatch as dispatch
except ImportError:
    HAS_RUNTIME = False
else:
    HAS_RUNTIME = True

@unittest.skipUnless(HAS_RUNTIME, "Requires main runtime dependencies.")
class RemainingTaskWiringTests(unittest.TestCase):
    def test_commands_reach_api_with_central_prompt(self):
        import requests
        from common.utils import gemini_client
        original_run = runpy.run_path
        from PIL import Image
        png = io.BytesIO()
        Image.new('RGB', (1536, 1024), 'gray').save(png, format='PNG')
        payload = png.getvalue()
        encoded = base64.b64encode(payload).decode('ascii')
        class Response:
            status_code = 200
            ok = True
            def raise_for_status(self): pass
            def json(self):
                return {'output': {'choices': [{'message': {'content': [{'image': 'https://image.invalid/result.png'}]}}]}}
            def iter_content(self, chunk_size): return iter([payload])
            def __enter__(self): return self
            def __exit__(self, *unused): pass
        credentials = dict(ARK_API_KEY='test-key', DASHSCOPE_API_KEY='test-key',
                           AIHUBMIX_API_KEY='test-key', AZURE_OPENAI_API_KEY='test-key',
                           AZURE_OPENAI_ENDPOINT='https://azure.invalid/', AZURE_OPENAI_API_VERSION='test-version')
        for target in ('normal', 'roughness', 'metallic'):
            module = importlib.import_module(f'experiments.main_stress.{target}.prompts')
            for model in dispatch.TARGET_MODELS[target]:
                with self.subTest(target=target, model=model), tempfile.TemporaryDirectory() as tmp:
                    assets = ROOT / 'examples/assets'
                    dataset = 'mainaxis' if target == 'metallic' else 'openroomff_mainaxis'
                    input_root = assets / 'pbr_rooms' if target == 'metallic' else assets
                    options = ['run', '--target', target, '--model', model, '--dataset', dataset,
                               '--input-root', str(input_root), '--output-root', tmp, '--max-generate', '1']
                    if target == 'normal':
                        options += ['--exemplar', 'E1']
                    with patch('common.utils.env.load_project_env'), patch.object(dispatch, 'load_project_env'):
                        args = dispatch.parse_args(options)
                    cmd = dispatch.commands(args, dataset)[0]
                    if model == 'gemini':
                        scenes = Path(cmd[cmd.index('--scene_list') + 1])
                        scenes.parent.mkdir(parents=True, exist_ok=True)
                        scenes.write_text('.\n')
                    captured = []
                    def intercept(*unused, **kwargs):
                        if 'prompt' in kwargs:
                            captured.append(kwargs['prompt'])
                        else:
                            messages = kwargs['json']['input']['messages'] if 'json' in kwargs else kwargs['messages']
                            captured.extend(p['text'] for m in messages if isinstance(m['content'], list)
                                            for p in m['content'] if 'text' in p)
                        if 'json' in kwargs:
                            return Response()
                        return SimpleNamespace(data=[SimpleNamespace(b64_json=encoded, url=None)],
                            choices=[SimpleNamespace(message=SimpleNamespace(
                                multi_mod_content=[{'inline_data': {'data': encoded}}]))])
                    client = SimpleNamespace(images=SimpleNamespace(generate=intercept, edit=intercept),
                        chat=SimpleNamespace(completions=SimpleNamespace(create=intercept)))
                    def invoke(path, **unused):
                        ns = original_run(str(path), run_name='wiring_test')
                        overrides = {'Ark': Mock(return_value=client), 'AzureOpenAI': Mock(return_value=client)}
                        if target == 'normal':
                            original_generate = ns['run_image_generation']
                            def check_order(**kwargs):
                                names = [Path(p).name for p in kwargs['image_paths']]
                                expected = (['0_im.png', 'image.png', 'normal.png'] if model == 'doubao'
                                            else ['query.png', 'example_rgb.png', 'example_normal.png'])
                                self.assertEqual(names, expected)
                                return original_generate(**kwargs)
                            overrides['run_image_generation'] = check_order
                        with patch.dict(ns['main'].__globals__, overrides):
                            ns['main']()
                    with patch('common.utils.env.load_project_env'), patch.dict(os.environ, credentials), \
                            patch.object(sys, 'argv', cmd[1:]), patch.object(requests, 'post', side_effect=intercept), \
                            patch.object(requests, 'get', return_value=Response()), \
                            patch.object(gemini_client, 'OpenAI', Mock(return_value=client)), \
                            patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')), \
                            contextlib.redirect_stdout(io.StringIO()):
                        if target == 'roughness' and model == 'gpt-image-2.0':
                            ns = original_run(cmd[1], run_name='wrapper_test')
                            with patch.object(runpy, 'run_path', side_effect=invoke):
                                ns['main']()
                        else:
                            invoke(cmd[1])
                    self.assertEqual(captured, [module.PROMPTS[dispatch.MODEL_SUFFIXES[model]]])
                    image, = dispatch.queries(args, dataset)
                    self.assertTrue(dispatch.complete_image(dispatch.prediction_path(args, dataset, image)))
                    self.assertEqual(dispatch.prediction_path(args, dataset, image).read_bytes(), payload)
                    if target == 'normal' and model == 'doubao':
                        import json
                        out = Path(cmd[cmd.index('--output_dir') + 1])
                        records = json.loads((out / 'meta/manifest.json').read_text())
                        self.assertEqual(len(records), 1)
                        self.assertEqual(records[0]['status'], 'completed')
                        self.assertEqual(records[0]['input_order'], ['query_rgb', 'example_rgb', 'example_normal'])

    def test_normal_exemplars_have_separate_outputs(self):
        with patch.object(dispatch, 'load_project_env'):
            args = dispatch.parse_args(['run', '--target', 'normal', '--model', 'doubao',
                                       '--dataset', 'openroomff_mainaxis'])
        commands = dispatch.commands(args, 'openroomff_mainaxis')
        self.assertEqual(len(commands), 3)
        outputs = [cmd[cmd.index('--output_dir') + 1] for cmd in commands]
        self.assertEqual(len(set(outputs)), 3)
        self.assertEqual(args.exemplars, ['E1', 'E2', 'E3'])

    def test_metallic_limit_and_prediction_preserve_camera_layout(self):
        with patch.object(dispatch, 'load_project_env'):
            args = dispatch.parse_args(['run', '--target', 'metallic', '--model', 'gemini',
                '--dataset', 'mainaxis', '--input-root', str(ROOT / 'examples/assets/pbr_rooms'), '--max-generate', '1'])
        image, = dispatch.queries(args, 'mainaxis')
        cmd, = dispatch.commands(args, 'mainaxis')
        expected = Path(cmd[cmd.index('--output_dir') + 1]) / (image.stem + '_metallic.png')
        self.assertEqual(dispatch.prediction_path(args, 'mainaxis', image), expected)
