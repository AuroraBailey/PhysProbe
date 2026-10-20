"""Exercise each real albedo entry point up to its intercepted API request."""
import contextlib
import io
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

try:
    import dotenv
    import requests
    from PIL import Image
except ImportError:
    HAS_RUNTIME = False
else:
    HAS_RUNTIME = True


class RequestCaptured(BaseException):
    """Stop before a network request, bypassing the runners' retry handling."""


@unittest.skipUnless(HAS_RUNTIME, 'Requires python-dotenv, requests and Pillow.')
class AlbedoPromptWiringTests(unittest.TestCase):
    def test_all_models_send_the_selected_shared_prompt(self):
        from common.utils import gemini_client
        from experiments.main_stress.albedo.prompts import make_albedo_prompt

        entries = {
            'doubao': 'Doubao_Seedream_5.0/albedo_doubao.py',
            'gpt_image_15': 'GPT-Image-1.5/albedo_gpt_image_15.py',
            'gpt_image_2': 'GPT-Image-2.0/albedo_gpt_image_2.py',
            'gemini': 'Gemini_3_Pro_Image/albedo_gemini.py',
            'qwen2': 'Qwen-Image-2.0/albedo_qwen2.py',
            'wan': 'Wan2.7-Image/albedo_wan.py',
        }
        assets = ROOT / 'examples/assets/openroomff_mainaxis'
        image_dir = assets / 'scene0586_02/main_xml'

        for model, relative in entries.items():
            with self.subTest(model=model), tempfile.TemporaryDirectory(prefix='albedo-prompt-test-') as tmp:
                captured = []

                def intercept(*args, **kwargs):
                    if 'prompt' in kwargs:  # Azure / ARK image request.
                        captured.append(kwargs['prompt'])
                    else:  # Gemini chat request or DashScope HTTP JSON.
                        messages = (kwargs['json']['input']['messages'] if 'json' in kwargs
                                    else kwargs['messages'])
                        captured.extend(part['text'] for message in messages
                                        if isinstance(message['content'], list)
                                        for part in message['content'] if 'text' in part)
                    raise RequestCaptured()

                client = SimpleNamespace(
                    images=SimpleNamespace(generate=intercept, edit=intercept),
                    chat=SimpleNamespace(completions=SimpleNamespace(create=intercept)),
                )
                path = ROOT / 'experiments/main_stress/albedo' / relative
                with patch('common.utils.env.load_project_env'):
                    namespace = runpy.run_path(str(path), run_name='albedo_wiring_test')
                globals_ = namespace['main'].__globals__
                argv = [str(path), '--input_dir', str(image_dir), '--output_dir', str(Path(tmp) / 'output'),
                        '--api_key', 'test-key', '--base_url', 'https://example.invalid/v1',
                        '--filename_suffix', '_im.png', '--max_generate', '1']
                if model == 'doubao':
                    argv.append('--independent_images')
                else:
                    argv += ['--gt_root', str(assets)]
                if model == 'gemini':
                    scenes = Path(tmp) / 'scenes.txt'
                    scenes.write_text('scene0586_02/main_xml\n')
                    argv += ['--scene_list', str(scenes), '--dataset_root', str(assets)]

                with patch.object(sys, 'argv', argv), \
                        patch.dict(globals_, {'Ark': Mock(return_value=client),
                                              'AzureOpenAI': Mock(return_value=client)}), \
                        patch.object(gemini_client, 'OpenAI', Mock(return_value=client)), \
                        patch.object(requests, 'post', side_effect=intercept), \
                        patch('socket.socket.connect', side_effect=AssertionError('Unexpected network access')), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(RequestCaptured):
                        namespace['main']()
                self.assertEqual(captured, [make_albedo_prompt(model)])
