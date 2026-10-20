"""Verify endpoint and image/model forwarding without an external API call."""
import base64
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common.utils import gemini_client


class GeminiConfigurationTests(unittest.TestCase):
    def test_environment_endpoint_and_cli_override(self):
        with patch.dict(os.environ, {'AIHUBMIX_BASE_URL': 'https://env.invalid/v1'}, clear=True), \
                patch.object(gemini_client, 'OpenAI') as sdk:
            client = gemini_client.GeminiImageClient('test-key', aspect_ratio='4:3', timeout=30)
            sdk.assert_called_once_with(api_key='test-key', base_url='https://env.invalid/v1', timeout=30)
            sdk.reset_mock()
            gemini_client.GeminiImageClient('override-key', aspect_ratio='4:3', base_url='https://cli.invalid/v1')
            sdk.assert_called_once_with(api_key='override-key', base_url='https://cli.invalid/v1')

    def test_model_and_original_image_bytes_are_forwarded(self):
        encoded = base64.b64encode(b'input-image-bytes').decode()
        output = base64.b64encode(b'output-image-bytes').decode()
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            multi_mod_content=[{'inline_data': {'data': output}}]))])
        with patch.object(gemini_client, 'OpenAI', Mock()) as sdk:
            sdk.return_value.chat.completions.create.return_value = response
            client = gemini_client.GeminiImageClient('test-key', aspect_ratio='4:3', base_url='https://test.invalid/v1')
            result = client.images.edit(model='selected-model', prompt='test prompt', image=f'data:image/png;base64,{encoded}')
            request = sdk.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(request['model'], 'selected-model')
            self.assertEqual(request['messages'][1]['content'][0]['image_url']['url'], f'data:image/png;base64,{encoded}')
            self.assertEqual(result.data[0].b64_json, output)

    def test_image_options_are_silently_ignored(self):
        import warnings
        output = base64.b64encode(b'output').decode()
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            multi_mod_content=[{'inline_data': {'data': output}}]))])
        with patch.object(gemini_client, 'OpenAI', Mock()) as sdk:
            sdk.return_value.chat.completions.create.return_value = response
            client = gemini_client.GeminiImageClient('test-key', aspect_ratio='4:3')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = client.images.edit(model='test-model', prompt='test',
                    size='1024x1024', quality='high', output_format='png', seed=42,
                    watermark=False, unknown_option=True)
            self.assertEqual(caught, [])
            request = sdk.return_value.chat.completions.create.call_args.kwargs
            self.assertFalse({'size', 'quality', 'output_format', 'seed', 'watermark', 'unknown_option'} & request.keys())
            self.assertEqual(result.request_parameters['unused_options']['seed'], 42)
            with self.assertRaises(ValueError):
                client.images.edit(model='test-model', prompt='test', n=2)

    def test_run_record_separates_options_without_affecting_other_providers(self):
        record = {'provider': 'gemini', 'size': '1024x1024', 'seed': 42, 'image_quality': 'high', 'prompt_text': 'test'}
        logged = gemini_client.provider_request_record(record, 'gemini')
        self.assertFalse({'size', 'seed', 'image_quality'} & logged.keys())
        self.assertEqual(logged['unused_options']['size'], '1024x1024')
        self.assertEqual(logged['prompt_text'], 'test')
        self.assertEqual(record['seed'], 42)
        self.assertEqual(gemini_client.provider_request_record(record, 'gpt'), record)

    def test_roughness_ablation_signature_records_effective_gemini_options(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from experiments.ablation.roughness import roughness_ablation_runner as runner
        signature = runner.build_run_signature(
            config=runner.get_variant_config('a0'),
            runtime=SimpleNamespace(provider='gemini', image_model='test', base_url='https://test.invalid'),
            prompt_text='test', example_rgb='', example_roughness='', timeout=30,
            sleep=0, request_retries=0, retry_backoff=0, retry_max_backoff=0,
            size='1024x1024', watermark=False)
        self.assertNotIn('size', signature)
        self.assertEqual(signature['unused_options']['size'], '1024x1024')
        self.assertEqual(signature['effective_request']['aspect_ratio'], '4:3')
