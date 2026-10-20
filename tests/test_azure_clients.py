"""Exercise client configuration without importing image runtimes or calling APIs."""
import argparse
import ast
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(os=os, argparse=argparse, Any=Any, Dict=Dict, Optional=Optional, Tuple=Tuple,
                     DEFAULT_API_VERSION='test-version', DEFAULT_AZURE_API_VERSION='test-version',
                     DEFAULT_AZURE_ENDPOINT='https://placeholder.invalid/', AzureOpenAI=Mock())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class AzureClientTests(unittest.TestCase):
    def test_main_clients_require_endpoint_and_use_azure(self):
        paths = list((ROOT / 'experiments/main_stress').glob('*/GPT-Image-*/*.py'))
        tested = 0
        for path in paths:
            tree = ast.parse(path.read_text())
            if not any(isinstance(n, ast.FunctionDef) and n.name == 'resolve_image_client_config' for n in tree.body):
                continue
            with self.subTest(path=path), patch.dict(os.environ, {}, clear=True):
                ns = load_functions(path, {'ensure_api_key', 'resolve_image_client_config'})
                args = argparse.Namespace(api_key='test-key', albedo_api_key=None,
                    albedo_endpoint=None, azure_endpoint=None, base_url=None, albedo_api_version=None,
                    api_version=None, albedo_deployment=None, albedo_model='test-model',
                    image_deployment=None, image_model='test-model')
                with self.assertRaises(ValueError):
                    ns['resolve_image_client_config'](args)
                ns['AzureOpenAI'].assert_not_called()
                os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://azure.invalid/'
                os.environ['AZURE_OPENAI_API_VERSION'] = 'explicit-version'
                ns['resolve_image_client_config'](args)
                ns['AzureOpenAI'].assert_called_once_with(
                    api_key='test-key', azure_endpoint='https://azure.invalid/', api_version='explicit-version')
                # A legacy official-service credential must not authenticate a GPT run.
                args.api_key = None
                os.environ['OPENAI_API_KEY'] = 'legacy-key'
                with self.assertRaises(RuntimeError):
                    ns['resolve_image_client_config'](args)
                tested += 1
        self.assertEqual(tested, 6)

    def test_diagnostic_clients_use_azure(self):
        for target in ('depth', 'albedo', 'normal'):
            path = ROOT / f'experiments/ablation/{target}/{target}_provider_utils.py'
            ns = load_functions(path, {'ensure_azure_import', 'build_gpt_client'})
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    ns['build_gpt_client']('', 'test-key', 'version', 30)
                _, kind = ns['build_gpt_client']('https://azure.invalid/', 'test-key', 'version', 30)
                self.assertEqual(kind, 'azure')
                ns['AzureOpenAI'].assert_called_once_with(api_key='test-key',
                    azure_endpoint='https://azure.invalid/', api_version='version', timeout=30)

    def test_normal_roles_share_env_and_cli_takes_precedence(self):
        paths = (ROOT / 'experiments/main_stress/normal/GPT-Image-1.5').glob('*.py')
        for path in paths:
            with self.subTest(path=path), patch.dict(os.environ, {
                'AZURE_OPENAI_API_KEY': 'shared-key',
                'AZURE_OPENAI_ENDPOINT': 'https://shared.invalid/',
                'AZURE_OPENAI_API_VERSION': 'shared-version',
            }, clear=True):
                ns = load_functions(path, {'resolve_azure_client_config'})
                args = argparse.Namespace(api_key=None, azure_endpoint=None, api_version=None)
                resolve = ns['resolve_azure_client_config']
                for role in ('normal',):
                    self.assertEqual(resolve(role, None, None, None, args),
                        ('shared-key', 'https://shared.invalid/', 'shared-version'))
                    args.api_key, args.azure_endpoint, args.api_version = (
                        'cli-key', 'https://cli.invalid/', 'cli-version')
                    self.assertEqual(resolve(role, None, None, None, args),
                        ('cli-key', 'https://cli.invalid/', 'cli-version'))
                    self.assertEqual(resolve(role, 'role-key', 'https://role.invalid/', 'role-version', args),
                        ('role-key', 'https://role.invalid/', 'role-version'))
                    args.api_key = args.azure_endpoint = args.api_version = None
                os.environ.pop('AZURE_OPENAI_ENDPOINT')
                with self.assertRaises(ValueError):
                    resolve('normal', None, None, None, args)


if __name__ == '__main__':
    unittest.main()
