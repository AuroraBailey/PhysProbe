"""Check retained prompt behavior and the shared albedo model mapping."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class MainStressPromptTests(unittest.TestCase):
    def test_prompt_text_matches_recorded_contract(self):
        cases = json.loads((ROOT / 'tests/fixtures/main_stress_prompt_hashes.json').read_text())
        for index, case in enumerate(cases):
            with self.subTest(index=index, module=case['module'], function=case['function']):
                module = importlib.import_module(case['module'])
                kwargs = dict(case['kwargs'])
                for name, (text, count) in case.get('repeat_kwargs', {}).items():
                    kwargs[name] = text * count
                function = getattr(module, case['function'])
                if 'error' in case:
                    with self.assertRaises(Exception) as caught:
                        function(**kwargs)
                    self.assertEqual(type(caught.exception).__name__, case['error'])
                    self.assertEqual(str(caught.exception), case['message'])
                else:
                    result = function(**kwargs)
                    self.assertEqual(hashlib.sha256(result.encode()).hexdigest(), case['sha256'])

    def test_no_prompt_builders_remain_in_api_runners(self):
        names = {'make_albedo_prompt', 'build_relative_depth_prompt', 'build_roughness_prompt',
                 'build_metallic_prompt', 'build_analysis_prompt', 'make_zero_shot_prompt',
                 'make_one_shot_prompt'}
        for path in (ROOT / 'experiments/main_stress').rglob('*.py'):
            if 'prompts' in path.parts or path.name == 'prompts.py':
                continue
            tree = ast.parse(path.read_text())
            definitions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            with self.subTest(path=path):
                self.assertFalse(names & definitions)

    def test_albedo_model_groups(self):
        from experiments.main_stress.albedo.prompts import (
            ALBEDO_FULL_PROMPT, make_albedo_prompt,
        )
        for model in ('doubao', 'gemini', 'gpt_image_15', 'gpt_image_2', 'qwen2', 'wan'):
            self.assertEqual(make_albedo_prompt(model), ALBEDO_FULL_PROMPT)
        from experiments.ablation.albedo.prompts import FULL_PROMPT_TEXT
        self.assertEqual(ALBEDO_FULL_PROMPT, FULL_PROMPT_TEXT)
        with self.assertRaises(ValueError):
            make_albedo_prompt('unknown')

    def test_roughness_gpt_image_2_shared_prompt(self):
        from experiments.main_stress.roughness.prompts import build_roughness_prompt
        self.assertEqual(build_roughness_prompt(model='gpt_image_15'),
                         build_roughness_prompt(model='gpt_image_2'))
        with self.assertRaises(ValueError):
            build_roughness_prompt('rgb_plus_seg', model='gpt_image_15')

    def test_depth_runners_import_the_same_builder(self):
        from experiments.main_stress.depth.prompts import build_relative_depth_prompt
        prompt = build_relative_depth_prompt('near_white')
        self.assertTrue(prompt.startswith('You are a senior monocular depth estimation'))
        self.assertIn('Brighter values indicate surfaces closer to the camera; darker values indicate surfaces farther from the camera.', prompt)
        from experiments.ablation.depth.prompts import build_direct_prompt, build_example_prompt, build_seg_prompt
        self.assertEqual(build_direct_prompt(), prompt)
        for builder in (build_example_prompt, build_seg_prompt):
            self.assertIn('Brighter values indicate surfaces closer to the camera; darker values indicate surfaces farther from the camera.', builder())
        self.assertIn('This is a relative depth prediction task', prompt)
        for path in (ROOT / 'experiments/main_stress/depth').glob('*/depth_*.py'):
            if path.parent.name == 'GPT-Image-2.0':
                continue  # This entry delegates to the shared GPT implementation.
            tree = ast.parse(path.read_text())
            self.assertTrue(any(isinstance(n, ast.ImportFrom) and
                n.module == 'experiments.main_stress.depth.prompts' and
                any(alias.name == 'build_relative_depth_prompt' for alias in n.names)
                for n in tree.body), path)

    def test_depth_rejects_removed_prompt_options(self):
        from experiments.main_stress.depth.prompts import build_relative_depth_prompt
        with self.assertRaises(TypeError):
            build_relative_depth_prompt('near_white', analysis_text='unused')
        with self.assertRaises(TypeError):
            build_relative_depth_prompt('near_white', input_mode='rgb_plus_example')

    def test_metallic_model_prompt_selection(self):
        from experiments.main_stress.metallic.prompts import (
            METALLIC_PROMPT, QWEN_WAN_METALLIC_PROMPT, TEXTURE_GUIDANCE, SUPPORTED_MODELS, build_metallic_prompt,
        )
        self.assertIn(TEXTURE_GUIDANCE, METALLIC_PROMPT)
        self.assertIn("Prefer a sparse, semantically correct material-category map", METALLIC_PROMPT)
        for model in SUPPORTED_MODELS:
            expected = QWEN_WAN_METALLIC_PROMPT if model in {"qwen2", "wan"} else METALLIC_PROMPT
            self.assertEqual(build_metallic_prompt(model=model), expected)
        self.assertEqual(len(QWEN_WAN_METALLIC_PROMPT), 2684)

    def test_normal_models_use_shared_query_first_prompt(self):
        from experiments.main_stress.normal.prompts import NORMAL_PROMPT, SUPPORTED_MODELS, make_one_shot_prompt
        self.assertTrue(NORMAL_PROMPT.startswith("You are given three images in order: (1) the query indoor RGB"))
        self.assertIn("Normals pointing to the camera-right (+X) should appear red / pink.", NORMAL_PROMPT)
        self.assertTrue(NORMAL_PROMPT.endswith("Output only the final normal map for image (1)."))
        for model in SUPPORTED_MODELS:
            self.assertEqual(make_one_shot_prompt(model=model), NORMAL_PROMPT)

    def test_roughness_all_models_use_ablation_a0_prompt(self):
        from experiments.main_stress.roughness.prompts import (
            ROUGHNESS_PROMPT, SUPPORTED_MODELS, build_roughness_prompt,
        )
        from experiments.ablation.roughness.prompts import build_prompt
        self.assertEqual(ROUGHNESS_PROMPT, build_prompt("a0"))
        self.assertTrue(ROUGHNESS_PROMPT.endswith("Output exactly one grayscale roughness map only."))
        for model in SUPPORTED_MODELS:
            self.assertEqual(build_roughness_prompt(model=model), ROUGHNESS_PROMPT)
