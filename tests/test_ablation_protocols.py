"""Verify paper setting identities, manifest selection, image evidence and region-fill behavior."""
import ast
from collections import Counter
import contextlib
import csv
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
try:
    import dotenv
    import numpy as np
    from PIL import Image
except ImportError:
    HAS_RUNTIME = False
else:
    HAS_RUNTIME = True

@unittest.skipUnless(HAS_RUNTIME, 'Requires dotenv, Pillow, numpy and main runtime dependencies.')
class AblationProtocolTests(unittest.TestCase):
    def test_paper_manifests_have_expected_balanced_subsets(self):
        for target in ('albedo', 'depth', 'normal', 'roughness'):
            with self.subTest(target=target):
                with (ROOT / f'data/manifests/ablation/{target}_manifest_ablation.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 96)
                counts = Counter((r['datasource'], r['split']) for r in rows)
                self.assertEqual(set(counts.values()), {48} if target == 'roughness' else {24})

    def test_all_paper_settings_run_with_expected_query_evidence(self):
        from experiments.ablation import tasks
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rel = Path('openroomff_mainaxis/scene/main_xml/0_im.png')
            query = tmp / 'data' / rel
            query.parent.mkdir(parents=True)
            Image.new('RGB', (8, 8), 'gray').save(query)
            Image.new('RGB', (8, 8), 'red').save(query.with_name('0_seg.png'))
            manifest = tmp / 'manifest.csv'
            with manifest.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['datasource', 'split', 'rgb_path'])
                writer.writeheader()
                writer.writerow(dict(datasource='openroomff', split='mainaxis', rgb_path=str(rel.with_name('0_im_5.rgbe'))))
            cache = tmp / 'cache'
            analysis = cache / rel.with_suffix('.json')
            analysis.parent.mkdir(parents=True)
            analysis.write_text(json.dumps({'scene_summary': 'SHARED_QUERY_ONLY_ANALYSIS', 'albedo_material_notes': ['wood'],
                                            'albedo_lighting_notes': ['cast shadow']}))
            second_rel = Path('openroomff_mainaxis/scene_b/main_xml/0_im.png')
            second = tmp / 'data' / second_rel
            second.parent.mkdir(parents=True)
            Image.new('RGB', (8, 8), 'blue').save(second)
            Image.new('RGB', (8, 8), 'green').save(second.with_name('0_seg.png'))
            with manifest.open('a', newline='') as handle:
                csv.writer(handle).writerow(['openroomff', 'mainaxis', str(second_rel.with_name('0_im_5.rgbe'))])
            second_analysis = cache / second_rel.with_suffix('.json')
            second_analysis.parent.mkdir(parents=True)
            second_analysis.write_bytes(analysis.read_bytes())
            for target in tasks.SETTINGS:
                for model in tasks.PAPER_MODELS[target]:
                    with self.subTest(target=target, model=model), patch.object(tasks, 'load_project_env'):
                        args = tasks.parse_args(['run', '--target', target, '--model', model, '--manifest', str(manifest),
                            '--input-root', str(tmp / 'data'), '--output-root', str(tmp / target / model), '--analysis-dir', str(cache), '--sleep', '0'])
                        selected = tasks.records(args)
                        self.assertEqual(selected[0]['relative'], rel)
                        tasks.stage(args, selected)
                        for job in tasks.jobs(args):
                            command = job['command']
                            if target == 'albedo' and job['setting'] == 'A1':
                                runner = importlib.import_module('experiments.ablation.albedo.albedo_analysis_runner')
                                entry = SimpleNamespace(main=runner.main)
                            elif target == 'roughness':
                                runner = importlib.import_module('experiments.ablation.roughness.roughness_ablation_runner')
                                entry = SimpleNamespace(main=runner.main)
                            else:
                                entry = importlib.import_module(f'experiments.ablation.{target}.run_{target}_{job["setting"].lower()}')
                                runner = importlib.import_module(f'experiments.ablation.{target}.{target}_ablation_runner')
                            runtime = SimpleNamespace(provider=tasks.MODELS[model], albedo_model='test', analysis_model='doubao-seed-2-0-pro-260215',
                                albedo_base_url='https://invalid', analysis_base_url='https://invalid', albedo_quality='medium', generation_mode='edit',
                                normal_model='test', normal_base_url='https://invalid', normal_quality='high',
                                base_url='https://invalid', image_model='test', gpt_backend='azure', image_quality='medium', seed=123)
                            calls = []
                            def generate(*positional, **kwargs):
                                prompt = kwargs.get('prompt', positional[1] if positional else '')
                                images = kwargs.get('image_paths', positional[2] if positional else [])
                                calls.append((prompt, [Path(p) for p in images]))
                                return object()
                            def save(runtime, response, path, **kwargs):
                                Image.new('RGB', (8, 8), 'gray').save(path)
                                return ''
                            name = {'albedo': 'run_albedo_generation_with_retry', 'normal': 'run_normal_generation',
                                    'depth': 'run_depth_generation_with_retry', 'roughness': 'run_depth_generation_with_retry'}[target]
                            with patch.object(sys, 'argv', command[1:]), patch.object(runner, 'resolve_provider_runtime', return_value=runtime), \
                                    patch.object(runner, name, side_effect=generate), patch.object(runner, 'save_generated_image', side_effect=save), \
                                    patch('socket.socket.connect', side_effect=AssertionError('No network in offline test')), contextlib.redirect_stdout(io.StringIO()):
                                if hasattr(entry, 'CONFIG'):
                                    runner.run_variant(entry.CONFIG)
                                else:
                                    entry.main()
                            self.assertEqual(len(calls), 2)
                            prompt, inputs = calls[0]
                            setting = job['setting']
                            expected_count = (3 if target == 'normal' and setting in {'A0', 'A2'} or target == 'albedo' and setting == 'B0' or
                                target == 'depth' and setting == 'A1' or target == 'roughness' and setting == 'A3' else
                                2 if target == 'depth' and setting == 'A2' or target == 'roughness' and setting in {'A1', 'A2'} else 1)
                            self.assertEqual(len(inputs), expected_count)
                            self.assertEqual(inputs[-1].name if expected_count == 3 and target != 'normal' else inputs[0].name, '0_im.png')
                            if target == 'albedo' and setting == 'A1':
                                self.assertIn('SHARED_QUERY_ONLY_ANALYSIS', prompt)
                                self.assertEqual((job['output'] / 'meta/per_image_analysis' / rel.with_suffix('.json')).read_bytes(), analysis.read_bytes())
                            self.assertTrue(tasks.prediction(job, target, rel).is_file())
                            self.assertTrue(tasks.prediction(job, target, second_rel).is_file())
                            run_records = json.loads((job['output'] / 'meta/manifest.json').read_text())
                            self.assertEqual(len(run_records), 2, 'Same-basename query records must not overwrite each other')

    def test_region_fill_preserves_disconnected_components(self):
        from experiments.ablation.roughness.roughness_ablation_runner import apply_region_fill_from_segmentation
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pred = np.array([[0, 20, 80, 160, 200]], dtype=np.uint8)
            seg = np.array([[[255, 0, 0], [255, 0, 0], [0, 255, 0], [255, 0, 0], [255, 0, 0]]], dtype=np.uint8)
            Image.fromarray(pred).save(tmp / 'pred.png')
            Image.fromarray(seg).save(tmp / 'seg.png')
            apply_region_fill_from_segmentation(tmp / 'pred.png', tmp / 'seg.png', tmp / 'filled.png')
            self.assertEqual(np.array(Image.open(tmp / 'filled.png')).tolist(), [[10, 10, 80, 180, 180]])

    def test_provider_qwen_is_distinct_from_wan(self):
        for target in ('albedo', 'depth'):
            module = importlib.import_module(f'experiments.ablation.{target}.{target}_provider_utils')
            for provider, expected in [('qwen2', 'qwen-image-2.0'), ('wan', 'wan2.7-image')]:
                args = SimpleNamespace(provider=provider, api_key='test', base_url=None, albedo_api_key=None, analysis_api_key=None,
                    albedo_base_url=None, analysis_base_url=None, analysis_model=None, image_model=None, albedo_model=None,
                    timeout=10, generation_mode='edit', albedo_quality='medium')
                runtime = module.resolve_provider_runtime(args, analysis_enabled=False) if target == 'albedo' else module.resolve_provider_runtime(args)
                self.assertEqual(runtime.albedo_model if target == 'albedo' else runtime.image_model, expected)
