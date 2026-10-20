"""Regression checks for image I/O and paper estimators without external APIs."""
import ast
import base64
import importlib.util
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from common.utils.image_policy import prepare_input, resized_upload
from tools.aggregation.aggregate_results import estimate, merge_records, cluster_ci, setting_of
from tools.evaluation.mask_policy import resolve_required_mask

class ImagePolicyTests(unittest.TestCase):
    def test_raw_inputs_and_responses_preserve_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            src=Path(directory)/'source.png';dst=Path(directory)/'upload.png'
            Image.new('RGB',(91,47),'orange').save(src)
            raw=src.read_bytes()
            prepare_input(src,dst,gpt=False)
            self.assertEqual(dst.read_bytes(),raw)
            prepare_input(src,dst,gpt=True)
            with Image.open(dst) as uploaded:
                self.assertEqual(uploaded.size,(1536,1024))
            for path in (ROOT/'experiments/main_stress').glob('*/*/*.py'):
                tree=ast.parse(path.read_text())
                for function in tree.body:
                    if isinstance(function,ast.FunctionDef) and function.name=='postprocess_generated_image_bytes':
                        ns={}
                        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),ns)
                        self.assertEqual(ns[function.name](raw,crop_bbox=[5,5,8,8],final_size=(2,2)),raw,path)

    def test_gpt_normal_exemplars_and_queries_use_fixed_upload_size(self):
        from experiments.ablation.normal import normal_provider_utils as module
        with tempfile.TemporaryDirectory() as directory:
            image=Path(directory)/'image.png';Image.new('RGB',(13,9),'blue').save(image)
            def edit(**kwargs):
                for upload in kwargs['image']:
                    self.assertEqual(Image.open(upload).size,(1536,1024))
                self.assertEqual(kwargs['size'],'1024x1024') # request size remains independent
                return SimpleNamespace(data=[])
            runtime=SimpleNamespace(normal_client=SimpleNamespace(images=SimpleNamespace(edit=edit)),normal_model='gpt-image-1.5',normal_quality='high')
            module.run_gpt_generation(runtime,'test',[image,image,image],'1024x1024')

    def test_default_masks_require_same_view_and_allow_explicit_opt_out(self):
        with tempfile.TemporaryDirectory() as directory:
            gt=Path(directory)/'4_roughness.png'
            with self.assertRaises(FileNotFoundError):resolve_required_mask(gt)
            mask=gt.with_name('4_mask.png');Image.new('L',(8,8),255).save(mask)
            self.assertEqual(resolve_required_mask(gt),mask)
            self.assertIsNone(resolve_required_mask(gt,mask,disabled=True))

    def test_metallic_half_is_positive_in_both_evaluation_paths(self):
        tree=ast.parse((ROOT/'tools/evaluation/metallic_eval.py').read_text())
        statements=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'gt_binary','pred_binary'} for t in n.targets) and isinstance(n.value,ast.Compare)]
        checked=0
        for statement in statements:
            if isinstance(statement.value.left,ast.Name) and statement.value.left.id in {'gt_valid','pred_valid'}:
                ns={'gt_valid':np.array([.499,.5,.501]),'pred_valid':np.array([.499,.5,.501]),'gt_threshold':.5,'pred_threshold':.5}
                exec(compile(ast.Module(body=[statement],type_ignores=[]),'threshold','exec'),ns)
                self.assertEqual(ns[statement.targets[0].id].tolist(),[False,True,True]);checked+=1
        self.assertEqual(checked,4)

class PaperAggregationTests(unittest.TestCase):
    def test_aggregation_runs_without_a_local_manuscript(self):
        import csv, json
        from tools.aggregation.aggregate_results import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'exports/1_mainaxis/roughness/Gemini_3_Pro_Image/openroomff_mainaxis'
            source.mkdir(parents=True)
            with (source / 'roughness_metrics.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['name', 'rmse', 'ssim', 'region_macro_mae', 'region_count'])
                writer.writeheader()
                writer.writerow(dict(name='scene0001_00/main_xml/0_im.png', rmse=.2, ssim=.8,
                                     region_macro_mae=.1, region_count=2))
            output = root / 'tables'
            main(['--evaluation-root', str(root / 'exports'), '--output-dir', str(output), '--resamples', '10'])
            rows = json.loads((output / 'tables.json').read_text())
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]['rmse'], .2)
            self.assertAlmostEqual(rows[0]['rmse_ci95_lower'], .2)
            self.assertTrue((output / 'audit.json').is_file())
            self.assertTrue((output / 'tables.csv').is_file())
            self.assertTrue((output / 'scene_resampling_indices.npz').is_file())
            self.assertEqual(list(output.glob('paper_*')), [])

    def rows(self):
        return [dict(_source='openroomff',_partition='mainaxis',_exemplar='none',_scene='s1',mae=1),
                dict(_source='openroomff',_partition='mainaxis',_exemplar='none',_scene='s1',mae=1),
                dict(_source='interiorverse',_partition='mainaxis',_exemplar='none',_scene='s2',mae=0)]
    def test_source_balance_is_not_image_pooling(self):
        self.assertEqual(estimate(self.rows(),'albedo','mae'),.5)
    def test_roughness_official_and_diagnostic_are_distinct(self):
        rows=[dict(_split='1_mainaxis',_partition='mainaxis',rmse=0,valid_pixels=1),dict(_split='1_mainaxis',_partition='mainaxis',rmse=1,valid_pixels=9)]
        self.assertEqual(estimate(rows,'roughness','rmse'),.5)
        rows.append(dict(_partition='stresstest',rmse=0,valid_pixels=1))
        self.assertAlmostEqual(estimate(rows,'roughness','rmse',True),np.sqrt(.9)/2)
    def test_regions_have_equal_weight_not_images(self):
        rows=[dict(region_macro_mae=0,region_count=1),dict(region_macro_mae=1,region_count=9)]
        self.assertAlmostEqual(estimate(rows,'roughness','region_macro_mae'),.9)
    def test_bootstrap_retains_entire_scenes_and_is_shared(self):
        a=cluster_ci(self.rows(),'albedo','mae',30,42)
        b=cluster_ci(list(reversed(self.rows())),'albedo','mae',30,42)
        self.assertEqual(a,b);self.assertEqual(a,[.5,.5])
    def test_setting_ids_follow_the_paper_without_remapping(self):
        for target in ('albedo','depth','normal','roughness'):
            self.assertEqual(setting_of(target,'A0'),'A0')
            self.assertEqual(setting_of(target,'a1'),'A1')
            self.assertEqual(setting_of(target,'A2'),'A2')
            self.assertIsNone(setting_of(target,'obsolete_setting'))
            self.assertIsNone(setting_of(target,'a0_full_prompt'))
            self.assertIsNone(setting_of(target,'example'))
        self.assertEqual(setting_of('albedo','B0'),'B0')
        self.assertIsNone(setting_of('depth','A3'))
        self.assertIsNone(setting_of('metallic','A0'))

    def test_canonical_diagnostic_directories_and_exemplars(self):
        import csv, json
        from tools.aggregation.aggregate_results import discover, exemplar_of
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for model in ('Gemini_3_Pro_Image','GPT-Image-2.0'):
                for setting in ('A0','A1','A2','A3','unsupported'):
                    folder=root/'3_comparison/roughness'/model/setting/'openroomff_mainaxis'
                    folder.mkdir(parents=True)
                    with (folder/'roughness_metrics.csv').open('w',newline='') as handle:
                        writer=csv.DictWriter(handle,fieldnames=['name','scene_id'])
                        writer.writeheader();writer.writerow({'name':'scene0001_00/main_xml/0_roughness.png','scene_id':'scene0001_00'})
            audit=[];rows=discover(root,audit)
            self.assertEqual(len(rows),8)
            self.assertEqual({row['_setting'] for row in rows},{'A0','A1','A2','A3'})
            self.assertEqual(sum(item['status']=='unsupported_setting' for item in audit),2)
        self.assertEqual(exemplar_of(('A0','E2','openroomff_mainaxis')),'E2')
        self.assertEqual(exemplar_of(('A0','example2','openroomff_mainaxis')),'none')

    def test_normal_editor_exports_require_canonical_exemplar_directories(self):
        import csv
        from tools.aggregation.aggregate_results import discover
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for exemplar in ('E1','E2','E3','unrecognized_exemplar'):
                folder=root/'1_mainaxis/normal/GPT-Image-1.5'/exemplar/'openroomff_mainaxis'
                folder.mkdir(parents=True)
                with (folder/'normal_metrics.csv').open('w',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=['name','scene_id','strategy'])
                    writer.writeheader();writer.writerow({'name':'scene0001_00/main_xml/0_normal.png','scene_id':'scene0001_00','strategy':'fixed'})
            audit=[];rows=discover(root,audit)
            self.assertEqual({row['_exemplar'] for row in rows},{'E1','E2','E3'})
            self.assertEqual(sum(item['status']=='unsupported_exemplar' for item in audit),1)

    def test_evaluator_bootstrap_retains_image_macro(self):
        from tools.evaluation.roughness_eval import SceneAccumulator, PairDiagnostics, bootstrap_summary
        scene=SceneAccumulator('scene','mainaxis')
        for pixels,sq_sum,ssim in [(1,0,.2),(9,9,.8)]:
            scene.update(PairDiagnostics(valid_pixels=pixels,abs_sum=0,sq_sum=sq_sum,
                bin_abs_sums=np.zeros(5),bin_sq_sums=np.zeros(5),bin_counts=np.zeros(5,dtype=np.int64),
                region_maes=[],region_rmses=[],ssim=ssim,lpips=None))
        summary=bootstrap_summary([scene],20,42)
        self.assertAlmostEqual(summary['rmse_image_macro']['ci95_low'],.5)
        self.assertAlmostEqual(summary['ssim_image_macro']['ci95_low'],.5)
        self.assertNotIn('rmse_scene_macro', summary)
        self.assertNotIn('ssim_scene_macro', summary)

if __name__=='__main__':unittest.main()

class PaperNamingAndPrivacyTests(unittest.TestCase):
    def test_normal_prompt_and_query_first_contract(self):
        from experiments.ablation.normal import prompts
        from experiments.main_stress.normal.prompts import NORMAL_PROMPT
        self.assertEqual(prompts.FULL_RGB_PLUS_EXAMPLE_PROMPT,NORMAL_PROMPT)
        self.assertTrue(prompts.MINIMAL_RGB_PLUS_EXAMPLE_PROMPT.startswith(prompts.INPUT_ORDER))
        self.assertIn('(1) the query indoor RGB image to convert, ',prompts.INPUT_ORDER)
        self.assertNotIn('image (3)',prompts.FULL_RGB_ONLY_PROMPT)
        self.assertNotIn('reference',prompts.FULL_RGB_ONLY_PROMPT.lower())

    def test_aggregation_report_uses_relative_input_paths(self):
        import csv, json, subprocess
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'private_account/evaluations'
            folder=root/'1_mainaxis/albedo/GPT-Image-1.5/openroomff_mainaxis'
            folder.mkdir(parents=True)
            with (folder/'albedo_metrics.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=['name','mae','psnr','ssim','lpips'])
                writer.writeheader();writer.writerow(dict(name='scene0001_00/main_xml/0_albedo.png',mae=.2,psnr=13,ssim=.5,lpips=.4))
            output=Path(directory)/'reports'
            subprocess.run([sys.executable,str(ROOT/'tools/aggregation/aggregate_results.py'),
                '--evaluation-root',str(root),'--output-dir',str(output),'--resamples','0'],check=True,capture_output=True)
            payload=(output/'audit.json').read_text()
            self.assertNotIn(directory,payload)
            self.assertNotIn('private_account',payload)
            self.assertEqual(json.loads(payload)['files'][0]['file'],'1_mainaxis/albedo/GPT-Image-1.5/openroomff_mainaxis/albedo_metrics.csv')

    def test_normal_backend_does_not_drop_query_or_references(self):
        from unittest.mock import patch
        from experiments.ablation.normal import normal_provider_utils as provider
        generate=Mock(side_effect=TypeError('unsupported image list'))
        runtime=SimpleNamespace(normal_model='custom',normal_client=SimpleNamespace(images=SimpleNamespace(generate=generate)))
        images=[Path('query.png'),Path('example_rgb.png'),Path('example_normal.png')]
        with patch.object(provider,'file_to_data_uri',side_effect=str):
            with self.assertRaises(RuntimeError):provider.run_doubao_generation(runtime,'prompt',images,'adaptive',False)
        self.assertEqual(generate.call_count,1)
        self.assertEqual(generate.call_args.kwargs['image'],[str(path) for path in images])
