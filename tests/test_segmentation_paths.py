import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/segmentation/generate_roughness_segmentation.py'
spec = importlib.util.spec_from_file_location('roughness_segmentation_runner', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SegmentationPathTests(unittest.TestCase):
    def write_manifest(self, path, rows):
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['datasource', 'rgb_path', 'segmentation_path'])
            writer.writeheader()
            writer.writerows(rows)

    def test_original_rgb_to_processed_image_and_segmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'manifest.csv'
            self.write_manifest(manifest, [{'datasource': 'openroomff',
                'rgb_path': 'openroomff_stresstest/main_xml_scene0010_00/12_im_5.rgbe',
                'segmentation_path': 'openroomff_stresstest/main_xml_scene0010_00/12_seg.png'}])
            jobs = runner.plan_jobs(manifest, root / 'inputs', root / 'outputs')
            self.assertEqual(jobs, [(
                root / 'inputs/openroomff_stresstest/main_xml_scene0010_00/12_im.png',
                root / 'outputs/openroomff_stresstest/main_xml_scene0010_00/12_seg.png')])

    def test_duplicate_destinations_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {'datasource': 'openroomff', 'rgb_path': '1_im_5.rgbe', 'segmentation_path': '1_seg.png'}
            self.write_manifest(root / 'manifest.csv', [row, row])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                runner.plan_jobs(root / 'manifest.csv', root, root)

    def test_method_loading_uses_functional_module_names(self):
        for method, inference in [('standard', 'scene_inference'),
                                  ('contextual', 'scene_inference_contextual')]:
            with self.subTest(method=method):
                builder = Mock(return_value='model')
                base = SimpleNamespace(build_sam3_image_model=builder, merge_scene=Mock())
                merging = SimpleNamespace(merge_scene=Mock())
                modules = {inference: base, 'region_merging': merging}
                with patch.object(runner, 'load_module', side_effect=modules.__getitem__) as load:
                    loaded, merge = runner.load_pipeline(method, Path('/weights/sam3.pt'))
                    self.assertIs(loaded, base)
                    self.assertIs(merge, base.merge_scene if method == 'standard' else merging.merge_scene)
                    base.build_sam3_image_model(device='cpu')
                    builder.assert_called_once_with(device='cpu', checkpoint_path='/weights/sam3.pt', load_from_HF=False)
                    self.assertEqual([c.args[0] for c in load.call_args_list],
                                     [inference] if method == 'standard' else [inference, 'region_merging'])

    def test_paths_cannot_escape_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                runner.below(Path(tmp), '../outside.png')
            with self.assertRaises(ValueError):
                runner.below(Path(tmp), '/outside.png')


if __name__ == '__main__':
    unittest.main()
