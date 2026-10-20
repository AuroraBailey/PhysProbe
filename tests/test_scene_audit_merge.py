import json
from pathlib import Path
import tempfile
import unittest

from annotation.scripts.merge_scene_audits import (
    MODEL_DIRS, SeededRandom, merge_directories, merge_records,
    resolve_confidence, resolve_vote,
)


class SceneAuditMergeTests(unittest.TestCase):
    def test_votes_and_confidence(self):
        self.assertEqual(resolve_vote(['good', 'bad', 'good']), 'good')
        self.assertEqual(resolve_vote(['Good', 'good', 'bad']), 'Good')
        self.assertEqual(resolve_vote(['a', 'b', 'c']), ['a', 'b', 'c'])
        self.assertIsNone(resolve_vote([None, None, 'good']))
        self.assertEqual(resolve_vote([False, 0, False]), False)
        self.assertEqual(resolve_confidence(['0.8', 0.8, 0.2]), 0.8)
        self.assertAlmostEqual(resolve_confidence([0.1, '0.3', 0.8]), 0.4)
        self.assertIsNone(resolve_confidence([None, 'invalid', []]))
        self.assertIsNone(resolve_confidence([float('nan'), float('inf'), None]))

    def test_dotnet_seeded_sequence(self):
        rng = SeededRandom(42)
        self.assertEqual([rng.next(2147483647) for _ in range(3)],
                         [1434747710, 302596119, 269548474])

    def test_record_shape_and_evidence(self):
        records = [dict(scene='scene1', images=['a_im.png'], raw=f'raw-{i}',
                        parsed=dict(scene_category='bedroom', layout_plausibility=label,
                                    confidence=confidence, issues=issues, reason_short=str(i),
                                    lighting_challenge_type=f'type-{i}'), computed={'model': i},
                        parse_error=None)
                   for i, (label, confidence, issues) in enumerate([
                       ('good', 0.1, ['issue-a']), ('good', 0.3, []), ('bad', 0.8, 'issue-b')])]
        records[1]['images'] = ['a_im.png', 'b_im.png']
        merged = merge_records(records, SeededRandom(42))
        self.assertEqual(merged['layout_plausibility'], 'good')
        self.assertEqual(merged['images'], records[1]['images'])
        self.assertEqual(merged['computed'], {'model': 1})
        self.assertEqual(merged['raw'], ['raw-0', 'raw-1', 'raw-2'])
        self.assertEqual(merged['issues'], [['issue-a'], [], ['issue-b']])
        self.assertEqual(merged['lighting_challenge_type'], ['type-0', 'type-1', 'type-2'])
        self.assertEqual(merged['parse_error'], [None, None, None])
        self.assertNotIn('parsed', merged)  # Reviewer expects top-level fields.
        records[2]['parsed'] = None
        records[2]['parse_error'] = 'json_not_found'
        merged = merge_records(records, SeededRandom(42))
        self.assertEqual(merged['parse_error'][-1], 'json_not_found')
        self.assertEqual(merged['layout_plausibility'], 'good')

    def test_recursive_directories_missing_file_and_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, directory in enumerate(MODEL_DIRS.values()):
                path = root / directory / 'group' / 'scene.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'scene': 'scene', 'images': ['a'],
                                'parsed': {'confidence': 0.5}, 'computed': {'model': i}}), encoding='utf-8-sig')
            (root / MODEL_DIRS['doubao'] / 'missing.json').write_text('{}')
            first = merge_directories(root, Path('merged'))
            expected = (root / 'merged/group/scene.json').read_text()
            second = merge_directories(root, Path('merged'))
            self.assertEqual(first, second)
            self.assertEqual((first['merged_files'], first['skipped_files']), (1, 1))
            self.assertEqual(first['skipped'][0]['missing_models'], ['glm', 'qwen'])
            self.assertEqual((root / 'merged/group/scene.json').read_text(), expected)
            with self.assertRaises(ValueError):
                merge_directories(root, Path(MODEL_DIRS['doubao']))
            (root / MODEL_DIRS['glm'] / 'group/scene.json').write_text('{broken')
            with self.assertRaisesRegex(ValueError, 'cannot read JSON'):
                merge_directories(root, Path('merged'))


if __name__ == '__main__':
    unittest.main()
