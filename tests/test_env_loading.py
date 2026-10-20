from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common.utils.env import load_project_env


class EnvironmentLoadingTests(unittest.TestCase):
    def test_precedence_aliases_and_literal_values(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'ARK_API_KEY': 'from-shell'}, clear=True):
            path = Path(tmp) / '.env'
            path.write_text("ARK_API_KEY=from-file\nDOUBAO_BASE_URL=https://example.invalid/v3\n"
                            "PHYS_TEST_SECRET='literal$HOME`value`'\nGLM_API_KEY=\n"
                            "AZURE_OPENAI_API_KEY=shared\nAZURE_OPENAI_ENDPOINT=https://azure.invalid/\n")
            load_project_env(path)
            self.assertEqual(os.environ['ARK_API_KEY'], 'from-shell')
            self.assertEqual(os.environ['ARK_BASE_URL'], 'https://example.invalid/v3')
            self.assertEqual(os.environ['PHYS_TEST_SECRET'], 'literal$HOME`value`')
            self.assertNotIn('GLM_API_KEY', os.environ)
            self.assertEqual(os.environ['AZURE_OPENAI_API_KEY'], 'shared')
            self.assertEqual(os.environ['AZURE_OPENAI_ENDPOINT'], 'https://azure.invalid/')

    def test_shell_export_roundtrip(self):
        # Use a temporary synthetic repo so the developer's local .env is never read.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / 'src/common/utils/env.py'
            script.parent.mkdir(parents=True)
            script.write_text((ROOT / 'src/common/utils/env.py').read_text())
            (root / '.env').write_text("PHYS_TEST_SECRET='a b;$(false)`false`$HOME'\n")
            exports = subprocess.check_output([sys.executable, str(script), '--shell'], text=True)
            result = subprocess.check_output(['bash', '-c', exports + '\nprintf "%s" "$PHYS_TEST_SECRET"'], text=True)
            self.assertEqual(result, 'a b;$(false)`false`$HOME')


if __name__ == '__main__':
    unittest.main()
