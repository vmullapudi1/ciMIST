"""Palette resources must load under PyMOL's run command and regular imports."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'pymol_palettes/pymol_palettes.py'


class PymolPalettes(unittest.TestCase):
    def load_palette(self, namespace):
        pymol = ModuleType('pymol')
        pymol.cmd = SimpleNamespace(extend=lambda function: None)
        namespace['__name__'] = 'palette_test'
        with patch.dict(sys.modules, {'pymol': pymol}):
            exec(compile(SCRIPT.read_text(), str(SCRIPT), 'exec'), namespace)
        self.assertEqual(Path(namespace['SPECTRA_DIR']), SCRIPT.parent)
        self.assertIn('algae', namespace['NEW_PALETTES'])
        self.assertIn('turbid', namespace['NEW_PALETTES'])

    def test_pymol_run_uses_script_path(self):
        self.load_palette({
            '__file__': '/missing/pymol/__init__.py',
            '__script__': str(SCRIPT),
        })

    def test_regular_import_uses_file_path(self):
        self.load_palette({'__file__': str(SCRIPT)})


if __name__ == '__main__':
    unittest.main()
