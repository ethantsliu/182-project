"""Trainer logging uses the selected destination without touching a legacy logs path."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from _paths import REPO


class TestRunLog(unittest.TestCase):
    def test_selected_destination_works_with_unusable_legacy_logs(self):
        source = str(Path(REPO) / "track_1_short/run_log.py")
        script = """
import runpy, sys, types
sys.modules['torch'] = types.ModuleType('torch')
sys.modules['triton'] = types.ModuleType('triton')
module = runpy.run_path(sys.argv[1])
print0, flush = module['start_run_log'](True, 'test-run')
print0('source and environment sentinel')
flush()
"""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "logs").write_text("legacy path unavailable")
            output = Path(tmp, "session with spaces", "trainer")
            result = subprocess.run([sys.executable, "-c", script, source], cwd=tmp,
                                    env=dict(os.environ, TRAIN_LOG_DIR=str(output)),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"TRAIN_LOG_PATH={output / 'test-run.txt'}", result.stdout)
            self.assertEqual((output / "test-run.txt").read_text(), "source and environment sentinel\n")
            self.assertEqual(Path(tmp, "logs").read_text(), "legacy path unavailable")
