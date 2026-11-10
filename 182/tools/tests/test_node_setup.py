"""182/tools/node_setup.sh's venv repair, CUDA 13 runtime path and stack check (fake Linux/CUDA commands, no GPU)."""
from pathlib import Path
import unittest

from _paths import TOOLS
import test_shell_tools


class TestNodeSetup(test_shell_tools.TestShellTools):

    def test_failed_venv_creation_is_removed_and_explained(self):
        """Fresh Ubuntu 24.04: `python3 -m venv` fails (no python3-venv) after writing pyvenv.cfg and bin/python."""
        self.cuda_runtime()
        self.command("uv", "exit 1\n")   # no uv on that box either
        self.command("python3.12", r'''
if [[ "${1:-}" == -m && "${2:-}" == venv ]]; then
  mkdir -p "$3/bin"; touch "$3/pyvenv.cfg"; printf '#!/bin/bash\nexit 1\n' > "$3/bin/python"; chmod +x "$3/bin/python"
  echo "ensurepip is not available" >&2; exit 1
fi
if [[ "${1:-}" == -c ]]; then echo 3.12; fi
exit 0
''')
        result = self.run_script("node_setup.sh", "--venv", "venv", "--skip-fa3", "--skip-data", "--skip-check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sudo apt install python3.12-venv", result.stdout)
        self.assertFalse((self.root / "venv").exists(), "the partial venv was left behind")
        # a leftover broken venv from an earlier run is recreated, not trusted
        (self.root / "venv/bin").mkdir(parents=True)
        (self.root / "venv/pyvenv.cfg").touch()
        (self.root / "venv/bin/python").write_text("#!/bin/bash\nexit 1\n")
        (self.root / "venv/bin/python").chmod(0o755)
        result = self.run_script("node_setup.sh", "--venv", "venv", "--skip-fa3", "--skip-data", "--skip-check")
        self.assertIn("has no working pip", result.stdout)
        self.assertIn("sudo apt install python3.12-venv", result.stdout)

    def test_cudart13_dir_is_written_into_activate(self):
        lib = self.root / "cuda/lib64"
        lib.mkdir()
        (lib / "libcudart.so.13").touch()
        self.mock_venv()
        (self.root / "venv/bin/activate").write_text("# venv activate\n")
        for _ in range(2):  # idempotent
            result = self.run_script("node_setup.sh", "--venv", "venv", "--skip-fa3", "--skip-data", "--skip-check")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        act = (self.root / "venv/bin/activate").read_text()
        self.assertEqual(act.count(f'export LD_LIBRARY_PATH="{lib}'), 1, act)
        self.assertIn("added for this run", result.stdout)

    def test_stack_check_requires_triton_3_6(self):
        src = (Path(TOOLS) / "node_setup.sh").read_text()
        self.assertIn('triton.__version__ != "3.6.0"', src)



for _name in [n for n in vars(test_shell_tools.TestShellTools) if n.startswith("test_")]:
    if _name not in vars(TestNodeSetup):
        setattr(TestNodeSetup, _name, None)

if __name__ == "__main__":
    unittest.main()
