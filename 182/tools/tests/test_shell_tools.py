"""Exercise setup/profiling entry points with a fake Linux/CUDA command surface, without GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from _paths import TOOLS


class TestShellTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.tools = self.root / "182/tools"
        self.tools.mkdir(parents=True)
        for name in ("node_setup.sh", "profile_nsys.sh"):
            shutil.copy(Path(TOOLS) / name, self.tools / name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"])
        self.env.pop("LD_LIBRARY_PATH", None)
        self.env["CUDA_HOME"] = str(self.root / "cuda")
        (self.root / "cuda/include").mkdir(parents=True)
        (self.root / "cuda/include/cuda_bf16.h").touch()
        self.command("nvidia-smi", '''
case "$*" in
  -L) for i in {0..7}; do echo "GPU $i: NVIDIA H100"; done;;
  *driver_version*) echo 580.1;;
  *persistence_mode*) echo Enabled;;
  *) echo 'Driver Version: 580.1 CUDA Version: 13.1';;
esac
''')
        self.command("ldconfig", "exit 0\n")
        self.command("nproc", "echo 64\n")
        self.command("python", 'if [[ "${1:-}" == - ]]; then cat >/dev/null; fi\nexit 0\n')
        self.command("torchrun", "exit 7\n")
        self.env["MOCK_BIN"] = str(self.bin)

    def tearDown(self):
        self.tmp.cleanup()

    def command(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)

    def run_script(self, name, *args):
        return subprocess.run(["bash", str(self.tools / name), *args], cwd=self.root, env=self.env,
                              text=True, capture_output=True, timeout=20)

    def mock_venv(self):
        venv = self.root / "venv/bin"
        venv.mkdir(parents=True)
        for name in ("python", "torchrun"):
            shutil.copy(self.bin / name, venv / name)

    def cuda_runtime(self):
        lib = self.root / "cuda/lib64"
        lib.mkdir()
        (lib / "libcudart.so.13").touch()
        self.env["LD_LIBRARY_PATH"] = str(lib)

    def test_missing_cuda_runtime_reports_failure_with_unset_library_path(self):
        self.mock_venv()
        result = self.run_script("node_setup.sh", "--venv", "venv", "--skip-fa3", "--skip-data", "--skip-check")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("unbound variable", result.stderr)
        self.assertIn("FAIL: no libcudart.so.13", result.stdout)

    def test_venv_gpu_check_failure_is_fatal(self):
        self.mock_venv()
        self.cuda_runtime()
        result = self.run_script("node_setup.sh", "--venv", "venv", "--skip-fa3", "--skip-data")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL: GPU / NCCL sanity", result.stdout)

    def test_uv_environment_seeds_pip(self):
        self.cuda_runtime()
        self.command("uv", '''
echo "$*" > uv_args.txt
for target in "$@"; do :; done
mkdir -p "$target/bin"
cp "$MOCK_BIN/python" "$target/bin/python"
cp "$MOCK_BIN/torchrun" "$target/bin/torchrun"
''')
        result = self.run_script("node_setup.sh", "--venv", "venv", "--skip-fa3", "--skip-data", "--skip-check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--seed", (self.root / "uv_args.txt").read_text())

    def test_docker_gpu_check_failure_is_fatal(self):
        self.command("docker", '''
if [[ "$*" == *torchrun* ]]; then exit 7; fi
if [[ "$*" == *"python -"* ]]; then cat >/dev/null; fi
exit 0
''')
        result = self.run_script("node_setup.sh", "--mode", "docker", "--skip-fa3", "--skip-data")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL: GPU / NCCL sanity (docker)", result.stdout)

    def test_invalid_setup_mode_does_not_succeed(self):
        result = self.run_script("node_setup.sh", "--mode", "typo")
        self.assertEqual(result.returncode, 2)

    def test_profile_preserves_quoted_regex_and_path_arguments(self):
        # The analyzer runs for real through this tiny stand-in, to inspect its exact argv.
        self.command("python", f'exec "{sys.executable}" "$@"\n')
        self.command("nsys", 'exit 0\n')
        self.command("torchrun", 'touch "$_NSYS_OUT/rank${_NSYS_RANK}.nsys-rep"\n')
        (self.tools / "analyze_nsys.py").write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
        result = self.run_script("profile_nsys.sh", "--out", "trace", "--analyze-args",
                                 '--step-marker "^gemm start$" --gemm-shapes "my shapes.json"')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads((self.root / "trace/analysis.md").read_text())
        self.assertEqual(args[-4:], ["--step-marker", "^gemm start$", "--gemm-shapes", "my shapes.json"])

    def test_profile_rejects_unreachable_rank_and_unknown_mode(self):
        for args in [("--rank", "8", "--nproc", "8"), ("--mode", "typo")]:
            result = self.run_script("profile_nsys.sh", *args)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
