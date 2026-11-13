"""Profile retries must not mistake partial or stale analysis for a successful attempt."""
import subprocess

from test_gpu_session import SessionRepo


class TestProfileResume(SessionRepo):
    def test_partial_analysis_is_retried_and_completed_tree_is_reused(self):
        out = self.sd / "profile/root"
        out.mkdir(parents=True)
        (out / "analysis.txt").write_text("Traceback: previous analyzer failed\n")
        (out / "analysis.json").write_text("{}\n")
        # Stand in for the expensive GPU profiling run. Its marker is written only on success.
        run = self.repo / "182/profile/run.sh"
        run.write_text('#!/usr/bin/env bash\nmkdir -p "$2"\n'
                       'echo attempted >> "$2/attempts"\n'
                       'if [[ "${FAKE_PROFILE_FAIL:-0}" == 1 ]]; then\n'
                       '  echo "analysis failed" > "$2/analysis.txt"; exit 1\nfi\n'
                       'echo "steps profiled: 20" > "$2/analysis.txt"\n'
                       'echo "{}" > "$2/analysis.json"\n'
                       'echo complete > "$2/.profile_complete"\n')
        rc, text = self.run_session("profile", PROFILE_TREES="root", FAKE_PROFILE_FAIL="1")
        self.assertEqual(rc, 1, text)
        self.assertEqual((out / "attempts").read_text().splitlines(), ["attempted"])
        self.assertNotIn("done earlier", text)
        rc, text = self.run_session("profile", PROFILE_TREES="root")
        self.assertEqual(rc, 0, text)
        self.assertEqual((out / "attempts").read_text().splitlines(), ["attempted", "attempted"])
        # Retry a phase with a valid prior tree and no phase PASS marker.
        (self.sd / "status/profile").unlink()
        rc, text = self.run_session("profile", PROFILE_TREES="root")
        self.assertEqual(rc, 0, text)
        self.assertIn("profile root: done earlier", text)
        self.assertEqual(len((out / "attempts").read_text().splitlines()), 2)
        # A completion marker cannot compensate for a missing required output.
        (out / "analysis.json").unlink()
        (self.sd / "status/profile").unlink()
        rc, text = self.run_session("profile", PROFILE_TREES="root")
        self.assertEqual(rc, 0, text)
        self.assertEqual(len((out / "attempts").read_text().splitlines()), 3)

    def test_run_script_clears_stale_completion_and_marks_only_successful_analysis(self):
        tree = self.repo / "variants/fake"
        (tree / "182").mkdir(parents=True)
        (tree / "182/run_node.sh").write_text(
            '#!/usr/bin/env bash\ntouch "$PROFILE_DIR/.shim_used"\n'
            '[[ "${FAKE_PROFILE_TRACE:-1}" == 1 ]] && echo trace > "$PROFILE_DIR/trace_rank0.json.gz"\nexit 0\n')
        analyzer = self.repo / "182/profile/analyze.py"
        analyzer.write_text('import os, pathlib, sys\nprint("analysis output")\n'
                            'if os.environ.get("FAKE_ANALYSIS_FAIL") == "1": raise SystemExit(1)\n'
                            'pathlib.Path(sys.argv[-1]).write_text("{}\\n")\n')
        out = self.root / "profile-output"
        out.mkdir()
        (out / ".profile_complete").write_text("complete\n")
        (out / "trace_rank0.json.gz").write_text("old trace")
        script = self.repo / "182/profile/run.sh"

        def invoke(**extra):
            return subprocess.run(["bash", str(script), str(tree), str(out)],
                                  env=dict(self.env, **extra), capture_output=True, text=True, timeout=30)

        result = invoke(FAKE_ANALYSIS_FAIL="1")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertTrue((out / "analysis.txt").exists())
        self.assertFalse((out / ".profile_complete").exists())
        result = invoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((out / ".profile_complete").exists())
        result = invoke(FAKE_PROFILE_TRACE="0")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("no trace_rank0.json.gz", result.stdout)
        self.assertFalse((out / ".profile_complete").exists())


if __name__ == "__main__":
    import unittest
    unittest.main()
