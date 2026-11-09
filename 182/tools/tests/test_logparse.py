import glob
import os
import statistics
import subprocess
import tempfile
import unittest

from _paths import ANVIL2, RECORDS, REPO
import logparse


class TestRealLogs(unittest.TestCase):
    def test_open_pr_full_logs(self):
        # Read full, source-embedded logs directly from fetched upstream objects, without a checkout.
        pools = [(397, "track_1_short/2026-10-31_ExactSpeedups/this_pr", 12, 30.6745833333, 3.2774333333),
                 (381, "track_1_short/2026-10-03_ExactCountChain/this_pr", 16, 9.65275, 3.27588125),
                 (261, "track_2_medium/2026-0427-MuddFormer/this_pr", 6, 860.4783333333, 2.9170666667)]
        for pr, folder, n, time_mean, loss_mean in pools:
            ref = f"refs/remotes/upstream/pr/{pr}"
            if subprocess.run(["git", "-C", REPO, "rev-parse", "--verify", ref], capture_output=True).returncode:
                self.skipTest(f"fetch upstream pull/{pr}/head:{ref} for full-log regression")
            paths = subprocess.check_output(["git", "-C", REPO, "ls-tree", "-r", "--name-only", ref,
                                             "records/" + folder], text=True).splitlines()
            runs = []
            for path in paths:
                if path.endswith(".txt"):
                    text = subprocess.check_output(["git", "-C", REPO, "show", ref + ":" + path], text=True)
                    parsed = logparse.parse_runs(text, path)
                    self.assertEqual(len(parsed), 1, path)
                    self.assertTrue(parsed[0].completed, path)
                    runs.extend(parsed)
            self.assertEqual(len(runs), n)
            self.assertAlmostEqual(statistics.mean(r.train_time_s for r in runs), time_mean, places=8)
            self.assertAlmostEqual(statistics.mean(r.val_loss for r in runs), loss_mean, places=8)

    def test_track1_anvil2(self):
        r = logparse.parse_file(os.path.join(ANVIL2, "this_pr", "01b2f094-4536-479c-b0dd-cb14eba4844c.txt"))
        self.assertEqual(r.fmt, "ms")
        self.assertEqual((r.final_step, r.total_steps), (1194, 1194))
        self.assertTrue(r.completed)
        self.assertAlmostEqual(r.val_loss, 3.2758)
        self.assertAlmostEqual(r.train_time_s, 40.020)
        self.assertAlmostEqual(r.step_avg_ms, 33.52)
        self.assertEqual(r.peak_mem_mib, 53424)
        self.assertEqual(r.torch_version, "2.10.0+cu128")
        self.assertEqual(r.last_train_step, 1194)
        self.assertEqual(len(r.val_curve), 1)  # val_loss_every = 0: only the final validation

    def test_track1_every_anvil2_log_parses(self):
        paths = logparse.expand([os.path.join(ANVIL2, "this_pr"), os.path.join(ANVIL2, "baseline")])
        self.assertEqual(len(paths), 17 + 9)  # README.md / statistics.md skipped
        for p in paths:
            r = logparse.parse_file(p)
            self.assertTrue(r.completed, p)
            self.assertTrue(3.27 < r.val_loss < 3.29, p)

    def test_track2_medium(self):
        p = os.path.join(RECORDS, "track_2_medium", "2025-12-31_BulkSmallTrackTransfer", "354be270-7d41-44b7-8064-f040923f024f.txt")
        r = logparse.parse_file(p)
        self.assertEqual(r.fmt, "ms")
        self.assertEqual((r.final_step, r.total_steps), (4740, 4740))
        self.assertAlmostEqual(r.val_loss, 2.9165)
        self.assertAlmostEqual(r.train_time_s, 1041.071)

    def test_track3_seconds_format_and_multi_run_file(self):
        # This file is result #12's whole n=20 pool, concatenated: 20 runs, readout at step 3325.
        p = os.path.join(RECORDS, "track_3_optimization", "results", "1bd8db7a-f3a3-4195-856d-cab7e0816443.txt")
        runs = logparse.parse_file_runs(p)
        self.assertEqual(len(runs), 20)
        r = runs[0]
        self.assertEqual((r.fmt, r.final_step, r.total_steps, r.run_index, r.runs_in_file), ("s", 3350, 3350, 0, 20))
        self.assertAlmostEqual(r.val_loss, 3.27874)
        self.assertAlmostEqual(r.train_time_s, 496.855)
        self.assertAlmostEqual(r.val_at(3325), 3.27931)
        self.assertEqual(r.first_step_below(3.28), 3325)
        self.assertIsNone(r.val_at(3326))
        self.assertAlmostEqual(runs[-1].val_loss, 3.27839)
        mean = sum(x.val_at(3325) for x in runs) / 20
        self.assertEqual(round(mean, 4), 3.2790)  # README result #12: "3.2790 (n=20)" at 3325 steps

    def test_track3_extra_fields(self):
        p = os.path.join(RECORDS, "track_3_optimization", "results", "20260619_cwd_rowfloor_tailema",
                         "A40_seed1_41f040b9-dab7-4b02-91ba-60edcb6f62a7.txt")
        r = logparse.parse_file(p)
        self.assertAlmostEqual(r.val_loss, 3.27096)
        self.assertEqual(r.extras, {"val_avg_loss": 3.27096, "val_ema_loss": 3.27138})
        self.assertAlmostEqual(r.train_time_s, 10675.775)

    def test_every_record_log_with_a_val_line_parses(self):
        # The survey behind the regex: every run log under records/ that has a val line yields one.
        n = 0
        for p in glob.glob(os.path.join(RECORDS, "**", "*.txt"), recursive=True):
            with open(p, errors="replace") as f:
                text = f.read()
            if "val_loss:" not in text or "\nstep:" not in text:
                continue
            r = logparse.parse_text(text, p)
            if any(l.startswith("step:") and " val_loss:" in l for l in text.splitlines()):
                self.assertIsNotNone(r.val_loss, p)
                n += 1
        self.assertGreater(n, 500)


class TestSynthetic(unittest.TestCase):
    SRC = 'print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms")\n'
    SEP = "=" * 100

    def test_overlapping_inputs_and_symlinks_do_not_inflate_n(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.txt")
            with open(path, "w") as f:
                f.write("step:10/10 val_loss:3.278 train_time:1000ms\n")
            alias = os.path.join(d, "alias.txt")
            os.symlink(path, alias)
            self.assertEqual(logparse.expand([path, d, os.path.join(d, "*.txt")]), [path])

    def test_source_dump_ignored_and_nan(self):
        text = (self.SRC + "step:5/10 val_loss:9.9 train_time:1ms step_avg:1ms\n" + self.SEP + "\n"
                "step:5/10 train_time:500ms step_avg:100.00ms\nstep:10/10 val_loss:nan train_time:1000ms step_avg:100.00ms\n")
        r = logparse.parse_text(text)
        self.assertTrue(r.is_nan)
        self.assertEqual(len(r.val_curve), 1)  # the fake line above the separator is ignored

    def test_crashed_run(self):
        text = self.SEP + "\nstep:25/100 train_time:500ms step_avg:20.00ms\nstep:50/100 train_time:1000ms step_avg:20.00ms\n"
        r = logparse.parse_text(text)
        self.assertIsNone(r.val_loss)
        self.assertFalse(r.completed)
        self.assertEqual((r.last_train_step, r.total_steps), (50, 100))

    def test_find_log_path(self):
        self.assertEqual(logparse.find_log_path("logs/abc-123.txt\nstep:1/2 ...\n"), "logs/abc-123.txt")
        self.assertIsNone(logparse.find_log_path("no path here\n"))


if __name__ == "__main__":
    unittest.main()
