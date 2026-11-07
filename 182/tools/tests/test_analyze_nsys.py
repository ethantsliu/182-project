import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

from _paths import FIXTURES
import analyze_nsys as an

sys.path.insert(0, FIXTURES)
import make_nsys_fixture  # noqa: E402

COMMITTED = os.path.join(FIXTURES, "synthetic_nsys.sqlite")
RULES = [{"match": "xmma_gemm", "mnk": [1000, 1000, 1000]}]


class TestIntervals(unittest.TestCase):
    def test_merge_and_overlap(self):
        m = an.merge([(5, 10), (0, 3), (2, 4), (10, 12), (20, 20)])
        self.assertEqual(m, [[0, 4], [5, 12]])
        self.assertEqual(an.length(m), 11)
        self.assertEqual(an.intersect_len(m, an.merge([(3, 6), (11, 30)])), 1 + 1 + 1)
        self.assertEqual(an.gaps(m, 0, 15), [(4, 5), (12, 15)])
        self.assertEqual(an.clip(m, 2, 6), [[2, 4], [5, 6]])

    def test_flops_from_sizes(self):
        self.assertEqual(an.parse_aten_nvtx("aten::mm, seq = 1, sizes = [[2, 3], [3, 4]], input_op_ids = []"), 2 * 2 * 3 * 4)
        self.assertEqual(an.parse_aten_nvtx("aten::addmm, sizes = [[4], [2, 3], [3, 4]]"), 48)
        self.assertEqual(an.parse_aten_nvtx("aten::bmm, sizes = [[5, 2, 3], [5, 3, 4]]"), 5 * 48)
        self.assertEqual(an.parse_aten_nvtx("aten::linear, sizes = [[8, 16, 32], [64, 32], [64]]"), 2 * 128 * 32 * 64)
        self.assertIsNone(an.parse_aten_nvtx("aten::add, sizes = [[2], [2]]"))


class TestFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.fresh = os.path.join(cls.tmp, "fresh.sqlite")
        make_nsys_fixture.build(cls.fresh)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def check_steps(self, res, n_steps):
        self.assertEqual(res["device"], 0)
        self.assertEqual(len(res["steps"]), n_steps)
        for s in res["steps"]:
            self.assertAlmostEqual(s["window_us"], 400.0)
            self.assertEqual(s["kernels"], 5)
            self.assertEqual(s["graph_kernels"], 1)
            self.assertAlmostEqual(s["busy_us"], 290.0)
            self.assertAlmostEqual(s["idle_us"], 110.0)
            self.assertEqual(s["gaps_over_threshold"], 3)
            self.assertAlmostEqual(s["largest_gap_us"], 70.0)
            self.assertAlmostEqual(s["nccl_us"], 100.0)
            self.assertAlmostEqual(s["exposed_nccl_us"], 90.0)
            self.assertAlmostEqual(s["gemm_us"], 150.0)  # flash-attn (cutlass in its name) excluded
            self.assertEqual(s["memcpy"], 1)
            self.assertEqual(s["memcpy_bytes"], 1 << 20)

    def test_nvtx_steps(self):
        for path in (self.fresh, COMMITTED):
            res = an.analyze(path, gemm_rules=RULES, peak_tflops=989)
            self.assertEqual(res["step_mode"], "nvtx")
            self.assertEqual([s["label"] for s in res["steps"]], ["step 300", "step 301", "step 302"])
            self.check_steps(res, 3)
            gt = {r["name"][:5]: r for r in res["gemm_table"]}
            self.assertAlmostEqual(gt["nvjet"]["tflops"], 2 * 2048 * 1024 * 4096 / 50e-6 / 1e12, places=6)
            self.assertEqual(gt["nvjet"]["source"], "nvtx")
            self.assertAlmostEqual(gt["sm90_"]["tflops"], 20.0)
            self.assertEqual(gt["sm90_"]["source"], "rules")
            self.assertAlmostEqual(gt["sm90_"]["pct_peak"], 100 * 20 / 989)
            g = res["gemm"]
            self.assertEqual((g["gemm_kernels"], g["with_shapes"]), (6, 6))
            self.assertAlmostEqual(g["achieved_tflops"], (3 * 2e9 + 3 * 2 * 2048 * 1024 * 4096) / 450e-6 / 1e12, places=6)
            top = res["top_kernels"]
            self.assertEqual(top[0]["count"], 3)
            self.assertAlmostEqual(sum(r["pct_time"] for r in top), 100.0)
            self.assertAlmostEqual(sum(r["total_us"] for r in top), 3 * 300.0)  # > busy: NCCL overlaps compute
            self.assertAlmostEqual(res["top_gaps"][0]["us"], 70.0)
            self.assertEqual(res["top_gaps"][0]["after"], "FlashAttnFwdSm90")

    def test_no_rules_only_nvtx_shapes(self):
        res = an.analyze(self.fresh)
        self.assertEqual(res["gemm"]["with_shapes"], 3)
        self.assertEqual(len(res["gemm_table"]), 1)

    def test_marker_steps(self):
        res = an.analyze(self.fresh, step_marker="xmma_gemm")
        self.assertEqual(res["step_mode"], "marker")
        self.check_steps(res, 4)

    def test_whole_trace_and_device_choice(self):
        res = an.analyze(self.fresh, step_regex="^nomatch$")
        self.assertEqual(res["step_mode"], "whole")
        self.assertEqual(res["steps"][0]["kernels"], 25)
        res1 = an.analyze(self.fresh, device=1, step_regex="^nomatch$")
        self.assertEqual(res1["steps"][0]["kernels"], 2)

    def test_old_schema_without_optional_columns(self):
        # older exports: no graphNodeId, no textId; NVTX text inline
        p = os.path.join(self.tmp, "old.sqlite")
        shutil.copy(self.fresh, p)
        con = sqlite3.connect(p)
        con.execute("UPDATE NVTX_EVENTS SET text = (SELECT value FROM StringIds WHERE id = textId) WHERE textId IS NOT NULL")
        con.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL DROP COLUMN graphNodeId")
        con.execute("ALTER TABLE NVTX_EVENTS DROP COLUMN textId")
        con.commit()
        con.close()
        res = an.analyze(p)
        self.assertEqual(len(res["steps"]), 3)
        self.assertEqual(res["steps"][0]["graph_kernels"], 0)
        self.assertAlmostEqual(res["steps"][0]["exposed_nccl_us"], 90.0)

    def test_cli_and_render(self):
        import contextlib, io, json
        js = os.path.join(self.tmp, "a.json")
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            an.main([self.fresh, "--json", js, "--gemm-shapes", os.path.join(FIXTURES, "gemm_shapes_example.json")])
        self.assertIn("| step 300 | 0.400 | 5 (1) | 0.290 | 0.110 |", buf.getvalue())
        with open(js) as f:
            self.assertEqual(json.load(f)["gemm"]["with_shapes"], 6)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            an.main([self.fresh, "--list-gemms"])
        self.assertEqual(len(buf.getvalue().strip().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
