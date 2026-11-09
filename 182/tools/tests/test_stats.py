import os
import random
import unittest

from _paths import ANVIL2, RECORDS
import stats

try:
    import scipy.stats as ss
except ImportError:  # the tools themselves never need scipy
    ss = None


class TestDistributions(unittest.TestCase):
    @unittest.skipIf(ss is None, "scipy not installed")
    def test_t_sf_matches_scipy(self):
        for df in [1, 2, 3, 5, 8, 17, 30, 100, 14.87]:
            for t in [-6, -2.5, -1, -0.1, 0, 0.3, 1, 2.0, 3.3, 8, 11.5]:
                self.assertAlmostEqual(stats.t_sf(t, df), ss.t.sf(t, df), delta=1e-12 + 1e-9 * ss.t.sf(t, df))

    @unittest.skipIf(ss is None, "scipy not installed")
    def test_ppf_matches_scipy(self):
        for df in [1, 4, 17, 60]:
            for p in [0.01, 0.2, 0.5, 0.8, 0.99]:
                self.assertAlmostEqual(stats.t_ppf(p, df), ss.t.ppf(p, df), places=8)
        self.assertAlmostEqual(stats.norm_ppf(0.999), ss.norm.ppf(0.999), places=8)

    def test_known_values(self):
        self.assertAlmostEqual(stats.t_sf(0.0, 7), 0.5)
        self.assertAlmostEqual(stats.t_sf(2.0, 1), 0.1475836176504333, places=12)  # Cauchy: 1/2 - atan(2)/pi
        self.assertAlmostEqual(stats.norm_ppf(0.999), 3.090232306167813, places=9)
        self.assertEqual(stats.t_sf(float("inf"), 7), 0)
        self.assertEqual(stats.t_sf(-float("inf"), 7), 1)


class TestAnvil2Record(unittest.TestCase):
    """records/track_1_short/2026-08-30_ANVIL2/{this_pr,baseline}/statistics.md, from the shipped logs."""

    def test_gate_reproduces_published_pool(self):
        # 17 logs ship; the 18th leg's row (39.969 s @ 3.2775) is in the README.
        vals, notes = stats._collect([os.path.join(ANVIL2, "this_pr")], [3.2775], "val", None, False)
        self.assertEqual(len(vals), 18)
        g = stats.gate_test(vals)
        self.assertAlmostEqual(g["mean"], 3.2773111111, places=9)
        self.assertAlmostEqual(g["sd"], 0.0009904775, places=9)
        self.assertEqual(round(g["t"], 1), 11.5)
        self.assertAlmostEqual(g["p"], 9.40e-10, delta=0.01e-10)
        self.assertEqual(g["df"], 17)
        self.assertTrue(g["passes"])

    def test_baseline_pool(self):
        t, _ = stats._collect([os.path.join(ANVIL2, "baseline")], None, "time", None, False)
        v, _ = stats._collect([os.path.join(ANVIL2, "baseline")], None, "val", None, False)
        m, sd = stats.mean_sd(t)
        self.assertAlmostEqual(m, 73.889, places=3)
        self.assertAlmostEqual(sd, 0.137, places=3)
        mv, sv = stats.mean_sd(v)
        self.assertAlmostEqual(mv, 3.2782777778, places=9)
        self.assertAlmostEqual(sv, 0.0020462024, places=9)
        self.assertFalse(stats.gate_test(v)["passes"])  # record #89's 9-run pool alone: p = 0.018

    def test_welch_time(self):
        a, _ = stats._collect([os.path.join(ANVIL2, "this_pr")], [39.969], "time", None, False)
        b, _ = stats._collect([os.path.join(ANVIL2, "baseline")], None, "time", None, False)
        w = stats.welch_test(a, b, "less")
        self.assertAlmostEqual(w["mean_a"], 39.914, places=3)
        self.assertAlmostEqual(-w["delta"], 33.975, places=3)  # README: "win 33.975 s"
        self.assertTrue(w["passes"])
        if ss is not None:
            ref = ss.ttest_ind(a, b, equal_var=False, alternative="less")
            self.assertAlmostEqual(w["t"], ref.statistic, places=9)
            self.assertAlmostEqual(w["p"] / ref.pvalue, 1.0, places=6)


class TestWelchScipy(unittest.TestCase):
    @unittest.skipIf(ss is None, "scipy not installed")
    def test_against_scipy(self):
        rng = random.Random(0)
        for _ in range(20):
            a = [rng.gauss(40, 0.1) for _ in range(rng.randrange(3, 12))]
            b = [rng.gauss(40.05, 0.2) for _ in range(rng.randrange(3, 12))]
            for alt in ["less", "greater", "two-sided"]:
                w = stats.welch_test(a, b, alt)
                ref = ss.ttest_ind(a, b, equal_var=False, alternative=alt)
                self.assertAlmostEqual(w["t"], ref.statistic, places=9)
                self.assertAlmostEqual(w["p"], ref.pvalue, places=9)
                self.assertAlmostEqual(w["df"], ref.df, places=9)
            g = stats.gate_test(a, 40.0)
            ref = ss.ttest_1samp(a, 40.0, alternative="less")
            self.assertAlmostEqual(g["p"], ref.pvalue, places=9)
            self.assertEqual(g["df"], len(a) - 1)
            self.assertAlmostEqual(g["t"], -ref.statistic, places=9)

    @unittest.skipIf(ss is None, "scipy not installed")
    def test_track2_gate_random_losses(self):
        rng = random.Random(182)
        for n in [2, 3, 8, 30, 100]:
            vals = [rng.gauss(2.919, 0.002) for _ in range(n)]
            got = stats.gate_test(vals, target=2.92)
            ref = ss.ttest_1samp(vals, 2.92, alternative="less")
            self.assertAlmostEqual(got["p"], ref.pvalue, places=10)
            self.assertEqual(got["passes"], ref.pvalue < 0.01)


class TestTrack3Rules(unittest.TestCase):
    """Numbers from records/track_3_optimization/README.md "Rules"."""

    def test_z3_examples(self):
        self.assertTrue(stats.z3_rule([3.2759])["passes"])       # "a single run below 3.276 suffices"
        self.assertFalse(stats.z3_rule([3.2761])["passes"])
        self.assertTrue(stats.z3_rule([3.2779] * 4)["passes"])   # "for n=4 runs, any average below 3.278"
        self.assertFalse(stats.z3_rule([3.2781] * 4)["passes"])
        self.assertAlmostEqual(stats.z3_rule([3.278] * 4)["max_mean_to_pass"], 3.278)
        self.assertFalse(stats.z3_rule([-float("inf")])["passes"])
        self.assertFalse(stats.z3_rule([float("nan")])["passes"])

    def test_z3_on_result_20_logs(self):
        # Result #20 "3030 steps, 3.2790 (n=30)": its logs run to 3040, readout at the fixed step 3030.
        d = os.path.join(RECORDS, "track_3_optimization", "results", "20260509_contra_soft_muon")
        vals, _ = stats._collect([d], None, "val", 3030, True)
        self.assertEqual(len(vals), 30)
        z = stats.z3_rule(vals)
        self.assertEqual(round(z["mean"], 4), 3.2790)
        self.assertTrue(z["passes"])

    def test_pairwise_examples(self):
        # #13 (3210, 3.2785, n=10) vs #11 (3225, 3.2785, n=16): "the LHS is 0.00167"
        r = stats.pair3_rule(3.2785, 10, 3210, 3.2785, 16, 3225)
        self.assertAlmostEqual(r["lhs"], 0.00167, places=5)
        self.assertFalse(r["passes"])
        # #13 vs #8 (3250, 3.2778, n=10): "= 0.0027"
        r = stats.pair3_rule(3.2785, 16, 3210, 3.2778, 10, 3250)
        self.assertAlmostEqual(r["lhs"], 0.0027, places=4)
        # #12 vs #6: final_loss_diff -0.0002, 50 steps, n = 20 each: "0.00648", passes
        r = stats.pair3_rule(3.2790, 20, 3325, 3.2788, 20, 3375)
        self.assertAlmostEqual(r["lhs"], 0.00648, places=5)
        self.assertTrue(r["passes"])


class TestRequiredN(unittest.TestCase):
    def _sim_gate(self, n, sigma, margin, trials=4000, seed=1):
        rng = random.Random(seed)
        ok = sum(stats.gate_test([3.28 - margin + rng.gauss(0, sigma) for _ in range(n)])["passes"] for _ in range(trials))
        return ok / trials

    def test_gate_n_reaches_power(self):
        for sigma, margin in [(0.0013, 0.0015), (0.0010, 0.0027), (0.0013, 0.0008)]:
            n = stats.required_n_gate(sigma, margin, power=0.8)["n"]
            self.assertGreaterEqual(self._sim_gate(n, sigma, margin), 0.74, (sigma, margin, n))
            if n > 3:
                self.assertLess(self._sim_gate(n - 2, sigma, margin), 0.80, (sigma, margin, n))

    def test_welch_n_reaches_power(self):
        n = stats.required_n_welch(0.12, 0.15, power=0.8)["n"]
        rng = random.Random(2)
        ok = 0
        for _ in range(3000):
            a = [rng.gauss(39.85, 0.12) for _ in range(n)]
            b = [rng.gauss(40.0, 0.12) for _ in range(n)]
            ok += stats.welch_test(a, b, "less")["passes"]
        self.assertGreaterEqual(ok / 3000, 0.75)

    def test_z3_n(self):
        r = stats.required_n_z3(3.2785, power=0.8)
        self.assertGreaterEqual(r["p_pass_at_n"], 0.8)
        smaller = stats.required_n_z3(3.2785, power=0.5)
        self.assertLessEqual(smaller["n"], r["n"])
        # mean 3.2785: zero-noise pass needs (0.004/0.0015)^2 = 7.1 -> 8 runs; noise pushes it up
        self.assertGreaterEqual(r["n"], 8)


class TestCli(unittest.TestCase):
    def test_cli_gate(self):
        res = stats.main(["gate", os.path.join(ANVIL2, "this_pr"), "--extra", "3.2775"])
        self.assertTrue(res["passes"])
        res = stats.main(["n-gate", "--sigma", "0.0013", "--expected-mean", "3.2785"])
        self.assertIsNotNone(res["n"])


if __name__ == "__main__":
    unittest.main()
