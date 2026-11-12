"""CPU checks of the scientific controls and split/selection contract."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = HERE / "fno_followup.py"
        cls.assert_module = path.exists()
        if path.exists():
            spec = importlib.util.spec_from_file_location("fno_followup", path)
            cls.f = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.f)

    def setUp(self):
        self.assertTrue(self.assert_module, "fno_followup.py must implement the follow-up queue")

    def test_mixed_blocks_invert_and_preserve_matrix_sizes(self):
        x = torch.arange(8 * 3 * 4).reshape(8, 3, 4).to(torch.complex64)
        view = self.f.mixed_view(x.shape, x.device)
        self.assertEqual(view.fwd(x).shape, x.shape)
        torch.testing.assert_close(view.bwd(view.fwd(x)), x)
        self.assertFalse(torch.equal(view.fwd(x), x))
        # Input-channel rows remain intact; blocks mix mode/output columns.
        for row in range(3):
            self.assertEqual(sorted(view.fwd(x)[:, row, :].real.flatten().tolist()),
                             sorted(x[:, row, :].real.flatten().tolist()))

    def test_complex_adam_matches_radial_first_step(self):
        p = torch.nn.Parameter(torch.zeros(2, dtype=torch.complex64))
        p.grad = torch.tensor([3 + 4j, -4 + 3j])
        opt = self.f.ComplexAdam([p], lr=0.1, decay=0)
        opt.step()
        torch.testing.assert_close(p, -0.1 * p.grad / p.grad.abs())
        self.assertFalse(opt.state[p]["v"].is_complex())

    def test_shared_soap_keeps_per_mode_adam_statistics(self):
        p = torch.nn.Parameter(torch.zeros(5, 3, 4, dtype=torch.complex64))
        opt = self.f.make_optimizer("soap_shared", [p], 0.003, refresh=10)
        p.grad = torch.randn_like(p)
        opt.step()
        st = opt.state[p]
        self.assertEqual(st["L"].shape, (1, 3, 3))
        self.assertEqual(st["R"].shape, (1, 4, 4))
        self.assertEqual(st["vr"].shape, p.shape)
        self.assertTrue(torch.isfinite(p).all())

    def test_split_ignores_old_test_and_has_no_overlap(self):
        d = {"x_train": torch.arange(1000)[:, None], "y_train": torch.arange(1000)[:, None],
             "x_test": torch.full((200, 1), -999), "y_test": torch.full((200, 1), -999)}
        tr, va = self.f.split_training(d)
        self.assertEqual(len(tr[0]), 800)
        self.assertEqual(len(va[0]), 200)
        self.assertFalse(set(tr[0].flatten().tolist()) & set(va[0].flatten().tolist()))
        self.assertEqual(len(set(torch.cat([tr[0], va[0]]).flatten().tolist())), 1000)
        self.assertTrue(torch.equal(self.f.split_training(d)[1][0], va[0]))

    def test_selection_uses_validation_only_and_rejects_all_failures(self):
        records = [{"lr": 0.003, "final": 2.0, "test_final": 0.1},
                   {"lr": 0.009, "final": 1.0, "test_final": 10.0},
                   {"lr": 0.027, "final": None}]
        self.assertEqual(self.f.select_best(records)["lr"], 0.009)
        with self.assertRaises(ValueError):
            self.f.select_best([{"final": None}])

    def test_manifest_has_independent_tuning_and_confirmation_seeds(self):
        stages = self.f.stage_specs()
        self.assertEqual(stages["controls-darcy"]["dataset"], "darcy")
        self.assertIn("soap_shared", [a["arm"] for a in stages["controls-darcy"]["arms"]])
        self.assertEqual(stages["ns-boundary"]["arms"][0]["center_lr"], 0.027)
        self.assertEqual(self.f.CONFIRM_SEEDS, (101, 102, 103))
        self.assertNotIn(self.f.TUNE_SEED, self.f.CONFIRM_SEEDS)

    def test_tuning_smoke_never_opens_test_data(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "train.pt"
            x = torch.randn(20, 32)
            torch.save(dict(x_train=x, y_train=.3 * x + .2), p)
            result = self.f.train(dict(source=str(p), fresh_test=str(Path(td) / "missing.pt"),
                                       dataset="burgers", arm="complex_adam", lr=.003,
                                       refresh=10, seed=100, evaluate_test=False, smoke=True, device="cpu"))
            self.assertIsNone(result["test_final"])
            self.assertTrue(torch.isfinite(torch.tensor(result["final"])))
            self.assertEqual(result["steps"], 2)


if __name__ == "__main__":
    unittest.main()
