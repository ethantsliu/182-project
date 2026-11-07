"""CPU tests for ws/structured: the Monarch MLP math, the fixed bases, and a tiny smoke run of every
Track 3 optimizer option on a toy model.

    TORCH_COMPILE_DISABLE=1 python -m pytest -q 182/structured/test_structured.py
"""
import importlib.util
import math
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # the trainer's muon_update is @torch.compile

import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from track_1_short.model import monarch as M  # noqa: E402  (no triton imports)

_spec = importlib.util.spec_from_file_location("train_gpt_rotated", REPO / "182/structured/train_gpt_rotated.py")
T3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T3)


def setUpModule():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group("gloo", rank=0, world_size=1)


def block_diag(blocks):
    return torch.block_diag(*blocks.unbind(0))


# ---------------------------------------------------------------------------------------------------
# Monarch MLP (Track 1)

class TestMonarch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_record_shapes(self):
        s = M.MonarchShape(768, 4, 2)
        self.assertEqual((s.p, s.q, s.l, s.hidden, s.blocks_per_slot), (192, 384, 8, 3072, 24))
        self.assertEqual(s.params_per_slot(), 24 * 384 * 192)  # 1.77M vs dense 2 * 2816 * 768 = 4.33M
        for k in (2, 4, 8, 16):
            for e in (1, 2, 3):
                s = M.MonarchShape(768, k, e)
                self.assertEqual(s.hidden, 768 * e * e)
                self.assertEqual(12 * s.blocks_per_slot % 8, 0)  # ANVIL shards the bank over 8 ranks

    def _random_slot(self, s, dtype=torch.float64):
        return torch.randn(s.blocks_per_slot, s.q, s.p, dtype=dtype)

    def test_up_matches_explicit_block_diag_permutation(self):
        """up == L_bd @ P @ R_bd with P the [k, q] -> [q, k] transpose (built independently)."""
        for s in (M.MonarchShape(24, 4, 2), M.MonarchShape(48, 2, 3), M.MonarchShape(768, 4, 2)):
            blocks = self._random_slot(s)
            r_up, l_up, _, _ = (blocks[x] for x in s.slot_slices())
            n = s.blocks * s.q
            perm = torch.zeros(n, n, dtype=torch.float64)
            for i in range(s.blocks):
                for j in range(s.q):
                    perm[j * s.blocks + i, i * s.q + j] = 1.0
            expected = block_diag(l_up) @ perm @ block_diag(r_up)
            w_up, _ = M.dense_equivalents(blocks, s)
            torch.testing.assert_close(w_up, expected)
            self.assertTrue((w_up != 0).all(), "every output must depend on every input (full mixing)")

    def test_down_is_transpose_structure(self):
        s = M.MonarchShape(48, 4, 2)
        blocks = self._random_slot(s)
        r_up, l_up, l_down, r_down = s.slot_slices()
        blocks[l_down] = blocks[l_up]
        blocks[r_down] = blocks[r_up]
        w_up, w_down = M.dense_equivalents(blocks, s)
        torch.testing.assert_close(w_down, w_up.T)

    def test_mlp_forward_backward_bf16(self):
        s = M.MonarchShape(768, 4, 2)
        bank = torch.empty(12, s.blocks_per_slot, s.q, s.p)
        M.init_monarch_bank_(bank, s)
        slot = bank[0].bfloat16().requires_grad_()
        x = torch.randn(64, 768, dtype=torch.bfloat16)
        out = M.monarch_mlp(x, slot, s)
        self.assertEqual(out.shape, (64, 768))
        self.assertEqual(out.abs().max().item(), 0.0)  # R_down is zero at init, like c_proj
        (out.float() * torch.randn(64, 768)).sum().backward()
        g = slot.grad
        r_up, l_up, l_down, r_down = s.slot_slices()
        self.assertGreater(g[r_down].abs().sum().item(), 0.0, "the zero factor must get a gradient")

    def test_init_variance(self):
        s = M.MonarchShape(768, 4, 2)
        bank = torch.empty(1, s.blocks_per_slot, s.q, s.p)
        M.init_monarch_bank_(bank, s)
        r_up, l_up, l_down, r_down = (bank[0, x] for x in s.slot_slices())
        x = torch.randn(4096, 768)
        pre = M.monarch_up(x, r_up, l_up, s)
        self.assertAlmostEqual(pre.var().item(), 0.25, delta=0.03)  # dense c_fc: 0.25 * Var(x)
        self.assertEqual(r_down.abs().max().item(), 0.0)
        self.assertGreater(l_down.abs().max().item(), 0.0)

    def test_matrix_lr_muls(self):
        s = M.MonarchShape(768, 4, 2)
        muls = M.monarch_matrix_lr_muls(s, 12, frozen_slots=(7,))
        self.assertEqual(len(muls), 12 * s.blocks_per_slot)
        per = s.blocks_per_slot
        self.assertEqual(set(muls[7 * per:8 * per]), {0.0})
        r_up, l_up, l_down, r_down = s.slot_slices()
        slot0 = muls[:per]
        self.assertEqual({slot0[j] for j in range(r_up.start, l_up.stop)}, {M.MONARCH_LR_MUL})
        self.assertEqual({slot0[j] for j in range(l_down.start, r_down.stop)}, {M.MONARCH_LR_MUL * M.MONARCH_DOWN_LR_MUL})

    def test_per_block_polar_is_polar_of_block_diag_factor(self):
        """Per-block Muon == Muon of the block-diagonal factor (but not of the composed weight)."""
        def polar(a):
            u, _, vh = torch.linalg.svd(a, full_matrices=False)
            return u @ vh
        blocks = torch.randn(4, 12, 6, dtype=torch.float64)
        torch.testing.assert_close(polar(block_diag(blocks)), block_diag(torch.stack([polar(b) for b in blocks])))


# ---------------------------------------------------------------------------------------------------
# Fixed bases (Track 3)

class TestBases(unittest.TestCase):
    def assert_orthonormal(self, Q, atol=1e-5):
        n = Q.size(0)
        torch.testing.assert_close(Q.double() @ Q.double().T, torch.eye(n, dtype=torch.float64), atol=atol, rtol=0)
        torch.testing.assert_close(Q.double().T @ Q.double(), torch.eye(n, dtype=torch.float64), atol=atol, rtol=0)

    def test_paley_12(self):
        H = T3.paley_hadamard_12()
        torch.testing.assert_close(H @ H.T, 12 * torch.eye(12))
        self.assertEqual(set(H.unique().tolist()), {-1.0, 1.0})

    def test_hadamard(self):
        for n in (2, 4, 8, 12, 24, 48, 64, 256, 768, 1024, 3072):
            Q = T3.hadamard_matrix(n)
            self.assertIsNotNone(Q, n)
            self.assert_orthonormal(Q)
            torch.testing.assert_close(Q.abs(), torch.full((n, n), n ** -0.5))
        for n in (3, 20, 36, 50304):
            self.assertIsNone(T3.hadamard_matrix(n), n)

    def test_dct(self):
        for n in (5, 64, 768):
            C = T3.dct_matrix(n)
            self.assert_orthonormal(C)
        # DCT of the k-th basis cosine is a delta at k
        n, k = 64, 5
        x = torch.cos(torch.pi * (torch.arange(n) + 0.5) * k / n)
        y = T3.dct_matrix(n) @ x
        self.assertEqual(y.abs().argmax().item(), k)
        self.assertLess((y.abs().sum() - y.abs().max()).item(), 1e-4)

    def test_random(self):
        Q = T3.random_orthogonal(768, seed=0)
        self.assert_orthonormal(Q)
        torch.testing.assert_close(Q, T3.random_orthogonal(768, seed=0))  # same on every rank

    def test_invertibility_and_basis_matrix(self):
        x = torch.randn(10, 768)
        for kind in ("hadamard", "dct", "random"):
            Q = T3.basis_matrix(kind, 768, "cpu")
            self.assert_orthonormal(Q)
            torch.testing.assert_close((x @ Q) @ Q.T, x, atol=1e-4, rtol=0)
            torch.testing.assert_close((x @ Q).norm(), x.norm(), atol=1e-3, rtol=0)
        self.assertIsNone(T3.basis_matrix("identity", 768, "cpu"))
        self.assertIsNone(T3.basis_matrix("dct", 50304, "cpu"))  # the vocab dim stays standard
        self.assertIsNone(T3.basis_matrix("hadamard", 20, "cpu"))

    def test_muon_is_basis_free(self):
        """Why there is no 'Muon in a rotated basis' arm: NS(QL.T G QR) == QL.T NS(G) QR."""
        torch.manual_seed(0)
        G = torch.randn(64, 32)
        QL, QR = T3.hadamard_matrix(64), T3.dct_matrix(32).T
        a = T3.zeropower_via_newtonschulz5(QL.T @ G @ QR).float()
        b = QL.T @ T3.zeropower_via_newtonschulz5(G).float() @ QR
        self.assertLess(((a - b).norm() / b.norm()).item(), 0.05)  # bf16 iteration noise only


# ---------------------------------------------------------------------------------------------------
# Optimizers (Track 3)

class TestOptimizers(unittest.TestCase):
    def test_rotated_adamw_identity_is_adamw(self):
        torch.manual_seed(0)
        p1 = torch.nn.Parameter(torch.randn(16, 32))
        p2 = torch.nn.Parameter(p1.detach().clone())
        kw = dict(lr=0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.01)
        o1, o2 = T3.RotatedAdamW([p1], **kw), torch.optim.AdamW([p2], **kw)
        for _ in range(20):
            g = torch.randn(16, 32)
            p1.grad, p2.grad = g.clone(), g.clone()
            o1.step()
            o2.step()
        torch.testing.assert_close(p1, p2, atol=1e-6, rtol=1e-6)

    def test_rotated_adamw_is_adam_on_rotated_param(self):
        """Adam in basis Q on W == plain Adam on W' = W @ Q (then mapped back)."""
        torch.manual_seed(0)
        Q = T3.basis_matrix("hadamard", 32, "cpu")
        w = torch.nn.Parameter(torch.randn(8, 32))
        w_rot = torch.nn.Parameter((w.detach() @ Q).clone())
        kw = dict(lr=0.01, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.01)
        o1 = T3.RotatedAdamW([w], basis="hadamard", **kw)
        o2 = T3.RotatedAdamW([w_rot], basis="identity", **kw)
        for _ in range(20):
            g = torch.randn(8, 32)
            w.grad, w_rot.grad = g.clone(), g @ Q
            o1.step()
            o2.step()
        torch.testing.assert_close(w.detach() @ Q, w_rot.detach(), atol=1e-5, rtol=1e-5)

    def test_soap_fixed_is_adam_on_two_sided_rotation(self):
        torch.manual_seed(0)
        QL, QR = T3.basis_matrix("hadamard", 24, "cpu"), T3.basis_matrix("hadamard", 16, "cpu")
        w = torch.nn.Parameter(torch.randn(24, 16))
        w_rot = torch.nn.Parameter((QL.T @ w.detach() @ QR).clone())
        kw = dict(lr=0.003, betas=(0.95, 0.95), eps=1e-8, weight_decay=0.1)
        o1 = T3.SOAPFixedBasis([w], basis="hadamard", **kw)
        o2 = T3.SOAPFixedBasis([w_rot], basis="identity", **kw)
        for _ in range(20):
            g = torch.randn(24, 16)
            w.grad, w_rot.grad = g.clone(), QL.T @ g @ QR
            o1.step()
            o2.step()
        torch.testing.assert_close(QL.T @ w.detach() @ QR, w_rot.detach(), atol=1e-5, rtol=1e-5)

    def test_basis_changes_adam(self):
        """Sanity: unlike Muon, Adam's update does depend on the basis."""
        torch.manual_seed(0)
        g = torch.randn(8, 32) * torch.logspace(-2, 2, 32)  # anisotropic columns
        ups = {}
        for kind in ("identity", "hadamard"):
            w = torch.nn.Parameter(torch.zeros(8, 32))
            w.grad = g.clone()
            T3.RotatedAdamW([w], lr=1.0, basis=kind).step()
            ups[kind] = w.detach().clone()
        self.assertGreater((ups["identity"] - ups["hadamard"]).norm().item(), 1.0)

    def test_significance(self):
        self.assertTrue(T3.significance([3.2759])[2])
        self.assertFalse(T3.significance([3.2761])[2])
        self.assertTrue(T3.significance([3.2779] * 4)[2])  # README: n=4 needs avg < 3.278
        self.assertFalse(T3.significance([3.2781] * 4)[2])


def toy_batches(vocab, batch, seq, steps, seed=0):
    """A learnable toy language: t[i+1] = (5 t[i] + 3) mod vocab with 10% noise tokens."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        t = torch.empty(batch, seq + 1, dtype=torch.long)
        t[:, 0] = torch.randint(vocab, (batch,), generator=g)
        for i in range(seq):
            t[:, i + 1] = (5 * t[:, i] + 3) % vocab
        noise = torch.rand(batch, seq + 1, generator=g) < 0.1
        t[noise] = torch.randint(vocab, (int(noise.sum()),), generator=g)
        yield t[:, :-1].contiguous(), t[:, 1:].contiguous()


def zeropower_fp32(G):
    """The trainer's Newton-Schulz iteration in fp32: bf16 matmuls are a slow reference path on arm64 CPUs."""
    X = G.float()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if G.size(-2) > G.size(-1) else X


class TestSmokeRuns(unittest.TestCase):
    """40 steps of the trainer's toy-sized GPT under every optimizer option: loss must fall and stay finite.
    CPU-only substitutions: fp32 embedding and fp32 Newton-Schulz (the trainer uses bf16 for both)."""

    @classmethod
    def setUpClass(cls):
        cls._ns = T3.zeropower_via_newtonschulz5
        T3.zeropower_via_newtonschulz5 = zeropower_fp32  # muon_update looks it up at call time

    @classmethod
    def tearDownClass(cls):
        T3.zeropower_via_newtonschulz5 = cls._ns
    CONFIGS = [
        ("standard", "all", "muon", "hadamard"),      # the baseline
        ("identity", "all", "muon", "hadamard"),      # A/A: inline AdamW
        ("dct", "all", "muon", "hadamard"),
        ("hadamard", "all", "muon", "hadamard"),
        ("hadamard", "2d", "muon", "hadamard"),
        ("random", "1d", "muon", "hadamard"),
        ("standard", "all", "soap_fixed", "hadamard"),
        ("standard", "all", "soap_fixed", "dct"),
        ("standard", "all", "soap_fixed", "identity"),
        ("hadamard", "all", "soap_fixed", "hadamard"),
    ]

    def run_config(self, adam_basis, scope, hidden_opt, hidden_basis, steps=40):
        torch.manual_seed(0)
        vocab, dim = 128, 256
        model = T3.GPT(vocab_size=vocab, num_layers=2, model_dim=dim)
        model.embed.float()  # bf16 CPU matmuls are a slow reference path on arm64; fp32 for the smoke run
        for name, p in model.named_parameters():  # the trainer's init
            w = p.data
            if name.endswith("weight"):
                if "proj" in name:
                    w.zero_()
                elif "embed" in name:
                    w.normal_()
                else:
                    w.normal_(std=0.33 ** 0.5 / w.size(-1) ** 0.5)
            elif name.endswith("bias"):
                w.zero_()
            else:
                w.fill_(1.0)
        os.environ["HIDDEN_WARMUP_STEPS"] = "10"
        try:
            opts = T3.build_optimizers(model, adam_basis, scope, hidden_opt, hidden_basis)
        finally:
            del os.environ["HIDDEN_WARMUP_STEPS"]
        params = {p for o in opts for g in o.param_groups for p in g["params"]}
        self.assertEqual(params, set(model.parameters()))
        for o in opts:
            for g in o.param_groups:
                g["initial_lr"] = g["lr"]
        init = {n: p.detach().clone() for n, p in model.named_parameters()}
        losses = []
        for step, (x, y) in enumerate(toy_batches(vocab, 8, 32, steps)):
            for o in opts:  # the trainer's warmup rule (no cooldown in a smoke run)
                for g in o.param_groups:
                    w = g.get("warmup_steps", 0)
                    g["lr"] = g["initial_lr"] * (min(1.0, (step + 1) / w) if w else 1.0)
            loss = model(x, y)
            loss.backward()
            for o in opts:
                o.step()
            model.zero_grad(set_to_none=True)
            losses.append(loss.item() / y.numel())
        unmoved = [n for n, p in model.named_parameters() if torch.equal(p.detach(), init[n])]
        return losses, unmoved

    def test_all_configs(self):
        for cfg in self.CONFIGS:
            with self.subTest(cfg=cfg):
                losses, unmoved = self.run_config(*cfg)
                self.assertEqual(unmoved, [], cfg)
                self.assertTrue(all(math.isfinite(l) for l in losses), cfg)
                first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
                print(f"  {cfg}: loss {first:.3f} -> {last:.3f}")
                self.assertLess(last, first - 0.5, cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
