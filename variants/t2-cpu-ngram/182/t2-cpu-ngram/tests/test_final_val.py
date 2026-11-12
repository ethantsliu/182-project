"""CPU tests of the final validation's memory handling (track_1_short/final_val.py, model/gpt.py's loss slabs) and of
the stack guard (train_gpt_medium_cpu.py, node.py).

The out-of-memory retry recomputes a batch with smaller loss slabs; the val loss is the mean over the same tokens of
the same per-token formula. Checked on CPU: eager, bit for bit at every slab size of 2 rows and up; through CPU
inductor, equal to float rounding; and the compiled function recompiles when final_val.py halves the constant.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
VARIANT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import cpu_stubs  # noqa: E402

cpu_stubs.install_all(VARIANT)
if not hasattr(os, "sched_getaffinity"):  # macOS: retrieval.py reads it at import
    os.sched_getaffinity = lambda pid: set(range(os.cpu_count() or 1))

import torch  # noqa: E402
from torch import nn  # noqa: E402

import node  # noqa: E402
from track_1_short import config, exact_counts, final_val, memory_budget  # noqa: E402
from track_1_short.model import gpt  # noqa: E402
from track_1_short.ngram_table import NgramTable  # noqa: E402
from track_1_short.perf.cuda_graphs.step_graphs import StepGraphs  # noqa: E402
from track_1_short.schedule import TrainingSchedule  # noqa: E402

VOCAB = 50304  # the padded vocabulary: the copy sink's column is 50257
T, D, IDS = 45, 32, 64  # rows, width, distinct input tokens (canon_mask rows)


class Head(nn.Module):  # model/layers.py CastedLinearT's validation forward, on a small width
    def __init__(self, d, v, g):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d, v, generator=g).bfloat16() * 0.2)

    def forward(self, x):
        return x @ self.weight.type_as(x)


def cplm_case(seed=0):
    g = torch.Generator().manual_seed(seed)
    lm_head = Head(D, VOCAB, g)
    xf = torch.randn(T, D, generator=g).bfloat16()
    inputs = torch.randint(0, IDS, (T,), generator=g)
    targets = torch.randint(0, 50257, (T,), generator=g)
    targets[::7] = inputs[::7]  # some targets the mask may drop
    canon = torch.randint(0, 256, (IDS, VOCAB // 8), generator=g, dtype=torch.uint8)
    p_copy = torch.rand(T, generator=g)
    a_sink = torch.rand(T, generator=g)
    gate = torch.full((1,), 0.7)
    return lm_head, xf, inputs, targets, canon, p_copy, a_sink, gate


@pytest.mark.parametrize("rows", [2, 7, 16, 22, 44])
def test_cplm_slabs_give_the_same_per_token_losses(rows):
    lm_head, xf, inputs, targets, canon, p_copy, a_sink, gate = cplm_case()
    with torch.no_grad():
        whole = gpt.cplm_eval_slab_losses(lm_head, xf, inputs, targets, canon, p_copy, a_sink, gate, T)
        part = gpt.cplm_eval_slab_losses(lm_head, xf, inputs, targets, canon, p_copy, a_sink, gate, rows)
    assert whole.shape == (T,) and torch.isfinite(whole).all()
    assert torch.equal(whole, part)  # bit for bit, every token
    assert torch.equal(whole.float().mean(), part.float().mean())  # and so the batch's contribution to val_loss


def test_one_row_slabs_differ_only_by_rounding():
    """The degenerate 1-row slab is the one case seen to differ on CPU: one token by ~2e-6 nats (the single-row
    reduction path, not the matmul). The retry never goes below EVAL_CE_MIN_SLAB_ROWS = 2048 rows."""
    lm_head, xf, inputs, targets, canon, p_copy, a_sink, gate = cplm_case()
    with torch.no_grad():
        whole = gpt.cplm_eval_slab_losses(lm_head, xf, inputs, targets, canon, p_copy, a_sink, gate, T)
        one = gpt.cplm_eval_slab_losses(lm_head, xf, inputs, targets, canon, p_copy, a_sink, gate, 1)
    assert (whole - one).abs().max() < 1e-5 and abs(float(whole.mean() - one.mean())) < 1e-6
    assert final_val.EVAL_CE_MIN_SLAB_ROWS >= 2


@pytest.mark.parametrize("rows", [1, 8, 30])
def test_plain_slabs_give_the_same_per_token_losses(rows):
    g = torch.Generator().manual_seed(1)
    v = 256
    lm_head = Head(D, v, g)
    x = torch.randn(T, D, generator=g).bfloat16()
    inputs, targets = torch.randint(0, v, (T,), generator=g), torch.randint(0, v, (T,), generator=g)
    canon = torch.randint(0, 256, (v, v // 8), generator=g, dtype=torch.uint8)
    with torch.no_grad():
        assert torch.equal(gpt.eval_slab_losses(lm_head, x, inputs, targets, canon, T),
                           gpt.eval_slab_losses(lm_head, x, inputs, targets, canon, rows))


def test_inductor_slab_sizes_agree_to_rounding(tmp_path):
    """The slab loop through CPU inductor (tests/inductor_probe.py, a subprocess: inductor cannot compile with the
    triton stub loaded) at one slab, half a slab and 8 rows. Eager is bit for bit (above); compiled, a different
    slab size may change the kernels' reduction split, so a token's loss may move at float rounding. Measured:
    at most 3.8e-6 nats per token, 2e-8 on the mean. On the GPU the bound is not measured (see the README)."""
    env = {**os.environ, "TMPDIR": str(tmp_path), "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor")}
    out = subprocess.run([sys.executable, str(HERE / "inductor_probe.py")], env=env, capture_output=True, text=True,
                         timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["graphs"] == 3  # one compile per slab size: the bare module constant is guarded, so the retry recompiles
    assert max(r["max_abs"].values()) < 1e-4 and max(r["mean_abs"].values()) < 1e-6


def test_eval_forward_retries_with_half_the_slab(monkeypatch, capsys):
    monkeypatch.setattr(gpt, "EVAL_CE_SLAB_ROWS", 32768)
    seen = []

    def model(x, scale=1):
        seen.append(gpt.EVAL_CE_SLAB_ROWS)
        if gpt.EVAL_CE_SLAB_ROWS > 8192:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 6.14 GiB")
        return x * scale

    assert final_val.eval_forward(model, 3, scale=2) == 6
    assert seen == [32768, 16384, 8192] and gpt.EVAL_CE_SLAB_ROWS == 8192
    out = capsys.readouterr().out
    assert "out of memory with 32768-row loss slabs" in out and "retrying the batch with 8192-row slabs" in out


def test_eval_forward_gives_up_at_the_floor(monkeypatch):
    monkeypatch.setattr(gpt, "EVAL_CE_SLAB_ROWS", 4096)

    def model():
        raise torch.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(torch.OutOfMemoryError):
        final_val.eval_forward(model)
    assert gpt.EVAL_CE_SLAB_ROWS == final_val.EVAL_CE_MIN_SLAB_ROWS

    def other():
        raise RuntimeError("not memory")

    with pytest.raises(RuntimeError):  # anything else is not retried
        final_val.eval_forward(other)


def test_release_drops_the_training_buffers():
    ex = object.__new__(exact_counts.ExactCounts)
    ex.corpus, ex.tab, ex.filt = torch.zeros(100, dtype=torch.int16), torch.zeros(10, dtype=torch.int64), torch.zeros(4, dtype=torch.int64)
    ex.q_chunk = torch.zeros(6, dtype=torch.int64)
    ex.q_tok, ex.q_tgt = torch.zeros(2, 6, dtype=torch.int32)
    ex.out = torch.zeros(6, memory_budget.EXACT_OUT_COLS, dtype=torch.int32)
    assert ex.release() == 200 + 80 + 32 + 48 + 48 + 6 * 14 * 4
    assert ex.corpus is None and ex.out is None and ex.q_tok is None and ex.release() == 0

    resets = []
    sg = object.__new__(StepGraphs)
    graph = SimpleNamespace(reset=lambda: resets.append(1))
    sg.captured = {"a": SimpleNamespace(forward=graph, backward=graph), "b": SimpleNamespace(forward=graph, backward=graph)}
    sg.unchecked, sg.current, sg.adam_accumulators, sg.ngram_grad_ring, sg.sealed = set(), "cs", {"p": 1}, [1, 2], True
    assert sg.release() == 4 and len(resets) == 4
    assert not sg.captured and sg.current is None and not sg.adam_accumulators and sg.ngram_grad_ring == [] and sg.sealed

    table = object.__new__(NgramTable)
    table.pending, table.grads_in_flight, table.grad_per_slot = [(1, 2)], (1,), torch.zeros(3)
    table.shard = torch.zeros(2)
    table.release_training_state()
    assert table.pending == [] and table.grads_in_flight is None and table.grad_per_slot is None and table.shard is not None


def test_release_for_final_validation_calls_everything():
    calls = []
    ex = SimpleNamespace(release=lambda: calls.append("exact") or 3 << 30)
    sg = SimpleNamespace(release=lambda: calls.append("graphs") or 24)
    table = SimpleNamespace(release_training_state=lambda: calls.append("table"))
    model = SimpleNamespace(zero_grad=lambda set_to_none: calls.append(("grad", set_to_none)))
    line = final_val.release_for_final_validation(ex, sg, table, model, before=0)
    assert calls == ["exact", "graphs", "table", ("grad", True)]
    assert line.startswith("final validation memory release:") and "exact counts 3.00 GiB, 24 step graphs" in line


def test_fit_steps_match_the_schedule():
    for scheduled in (5790, 5200, 4600, 3000, 715):
        s = TrainingSchedule(config.TRAINING_STAGES, scheduled, 10, device=torch.device("cpu"), cooldown_frac=0.7,
                             split_embed_stage=config.SPLIT_EMBED_STAGE, ws_post_yarn_ext=config.WS_POST_YARN_EXT)
        assert memory_budget.fit_steps_count(s.total_steps) == len(range(s.boundaries[3][0], s.total_steps, 4))


def test_final_validation_estimate_at_d1024():
    held, transient = memory_budget.final_validation(1024, 16, 4096, 1048576, 5800)
    gib = {k: round(v / memory_budget.GiB, 2) for k, v in {**held, **transient}.items()}
    assert gib["fit steps' hidden states (list + concatenation)"] == 5.62  # 90 fit steps x 16384 tokens x 1024 x bf16, twice
    assert gib["exact counts (corpus share + table)"] == 6.25
    assert gib["loss slab, fused (bf16 matmul + fp32 logits)"] == 9.21  # 32768 x 50304 x 6 bytes
    text = memory_budget.report(1024, 16, 4096, 1048576, 5800)
    assert "the release drops 13.5 GiB plus the step graphs' pool" in text


# ---- the stack guard ------------------------------------------------------------------------------------

def wrapper_ns():
    src = (VARIANT / "train_gpt_medium_cpu.py").read_text()
    ns = {}
    exec("STACK = " + src.split("STACK = ", 1)[1].split('if __name__ == "__main__":')[0], ns)
    return ns


def test_wrapper_stops_on_another_stack():
    ns = wrapper_ns()
    assert ns["STACK"] == {"torch": node.TORCH, "triton": node.TRITON}
    assert ns["stack_error"]({"torch": "2.10.0+cu128", "triton": "3.6.0", "cuda": "12.8"}, allow=False) is None
    why = ns["stack_error"]({"torch": "2.11.0+cu128", "triton": "3.6.0"}, allow=False)
    assert why.startswith("wrong stack: torch 2.11.0+cu128 (needs 2.10.0+cu128)") and "node_setup.sh --mode venv" in why
    assert "ALLOW_STACK=1" in why and node.runlib.failure_reason(why + "\n", 1).startswith("exit 1: wrong stack")
    assert ns["stack_error"]({"torch": "2.11.0", "triton": "3.7.0"}, allow=True) is None


def test_check_fails_on_another_stack_unless_allowed():
    fake = {"torch": "2.11.0+cu128", "triton": "3.6.0"}.get
    lines = node.stack_problems(False, version=fake)
    assert lines[0][0] == "FAIL" and "2.11.0+cu128" in lines[0][2] and "pip install torch==2.10.0" in lines[0][2]
    assert "ALLOW_STACK=1" in lines[0][2] and lines[1] == ("PASS", "triton", "3.6.0")
    assert node.stack_problems(True, version=fake)[0][0] == "WARN"
    assert node.stack_problems(True, version={}.get)[0][0] == "FAIL"  # not installed is never allowed


def test_skip_check_still_checks_the_stack(monkeypatch, capsys):
    monkeypatch.setenv("SKIP_CHECK", "1")
    monkeypatch.delenv("ALLOW_STACK", raising=False)
    monkeypatch.setattr(node, "installed_version", {"torch": "2.11.0+cu128", "triton": "3.6.0"}.get)
    reason = node.preflight(dry=False)
    assert reason and "1 FAIL" in reason and "FAIL torch: 2.11.0+cu128" in capsys.readouterr().out
    monkeypatch.setenv("ALLOW_STACK", "1")
    assert node.preflight(dry=False) is None
    assert node.preflight(dry=True) is None


def test_result_line_carries_torch():
    r = dict(val_loss=2.877, train_time_ms=221500, steps=5800)
    assert node.torch_version("x\nstack: torch=2.10.0+cu128 triton=3.6.0 cuda=12.8\n") == "2.10.0+cu128"
    line = node.result_line(r, "logs/a.txt", torch="2.10.0+cu128")
    assert line.endswith("hit=1 torch=2.10.0+cu128 log=logs/a.txt")
    bad = node.result_line(None, None, "exit 1: boom", torch="2.11.0")
    assert bad.startswith("RESULT variant=t2-cpu-ngram status=failed torch=2.11.0 reason=exit 1: boom")
