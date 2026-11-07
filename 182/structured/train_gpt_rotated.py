"""
train_gpt_rotated.py

records/track_3_optimization/train_gpt_simple.py (the tuned Muon + aux AdamW baseline, result #36) plus
two env-flagged fixed-basis optimizer tests, inline (Track 3 rule 4: no third-party optimizer imports).
With no env vars set it runs the baseline: fused AdamW + Muon, 3250 steps.

    ADAM_BASIS=standard|identity|dct|hadamard|random   Adam params' basis (identity = inline AdamW, A/A)
    ADAM_BASIS_PARAMS=all|2d|1d                         which Adam params rotate (default all)
    HIDDEN_OPT=muon|soap_fixed                          hidden-matrix optimizer
    HIDDEN_BASIS=hadamard|dct|random|identity           soap_fixed's fixed basis (identity = plain Adam)
    HIDDEN_LR=0.003 HIDDEN_BETAS=0.95,0.95 HIDDEN_EPS=1e-8 HIDDEN_WD=0.1 HIDDEN_WARMUP_STEPS=250
    TRAIN_STEPS=3250                                    BASIS_SEED=0 (random basis)

Run from the repo root: torchrun --standalone --nproc_per_node=8 182/structured/train_gpt_rotated.py [num_trials]

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


########################################
#   Fixed-basis optimizers (182 ws/structured)
########################################
# Two falsification tests of "the basis is the preconditioner", both off by default (the baseline's
# fused AdamW + Muon run unchanged):
#   ADAM_BASIS     Adam for the Adam-managed params (embed, lm head, the 1-D params) run in a fixed
#                  orthogonal basis of the feature dim: g' = g @ Q, Adam moments on g', u = u' @ Q.T.
#   HIDDEN_OPT=soap_fixed   SOAP (Adam in a two-sided rotated basis, https://arxiv.org/abs/2409.11321)
#                  with the Shampoo eigenbases replaced by FIXED ones: g' = QL.T @ g @ QR,
#                  u = QL @ adam(g') @ QR.T, instead of Muon for the hidden matrices.
# Muon itself is not tested in a rotated basis: polar(QL.T G QR) == QL.T polar(G) QR for orthogonal
# QL, QR, so Muon is basis-free and "Muon in the Hadamard basis" is just Muon. Only elementwise
# (diagonal) preconditioners can see the basis.
#
# Bases (orthonormal, fp32): "hadamard" (Sylvester for 2^a, Paley-12 (x) Sylvester for 12 * 2^a, which
# covers 768 and 3072), "dct" (DCT-II), "random" (seeded Haar orthogonal), "identity". A dim with no
# construction (or larger than MAX_BASIS_DIM, e.g. the 50304-wide vocab) stays in the standard basis.

MAX_BASIS_DIM = 4096

def sylvester_hadamard(n: int) -> Tensor:
    assert n >= 1 and n & (n - 1) == 0, f"Sylvester Hadamard needs a power of two, got {n}"
    H = torch.ones(1, 1)
    while H.size(0) < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H

def paley_hadamard_12() -> Tensor:
    """Paley construction I for q = 11: H = I + [[0, 1^T], [-1, Q]], Q the Jacobsthal matrix."""
    q = 11
    residues = {(x * x) % q for x in range(1, q)}
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in residues else -1)
    S = torch.zeros(q + 1, q + 1)
    S[0, 1:] = 1
    S[1:, 0] = -1
    for i in range(q):
        for j in range(q):
            S[1 + i, 1 + j] = chi(j - i)
    return torch.eye(q + 1) + S

def hadamard_matrix(n: int) -> Tensor | None:
    """Orthonormal Hadamard (H @ H.T = I) of order n, or None when n is not 2^a or 12 * 2^a."""
    if n & (n - 1) == 0:
        H = sylvester_hadamard(n)
    elif n % 12 == 0 and (n // 12) & (n // 12 - 1) == 0:
        H = torch.kron(paley_hadamard_12(), sylvester_hadamard(n // 12))
    else:
        return None
    return H / n ** 0.5

def dct_matrix(n: int) -> Tensor:
    """Orthonormal DCT-II: C[k, i] = s_k cos(pi (i + 1/2) k / n); x -> C @ x is the transform."""
    k = torch.arange(n, dtype=torch.float64)[:, None]
    i = torch.arange(n, dtype=torch.float64)[None, :]
    C = torch.cos(torch.pi * (i + 0.5) * k / n) * (2.0 / n) ** 0.5
    C[0] /= 2 ** 0.5
    return C.float()

def random_orthogonal(n: int, seed: int) -> Tensor:
    g = torch.Generator().manual_seed(seed + n)
    Q, R = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    return (Q * torch.sign(torch.diagonal(R))).float()  # sign fix: Haar-distributed

_BASIS_CACHE = {}
def basis_matrix(kind: str, n: int, device) -> Tensor | None:
    """Q [n, n] with orthonormal columns (the basis vectors), or None for the standard basis."""
    if kind == "identity" or n > MAX_BASIS_DIM or n < 2:
        return None
    key = (kind, n, str(device))
    if key not in _BASIS_CACHE:
        if kind == "hadamard":
            Q = hadamard_matrix(n)
        elif kind == "dct":
            Q = dct_matrix(n).T  # columns are the DCT basis vectors: g @ Q = (C @ g.T).T
        elif kind == "random":
            Q = random_orthogonal(n, seed=int(os.environ.get("BASIS_SEED", "0")))
        else:
            raise ValueError(f"unknown basis {kind!r}")
        _BASIS_CACHE[key] = None if Q is None else Q.to(device)
    return _BASIS_CACHE[key]

class RotatedAdamW(torch.optim.Optimizer):
    """AdamW whose moments live in a fixed orthogonal basis of each param's last dim (group["basis"]).
    With basis "identity" it is torch's AdamW (decoupled decay, bias correction; moments stored in the
    param's dtype, math in fp32, as the fused kernel does): the A/A control for ADAM_BASIS."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, basis="identity"):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, basis=basis))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["Q"] = basis_matrix(group["basis"], p.size(-1), p.device)
                Q = state["Q"]
                state["step"] += 1
                g = p.grad.float()
                if Q is not None:
                    g = g @ Q
                m = state["exp_avg"].float().lerp_(g, 1 - b1)
                v = state["exp_avg_sq"].float().mul_(b2).addcmul_(g, g, value=1 - b2)
                state["exp_avg"].copy_(m)
                state["exp_avg_sq"].copy_(v)
                bc1, bc2 = 1 - b1 ** state["step"], 1 - b2 ** state["step"]
                u = (m / bc1) / ((v / bc2).sqrt() + group["eps"])
                if Q is not None:
                    u = u @ Q.T
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u.to(p.dtype), alpha=-group["lr"])

class SOAPFixedBasis(torch.optim.Optimizer):
    """SOAP with fixed bases: Adam on QL.T @ G @ QR for each 2-D param (QL, QR from group["basis"],
    per side; a side with no construction stays standard), mapped back with QL @ U @ QR.T. Every rank
    holds every param and computes the same update from the same all-reduced gradient (no gather).
    group["warmup_steps"] is read by the trainer's lr schedule."""
    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), eps=1e-8, weight_decay=0.0, basis="hadamard",
                 warmup_steps=0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, basis=basis,
                                      warmup_steps=warmup_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                assert p.ndim == 2
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["QL"] = basis_matrix(group["basis"], p.size(0), p.device)
                    state["QR"] = basis_matrix(group["basis"], p.size(1), p.device)
                QL, QR = state["QL"], state["QR"]
                state["step"] += 1
                g = p.grad.float()
                if QL is not None:
                    g = QL.T @ g
                if QR is not None:
                    g = g @ QR
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.lerp_(g, 1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1, bc2 = 1 - b1 ** state["step"], 1 - b2 ** state["step"]
                u = (m / bc1) / ((v / bc2).sqrt() + group["eps"])
                if QL is not None:
                    u = QL @ u
                if QR is not None:
                    u = u @ QR.T
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u.to(p.dtype), alpha=-group["lr"])

def env_floats(name, default):
    return tuple(float(x) for x in os.environ.get(name, default).split(","))

def build_optimizers(model, adam_basis: str, adam_scope: str, hidden_opt: str, hidden_basis: str):
    """The baseline's AdamW + Muon (the defaults), or the fixed-basis variants. Same groups and lrs."""
    adam_groups = [dict(params=[model.embed.weight], lr=0.7),
                   dict(params=[model.proj.weight], lr=0.004),
                   dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015)]
    adam_kw = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001)
    if adam_basis == "standard":
        optimizer1 = AdamW(adam_groups, **adam_kw, fused=True)
    else:
        rotate_2d, rotate_1d = adam_scope in ("all", "2d"), adam_scope in ("all", "1d")
        for g in adam_groups:
            is_1d = g["params"][0].ndim < 2
            g["basis"] = adam_basis if (rotate_1d if is_1d else rotate_2d) else "identity"
        optimizer1 = RotatedAdamW(adam_groups, **adam_kw)
    hidden = [p for p in model.blocks.parameters() if p.ndim >= 2]
    if hidden_opt == "muon":
        optimizer2 = Muon(hidden, lr=0.025, weight_decay=0.05)
    elif hidden_opt == "soap_fixed":
        optimizer2 = SOAPFixedBasis(hidden, lr=float(os.environ.get("HIDDEN_LR", "0.003")),
                                    betas=env_floats("HIDDEN_BETAS", "0.95,0.95"),
                                    eps=float(os.environ.get("HIDDEN_EPS", "1e-8")),
                                    weight_decay=float(os.environ.get("HIDDEN_WD", "0.1")),
                                    basis=hidden_basis,
                                    warmup_steps=int(os.environ.get("HIDDEN_WARMUP_STEPS", "250")))
    else:
        raise ValueError(f"unknown HIDDEN_OPT {hidden_opt!r}")
    return [optimizer1, optimizer2]

def significance(losses, target=3.28):
    """Track 3 rule: (target - avg) * sqrt(n) >= 0.004 (one-sided z-test at sigma = 0.0013, p < .001)."""
    n = len(losses)
    avg = sum(losses) / n
    stat = (target - avg) * n ** 0.5
    return avg, stat, stat >= 0.004


def main():
    ########################################
    #                Setup                 #
    ########################################

    # torchrun sets these env variables
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    # this code can be run equivalently with 1, 2, 4, or 8 gpus.
    assert 8 % dist.get_world_size() == 0

    # logging setup
    if dist.get_rank() == 0:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{uuid.uuid4()}.txt"
        print(logfile)
    def print0(s, console=False, log=True):
        if dist.get_rank() == 0:
            if console:
                print(s)
            if log:
                with open(logfile, "a") as f:
                    print(s, file=f)

    # we begin by logging this file itself
    print0(code)
    print0("="*100)
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
           + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
    print0("="*100)

    val_tokens = 20 * 524288
    batch_size = 8 * 64 * 1024
    mbs = 64
    val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

    model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
    model.compile(dynamic=False)


    num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

    adam_basis = os.environ.get("ADAM_BASIS", "standard")
    adam_scope = os.environ.get("ADAM_BASIS_PARAMS", "all")
    hidden_opt = os.environ.get("HIDDEN_OPT", "muon")
    hidden_basis = os.environ.get("HIDDEN_BASIS", "hadamard")
    print0(f"config: ADAM_BASIS={adam_basis} ADAM_BASIS_PARAMS={adam_scope} HIDDEN_OPT={hidden_opt}"
           + (f" HIDDEN_BASIS={hidden_basis}" if hidden_opt != "muon" else ""), console=True)
    final_losses = []

    for _ in range(num_trials):


        ########################################
        #       Init & Optim Hyperparams       #
        ########################################

        # we want to minimize this while still reaching 3.28 val loss
        train_steps = int(os.environ.get("TRAIN_STEPS", "3250"))

        # initialize model parameters
        for name, p in model.named_parameters():
            w = p.data
            if name.endswith("weight"):
                if "proj" in name:
                    w.zero_()
                elif "embed" in name:
                    w.normal_()  # default torch init
                else:
                    w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
            elif name.endswith("bias"):
                w.zero_()
            elif name.endswith("gains"):
                w.normal_(mean=1, std=0)
            else:
                raise Exception(f"Uninitialized parameter: {name}")

        # create the optimizer(s): the baseline's unless the fixed-basis env vars say otherwise
        optimizers = build_optimizers(model, adam_basis, adam_scope, hidden_opt, hidden_basis)
        assert set(p for opt in optimizers for group in opt.param_groups
                   for p in group["params"]) == set(model.parameters())
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        # learning rate schedule: stable then decay
        def set_hparams(step, cooldown_frac=0.7):
            progress = step / train_steps
            assert 0 <= progress < 1
            if progress < 1 - cooldown_frac:
                eta = 1.0
            else:
                eta = (1 - progress) / cooldown_frac
            for opt in optimizers:
                for group in opt.param_groups:
                    warmup = group.get("warmup_steps", 0)  # only SOAPFixedBasis groups set it
                    group["lr"] = group["initial_lr"] * eta * (min(1.0, (step + 1) / warmup) if warmup else 1.0)


        ########################################
        #        Training and Validation       #
        ########################################

        train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
        for p in model.parameters():
            dist.broadcast(p.detach(), 0)
        # start the clock
        training_time = 0
        last_val_step = 0
        dist.barrier()
        t0 = time.perf_counter()
        for step in range(train_steps + 1):

            # --------------- VALIDATION SECTION -----------------
            val_step_freq = 125 if step / train_steps < 0.9 else 25
            if step == train_steps or step % val_step_freq == 0:
                # stop the clock
                dist.barrier()
                time_since_last_val = time.perf_counter() - t0
                step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
                last_val_step = step
                training_time += time_since_last_val
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    assert len(val_inputs) % mbs == 0
                    for i in range(len(val_inputs) // mbs):
                        val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
                dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                val_loss /= val_tokens
                print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                       + f" step_avg:{1000*step_avg:.2f}ms", console=True)
                model.train()
                # start the clock again
                dist.barrier()
                t0 = time.perf_counter()

            if step == train_steps:
                final_losses.append(val_loss.item())
                break

            # --------------- TRAINING SECTION -----------------
            inputs, targets = next(train_loader)
            # accumulate across microbatches in case we are running with fewer than 8 gpus
            assert len(inputs) % mbs == 0
            for i in range(len(inputs) // mbs):
                model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
            for name, p in model.named_parameters():
                assert p.grad is not None, name
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            # set optimization hyperparameters and take a step
            set_hparams(step)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            approx_training_time = training_time + (time.perf_counter() - t0)
            print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
                   + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

    avg, stat, ok = significance(final_losses)
    print0(f"final val losses (n={len(final_losses)}): {[round(x, 5) for x in final_losses]} avg={avg:.5f} "
           f"(3.28-avg)*sqrt(n)={stat:.5f} {'>=' if ok else '<'} 0.004: {'PASS' if ok else 'FAIL'}", console=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
