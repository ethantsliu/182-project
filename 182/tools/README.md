# 182/tools: shared harness

Tools that every workstream uses: node setup, interleaved A/B runs, speedrun statistics, and nsys
profiling. The A/B runner, statistics, log parser, session summary and nsys analysis are stdlib-only Python
(no numpy or scipy), so they run in any env; `node_check.py` needs torch (it runs under torchrun on the node),
and `triton_cpu/` runs in its own docker image (torch + triton). Nothing here
changes a trainer: the A/B runner and the nsys hook wrap `train_gpt.py` / `train_gpt_medium.py` /
`train_gpt_simple.py` as they are.

| file | what it does |
|---|---|
| `node_setup.sh` | Writes a node report (GPUs, driver, NVLink/topology, CPU, NUMA, RAM, disk, CUDA runtimes). Then, with `--mode venv` or `--mode docker`, it builds the env: `torch==2.10` cu128, `requirements.txt`, a `libcudart.so.13` + `CUDA_HOME` headers check, the pinned patched FA3 (`devenpzak/flash-attn3-12864`, revision and sha256 read from `track_1_short/model/attention.py`, cache seeded for `HF_HUB_OFFLINE=1`), `data/cached_fineweb10B.py 9`, and `node_check.py`. |
| `node_check.py` | Under torchrun, measures bf16 GEMM TFLOP/s per GPU and flags any GPU >5% below the median. Also measures all_reduce bus bandwidth. |
| `logparse.py` | Parser for every log format in `records/`: Track 1/2 `train_time:Nms`, Track 3 `train_time:N.Ns`, Track 3's extra `val_avg_loss`/`val_ema_loss` fields, and Track 3 files that hold many runs (split where the step index goes down). It also handles NaN, crashed and incomplete runs. |
| `stats.py` | `gate`: one-sided t-test that mean val ≤ target, needs p<0.01. `welch`: Welch test of a time or val delta. `z3`: Track 3 rule `(3.28-mean)·√n ≥ 0.004`. `pair3`: Track 3 pairwise rule. `n-gate` / `n-welch` / `n-z3`: runs needed, given σ, margin and power. |
| `ab_run.py` | Pre-registered, interleaved A/B runner across git refs (one pinned worktree per commit) or directories, each arm with its own env vars. Every run is counted. It can resume after a crash. It writes `report.md` and `runs.csv`. |
| `profile_nsys.sh` + `nsys_step_hook.py` | Records nsys on rank 0 only, for steps `[start, start+steps)` of the **timed** loop. Each step gets an NVTX range `step N`, `cudaProfilerApi` sets the capture range, and `--cuda-graph-trace=node` is on. The script then exports to sqlite and runs the analyzer. |
| `summarize_session.py` | Reads a `182/gpu_session.sh` bundle (or its directory). First a table of the methods (root record and the four variants: runs, mean val loss, mean time, best reported, official record, target hits), then each variant's runs (the `RESULT` lines of its `run_node.sh`), then the A/B phases: per arm n, val/train_time mean ± sd, step-sum, ms/step per schedule stage, flags; paired deltas; the FP8 gate decision (gap mean ± SE, 95% CI vs 0.5 mnat); the `FP8_ATTN_O` net seconds at 164 ms per millinat. `--gate DIR --write-decision F`, `--parse-run`, `--variant`, `--base`, `--attno` are what the session script uses. |
| `triton_cpu/` | Docker harness for the Triton kernels without a GPU: the interpreter with a numerics shim, plus an sm_90a compile check (see below). |
| `analyze_nsys.py` | Per step: kernel count (and how many came from graphs), GPU busy/idle time and the largest gaps (with the kernels on either side), NCCL time and exposed NCCL time, GEMM time. Also the top kernels, and GEMM TFLOP/s wherever a shape can be recovered. |

## Commands on the 8xH100 box

```bash
git clone git@github.com:ethantsliu/182-project.git && cd 182-project
bash 182/tools/node_setup.sh --mode docker          # or --mode venv (bare metal: needs libcudart.so.13); writes node_report.txt
# venv: source .venv-run/bin/activate; export HF_HUB_OFFLINE=1 if get_kernel returns 401

# A/B: the record config vs a flag (same tree), or vs a variant folder (--arm new=path:variants/t1-gpu-stack), 8 interleaved runs each, cold compile caches (as in certifications)
python 182/tools/ab_run.py plan --out ab/fp8 --repo . --arm base=ref:main --arm new=ref:main \
    --env new:HADAMARD_FP8=1 --n 8 --cache cold
python 182/tools/ab_run.py run ab/fp8            # resumable; then: python 182/tools/ab_run.py report ab/fp8
# faster iteration: --cache warm --warmup 1 (a per-arm cache, plus one excluded warm-up run per arm)
# Track 2: --script train_gpt_medium.py --target 2.92;  Track 3: --script records/track_3_optimization/train_gpt_simple.py --at-step 3250
# docker: --cmd 'docker run --rm --gpus all --ipc=host --network=host -v {worktree}:{worktree} -v {work_root}:{work_root}
#                -v {data_dir}:{worktree}/data/fineweb10B -v ~/.cache/huggingface:/root/.cache/huggingface -w {worktree} {env_flags}
#                modded-nanogpt torchrun --standalone --nproc_per_node={nproc} {script}'

# Stats on any logs
python 182/tools/stats.py gate ab/fp8/logs/*_new_*.txt                     # mean val <= 3.28, p < 0.01?
python 182/tools/stats.py welch --a ab/fp8/logs/*_new_*.txt --b ab/fp8/logs/*_base_*.txt
python 182/tools/stats.py n-gate --sigma 0.0013 --margin 0.0015             # -> n=11 at 80% power
python 182/tools/stats.py n-welch --sigma 0.12 --delta 0.15                 # -> 15 runs per arm for a 150 ms win
python 182/tools/ab_run.py parse --group base='records/track_1_short/2026-08-30_ANVIL2/baseline' --group new='...'

# Profile 5 timed-loop steps on rank 0 (not a timing run)
bash 182/tools/profile_nsys.sh -o nsys/master --start 300 --steps 5 --analyze-args "--peak-tflops 989"
python 182/tools/analyze_nsys.py nsys/master/rank0.sqlite --list-gemms     # then write a --gemm-shapes JSON
# train_gpt_medium.py has its loop inline (no step function): --mode window --delay <s> --duration 3
#   --analyze-args "--step-marker <regex of the step's first kernel>"
```

## Triton kernels on CPU (`triton_cpu/`, Docker)

Checks a kernel before it costs GPU time. Two modes, same image (torch 2.10.0 CPU + triton 3.6.0, the pair
the record's `torch==2.10` installs; `TRITON_CPU_VERSION=3.5.1` for torch 2.9.1 + triton 3.5.1):

```bash
bash 182/tools/triton_cpu/run.sh python -m pytest 182/tools/triton_cpu 182/hadamard-fp8/tests -q   # interpreter
bash 182/tools/triton_cpu/run.sh aot [--base <sha>]                                               # sm_90a compile
```

- **Interpreter** (`TRITON_INTERPRET=1`): runs the real kernels on CPU tensors. The stock interpreter is
  numerically wrong for bf16 (dot, elementwise ops and int/fp32->bf16 casts act on raw uint16 bits) and
  for fp32->fp8 (ties away from zero, lost mantissa carry), in triton 3.5.1 through 3.9.0. `shim.py`, loaded
  by `sitecustomize.py`, routes casts through torch (RNE, fp8 satfinite), dots through fp64 and bf16 math
  through fp32, and sets `torch.cuda.get_device_properties` to 3 SMs, so persistent loops run several tiles
  per program. `test_interp_probes.py` checks each fix (73 probes; `TRITON_CPU_NO_SHIM=1` shows 20 failing raw).
  Not modelled: tf32 for fp32 dots, GPU accumulation order, approximate `tl.sqrt`, races between programs.
  The GPU kernel tests fall back to CPU under the interpreter. Anything that compiles CUDA C (nvrtc) stays GPU-only.
- **Compile check** (`aot`, `compile_check.py`): a fake sm_90 driver turns each launch into a JIT warmup.
  Covered: the MLP kernels only (linear_relu_square bf16 / fp8 forward / fp8 backward, the weight-cache and
  fp8-transpose quantizes, and every HADAMARD_BLOCK x rotation subset of the Hadamard kernels), called
  through their launchers at the record's shapes (H=2816, D=768, the per-rank token counts of every
  training stage plus the validation batch, an aligned and an unaligned per-slot scale pointer), so the
  constexprs, integer / pointer specializations and stage choice match training. `hadamard_t_quant` is
  called through its Python body. Kernels go through the frontend, TTGIR/LLVM and the wheel's ptxas. Per
  kernel it reports shared memory for each `num_stages` (H100 limit 232448 B), the stage the launcher's
  OutOfResources step-down would pick, ptxas registers/spills, and which geometries share a binary. Not
  covered: inductor's wrapper around these kernels, cuBLAS `_scaled_mm`, kernels outside the MLP. With
  `--base <sha>` it also compiles the record (flags-off) variants from `git archive <sha>` through the API
  upstream has and diffs their normalized SASS and PTX (a cuobjdump failure or an empty disassembly is an
  error, never "identical"): identical SASS means the change cannot touch the record's MLP kernels. `--only
  REGEX` restricts the calls. Nothing runs.

Labels: interpreter-tested = logic and indexing; compile-tested = it builds for H100 (smem, regs). Neither
says anything about speed, and neither replaces a GPU run.

## Ab_run rules (why the runner is strict)

- `plan.json` is the pre-registration: arms pinned to commit SHAs (or, for a directory arm, its HEAD plus a hash of the uncommitted diff), seeds, order and cache policy. A hash of the plan is stored next to it. `run` refuses to start if the plan was edited, or if a directory arm's code changed since the plan (unless you pass `--allow-changed`).
- Relative `path:` arms resolve from `--repo` and run from that variant directory. By default, each arm gets an absolute `DATA_PATH` for its run directory, whose `data/fineweb10B` links to the plan's shared root data. This overrides an inherited `DATA_PATH`; an explicit `--env NAME:DATA_PATH=...` or `--common-env DATA_PATH=...` still takes precedence. Existing data directories/links must refer to the plan's data, preventing a stale link from selecting a different corpus. Docker commands receive the same setting through `{env_flags}`.
- Order: `blocked` (the default) runs every arm once per rep, in a seeded random order. That spreads node drift and thermal effects evenly across arms. `abba` and `alternate` are also available.
- Seeds: `none` (the default) leaves runs unseeded, as certifications are. `auto` or a list gives the same seed to every arm within a rep, and the report then adds a paired per-rep test. The trainer reads the seed from `TRAIN_SEED` by default (`--seed-env` changes that).
- Every trainer outcome is recorded and counted: ok, failed, nan, timeout, incomplete. Nothing is retried or dropped, and the report flags arms with failures. The one exception is a slot cut short by a harness crash: it is logged as `interrupted` and re-run.
- `--cache cold` gives each run its own fresh `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR`, deleted afterwards, so the ~7 min compile is in every run's wall time but not in `train_time`. `warm` keeps one cache per arm, so arms never share compiled code.

## nsys analysis notes

- The hook wraps `train_gpt.py`'s `train_step` without editing the file. Warmup also calls `train_step`, so the hook profiles the second pass, where the step index restarts (`NSYS_HOOK_PHASE=1`). The capture covers one unannotated step on each side of the profiled range. It synchronizes twice, at the capture boundaries, on every rank.
- Kernels are assigned to a step by the time of their launch API call, which works for CUDA-graph replays. A step's GPU window runs from its first kernel to the next step's first kernel, so idle time includes bubbles between steps. Exposed NCCL = NCCL kernel time not overlapped by any non-NCCL kernel on the same device.
- GEMM shapes come from NVTX `aten::mm ... sizes = [...]` ranges (eager trainers, `NSYS_HOOK_EMIT_NVTX=1`) or from a `--gemm-shapes` JSON (`[{"match": regex, "mnk": [M,N,K]}]`). CUDA-graph replays have no CPU ops, so Track 1 needs the JSON. The fixture's example is `tests/fixtures/gemm_shapes_example.json`. GEMMs without a shape are reported with time only.

## Tests (CPU, no GPU)

`python -m pytest -q 182/tools/tests` (or `cd 182/tools && python -m unittest discover -s tests`). The tests run on CPU; scipy cross-checks are skipped if scipy is absent. Open-PR full-log regressions read fetched `upstream/pr/397`, `upstream/pr/381`, and `upstream/pr/261` objects and skip if those refs are absent. They are checked against real records:

- ANVIL2 pool: 17 shipped logs plus the README's 18th row give t=11.5, p=9.40e-10, mean 3.2773111, sd 0.00099048. Baseline: 73.889 ± 0.137 s. Win: 33.975 s.
- Track 3 #12: n=20 parsed out of one file, 3.2790 at step 3325. Track 3 #20: 3.2790 (n=30) at step 3030.
- The README's `pair3` examples: 0.00167, 0.0027 and 0.00648.
- t, ppf and Welch match scipy. A simulation confirms the required-n results reach their stated power.
- `ab_run` end to end: a fake trainer in a throwaway git repo, with same-ref env arms and variant-directory arms sharing data, ref worktrees, resume including a torn ledger append, launch/crash/NaN/timeout counting, tamper detection and the s-format logs.
- `gpu_session.sh` end to end against the fake trainer and a fake `run_node.sh` (`tests/fixtures/fake_run_node.sh`): phase order and estimates, the root stack gate (root phases skipped, variants still run), variant check FAIL => skip, prep inline and in the background, failed / result-less / timed-out runs, resume and retry, `RESULT` parsing, the base and attno A/Bs, read-only `DRY_RUN`, `QUICK`.
- Setup and profile shell entry points against CPU stubs, including missing CUDA runtime, uv pip seeding, venv/docker GPU-check failures and quoted analyzer arguments. Real driver/kernel/container behavior still requires a GPU node.
- The nsys hook, against a stub torch.
- `analyze_nsys` on `tests/fixtures/synthetic_nsys.sqlite`, built by `make_nsys_fixture.py`, whose docstring gives the expected numbers. It is also run on an older-schema variant of the fixture.

## Not yet verified on GPU

- `node_setup.sh` venv and docker modes, and `node_check.py`. Only `--mode report` has run (on macOS, without a GPU).
- `profile_nsys.sh` under real nsys and torchrun. Specifically: `--no-python` rank wrapping, the `cudaProfilerApi` capture with CUDA graphs, and whether real nsys exports name tables and columns as the fixture assumes. That assumption covers `StringIds`, `CUPTI_ACTIVITY_KIND_KERNEL.demangledName/shortName/graphNodeId`, `NVTX_EVENTS.text/textId` and `CUPTI_ACTIVITY_KIND_RUNTIME.correlationId/globalTid`. The analyzer tolerates missing optional columns. On the first real trace, check the kernel count against `nsys stats`.
- `ab_run.py` with real torchrun and docker launches, and the cold-cache wall time.
- `gpu_session.sh` on a real node: every phase has only run against CPU fakes. The node-minute estimates are guesses (the variant ones assume the compile cache on disk is reused between runs). The variant phases depend on each `variants/<name>/182/run_node.sh` honouring the contract (`check` prints `PASS|WARN|FAIL` lines and names `prep` in a FAIL that prep fixes; `run` ends with a `RESULT` line); `SESSION_DOCKER` mode needs an image that also carries what a variant needs.
- The default GEMM and NCCL name regexes. Real kernel names (cuBLAS `nvjet`/`sm90_xmma`, Triton `triton_tem_*`, this repo's custom fp8 kernels) may need `--gemm-regex` or `--gemm-exclude` adjustments. Check with `--list-gemms`.
