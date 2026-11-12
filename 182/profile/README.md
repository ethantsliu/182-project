# 182/profile: where the step time goes

Before we build another kernel we need to know where a training step spends its time. Our two earlier kernel ideas
did not pay: HADAMARD_FP8 measured 3.17 s slower, and FP8_ATTN_O has a cost-model centre of +0.2 s. Both targeted
parts of the step whose share of the time we had never measured. This directory profiles about 20 steady-state
steps of the timed loop with torch.profiler and turns the trace into a table of time per category.

## What it does

- `hook.py` runs a trainer (`train_gpt.py` of the root record or of a variant) with its `train_step` wrapped. The
  trainer file is not edited. The approach is the one `182/tools/nsys_step_hook.py` uses: the script is executed as
  a module and its `main()` is called. The profiler window covers steps `START..START+COUNT-1` of the timed loop
  (`PROFILE_STEPS=START:COUNT`, default `300:20`). Warmup also calls `train_step`, so the window opens only at a call
  that follows the unbroken run of calls 0, 1, ..., START-1, which only the timed loop makes. Every rank profiles
  the same steps, with a `torch.cuda.synchronize()` at both ends of the window. Settings: CUDA + CPU activities,
  `record_shapes` on, no stack traces, no memory profiling. Rank 0 writes `trace_rank0.json.gz` (a Chrome/Perfetto
  trace). Every rank writes `rank<R>_summary.json`, which is analyze.py run on its own trace, so a slow rank stands
  out. Every rank also writes `meta_rank<R>.json`.
- `shim/torchrun` lets a variant's own `182/run_node.sh run` (data path, seed, checks, RESULT line) start the trainer
  through `hook.py`. `run.sh` puts it first on PATH. It inserts `hook.py` before the trainer script and execs the
  real torchrun. If a run never went through the shim, `run.sh` fails and says so; it never passes off an
  unprofiled run as a result.
- `run.sh TREE OUT` makes one profiled run, then runs `analyze.py`. `TREE` is a variant directory or `.` for the
  root. The session's `profile` phase calls it once per entry of `PROFILE_TREES` (default `t1gpu root`).
- `analyze.py OUT` (stdlib only) prints:
  - a per-category table in ms/step and % of the step's GPU window: attention fwd/bwd, MoE grouped GEMMs, dense
    GEMMs, fp8 quantize/scales, optimizer (polar/Newton-Schulz, Adam), loss/head, row gather/scatter, NCCL,
    memcpy, elementwise/other
  - idle time (the window minus the union of all kernels)
  - NCCL kernel time, and how much of it is exposed, meaning not under any compute kernel
  - the largest idle gaps, with the kernels on either side
  - the top 25 kernels
  - a per-rank table

  Categories are regexes on kernel names (`--categories FILE` replaces them), and the first match wins. Kernels on
  different streams overlap, so the category rows can sum to more than 100%.

## CUDA graphs

Both trainers replay the forward/backward, the optimizer and the fp8 refresh as CUDA graphs. CUPTI reports the
kernels inside a graph replay only if it was initialised before the graphs were captured (pytorch issue 75504).
The trainers capture during warmup, long before the window opens. So `hook.py` opens and closes an empty CUDA
profiler session before the trainer starts (`PROFILE_CUPTI_INIT=1`, the default), with `TEARDOWN_CUPTI=0` and
`DISABLE_CUPTI_LAZY_REINIT=1`. That is the workaround torch's own `_init_for_cuda_graphs` uses. analyze.py prints
the graph coverage: the number of `cudaGraphLaunch` calls and the number of kernels traced from them. If the second
is 0, it warns that the breakdown covers eager kernels only.

There is no graphs-off mode. In both trees the step, optimizer and fp8-refresh graphs are structural: no flag or
environment variable gives an eager path, and adding one would mean editing trainer files that this directory does
not own. Such a mode would also bias the breakdown. Without graphs every kernel pays its CPU launch, so idle gaps and
"elementwise/other" would grow, and inductor's graph-only fusions would differ. The fallback, if graph kernels are
missing from the trace, is `bash 182/tools/profile_nsys.sh`, which uses nsys `--cuda-graph-trace=node` on rank 0
for the same timed-loop steps.

## Cost to a normal run

None. The trainers contain no profiling code, and nothing here runs unless you go through `run.sh`, `hook.py` or the
session's `profile` phase. The root flags-off SASS/PTX check is unaffected, because no root file changed. A profiled
run is not a timing run: the profiler adds overhead, and so do the two synchronizes. Its RESULT line is printed as
"profiled run (NOT a timing run)".

## How to run

```bash
bash 182/gpu_session.sh profile                                  # t1gpu then root; output in 182/results/<session>/profile/
bash 182/profile/run.sh variants/t1-gpu-stack /tmp/prof_t1gpu    # one tree by hand
bash 182/profile/run.sh . /tmp/prof_root                         # the root record
PROFILE_STEPS=500:10 bash 182/profile/run.sh . /tmp/prof_late    # another window
python 182/profile/analyze.py /tmp/prof_t1gpu --json a.json --top 40
```

Each run takes about one training run plus profiler export. The session estimates 5 node-minutes per tree
(`PROFILE_RUN_MIN`) and caps each tree at `PROFILE_TIMEOUT_S` (1800). A trace bigger than `PROFILE_BUNDLE_MB` (150)
stays out of the session bundle. analysis.txt/json and the rank summaries are always bundled.

## Tests

`python -m pytest -q 182/profile/tests` runs on CPU. It covers analyze.py on a synthetic Kineto trace with
hand-computed numbers (windows, busy/idle, exposed NCCL, categories, graph coverage, gaps, the CLI with per-rank
summaries), the window logic of hook.py, an end-to-end hooked run against a stub torch, the shim, and run.sh
through a tree's run_node.sh and the root, including the venv-activation case and the bypassed-shim failure.

## Unverified

- Whether the CUPTI early initialisation really makes torch 2.10's Kineto record graph-replay kernels on this
  node. Check the "graph coverage" line on the first trace.
- The kernel-name regexes against real kernel names (FA3, cuBLAS nvjet/sm90_xmma, Gluon MoE, inductor
  `triton_*`). Kernels that fall into "elementwise/other" in the top-25 list show what needs a new rule.
- Trace size and export time at 20 steps with `record_shapes`. On all 8 ranks, every rank exports a trace to
  summarize it. Only rank 0's trace is kept.
- The default window, 300:20. Both trees run about 1050-1200 steps, and step cost changes with the schedule stage
  (sequence length and batch). One window describes one stage only. To cover the others, profile more windows
  (`PROFILE_STEPS`).

Nothing here has run on a GPU.
