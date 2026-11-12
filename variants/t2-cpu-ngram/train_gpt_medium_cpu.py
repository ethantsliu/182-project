"""NanoGPT speedrun, track 2 (GPT-2 Medium, 2.92 FineWeb val loss on 8xH100), on PR #380's CPU-retrieval stack.

Launch: bash 182/run_node.sh run (check and prep first), or directly, from this folder:
DATA_PATH=../.. torchrun --standalone --nproc_per_node=8 train_gpt_medium_cpu.py, after `pip install ./exact_match`.

This is train_gpt.py's run (PR #380 by devenpzak, on PR #367's exact-match retrieval) at the Track 2 size
ldmberman posted in the PR #380 thread (2026-10-04, 2.8769 in 221 s): 16 layers, d_model 1024, MLP hidden
4096, the lr cooldown over the last 70% of the main stages down to a 0.10 floor, the validation batch halved
to 1048576 tokens to fit in memory, and 5790 scheduled + 10 extension = 5800 steps. The max train batch stays 131072 tokens, and the
sampled-softmax candidate schedule stretches with the stage boundaries (10240 / 14336 / 24576 / full at
steps 0 / 3513 / 4800 / 5443 for 5790 steps). The run used PR #380's sampled softmax only; the hierarchical
and adaptive output-head variants in that attachment are not ported.

Every value below is only a default: an environment variable of the same name overrides it (the step-count
sweep in 182/t2-cpu-ngram/ sets NUM_SCHEDULED_ITERATIONS). The run log holds this file, train_gpt.py and
the package, as for Track 1.

COORD_DESCENT_TUNING=1 turns on torch._inductor.config.coordinate_descent_tuning, which the README allows on
the GPT-2 Medium track only (it is banned on Track 1). It adds a long untimed compile, so use it for final
record runs, not sweeps; its effect on this stack (most hot kernels are hand-written Triton or CUDA-graph
replays) has not been measured.
"""
import os
import sys

TRACK2_DEFAULTS = {
    "MODEL_DIM": "1024",
    "MODEL_LAYERS": "16",
    "MLP_HIDDEN_DIM": "4096",
    "LR_COOLDOWN_FRAC": "0.70",
    "LR_FLOOR": "0.10",
    "VAL_BATCH_SIZE": str(2 * 64 * 1024 * 8),
    "NUM_SCHEDULED_ITERATIONS": "5790",
    "TARGET_LOSS": "2.92",
}
for key, value in TRACK2_DEFAULTS.items():
    os.environ.setdefault(key, value)  # before track_1_short.config is imported

if os.environ.get("COORD_DESCENT_TUNING", "0") == "1":
    import torch._inductor.config
    torch._inductor.config.coordinate_descent_tuning = True  # GPT-2 Medium track only (README, Rules)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_gpt  # noqa: E402  (train_gpt reads sys.argv[0], this file, for the run log)

if __name__ == "__main__":
    if os.environ.get("RANK", "0") == "0":
        print(f"track 2 settings: { {k: os.environ[k] for k in TRACK2_DEFAULTS} } "
              f"coordinate_descent_tuning={os.environ.get('COORD_DESCENT_TUNING', '0') == '1'}")
        # The exact-count corpus is every train shard on disk the loader cannot reach, so a partial download
        # changes the hints and the loss without any error. ldmberman's run had all 103.
        train_files = train_gpt.Hyperparameters().train_files
        n_train = len(train_gpt.glob.glob(train_files))
        if n_train != 103:
            print(f"WARNING: {n_train} train shards match {train_files}, not 103 (bash 182/run_node.sh "
                  f"prep downloads them); the exact-count hints, and so the val loss, will not match ldmberman's run")
    train_gpt.main()
