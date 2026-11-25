"""Does the 1.30x map-sequence speedup survive a compiled record baseline? (one H100, MLP group)

The standard microbenchmark (bench.py, matrix.* cases) times the record's maps with reference_cascade, an
uncompiled kernel-for-kernel copy, while the trainer runs the record's maps inside torch.compile'd
anvil_cascade. This script times the record's copy and the Gram-space cascade both eager and under
torch.compile (dynamic=False), each as a CUDA-graph replay, interleaved, median and p10/p90 over --reps.

    python 182/optim-kernel/compile_bench.py [--reps 300]
"""
import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench as BB  # noqa: E402  (the standard microbenchmark's helpers and constants)
import torch  # noqa: E402

AK = BB.AK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device("cuda")
    timer = BB.Timer(dev, args.reps, args.warmup, graph=True)
    X = BB.synthetic(BB.chunk_shape("mlp_bank", 1), "p05", dev, 2)
    split = X.size(-2) > BB.SPLIT_MIN_ROWS

    def rec(X):
        return AK.reference_cascade(X, AK.ops_for(X), BB.MAPS, BB.MARGIN, BB.EPS, split)

    def gram(X):
        return AK.gram_cascade(X, AK.ops_for(X, True), BB.MAPS, (2, 2, 2), BB.MARGIN, BB.EPS)

    rec_c = torch.compile(rec, dynamic=False)
    gram_c = torch.compile(gram, dynamic=False)
    fns = {"record_eager": lambda: rec(X), "record_compiled": lambda: rec_c(X),
           "gram_eager": lambda: gram(X), "gram_compiled": lambda: gram_c(X)}
    print(f"COMPILE_BENCH gpu={torch.cuda.get_device_name(dev).replace(' ', '_')} torch={torch.__version__} "
          f"shape={tuple(X.shape)} reps={args.reps}", flush=True)
    with torch.no_grad():
        outs = {k: f().float() for k, f in fns.items()}
        for k in ("record_compiled", "gram_compiled"):
            ref = outs[k.replace("compiled", "eager")]
            print(f"COMPILE_BENCH check {k} vs eager: rel_fro={((outs[k]-ref).norm()/ref.norm()).item():.2e}", flush=True)
        for a, b in [("record_eager", "gram_eager"), ("record_compiled", "gram_eager"),
                     ("record_compiled", "gram_compiled"), ("record_eager", "record_compiled"),
                     ("gram_eager", "gram_compiled")]:
            ta, tb = timer.pair(fns[a], fns[b])
            print(f"COMPILE_BENCH {a} {ta['ms']:.4f} [{ta['p10']:.4f},{ta['p90']:.4f}] ms | {b} {tb['ms']:.4f} "
                  f"[{tb['p10']:.4f},{tb['p90']:.4f}] ms | ratio {ta['ms']/tb['ms']:.3f}", flush=True)


if __name__ == "__main__":
    main()
