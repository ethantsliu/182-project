"""GPU / NCCL sanity check for a fresh node (a few seconds; run under torchrun, one process per GPU):

    torchrun --standalone --nproc_per_node=8 182/tools/node_check.py

Per GPU: bf16 GEMM throughput (8192^3) and SM clock; across GPUs: all_reduce bus bandwidth (256 MiB
bf16). Flags any GPU more than 5% below the median GEMM throughput -- a slow or throttled GPU slows
every step of a data-parallel run. Reference on a healthy 8xH100 SXM node: roughly 600-750 TFLOP/s
bf16 GEMM per GPU and ~350-480 GB/s all_reduce bus bandwidth (indicative, not a spec).
"""
import os
import subprocess
import time

import torch
import torch.distributed as dist


def gemm_tflops(n=8192, iters=50):
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return 2 * n ** 3 * iters / (time.perf_counter() - t0) / 1e12


def allreduce_busbw(numel=128 * 1024 * 1024, iters=20):
    x = torch.ones(numel, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    w = dist.get_world_size()
    return x.numel() * x.element_size() * 2 * (w - 1) / w / dt / 1e9


def main():
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    tf = gemm_tflops()
    clk = subprocess.run(["nvidia-smi", "-i", str(local), "--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    res = [None] * world
    dist.all_gather_object(res, dict(rank=rank, gpu=local, name=torch.cuda.get_device_name(), tflops=tf, clk_temp=clk))
    bw = allreduce_busbw() if world > 1 else float("nan")
    if rank == 0:
        med = sorted(r["tflops"] for r in res)[world // 2]
        for r in res:
            flag = "  <-- SLOW" if r["tflops"] < 0.95 * med else ""
            print(f"gpu {r['gpu']} {r['name']}: bf16 GEMM {r['tflops']:.0f} TFLOP/s  (sm clock MHz, temp C: {r['clk_temp']}){flag}")
        print(f"all_reduce bus bandwidth ({world} ranks, 256 MiB bf16): {bw:.0f} GB/s")
        print(f"torch {torch.__version__}, NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
