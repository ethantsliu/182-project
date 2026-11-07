"""Build synthetic_nsys.sqlite: a tiny trace in the schema `nsys export --type sqlite` writes (the
subset of tables and columns analyze_nsys.py reads), with hand-computable answers.

Timeline (ns). Five identical steps k = 0..4 on device 0, GPU base B_k = 10_000_000 + 400_000 k.
NVTX "step 300".."step 302" annotate k = 1, 2, 3 (k = 0 and 4 are unannotated neighbours). Each step,
relative to B_k (stream in brackets):
    [7] sm90_xmma_gemm (cuBLAS)     [0, 100_000)        graph node; rule-file GEMM (1000^3 -> 2e9 flop)
    [7] triton_poi_fused_add_mul_0  [110_000, 130_000)
    [9] ncclDevKernel_AllReduce     [120_000, 220_000)  overlaps the triton kernel for 10 us
    [7] nvjet_tst GEMM              [250_000, 300_000)  launched inside NVTX "aten::mm ... sizes = [[2048, 1024], [1024, 4096]]"
    [7] flash::FlashAttnFwdSm90     [300_000, 330_000)  (cutlass in its name: must not count as a GEMM)
    H2D memcpy 1 MiB                [340_000, 350_000)
Expected per annotated step: window 400 us, 5 kernels (1 from a graph), busy 290 us, idle 110 us in three
gaps (10, 30, 70 us), NCCL 100 us of which 90 us exposed, GEMM 150 us, 1 memcpy. GEMM throughput:
nvjet 2*2048*1024*4096 flop / 50 us = 343.597 TFLOP/s; xmma (with the rule) 2e9 / 100 us = 20 TFLOP/s.
Device 1 carries two stray kernels (device selection must pick device 0).
"""
import os
import sqlite3
import sys

PID = 1234
GPID = PID << 24
GTID = GPID | PID

NAMES = {
    1: "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas",
    2: "triton_poi_fused_add_mul_0",
    3: "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
    4: "ncclDevKernel_AllReduce_Sum_bf16_RING_LL",
    5: "nvjet_tst_128x256_64x4_1x2_h_bz_coopB_TNT",
    6: "void flash::FlashAttnFwdSm90<cutlass::gemm::KernelTmaWarpSpecializedCooperative, 128, 128>(Params)",
    7: "FlashAttnFwdSm90",
    8: "cudaGraphLaunch_v10000",
    9: "cudaLaunchKernel_v7000",
    10: "step 300", 11: "step 301", 12: "step 302",
    13: "aten::mm, seq = 7, op_id = 9, sizes = [[2048, 1024], [1024, 4096]], input_op_ids = [(1, 0), (2, 0)]",
    14: "stray_kernel",
}

# (demangled id, short id, start, end, stream, graph node, launch api id)
STEP_KERNELS = [
    (1, 1, 0, 100_000, 7, 1, 8),
    (2, 2, 110_000, 130_000, 7, None, 9),
    (3, 4, 120_000, 220_000, 9, None, 9),
    (5, 5, 250_000, 300_000, 7, None, 9),
    (6, 7, 300_000, 330_000, 7, None, 9),
]


def build(path):
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    c = con.cursor()
    c.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
    c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER NOT NULL, end INTEGER NOT NULL,
        deviceId INTEGER NOT NULL, contextId INTEGER, streamId INTEGER NOT NULL, correlationId INTEGER,
        globalPid INTEGER, demangledName INTEGER NOT NULL, shortName INTEGER NOT NULL, mangledName INTEGER,
        gridX INTEGER, gridY INTEGER, gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER,
        registersPerThread INTEGER, staticSharedMemory INTEGER, dynamicSharedMemory INTEGER, graphNodeId INTEGER)""")
    c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER NOT NULL, end INTEGER NOT NULL,
        eventClass INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER, returnValue INTEGER)""")
    c.execute("""CREATE TABLE NVTX_EVENTS (start INTEGER NOT NULL, end INTEGER, eventType INTEGER NOT NULL,
        rangeId INTEGER, category INTEGER, color INTEGER, text TEXT, globalTid INTEGER, endGlobalTid INTEGER,
        textId INTEGER, domainId INTEGER)""")
    c.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER NOT NULL, end INTEGER NOT NULL,
        deviceId INTEGER NOT NULL, streamId INTEGER, correlationId INTEGER, globalPid INTEGER, bytes INTEGER,
        copyKind INTEGER, srcKind INTEGER, dstKind INTEGER)""")
    c.executemany("INSERT INTO StringIds VALUES (?, ?)", sorted(NAMES.items()))
    corr = 100
    for k in range(5):
        B = 10_000_000 + 400_000 * k
        C = B - 50_000  # CPU launches run ahead of the GPU
        if 1 <= k <= 3:
            # the step range uses textId (newer nsys); the aten range uses inline text (older nsys)
            c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text, globalTid, textId) VALUES (?,?,?,?,?,?)",
                      (C - 5_000, C + 20_000, 59, None, GTID, 9 + k))
        for j, (dn, sn, s, e, stream, node, api) in enumerate(STEP_KERNELS):
            corr += 1
            launch = C + j * 1_000
            c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, eventClass, globalTid, correlationId, nameId, returnValue) "
                      "VALUES (?,?,?,?,?,?,0)", (launch, launch + 500, 1, GTID, corr, api))
            if dn == 5:
                c.execute("INSERT INTO NVTX_EVENTS (start, end, eventType, text, globalTid) VALUES (?,?,?,?,?)",
                          (launch - 100, launch + 600, 59, NAMES[13], GTID))
            c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, deviceId, contextId, streamId, correlationId, "
                      "globalPid, demangledName, shortName, mangledName, gridX, gridY, gridZ, blockX, blockY, blockZ, graphNodeId) "
                      "VALUES (?,?,0,1,?,?,?,?,?,NULL,132,1,1,384,1,1,?)",
                      (B + s, B + e, stream, corr, GPID, dn, sn, node))
        c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY (start, end, deviceId, streamId, bytes, copyKind) VALUES (?,?,0,11,?,1)",
                  (B + 340_000, B + 350_000, 1 << 20))
    for i in range(2):
        c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, deviceId, streamId, correlationId, globalPid, "
                  "demangledName, shortName) VALUES (?,?,1,7,?,?,14,14)", (10_000_000 + i * 1000, 10_000_500 + i * 1000, 9000 + i, GPID))
    con.commit()
    con.close()


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_nsys.sqlite"))
