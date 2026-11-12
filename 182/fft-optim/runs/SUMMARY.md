# Queue results (2026-11-12)

## nanoGPT variant runs

```
--- t2-gpu-port
NameError: name 'sys' is not defined. Did you forget to import 'sys'?
--- t1-cpu-ngram
[rank3]: AssertionError("Loop-carried variable m_i has initial type <['64'], fp32> but is re-assigned to <['64'], fp64> in loop! Please make sure that the type stays consistent.")
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
train_gpt.py FAILED
--- t2-cpu-ngram
[rank0]: AssertionError("Loop-carried variable m_i has initial type <['64'], fp32> but is re-assigned to <['64'], fp64> in loop! Please make sure that the type stays consistent.")
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
train_gpt_medium_cpu.py FAILED
```

## Experiment 2 kill test: system identification (CPU)

L=1024, 16 channels, AR(1) rho=0.95 inputs, noise 10% of output std, 3000 steps, batch 8, warmup 5% + linear decay, LR grid ratio 3 (5 points + 1 edge extension), 3 seeds at the best LR.
Metric: excess eval MSE vs the noise-free output. Adam's median final = 2.572e-05.

| arm | best lr | final excess (median of 3) | steps to Adam's final (median) | ≤ half? | roughness >L/8 | ms/step | opt MB |
|---|---|---|---|---|---|---|---|
| adam | 0.001 | 2.572e-05 | None | no | 0.00807 | 8.44 | 0.13 |
| fftadam | 0.001 | 3.939e-05 | None | no | 0.0086 | 8.63 | 0.09 |
| fftadam_p0.25 | 0.001 | 2.134e-05 | 2900 | no | 0.00851 | 8.49 | 0.09 |
| fftadam_shared | 0.001 | 3.533e-05 | None | no | 0.00856 | 8.64 | 0.06 |
| fftadam_cap3 | 0.001 | 3.939e-05 | None | no | 0.0086 | 8.65 | 0.09 |
| fftadam_cap10 | 0.001 | 3.939e-05 | None | no | 0.0086 | 8.58 | 0.09 |

h* roughness (energy above L/8): 0.0081

**Verdict: NO-GO** (GO = some FFT-Adam variant reaches Adam's final loss in ≤ 1500 steps).

Notes (added after inspection):
- The κ=3 and κ=10 cap rows are bit-identical to uncapped FFT-Adam, and this is not a bug: on a random-gradient check the
  cap clips 7.7% of frequencies. Here most of the 513 frequencies sit at the high-frequency noise floor, so the median gain
  *is* the noise-floor gain and no frequency exceeds 3x it. A cap defined relative to the median cannot limit
  high-frequency boosting in this regime; a cap relative to, e.g., the DC or low-band gain would be needed to test the
  smoothness question.
- Partial whitening (p=0.25) has the best final loss, 2.13e-5 vs Adam's 2.57e-5 (~17% lower), but it reaches Adam's final
  only at step 2900 of 3000, far from the ≤1500 the kill test requires.
- The roughness of learned kernels (0.0081–0.0086) stays close to h*'s 0.0081 for every arm, so whitening did not roughen
  kernels at this noise level.
- The "None" in Adam's own steps-to-target row means one of its 3 seeds ended above the median target.

## Experiment 1 kill test: per-frequency vs tied Shampoo directions (ResNet-18 / CIFAR-10, AdamW 5 epochs)

Statistics summed over 485 steps; update = L^-1/4 M R^-1/4 with M = Adam's first moment at the end. AdamW run: test acc 0.845.

| layer | shape | cos(freq, tied) | cos(freq, M) | cos(tied, M) | DC share of M |
|---|---|---|---|---|---|
| 0 | [64, 3, 3, 3] | 0.4813 | 0.4340 | 0.8768 | 0.942 |
| 1 | [64, 64, 3, 3] | 0.7153 | 0.5057 | 0.8212 | 0.865 |
| 2 | [64, 64, 3, 3] | 0.7240 | 0.5987 | 0.9054 | 0.826 |
| 3 | [64, 64, 3, 3] | 0.7453 | 0.6248 | 0.9066 | 0.795 |
| 4 | [64, 64, 3, 3] | 0.7727 | 0.6511 | 0.9094 | 0.762 |
| 5 | [128, 64, 3, 3] | 0.7829 | 0.6672 | 0.9063 | 0.724 |
| 6 | [128, 128, 3, 3] | 0.8420 | 0.6870 | 0.8729 | 0.690 |
| 7 | [128, 128, 3, 3] | 0.8146 | 0.6608 | 0.8747 | 0.738 |
| 8 | [128, 128, 3, 3] | 0.8014 | 0.6474 | 0.8773 | 0.763 |
| 9 | [256, 128, 3, 3] | 0.8354 | 0.6834 | 0.8779 | 0.705 |
| 10 | [256, 256, 3, 3] | 0.8679 | 0.6998 | 0.8535 | 0.632 |
| 11 | [256, 256, 3, 3] | 0.8519 | 0.6824 | 0.8536 | 0.679 |
| 12 | [256, 256, 3, 3] | 0.8323 | 0.6539 | 0.8480 | 0.720 |
| 13 | [512, 256, 3, 3] | 0.8721 | 0.7044 | 0.8479 | 0.608 |
| 14 | [512, 512, 3, 3] | 0.8819 | 0.6676 | 0.7979 | 0.586 |
| 15 | [512, 512, 3, 3] | 0.8636 | 0.6228 | 0.7813 | 0.683 |
| 16 | [512, 512, 3, 3] | 0.8235 | 0.4916 | 0.6844 | 0.829 |

median cos(freq, tied) = 0.8235, min = 0.4813, max = 0.8819
**Verdict: GO** (GO if median < 0.9; NO-GO if every layer > 0.97).

### Experiment 1 main: ResNet-18 / CIFAR-10, 12 epochs, optimizer on 3x3 convs only

target for steps-to-target = worst arm's median final test acc = 0.9043

| arm | best lr | final test acc (mean ± std, n) | median | steps-to-target (median) | ms/step | opt state MB |
|---|---|---|---|---|---|---|
| adamw | 0.003 | 0.9058 ± 0.0018 (3) | 0.9049 | 1164 | 10.4 | 83.8 |
| sgd | 0.1 | 0.9047 ± 0.00091 (3) | 0.9043 | 1164.0 | 10.6 | 41.9 |
| muon | 0.02 | 0.9324 ± 0.0036 (3) | 0.9322 | 776 | 16.8 | 41.9 |
| muon_c | 0.00667 | 0.9265 ± 0.0029 (3) | 0.927 | 776 | 23.3 | 41.9 |
| shampoo | 0.003 | 0.9279 ± 0.0014 (3) | 0.9275 | 873 | 37.3 | 371.6 |
| soap | 0.027 | 0.9311 ± 0.0028 (3) | 0.9297 | 679 | 38.8 | 371.6 |
| tied | 0.003 | 0.9267 ± 0.00096 (3) | 0.9269 | 970 | 17.4 | 103.1 |
| shampoo_c | 0.009 | 0.9265 ± 0.00071 (3) | 0.9266 | 873 | 41.5 | 315.6 |
| soap_c | 0.009 | 0.9288 ± 0.0017 (3) | 0.9279 | 679 | 43.8 | 301.6 |

LR grids (seed 0): adamw: 0.000333->0.8874, 0.001->0.8999, 0.003->0.9049, 0.009->0.8957; sgd: 0.0333->0.8972, 0.1->0.904, 0.3->0.9002; muon: 0.00667->0.9268, 0.02->0.9322, 0.06->0.9312; muon_c: 0.00222->0.9172, 0.00667->0.9292, 0.02->0.9275, 0.06->0.9254; shampoo: 0.000333->0.8999, 0.001->0.9213, 0.003->0.9295, 0.009->0.9279; soap: 0.001->0.9148, 0.003->0.928, 0.009->0.9286, 0.027->0.9343; tied: 0.000333->0.8969, 0.001->0.9201, 0.003->0.9257, 0.009->0.9183; shampoo_c: 0.000333->0.891, 0.001->0.9148, 0.003->0.9245, 0.009->0.9271; soap_c: 0.001->0.9166, 0.003->0.9254, 0.009->0.9308, 0.027->0.9294

## Experiment 3 kill test: conditioning of per-mode Shampoo factors (FNO-1D, Burgers, AdamW)

L_k = sum over the last 500 steps of G_k G_k^H (Cin x Cin, complex), per layer and kept mode. AdamW run rel. L2 test error 0.001561.

| layer | median cond(L_k) | min | max | median cond(R_k) |
|---|---|---|---|---|
| 0 | inf | inf | inf | 2.47e+04 |
| 1 | 2.09e+05 | 6.19e+04 | inf | 2.92e+04 |
| 2 | 8.59e+03 | 2.83e+03 | inf | 4.01e+05 |
| 3 | 1.52e+05 | 1.21e+04 | inf | inf |

median cond(L_k) over all layers and modes = 9.593e+05
**Verdict: GO for the Shampoo variants** (threshold 10).

### Experiment 3 main: burgers

target for steps-to-target = worst arm's median final rel. L2 error = 0.001563

| arm | best lr | final rel. L2 error (mean ± std, n) | median | steps-to-target (median) | ms/step | opt state MB |
|---|---|---|---|---|---|---|
| adamw | 0.003 | 0.001231 ± 8.6e-05 (3) | 0.001199 | 5000 | 3.6 | 4.0 |
| mode_muon | 0.06 | 0.0009319 ± 1.1e-05 (3) | 0.0009367 | 5000 | 6.0 | 2.0 |
| mode_shampoo | 0.009 | 0.0009058 ± 8.2e-05 (3) | 0.000927 | 5000 | 15.6 | 11.0 |
| mode_soap | 0.003 | 0.0004153 ± 3.2e-05 (3) | 0.00041 | 4000 | 16.0 | 11.0 |
| muon_flat | 0.00667 | 0.001566 ± 0.00014 (3) | 0.001563 | 5000.0 | 5.8 | 2.0 |
| soap_flat | 0.009 | 0.0004118 ± 6.3e-06 (3) | 0.0004111 | 4250 | 11.3 | 67.2 |
| shampoo_shared | 0.000333 | 0.001195 ± 8.1e-06 (3) | 0.001199 | 4750 | 5.6 | 3.5 |

LR grids (seed 0): adamw: 0.000333->0.002327, 0.001->0.001559, 0.003->0.001166, 0.009->0.001833; mode_muon: 0.00667->0.002072, 0.02->0.001386, 0.06->0.0009367, 0.18->0.005922; mode_shampoo: 0.000333->0.002132, 0.001->0.00161, 0.003->0.001004, 0.009->0.000927; mode_soap: 0.001->0.0007229, 0.003->0.00041, 0.009->0.0005424; muon_flat: 0.00222->0.002476, 0.00667->0.001563, 0.02->0.001581, 0.06->0.001611; soap_flat: 0.001->0.0008503, 0.003->0.0004593, 0.009->0.0004185, 0.027->0.0007889; shampoo_shared: 0.000111->0.001273, 0.000333->0.0012, 0.001->0.001507, 0.003->0.002288

### Experiment 3 main: darcy

target for steps-to-target = worst arm's median final rel. L2 error = 0.01051

| arm | best lr | final rel. L2 error (mean ± std, n) | median | steps-to-target (median) | ms/step | opt state MB |
|---|---|---|---|---|---|---|
| adamw | 0.003 | 0.01048 ± 0.0001 (3) | 0.01051 | 4750.0 | 5.1 | 18.0 |
| mode_muon | 0.02 | 0.007602 ± 0.00012 (3) | 0.007656 | 3000 | 7.3 | 9.0 |
| mode_shampoo | 0.003 | 0.008818 ± 0.00025 (3) | 0.008959 | 3000 | 7.3 | 49.5 |
| mode_soap | 0.009 | 0.006647 ± 0.0001 (3) | 0.006671 | 1250 | 6.7 | 49.5 |
| muon_flat | 0.06 | 0.008315 ± 0.00013 (3) | 0.008302 | 3500 | 7.4 | 9.0 |
| soap_flat | 0.009 | 0.006901 ± 0.00011 (3) | 0.006939 | 1750 | 57.2 | 662.0 |
| shampoo_shared | 0.000333 | 0.00903 ± 0.00017 (3) | 0.008974 | 3750 | 6.9 | 13.6 |

LR grids (seed 0): adamw: 0.000333->0.01411, 0.001->0.011, 0.003->0.01055, 0.009->0.01155; mode_muon: 0.00667->0.0078, 0.02->0.00746, 0.06->0.008996; mode_shampoo: 0.000333->0.01121, 0.001->0.009237, 0.003->0.008959, 0.009->0.00921; mode_soap: 0.001->0.007814, 0.003->0.006703, 0.009->0.006535, 0.027->0.006931; muon_flat: 0.00667->0.01113, 0.02->0.00877, 0.06->0.008455, 0.18->0.0139; soap_flat: 0.001->0.009572, 0.003->0.00755, 0.009->0.006939, 0.027->0.007579; shampoo_shared: 0.000111->0.01078, 0.000333->0.009216, 0.001->0.009375, 0.003->0.01235

### Experiment 3 main: ns

target for steps-to-target = worst arm's median final rel. L2 error = 0.006132

| arm | best lr | final rel. L2 error (mean ± std, n) | median | steps-to-target (median) | ms/step | opt state MB |
|---|---|---|---|---|---|---|
| adamw | 0.003 | 0.006085 ± 0.00035 (3) | 0.006132 | 4750.0 | 4.8 | 18.0 |
| mode_muon | 0.02 | 0.00311 ± 9.5e-05 (3) | 0.0031 | 4000 | 7.2 | 9.0 |
| mode_shampoo | 0.003 | 0.004739 ± 0.00026 (3) | 0.004849 | 4000 | 6.7 | 49.5 |
| mode_soap | 0.027 | 0.001711 ± 4.4e-05 (3) | 0.00173 | 3250 | 7.3 | 49.5 |
| muon_flat | 0.06 | 0.004447 ± 4e-05 (3) | 0.004427 | 4500 | 6.9 | 9.0 |
| soap_flat | 0.009 | 0.001788 ± 7.9e-05 (3) | 0.001806 | 2750 | 57.0 | 662.0 |
| shampoo_shared | 0.000333 | 0.005533 ± 0.00022 (3) | 0.005619 | 4750 | 6.6 | 13.6 |

LR grids (seed 0): adamw: 0.000333->0.008306, 0.001->0.007068, 0.003->0.006132, 0.009->0.00721; mode_muon: 0.00667->0.003575, 0.02->0.0031, 0.06->0.003237; mode_shampoo: 0.000333->0.008981, 0.001->0.005985, 0.003->0.004446, 0.009->0.00448; mode_soap: 0.001->0.005445, 0.003->0.003174, 0.009->0.001825, 0.027->0.00173; muon_flat: 0.00667->0.006487, 0.02->0.004915, 0.06->0.004421, 0.18->0.008316; soap_flat: 0.001->0.005248, 0.003->0.002739, 0.009->0.001857, 0.027->0.002088; shampoo_shared: 0.000111->0.005833, 0.000333->0.005619, 0.001->0.006771, 0.003->0.008895

### Experiment 4: rescale mode (Adam in a fixed basis), 30M GPT, 300M tokens

target for steps-to-target = worst arm's median final val loss = 4.169

| arm | best lr | final val loss (mean ± std, n) | median | steps-to-target (median) | ms/step | opt state MB |
|---|---|---|---|---|---|---|
| identity | 0.002 | 4.169 ± 0.0076 (3) | 4.169 | 4538.5 | 40.6 | 81.0 |
| dct | 0.002 | 4.174 ± 0.064 (3) | 4.149 | 4500.0 | 41.9 | 81.0 |
| random | 0.002 | 4.183 ± 0.045 (3) | 4.159 | 4500.0 | 41.5 | 81.0 |

LR grids (seed 0): identity: 0.000667->4.211, 0.002->4.169, 0.006->6.57; dct: 0.000667->4.203, 0.002->4.149, 0.006->6.575; random: 0.000667->4.217, 0.002->4.159, 0.006->6.366

DCT - random: mean -0.0092 nats, Welch t = -0.20 (|t| < 2.78 = within noise at n=3+3)

**Finding: DCT and random are within seed noise.**

### Truncate mode (top 25% of columns of G Q by norm), at each basis's rescale-best LR

| basis | final val loss (mean ± std, n) | ms/step |
|---|---|---|
| identity | 4.4337 ± 0.2006 (3) | 43.5 |
| dct | 4.2946 ± 0.0044 (3) | 44.2 |
| random | 4.3285 ± 0.0195 (3) | 44.2 |

## Queue status
```
step	start	end	exit
t2-gpu-port	2026-11-12T04:50:29Z	2026-11-12T04:56:28Z	0
t1-cpu-ngram	2026-11-12T04:56:28Z	2026-11-12T04:58:52Z	1
t2-cpu-ngram	2026-11-12T04:58:52Z	2026-11-12T05:01:27Z	1
prep	2026-11-12T05:01:27Z	2026-11-12T05:01:28Z	0
exp2	2026-11-12T05:01:28Z	2026-11-12T05:02:44Z	0
exp1	2026-11-12T05:02:44Z	2026-11-12T05:08:12Z	0
exp3	2026-11-12T05:08:12Z	2026-11-12T05:47:01Z	0
exp4	2026-11-12T05:47:01Z	2026-11-12T06:04:46Z	0
```

## retry: t1-cpu-ngram-retry (exit 0)
```
step:685/685 val_loss:3.3008 train_time:23196ms step_avg:33.86ms
peak memory allocated: 53983 MiB reserved: 60394 MiB
```

## retry: t2-cpu-ngram-retry (exit 1)
```
[rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.07 GiB. GPU 1 has a total capacity of 79.10 GiB of which 741.88 MiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 73.31 GiB is allocated by PyTorch, with 5.72 GiB allocated in private pools (e.g., CUDA Graphs), and 503.26 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
[rank3]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.07 GiB. GPU 3 has a total capacity of 79.10 GiB of which 1021.88 MiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 73.12 GiB is allocated by PyTorch, with 5.72 GiB allocated in private pools (e.g., CUDA Graphs), and 414.45 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.07 GiB. GPU 0 has a total capacity of 79.10 GiB of which 881.88 MiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 73.31 GiB is allocated by PyTorch, with 5.72 GiB allocated in private pools (e.g., CUDA Graphs), and 362.77 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
[rank2]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.07 GiB. GPU 2 has a total capacity of 79.10 GiB of which 1.08 GiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 73.22 GiB is allocated by PyTorch, with 5.72 GiB allocated in private pools (e.g., CUDA Graphs), and 228.44 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
```

## retry: t2-gpu-port ACT_MEM_BUDGET=0.6 (exit 2)
```
```

## retry: t2-gpu-port ACT_MEM_BUDGET=0.4 (exit 2)
```
```
