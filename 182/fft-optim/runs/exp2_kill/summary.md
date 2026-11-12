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
