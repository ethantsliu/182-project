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
