# FNO follow-up results

LR selection: held-out validation only; final test: independent PDE samples.
Confirmation seeds: 101, 102, 103. Original 200-sample test sets are unused.
Times exclude compilation (these FNO runs are eager); train time excludes evaluation.
Targets below are fixed exploratory thresholds, not selected from this round.

## controls-darcy

Fixed validation error target: 0.01051.

| arm | refresh | best LR | test rel L2 mean ± std | successful / planned | train s | ms/step | spectral state MB | peak allocated MB | target reached / planned | train s to target (reached only) |
|---|---:|---:|---|---|---|---|---|---|---|---|
| adamw | 10 | 0.003 | 0.012777 ± 0.00037 | 3/3 | 20.018 ± 1.2 | 5.0044 ± 0.29 | 18 ± 0 | 332 ± 0 | 0/3 | — |
| adamw_matched | 10 | 0.003 | 0.011979 ± 0.00031 | 3/3 | 20.331 ± 1.1 | 5.0828 ± 0.28 | 18 ± 0 | 332 ± 0 | 0/3 | — |
| complex_adam | 10 | 0.009 | 0.012232 ± 0.00044 | 3/3 | 21.131 ± 1.1 | 5.2828 ± 0.28 | 13.5 ± 0 | 328 ± 0 | 0/3 | — |
| soap_shared | 10 | 0.003 | 0.0086457 ± 7.1e-05 | 3/3 | 26.443 ± 1.3 | 6.6108 ± 0.32 | 13.625 ± 0 | 328.13 ± 0 | 3/3 | 14.323 ± 1.3 |
| mode_soap | 10 | 0.009 | 0.0077332 ± 0.00014 | 3/3 | 27.121 ± 1.2 | 6.7803 ± 0.31 | 49.5 ± 0 | 597.73 ± 0 | 3/3 | 11.476 ± 1.3 |
| soap_mixed | 10 | 0.009 | 0.0088107 ± 0.00035 | 3/3 | 28.099 ± 1.3 | 7.0248 ± 0.33 | 49.5 ± 0 | 601.29 ± 0 | 3/3 | 18.882 ± 1.9 |

All tuning points, validation/time curves, failure records and per-job consoles are retained in JSON/log files.

## ns-boundary

Fixed validation error target: 0.006132.

| arm | refresh | best LR | test rel L2 mean ± std | successful / planned | train s | ms/step | spectral state MB | peak allocated MB | target reached / planned | train s to target (reached only) |
|---|---:|---:|---|---|---|---|---|---|---|---|
| mode_soap | 10 | 0.027 | 0.0021595 ± 0.00013 | 3/3 | 27.757 ± 1.8 | 6.9392 ± 0.46 | 49.5 ± 0 | 597.73 ± 0 | 3/3 | 20.371 ± 0.89 |
| adamw | 10 | 0.003 | 0.0077202 ± 0.00069 | 3/3 | 21.18 ± 1.8 | 5.2949 ± 0.46 | 18 ± 0 | 332 ± 0 | 0/3 | — |

All tuning points, validation/time curves, failure records and per-job consoles are retained in JSON/log files.

## refresh-darcy

Fixed validation error target: 0.01051.

| arm | refresh | best LR | test rel L2 mean ± std | successful / planned | train s | ms/step | spectral state MB | peak allocated MB | target reached / planned | train s to target (reached only) |
|---|---:|---:|---|---|---|---|---|---|---|---|
| adamw | 10 | 0.003 | 0.012777 ± 0.00037 | 3/3 | 20.018 ± 1.2 | 5.0044 ± 0.29 | 18 ± 0 | 332 ± 0 | 0/3 | — |
| soap_flat | 10 | 0.009 | 0.0080544 ± 0.00024 | 3/3 | 226.48 ± 2 | 56.619 ± 0.49 | 662 ± 0 | 1375.2 ± 0 | 3/3 | 121.36 ± 18 |
| mode_soap | 10 | 0.009 | 0.0077332 ± 0.00014 | 3/3 | 27.687 ± 1.2 | 6.9218 ± 0.29 | 49.5 ± 0 | 597.73 ± 0 | 3/3 | 11.835 ± 1.9 |
| mode_soap | 25 | 0.003 | 0.0078807 ± 0.0004 | 3/3 | 26.57 ± 1.7 | 6.6424 ± 0.44 | 49.5 ± 0 | 597.73 ± 0 | 3/3 | 9.9323 ± 1.5 |
| mode_soap | 50 | 0.003 | 0.0081159 ± 0.0002 | 3/3 | 25.227 ± 1.2 | 6.3068 ± 0.3 | 49.5 ± 0 | 597.73 ± 0 | 3/3 | 10.824 ± 1.2 |
| mode_soap | 100 | 0.001 | 0.0088961 ± 0.0002 | 3/3 | 24.819 ± 0.84 | 6.2046 ± 0.21 | 49.5 ± 0 | 597.73 ± 0 | 3/3 | 15.484 ± 0.45 |

All tuning points, validation/time curves, failure records and per-job consoles are retained in JSON/log files.

## refresh-ns

Fixed validation error target: 0.006132.

| arm | refresh | best LR | test rel L2 mean ± std | successful / planned | train s | ms/step | spectral state MB | peak allocated MB | target reached / planned | train s to target (reached only) |
|---|---:|---:|---|---|---|---|---|---|---|---|
| adamw | 10 | 0.003 | 0.0077202 ± 0.00069 | 3/3 | 21.18 ± 1.8 | 5.2949 ± 0.46 | 18 ± 0 | 332 ± 0 | 0/3 | — |
| soap_flat | 10 | 0.009 | 0.002289 ± 0.00011 | 3/3 | 227.07 ± 2.4 | 56.767 ± 0.59 | 662 ± 0 | 1375.2 ± 0 | 3/3 | 125.32 ± 1.4 |
| mode_soap | 10 | 0.027 | 0.0021595 ± 0.00013 | 3/3 | 27.757 ± 1.8 | 6.9392 ± 0.46 | 49.5 ± 0 | 597.73 ± 0 | 3/3 | 20.371 ± 0.89 |
| mode_soap | 25 | 0.009 | 0.0025703 ± 0.00011 | 3/3 | 26.208 ± 0.85 | 6.5521 ± 0.21 | 49.5 ± 0 | 597.73 ± 0 | 3/3 | 14.15 ± 0.57 |
| mode_soap | 50 | 0.027 | 0.0023484 ± 0.0001 | 3/3 | 24.945 ± 1.2 | 6.2362 ± 0.31 | 49.5 ± 0 | 597.73 ± 0 | 3/3 | 17.639 ± 0.82 |
| mode_soap | 100 | 0.009 | 0.0027976 ± 0.0003 | 3/3 | 25.623 ± 1.3 | 6.4057 ± 0.33 | 49.5 ± 0 | 597.73 ± 0 | 3/3 | 12.032 ± 1.3 |

All tuning points, validation/time curves, failure records and per-job consoles are retained in JSON/log files.
