# PBC Laplace3D MPI Benchmark Results

**Date:** 2026-04-10
**Cluster:** Rusty (Genoa nodes, AMD EPYC 9474F 48-Core, 96 cores/node, InfiniBand)
**Code:** FIDMK @ commit 2a38ea2
**Kernel:** Laplace3D, pot+grad (src+trg), PBC enabled
**Build:** GCC 13.3 + OpenMPI 4.1.8, `-O3 -march=native`

## 1. Accuracy: MPI Decomposition Correctness

### Methodology

This test verifies that distributing the problem across multiple MPI ranks
produces the same result as a single-rank run. It does **not** test absolute
accuracy against an independent reference (Ewald); that is covered separately
in `bench_pbc_accuracy.cpp`.

**Setup:**
- N total particles (sources and targets) are generated on rank 0 with a
  uniform random distribution in `[0.01, 0.99]^3`, with charge neutrality
  enforced.
- Particles are scattered evenly: each of P ranks receives N/P sources and
  N/P targets via `MPI_Scatter`.
- Each rank calls `pdmk_tree_create` / `pdmk_tree_eval` with `MPI_COMM_WORLD`.
- Results are gathered back to rank 0 via `MPI_Gather`.
- On rank 0, a **serial reference** is computed by running the same problem
  with all N particles on a single rank using `MPI_COMM_SELF`.
- The relative L2 error between MPI and serial results is reported for
  potential and gradient, at sources and targets.

**Parameters swept:**
- N = 2000, 5000, 10000
- eps = 1e-3, 1e-6, 1e-9 (3, 6, 9 digit precision)
- MPI ranks = 1, 2, 4, 8, 16, 32 (all on 1 Genoa node, 96 cores total)
- OMP threads per rank = 96 / ranks

### Results

**1 MPI rank** (baseline — MPI and serial reference are identical code paths):

All errors are exactly 0. This is expected since both runs execute the same
serial code with identical data.

**2-32 MPI ranks** (worst-case error across all N and eps values):

| MPI Ranks | OMP/rank | pot_src     | pot_trg     | grad_src    | grad_trg    |
|-----------|----------|-------------|-------------|-------------|-------------|
| 2         | 48       | 3.38e-16    | 3.12e-16    | 6.98e-17    | 7.50e-17    |
| 4         | 24       | 4.25e-16    | 3.97e-16    | 8.28e-17    | 8.94e-17    |
| 8         | 12       | 9.07e-16    | 8.41e-16    | 3.75e-16    | 1.85e-16    |
| 16        | 6        | 1.49e-15    | 1.45e-15    | 5.29e-16    | 7.72e-16    |
| 32        | 3        | 1.48e-15    | 1.45e-15    | 4.15e-16    | 8.56e-16    |

Note: with 8 ranks and N=2000 (250 particles/rank), errors are exactly 0.
At such small local sizes, SCTL's distributed tree likely degenerates to a
trivially identical decomposition as the serial case.

**Conclusion:** MPI decomposition is correct. All errors are at double-precision
roundoff level (~1e-16 to ~1e-15), caused by floating-point reordering of
summations across ranks. There is no accuracy degradation from MPI distribution.

### Full Accuracy Data

Format: `N, mpi_ranks, digits, eps, pot_src_l2, pot_trg_l2, grad_src_l2, grad_trg_l2`

**2 ranks:**
```
 2000, 2, 3, 1e-03, 1.23e-16, 1.18e-16, 2.45e-17, 3.36e-17
 2000, 2, 6, 1e-06, 1.66e-16, 1.73e-16, 3.99e-17, 4.64e-17
 2000, 2, 9, 1e-09, 1.83e-16, 1.71e-16, 4.28e-17, 5.56e-17
 5000, 2, 3, 1e-03, 1.69e-16, 1.69e-16, 3.91e-17, 2.43e-17
 5000, 2, 6, 1e-06, 2.68e-16, 2.61e-16, 5.40e-17, 3.61e-17
 5000, 2, 9, 1e-09, 3.38e-16, 3.12e-16, 6.98e-17, 4.40e-17
10000, 2, 3, 1e-03, 2.16e-16, 2.13e-16, 2.02e-17, 3.99e-17
10000, 2, 6, 1e-06, 2.83e-16, 2.76e-16, 3.27e-17, 5.42e-17
10000, 2, 9, 1e-09, 3.05e-16, 2.98e-16, 4.86e-17, 7.50e-17
```

**4 ranks:**
```
 2000, 4, 3, 1e-03, 1.31e-16, 1.31e-16, 2.58e-17, 3.40e-17
 2000, 4, 6, 1e-06, 1.33e-16, 1.30e-16, 3.19e-17, 4.40e-17
 2000, 4, 9, 1e-09, 1.89e-16, 2.01e-16, 4.31e-17, 6.06e-17
 5000, 4, 3, 1e-03, 2.42e-16, 2.40e-16, 4.64e-17, 2.96e-17
 5000, 4, 6, 1e-06, 3.36e-16, 3.23e-16, 6.48e-17, 4.29e-17
 5000, 4, 9, 1e-09, 4.25e-16, 3.97e-16, 8.28e-17, 5.11e-17
10000, 4, 3, 1e-03, 2.32e-16, 2.29e-16, 2.17e-17, 4.30e-17
10000, 4, 6, 1e-06, 3.09e-16, 3.20e-16, 3.52e-17, 6.36e-17
10000, 4, 9, 1e-09, 3.02e-16, 3.02e-16, 5.18e-17, 8.94e-17
```

**8 ranks:**
```
 2000, 8, 3, 1e-03, 0        , 0        , 0        , 0
 2000, 8, 6, 1e-06, 0        , 0        , 0        , 0
 2000, 8, 9, 1e-09, 0        , 0        , 0        , 0
 5000, 8, 3, 1e-03, 4.20e-16, 3.93e-16, 9.43e-17, 5.73e-17
 5000, 8, 6, 1e-06, 6.31e-16, 5.93e-16, 1.87e-16, 1.11e-16
 5000, 8, 9, 1e-09, 9.07e-16, 8.41e-16, 3.75e-16, 1.85e-16
10000, 8, 3, 1e-03, 3.25e-16, 3.19e-16, 3.19e-17, 6.16e-17
10000, 8, 6, 1e-06, 4.53e-16, 4.66e-16, 6.56e-17, 1.14e-16
10000, 8, 9, 1e-09, 6.16e-16, 5.92e-16, 1.29e-16, 1.66e-16
```

**16 ranks:**
```
 2000, 16, 3, 1e-03, 5.26e-16, 5.27e-16, 1.37e-16, 1.88e-16
 2000, 16, 6, 1e-06, 9.61e-16, 1.03e-15, 3.13e-16, 4.30e-16
 2000, 16, 9, 1e-09, 1.39e-15, 1.44e-15, 5.29e-16, 7.72e-16
 4992, 16, 3, 1e-03, 6.38e-16, 6.23e-16, 1.19e-16, 1.01e-16
 4992, 16, 6, 1e-06, 1.02e-15, 9.89e-16, 2.50e-16, 2.04e-16
 4992, 16, 9, 1e-09, 1.49e-15, 1.45e-15, 4.15e-16, 3.77e-16
10000, 16, 3, 1e-03, 3.82e-16, 3.78e-16, 3.75e-17, 7.26e-17
10000, 16, 6, 1e-06, 5.96e-16, 6.01e-16, 7.60e-17, 1.45e-16
10000, 16, 9, 1e-09, 7.70e-16, 7.57e-16, 1.49e-16, 2.22e-16
```

**32 ranks:**
```
 1984, 32, 3, 1e-03, 5.23e-16, 5.38e-16, 1.05e-16, 2.24e-16
 1984, 32, 6, 1e-06, 9.56e-16, 1.05e-15, 2.33e-16, 4.68e-16
 1984, 32, 9, 1e-09, 1.43e-15, 1.43e-15, 3.82e-16, 8.56e-16
 4992, 32, 3, 1e-03, 6.75e-16, 6.67e-16, 1.22e-16, 1.02e-16
 4992, 32, 6, 1e-06, 1.03e-15, 1.00e-15, 2.51e-16, 2.05e-16
 4992, 32, 9, 1e-09, 1.48e-15, 1.45e-15, 4.15e-16, 3.76e-16
 9984, 32, 3, 1e-03, 4.08e-16, 4.03e-16, 5.65e-17, 7.35e-17
 9984, 32, 6, 1e-06, 5.95e-16, 5.99e-16, 1.18e-16, 1.54e-16
 9984, 32, 9, 1e-09, 7.91e-16, 7.85e-16, 2.08e-16, 2.45e-16
```

(N for 16/32 ranks is rounded down to the nearest multiple of rank count.)

## 2. Speed: Strong Scaling

### Methodology

This test measures wall-clock time for `pdmk_tree_create` and `pdmk_tree_eval`
as MPI rank count increases, with a fixed total problem size (strong scaling).

**Setup:**
- N total source particles generated on rank 0 (uniform random in
  `[0.01, 0.99]^3`, charge-neutral), scattered to P ranks (N/P per rank).
  Same N target particles, also scattered.
- `MPI_Barrier` before and after each timed section to measure wall-clock
  across all ranks.
- 1 warmup eval, then 3 timed evals; minimum eval time reported.
- Configurations:
  - 1 node: 1, 2, 4, 8 MPI ranks (OMP threads = 96/ranks)
  - 2 nodes: 2, 4, 8, 16 MPI ranks (OMP threads = 96/ranks_per_node)
  - 4 nodes: 4, 8, 16, 32 MPI ranks (OMP threads = 96/ranks_per_node)
- N = 500K, 1M, 2M; eps = 1e-3, 1e-6
- `n_per_leaf = 500`, `pgh = pot+grad` for both src and trg

### 2a. Single-Node Scaling (1 Genoa node, 96 cores)

**N=1M, eps=1e-3:**

| Ranks | OMP/rank | Create (s) | Eval (s) | Total (s) | Mpts/s | vs 1-rank |
|-------|----------|-----------|----------|-----------|--------|-----------|
| 1     | 96       | 0.18      | 0.17     | 0.35      | 5.79   | 1.00x     |
| 2     | 48       | 10.1      | 1.75     | 11.9      | 0.57   | 0.03x     |
| 4     | 24       | 30.1      | 5.26     | 35.4      | 0.19   | 0.01x     |
| 8     | 12       | 49.3      | 7.81     | 57.1      | 0.13   | 0.006x    |

**N=1M, eps=1e-6:**

| Ranks | OMP/rank | Create (s) | Eval (s) | Total (s) | Mpts/s | vs 1-rank |
|-------|----------|-----------|----------|-----------|--------|-----------|
| 1     | 96       | 0.19      | 0.37     | 0.56      | 2.71   | 1.00x     |
| 2     | 48       | 13.6      | 1.11     | 14.7      | 0.90   | 0.04x     |
| 4     | 24       | 29.9      | 6.19     | 36.1      | 0.16   | 0.02x     |
| 8     | 12       | 49.6      | 9.55     | 59.2      | 0.10   | 0.009x    |

**N=2M, eps=1e-3:**

| Ranks | OMP/rank | Create (s) | Eval (s) | Total (s) | Mpts/s | vs 1-rank |
|-------|----------|-----------|----------|-----------|--------|-----------|
| 1     | 96       | 0.43      | 0.28     | 0.71      | 7.06   | 1.00x     |
| 2     | 48       | 11.3      | 1.21     | 12.5      | 1.65   | 0.06x     |
| 4     | 24       | 31.3      | 5.16     | 36.4      | 0.39   | 0.02x     |
| 8     | 12       | 50.5      | 8.88     | 59.4      | 0.23   | 0.01x     |

### 2b. Multi-Node Scaling (N=1M, eps=1e-3)

| Nodes | Ranks | OMP/rank | Create (s) | Eval (s) | Total (s) | Mpts/s |
|-------|-------|----------|-----------|----------|-----------|--------|
| 1     | 1     | 96       | 0.18      | 0.17     | 0.35      | 5.79   |
| 1     | 2     | 48       | 10.1      | 1.75     | 11.9      | 0.57   |
| 2     | 2     | 96       | 33.4      | 3.68     | 37.1      | 0.27   |
| 2     | 4     | 48       | 68.2      | 7.32     | 75.5      | 0.14   |
| 4     | 4     | 96       | 134.7     | 16.3     | 151.0     | 0.06   |
| 4     | 8     | 48       | 174.8     | 24.7     | 199.5     | 0.04   |
| 4     | 16    | 24       | 187.4     | 26.2     | 213.6     | 0.04   |
| 4     | 32    | 12       | 248.4     | 33.4     | 281.8     | 0.03   |

### 2c. Full 1-Node Results (all N and eps)

| N     | Ranks | OMP | eps  | Create (s) | Eval (s) | Total (s) | Mpts/s |
|-------|-------|-----|------|-----------|----------|-----------|--------|
| 500K  | 1     | 96  | 1e-3 | 0.15      | 0.06     | 0.20      | 8.74   |
| 500K  | 1     | 96  | 1e-6 | 0.13      | 0.11     | 0.25      | 4.41   |
| 1M    | 1     | 96  | 1e-3 | 0.18      | 0.17     | 0.35      | 5.79   |
| 1M    | 1     | 96  | 1e-6 | 0.19      | 0.37     | 0.56      | 2.71   |
| 2M    | 1     | 96  | 1e-3 | 0.43      | 0.28     | 0.71      | 7.06   |
| 2M    | 1     | 96  | 1e-6 | 0.45      | 0.51     | 0.96      | 3.93   |
| 500K  | 2     | 48  | 1e-3 | 15.3      | 2.78     | 18.1      | 0.18   |
| 500K  | 2     | 48  | 1e-6 | 15.8      | 0.50     | 16.3      | 1.00   |
| 1M    | 2     | 48  | 1e-3 | 10.1      | 1.75     | 11.9      | 0.57   |
| 1M    | 2     | 48  | 1e-6 | 13.6      | 1.11     | 14.7      | 0.90   |
| 2M    | 2     | 48  | 1e-3 | 11.3      | 1.21     | 12.5      | 1.65   |
| 2M    | 2     | 48  | 1e-6 | 9.37      | 2.86     | 12.2      | 0.70   |
| 500K  | 4     | 24  | 1e-3 | 30.1      | 4.99     | 35.1      | 0.10   |
| 500K  | 4     | 24  | 1e-6 | 33.0      | 5.18     | 38.2      | 0.10   |
| 1M    | 4     | 24  | 1e-3 | 30.1      | 5.26     | 35.4      | 0.19   |
| 1M    | 4     | 24  | 1e-6 | 29.9      | 6.19     | 36.1      | 0.16   |
| 2M    | 4     | 24  | 1e-3 | 31.3      | 5.16     | 36.4      | 0.39   |
| 2M    | 4     | 24  | 1e-6 | 31.2      | 5.52     | 36.8      | 0.36   |
| 500K  | 8     | 12  | 1e-3 | 49.7      | 7.42     | 57.1      | 0.07   |
| 500K  | 8     | 12  | 1e-6 | 49.6      | 7.74     | 57.3      | 0.06   |
| 1M    | 8     | 12  | 1e-3 | 49.3      | 7.81     | 57.1      | 0.13   |
| 1M    | 8     | 12  | 1e-6 | 49.6      | 9.55     | 59.2      | 0.10   |
| 2M    | 8     | 12  | 1e-3 | 50.5      | 8.88     | 59.4      | 0.23   |
| 2M    | 8     | 12  | 1e-6 | 50.7      | 10.3     | 61.0      | 0.19   |

### 2d. Full 2-Node Results

| N     | Ranks | OMP | eps  | Create (s) | Eval (s) | Total (s) | Mpts/s |
|-------|-------|-----|------|-----------|----------|-----------|--------|
| 500K  | 2     | 96  | 1e-3 | 32.4      | 2.74     | 35.2      | 0.18   |
| 500K  | 2     | 96  | 1e-6 | 24.4      | 3.14     | 27.5      | 0.16   |
| 1M    | 2     | 96  | 1e-3 | 33.4      | 3.68     | 37.1      | 0.27   |
| 1M    | 2     | 96  | 1e-6 | 29.6      | 4.31     | 33.9      | 0.23   |
| 2M    | 2     | 96  | 1e-3 | 27.3      | 4.70     | 32.0      | 0.43   |
| 2M    | 2     | 96  | 1e-6 | 26.1      | 3.93     | 30.1      | 0.51   |
| 500K  | 4     | 48  | 1e-3 | 65.5      | 8.50     | 74.0      | 0.06   |
| 500K  | 4     | 48  | 1e-6 | 64.9      | 7.60     | 72.5      | 0.07   |
| 1M    | 4     | 48  | 1e-3 | 68.2      | 7.32     | 75.5      | 0.14   |
| 1M    | 4     | 48  | 1e-6 | 63.3      | 10.6     | 74.0      | 0.09   |
| 2M    | 4     | 48  | 1e-3 | 64.7      | 9.09     | 73.8      | 0.22   |
| 2M    | 4     | 48  | 1e-6 | 64.6      | 9.34     | 73.9      | 0.21   |
| 500K  | 8     | 24  | 1e-3 | 87.0      | 12.3     | 99.3      | 0.04   |
| 500K  | 8     | 24  | 1e-6 | 86.2      | 12.4     | 98.6      | 0.04   |
| 1M    | 8     | 24  | 1e-3 | 86.8      | 12.7     | 99.5      | 0.08   |
| 1M    | 8     | 24  | 1e-6 | 86.5      | 13.6     | 100.1     | 0.07   |
| 2M    | 8     | 24  | 1e-3 | 87.8      | 13.5     | 101.4     | 0.15   |
| 2M    | 8     | 24  | 1e-6 | 86.5      | 14.4     | 100.8     | 0.14   |
| 500K  | 16    | 12  | 1e-3 | 116.4     | 16.4     | 132.8     | 0.03   |
| 500K  | 16    | 12  | 1e-6 | 116.4     | 16.8     | 133.2     | 0.03   |
| 1M    | 16    | 12  | 1e-3 | 116.6     | 17.4     | 134.0     | 0.06   |
| 1M    | 16    | 12  | 1e-6 | 116.2     | 19.3     | 135.5     | 0.05   |
| 2M    | 16    | 12  | 1e-3 | 117.7     | 18.3     | 136.0     | 0.11   |
| 2M    | 16    | 12  | 1e-6 | 117.9     | 20.5     | 138.5     | 0.10   |

### 2e. Full 4-Node Results

| N     | Ranks | OMP | eps  | Create (s) | Eval (s) | Total (s) | Mpts/s |
|-------|-------|-----|------|-----------|----------|-----------|--------|
| 500K  | 4     | 96  | 1e-3 | 130.2     | 18.3     | 148.5     | 0.03   |
| 500K  | 4     | 96  | 1e-6 | 129.0     | 19.2     | 148.3     | 0.03   |
| 1M    | 4     | 96  | 1e-3 | 134.7     | 16.3     | 151.0     | 0.06   |
| 1M    | 4     | 96  | 1e-6 | 128.9     | 19.6     | 148.5     | 0.05   |
| 2M    | 4     | 96  | 1e-3 | 131.4     | 17.0     | 148.5     | 0.12   |
| 2M    | 4     | 96  | 1e-6 | 129.2     | 19.4     | 148.5     | 0.10   |
| 500K  | 8     | 48  | 1e-3 | 175.8     | 25.2     | 201.0     | 0.02   |
| 500K  | 8     | 48  | 1e-6 | 175.5     | 25.3     | 200.8     | 0.02   |
| 1M    | 8     | 48  | 1e-3 | 174.8     | 24.7     | 199.5     | 0.04   |
| 1M    | 8     | 48  | 1e-6 | 175.6     | 25.0     | 200.6     | 0.04   |
| 2M    | 8     | 48  | 1e-3 | 176.7     | 26.3     | 203.0     | 0.08   |
| 2M    | 8     | 48  | 1e-6 | 177.1     | 25.4     | 202.5     | 0.08   |
| 500K  | 16    | 24  | 1e-3 | 187.7     | 25.7     | 213.4     | 0.02   |
| 500K  | 16    | 24  | 1e-6 | 187.5     | 26.0     | 213.5     | 0.02   |
| 1M    | 16    | 24  | 1e-3 | 187.4     | 26.2     | 213.6     | 0.04   |
| 1M    | 16    | 24  | 1e-6 | 187.3     | 27.2     | 214.5     | 0.04   |
| 2M    | 16    | 24  | 1e-3 | 189.7     | 28.1     | 217.8     | 0.07   |
| 2M    | 16    | 24  | 1e-6 | 189.4     | 29.3     | 218.6     | 0.07   |
| 500K  | 32    | 12  | 1e-3 | 248.5     | 32.7     | 281.2     | 0.02   |
| 500K  | 32    | 12  | 1e-6 | 248.6     | 33.1     | 281.7     | 0.02   |
| 1M    | 32    | 12  | 1e-3 | 248.4     | 33.4     | 281.8     | 0.03   |
| 1M    | 32    | 12  | 1e-6 | 248.5     | 35.5     | 283.9     | 0.03   |
| 2M    | 32    | 12  | 1e-3 | 251.2     | 36.2     | 287.4     | 0.06   |
| 2M    | 32    | 12  | 1e-6 | 251.6     | 38.4     | 289.9     | 0.05   |

## 3. Analysis

### Tree Creation Dominates

The `pdmk_tree_create` call accounts for 80-90% of total time in multi-rank
runs. The cost scales roughly linearly with MPI rank count and is nearly
independent of problem size:

```
1 rank          :    0.18 s   (baseline)
2 ranks, 1 node :   10-15 s   (56-83x slower)
4 ranks, 1 node :   30-32 s   (167-178x slower)
8 ranks, 1 node :   49-51 s   (272-283x slower)
```

Cross-node communication adds further penalty:

```
2 ranks, 1 node  :  10-15 s
2 ranks, 2 nodes :  24-33 s   (2-3x worse than same-node)
4 ranks, 4 nodes : 129-135 s  (4x worse than same-node 4 ranks)
```

### Eval Also Degrades

Even after tree construction, evaluation time grows with rank count due to
halo exchange and distributed tree traversal (N=1M, eps=1e-3, 1 node):

```
1 rank  :  0.17 s  (baseline)
2 ranks :  1.75 s  (10x)
4 ranks :  5.26 s  (31x)
8 ranks :  7.81 s  (46x)
```

### Root Cause

The SCTL distributed octree construction involves global sort, particle
redistribution, halo ghost zone identification and exchange, and distributed
interaction list construction. For PBC, additional periodic image handling
amplifies the communication. At these problem sizes (500K-2M), the per-rank
local work is small relative to the MPI collective overhead.

## 4. Recommendations

1. **For N <= ~5M: use 1 MPI rank + max OMP threads.** Single-rank with 96
   OpenMP threads delivers 3-9 Mpts/s. No multi-rank configuration comes close.

2. **MPI may become beneficial at much larger N** (tens of millions+) where
   per-rank work dominates communication. Further testing needed.

3. **Tree creation is the primary optimization target** for MPI scaling. The
   SCTL distributed octree setup overhead grows with rank count nearly
   independently of problem size.

## 5. Files

- `bench_pbc_mpi.cpp` — benchmark source (accuracy + speed modes)
- `run_pbc_mpi_accuracy.slurm` — accuracy test script (1 node, 1-32 ranks)
- `run_pbc_mpi_scaling.slurm` — scaling test script (submit with `-N1`/`-N2`/`-N4`)
- `results/2026-04-10-pbc-mpi-accuracy.txt` — raw accuracy output
- `results/2026-04-10-pbc-mpi-scaling-1nodes.csv` — 1-node timing CSV
- `results/2026-04-10-pbc-mpi-scaling-2nodes.csv` — 2-node timing CSV
- `results/2026-04-10-pbc-mpi-scaling-4nodes.csv` — 4-node timing CSV
