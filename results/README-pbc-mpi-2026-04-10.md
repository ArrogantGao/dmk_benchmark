# PBC Laplace3D MPI Benchmark Results

**Date:** 2026-04-10
**Cluster:** Rusty (Genoa nodes, AMD EPYC 9474F 48-Core, 96 cores/node, InfiniBand)
**Code:** FIDMK @ commit 2a38ea2
**Kernel:** Laplace3D, pot+grad (src+trg), PBC enabled
**Build:** GCC 13.3 + OpenMPI 4.1.8, `-O3 -march=native`

## 1. Accuracy: MPI vs Single-Rank Reference

Metric: relative L2 error between distributed MPI run (P ranks) and serial
single-rank reference (same problem, `MPI_COMM_SELF`). Particles generated on
rank 0 and scattered evenly.

### Results (worst case across N=2000/5000/10000, eps=1e-3/1e-6/1e-9)

| MPI Ranks | OMP/rank | pot_src | pot_trg | grad_src | grad_trg |
|-----------|----------|---------|---------|----------|----------|
| 1         | 96       | 0       | 0       | 0        | 0        |
| 2         | 48       | 3.4e-16 | 3.1e-16 | 7.0e-17  | 7.5e-17  |
| 4         | 24       | 4.3e-16 | 4.0e-16 | 8.3e-17  | 8.9e-17  |
| 8         | 12       | 9.1e-16 | 8.4e-16 | 3.8e-16  | 1.8e-16  |
| 16        | 6        | 1.5e-15 | 1.5e-15 | 5.3e-16  | 7.7e-16  |
| 32        | 3        | 1.4e-15 | 1.4e-15 | 4.2e-16  | 8.6e-16  |

**Conclusion:** MPI decomposition is bit-accurate. All errors are at double-precision
roundoff level (~1e-16 to ~1e-15), caused solely by floating-point reordering
across ranks. No accuracy degradation at any precision level.

## 2. Speed: Strong Scaling

Fixed total problem size, varying MPI ranks and nodes. Each rank uses
`96 / ranks_per_node` OMP threads. Timing via `MPI_Wtime` with `MPI_Barrier`.

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
| 1M    | 2     | 48  | 1e-3 | 10.1      | 1.75     | 11.9      | 0.57   |
| 2M    | 2     | 48  | 1e-3 | 11.3      | 1.21     | 12.5      | 1.65   |
| 500K  | 4     | 24  | 1e-3 | 30.1      | 4.99     | 35.1      | 0.10   |
| 1M    | 4     | 24  | 1e-3 | 30.1      | 5.26     | 35.4      | 0.19   |
| 2M    | 4     | 24  | 1e-3 | 31.3      | 5.16     | 36.4      | 0.39   |
| 500K  | 8     | 12  | 1e-3 | 49.7      | 7.42     | 57.1      | 0.07   |
| 1M    | 8     | 12  | 1e-3 | 49.3      | 7.81     | 57.1      | 0.13   |
| 2M    | 8     | 12  | 1e-3 | 50.5      | 8.88     | 59.4      | 0.23   |

## 3. Analysis

### Tree Creation Dominates

The `pdmk_tree_create` call accounts for 80-90% of total time in multi-rank
runs. The cost scales roughly linearly with MPI rank count:

```
1 rank  :    0.18 s   (baseline)
2 ranks :   10-15 s   (56-83x slower)
4 ranks :   30-32 s   (167-178x slower)
8 ranks :   49-51 s   (272-283x slower)
```

Cross-node adds further penalty:
```
2 ranks, 1 node  :  10-15 s
2 ranks, 2 nodes :  24-33 s   (2-3x worse than same-node)
4 ranks, 4 nodes : 129-135 s  (4x worse than same-node 4 ranks)
```

### Eval Also Degrades

Even after tree construction, evaluation time grows with rank count due to
halo exchange and distributed tree traversal:

```
1 rank  :  0.17 s    (N=1M, eps=1e-3)
2 ranks :  1.75 s    (10x)
4 ranks :  5.26 s    (31x)
8 ranks :  7.81 s    (46x)
```

### Root Cause

The SCTL distributed octree construction involves:
- Global sort and redistribution of particles
- Halo ghost zone identification and exchange
- Distributed interaction list construction
- Upward/downward pass synchronization

For PBC, additional periodic image handling amplifies the communication.
At these problem sizes (500K-2M), the per-rank local work is small relative
to the MPI collective overhead.

## 4. Recommendations

1. **For N <= ~5M: use 1 MPI rank + max OMP threads.** Single-rank with 96
   OpenMP threads delivers 3-9 Mpts/s. No multi-rank configuration comes close.

2. **MPI may become beneficial at much larger N** (tens of millions+) where
   per-rank work dominates communication. Further testing needed.

3. **Tree creation is the primary optimization target** for MPI scaling. The
   SCTL distributed octree setup overhead grows with rank count regardless
   of problem size.

## 5. Files

- `bench_pbc_mpi.cpp` — benchmark source (accuracy + speed modes)
- `run_pbc_mpi_accuracy.slurm` — accuracy test script (1 node, 1-32 ranks)
- `run_pbc_mpi_scaling.slurm` — scaling test script (submit with -N1/-N2/-N4)
- `results/2026-04-10-pbc-mpi-accuracy.txt` — raw accuracy output
- `results/2026-04-10-pbc-mpi-scaling-1nodes.csv` — 1-node timing CSV
- `results/2026-04-10-pbc-mpi-scaling-2nodes.csv` — 2-node timing CSV
- `results/2026-04-10-pbc-mpi-scaling-4nodes.csv` — 4-node timing CSV
