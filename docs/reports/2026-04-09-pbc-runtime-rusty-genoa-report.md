# PBC Runtime Benchmark Report

**Date:** 2026-04-09  
**Machine:** Rusty, Genoa node  
**Kernel:** 3D Laplace  
**Mode:** potential + gradient  
**MPI ranks:** 1  
**OMP threads:** `1, 4, 8, 16, 32, 64, 96`  
**Particles:** `500000:100000:1000000`  
**Tolerance:** `eps = 1e-3` (`3` digits)  
**Leaf size:** `n_per_leaf = 500`

## Data

- Raw runtime CSV: [`2026-04-09-pbc-runtime-rusty-genoa.csv`](/mnt/home/xgao1/work/dmk_benchmark/results/2026-04-09-pbc-runtime-rusty-genoa.csv)
- Summary CSV: [`2026-04-09-pbc-runtime-rusty-genoa-summary.csv`](/mnt/home/xgao1/work/dmk_benchmark/results/2026-04-09-pbc-runtime-rusty-genoa-summary.csv)
- Workstation validation notes: [`README-pbc-runtime-2026-04-09.md`](/mnt/home/xgao1/work/dmk_benchmark/results/README-pbc-runtime-2026-04-09.md)

## Method

- Each run used one deterministic charge-neutral source set.
- `pbc=false` and `pbc=true` were run on the same particle set for each `N`.
- The reported performance numbers below use `eval_min_seconds` from the raw CSV.
- OpenMP thread counts were validated locally before the cluster run.

## Main Findings

- The best runtime for every tested `N` occurred at `96` threads for both `pbc=false` and `pbc=true`.
- The periodic case scaled well and stayed close to the free-space case.
- At the best-thread setting, periodic overhead was modest: about `2.6%` to `7.4%` depending on `N`.
- For the largest case, `N = 1000000`, the best eval times were:
  - `free`: `0.087904 s`
  - `pbc`: `0.092235 s`
  - periodic overhead: `4.93%`

## Best Runtime By Problem Size

| N | free best threads | free eval min (s) | free Mpts/s | pbc best threads | pbc eval min (s) | pbc Mpts/s | pbc overhead at best |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500000 | 96 | 0.031319 | 15.965 | 96 | 0.033166 | 15.076 | 5.90% |
| 600000 | 96 | 0.038787 | 15.469 | 96 | 0.041666 | 14.400 | 7.42% |
| 700000 | 96 | 0.049104 | 14.256 | 96 | 0.052725 | 13.276 | 7.37% |
| 800000 | 96 | 0.060621 | 13.197 | 96 | 0.064749 | 12.355 | 6.81% |
| 900000 | 96 | 0.075573 | 11.909 | 96 | 0.077544 | 11.606 | 2.61% |
| 1000000 | 96 | 0.087904 | 11.376 | 96 | 0.092235 | 10.842 | 4.93% |

## 1-To-96 Thread Scaling

| N | free eval 1T (s) | free eval 96T (s) | free speedup | pbc eval 1T (s) | pbc eval 96T (s) | pbc speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 500000 | 1.689824 | 0.031319 | 53.96x | 1.842143 | 0.033166 | 55.54x |
| 600000 | 2.370860 | 0.038787 | 61.13x | 2.591176 | 0.041666 | 62.19x |
| 700000 | 3.191770 | 0.049104 | 65.00x | 3.490326 | 0.052725 | 66.20x |
| 800000 | 4.138185 | 0.060621 | 68.26x | 4.523187 | 0.064749 | 69.86x |
| 900000 | 5.182507 | 0.075573 | 68.58x | 5.658569 | 0.077544 | 72.97x |
| 1000000 | 6.346972 | 0.087904 | 72.20x | 6.935125 | 0.092235 | 75.19x |

## Interpretation

- The periodic path does not show a scaling collapse relative to free space.
- The absolute periodic overhead is small enough that `pbc=true` remains competitive through the full tested range.
- Build time is not the dominant cost once thread count is moderately high; the benchmark is primarily eval-bound for these sizes.

## Conclusion

For this single-rank Genoa benchmark at `3` digits and `n_per_leaf=500`, `pbc=true` scales strongly with OpenMP and remains within about `5%` of `pbc=false` at `N = 10^6` on the best thread count. The strongest configuration tested here is `96` threads for both modes across the whole `5e5` to `1e6` range.
