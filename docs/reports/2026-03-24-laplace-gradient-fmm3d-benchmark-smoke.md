# Laplace Gradient FMM3D Benchmark Report

**Date:** 2026-03-24
**Kernel:** 3D Laplace
**Sources:** 128
**Targets:** 32
**Reference:** FMM3D at eps=1.0e-14

## Formulas

\[
\mathrm{relL2}(g, g^{ref}) = \sqrt{\frac{\sum_i \|g_i - g_i^{ref}\|_2^2}{\sum_i \|g_i^{ref}\|_2^2}}
\]

\[
\mathrm{maxRel}(g, g^{ref}) = \max_i \frac{\|g_i - g_i^{ref}\|_2}{\|g_i^{ref}\|_2}
\]

\[
S(p, t) = \frac{T_{1,1}}{T_{p,t}}, \qquad E(p, t) = \frac{S(p, t)}{pt}
\]

## Notes

- FMM3D uses the `1/(4*pi*r)` Laplace normalization.
- DMK gradients are scaled by `1/(4*pi)` before comparison.
- Times below are medians over the measured runs in each CSV.

## Baseline

Baseline total time for `(digits=3, mpi=1, omp=1)` is `0.007` seconds.

## 3 Digits

| MPI | OMP | build(s) | eval(s) | total(s) | eval pts/s | src relL2 | trg relL2 | src maxRel | trg maxRel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.005 | 0.001 | 0.007 | 109735.27 | 6.480e-05 | 5.916e-05 | 3.875e-02 | 7.773e-03 |
| 1 | 2 | 0.007 | 0.002 | 0.009 | 80275.56 | 6.480e-05 | 5.916e-05 | 3.875e-02 | 7.773e-03 |
| 2 | 1 | 0.004 | 0.001 | 0.006 | 114277.17 | 6.480e-05 | 5.916e-05 | 3.875e-02 | 7.773e-03 |
| 2 | 2 | 0.006 | 0.002 | 0.008 | 100495.40 | 6.480e-05 | 5.916e-05 | 3.875e-02 | 7.773e-03 |

