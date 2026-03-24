# Laplace Potential And Gradient FMM3D Benchmark Report

**Date:** 2026-03-24
**Kernel:** 3D Laplace
**Sources:** 128
**Targets:** 32
**Reference:** FMM3D at eps=1.0e-14

## Formulas

\[
\mathrm{relL2}(u, u^{ref}) = \sqrt{\frac{\sum_i |u_i - u_i^{ref}|^2}{\sum_i |u_i^{ref}|^2}}, \qquad \mathrm{relL2}(g, g^{ref}) = \sqrt{\frac{\sum_i \|g_i - g_i^{ref}\|_2^2}{\sum_i \|g_i^{ref}\|_2^2}}
\]

\[
\mathrm{maxRel}(u, u^{ref}) = \max_i \frac{|u_i - u_i^{ref}|}{|u_i^{ref}|}, \qquad \mathrm{maxRel}(g, g^{ref}) = \max_i \frac{\|g_i - g_i^{ref}\|_2}{\|g_i^{ref}\|_2}
\]

\[
S(p, t) = \frac{T_{1,1}}{T_{p,t}}, \qquad E(p, t) = \frac{S(p, t)}{pt}, \qquad H = \frac{T_{\mathrm{pot+grad}}}{T_{\mathrm{pot}}}
\]

## Notes

- FMM3D uses the `1/(4*pi*r)` Laplace normalization.
- DMK potentials and gradients are scaled by `1/(4*pi)` before comparison.
- `pot-only` timings are the baseline; `pot+grad` timings report the cost of computing both outputs.
- Times below are medians over the measured runs in each CSV.

## Baseline

Baseline for `(digits=3, mpi=1, omp=1)`: pot-only total `0.013` s, pot+grad total `0.007` s, overhead `0.56x`.

## 3 Digits

### Timing

| MPI | OMP | pot eval(s) | pot total(s) | pot+grad eval(s) | pot+grad total(s) | eval overhead | total overhead |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.002 | 0.013 | 0.002 | 0.007 | 1.07x | 0.56x |
| 1 | 2 | 0.001 | 0.004 | 0.002 | 0.005 | 1.83x | 1.04x |
| 2 | 1 | 0.001 | 0.007 | 0.002 | 0.006 | 1.04x | 0.85x |
| 2 | 2 | 0.002 | 0.007 | 0.002 | 0.006 | 1.05x | 0.81x |


### Accuracy

| MPI | OMP | pot src relL2 | pot trg relL2 | pot src maxRel | pot trg maxRel | grad src relL2 | grad trg relL2 | grad src maxRel | grad trg maxRel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8.856e-02 | 1.552e-04 | 1.272e+01 | 8.695e-01 | 6.480e-05 | 5.916e-05 | 3.875e-02 | 7.773e-03 |
| 1 | 2 | 8.856e-02 | 1.552e-04 | 1.272e+01 | 8.695e-01 | 6.480e-05 | 5.916e-05 | 3.875e-02 | 7.773e-03 |
| 2 | 1 | 1.267e-04 | 1.552e-04 | 4.546e-02 | 8.695e-01 | 6.480e-05 | 5.916e-05 | 3.875e-02 | 7.773e-03 |
| 2 | 2 | 1.267e-04 | 1.552e-04 | 4.546e-02 | 8.695e-01 | 6.480e-05 | 5.916e-05 | 3.875e-02 | 7.773e-03 |

