# PBC Ewald Accuracy Benchmark Design

## Goal

Benchmark `pbc=true` accuracy against a self-contained Ewald summation reference for small 3D Laplace systems, covering both potential-only and potential+gradient outputs on sources and targets.

## Scope

- Workspace: `/mnt/home/xgao1/work/dmk_benchmark`
- Kernel: 3D Laplace
- Boundary condition: `use_periodic = true`
- Sizes: `n_src = n_trg = N` with `N ∈ {1000, 2000, 5000, 10000}`
- Modes: `pot` and `pot+grad`
- Single-process only for this first step
- Accuracy only; thread scaling and performance sweeps come later

## Approach

Use a dedicated benchmark driver that calls the public DMK API (`pdmk_tree_create`, `pdmk_tree_eval`) and compares outputs against an internal Ewald reference. Reuse the same Ewald formulation already validated in the main repo doctests, but keep the benchmark logic local to this workspace.

The driver will generate deterministic charge-neutral random particle sets in `[0.01, 0.99)^3`, run DMK in `pot` and `pot+grad` modes, evaluate the Ewald reference on the same points, and report relative L2 errors separately for source potential, target potential, source gradient, and target gradient.

## Output

The benchmark will print CSV to stdout with one row per `(N, mode)`:

- `N`
- `mode`
- `pot_src_l2`
- `pot_trg_l2`
- `grad_src_l2`
- `grad_trg_l2`

For `pot` mode, gradient columns will be reported as `nan`.

## File Plan

- Add: `/mnt/home/xgao1/work/dmk_benchmark/bench_pbc_accuracy.cpp`
  - Dedicated accuracy benchmark driver
- Modify: `/mnt/home/xgao1/work/dmk_benchmark/CMakeLists.txt`
  - Add build target for the new binary
- Optionally write results to:
  - `/mnt/home/xgao1/work/dmk_benchmark/results/2026-04-09-pbc-ewald-accuracy.csv`

## Key Assumptions

- Ewald reference will use the same truncation parameters as the validated doctest path: `alpha = 10`, `n_ewald = 15`.
- Full-set error computation is acceptable at the requested sizes for this first step.
- Accuracy will be evaluated at `eps = 1e-6`, which is the current default validation point for periodic pipeline tests.
