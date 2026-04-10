# PBC Runtime Benchmark Design

**Date:** 2026-04-09

**Goal**

Benchmark single-rank OpenMP runtime for the 3D Laplace `pot+grad` path with `n_per_leaf = 500` and `eps = 1e-3`, comparing `pbc=true` against `pbc=false` across particle counts and thread counts. Validate thread behavior on the local workstation first, then run the full sweep on a Rusty Genoa node.

**Scope**

- Benchmark only one MPI rank per run.
- Sweep `N = 500000, 600000, 700000, 800000, 900000, 1000000`.
- Sweep `OMP_NUM_THREADS = 1, 4, 8, 16, 32, 64, 96`.
- Compare `use_periodic = 0` and `use_periodic = 1`.
- Use `kernel = DMK_LAPLACE`, `pgh_src = pgh_trg = DMK_POTENTIAL_GRAD`, `n_per_leaf = 500`, `eps = 1e-3`.

**Non-goals**

- No MPI scaling study in this step.
- No new accuracy study in this step.
- No change to the main FIDMK library behavior.

**Approach**

Add a dedicated benchmark driver in `/mnt/home/xgao1/work/dmk_benchmark` instead of overloading the accuracy harness. The driver should generate one deterministic charge-neutral dataset per `N`, reuse it for both `pbc` modes, time tree creation and evaluation separately, and write a machine-readable CSV.

For local validation, run a smaller subset on the workstation with different `OMP_NUM_THREADS` values and confirm:

- the program reports the requested and actual thread count,
- timings change in the expected direction,
- no oversubscription or obvious runtime anomaly appears.

For the cluster run, use a Rusty Genoa batch job with one task per node and explicit OpenMP pinning. The batch script should loop over thread counts and particle counts, append CSV rows to a single output file, and store stdout/stderr logs for later inspection.

**Artifacts**

- New runtime benchmark source file.
- CMake target for the new benchmark binary.
- One local validation CSV/log.
- One Rusty Genoa Slurm script.
- One final cluster CSV for the full matrix.

**Output Schema**

Each CSV row should contain at least:

- `mode` (`pbc` or `free`)
- `n_particles`
- `omp_threads_requested`
- `omp_threads_actual`
- `eps`
- `n_per_leaf`
- `create_seconds`
- `eval_warmup_seconds`
- `eval_min_seconds`
- `eval_avg_seconds`
- `eval_max_seconds`
- `total_min_seconds`
- `mpts_per_sec`

**Risks**

- Very high thread counts on the workstation may not be meaningful if the machine has fewer cores than the Genoa node. That is acceptable for validation; the workstation run is only a correctness/sanity gate.
- `pbc=true` and `pbc=false` must use exactly the same particle set for each `N`, or the comparison is not trustworthy.
- Thread pinning matters on Genoa. The Slurm script should set `OMP_PROC_BIND=close` and `OMP_PLACES=cores`.
