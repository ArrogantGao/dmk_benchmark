# PBC Runtime Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-rank OpenMP runtime benchmark that compares `pbc=true` and `pbc=false` for 3D Laplace `pot+grad` at `eps=1e-3`, `n_per_leaf=500`, validates thread behavior locally, and prepares the full Rusty Genoa run.

**Architecture:** Implement a dedicated benchmark driver in the benchmark workspace, reusing the deterministic dataset generation pattern from the existing local benchmark code. The driver will emit one CSV row per `(pbc mode, N, threads)` run. A separate Slurm script will drive the full matrix on Rusty Genoa with one rank and pinned OpenMP threads.

**Tech Stack:** C++, MPI, OpenMP, CMake, Slurm, Rusty Genoa CPU nodes

---

### Task 1: Add the runtime benchmark source and build target

**Files:**
- Create: `/mnt/home/xgao1/work/dmk_benchmark/bench_pbc_runtime.cpp`
- Modify: `/mnt/home/xgao1/work/dmk_benchmark/CMakeLists.txt`

- [ ] Implement a benchmark binary that:
  - parses `--n`, `--eps`, `--n-per-leaf`, `--n-runs`, and `--pbc both|true|false`,
  - reports requested and actual OpenMP thread count,
  - generates a deterministic charge-neutral dataset once per `N`,
  - runs `pbc=false` and `pbc=true` on that same dataset,
  - times tree creation and evaluation separately,
  - writes CSV to stdout.

- [ ] Add a CMake target `bench_pbc_runtime` linked against `dmk`.

### Task 2: Validate OpenMP behavior on the workstation

**Files:**
- Create: `/mnt/home/xgao1/work/dmk_benchmark/results/2026-04-09-pbc-runtime-local-validation.csv`

- [ ] Run a small local subset, for example:
  - `OMP_NUM_THREADS=1`
  - `OMP_NUM_THREADS=4`
  - `OMP_NUM_THREADS=8`

- [ ] Use a reduced problem size for validation, such as `N=500000`, to confirm:
  - actual threads match requested threads,
  - the program runs both `pbc` modes successfully,
  - timings are sensible.

### Task 3: Add the Rusty Genoa batch script

**Files:**
- Create: `/mnt/home/xgao1/work/dmk_benchmark/run_pbc_runtime_rusty_genoa.slurm`

- [ ] Create a Slurm script that:
  - requests one Rusty Genoa node,
  - loads `gcc`, `fftw`, and `openmpi`,
  - sets `OMP_PROC_BIND=close` and `OMP_PLACES=cores`,
  - loops over thread counts `1 4 8 16 32 64 96`,
  - loops over `N = 500000 600000 700000 800000 900000 1000000`,
  - runs the benchmark with one MPI rank,
  - appends rows to a single CSV in `/mnt/home/xgao1/work/dmk_benchmark/results/`.

### Task 4: Summarize local results and cluster execution command

**Files:**
- Create: `/mnt/home/xgao1/work/dmk_benchmark/results/README-pbc-runtime-2026-04-09.md`

- [ ] Record:
  - the exact local validation commands that were run,
  - the observed requested vs actual thread counts,
  - the exact `sbatch` command for the Rusty Genoa run,
  - the location of the final cluster CSV.
