# PBC Ewald Accuracy Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated benchmark binary in `/mnt/home/xgao1/work/dmk_benchmark` that measures `pbc=true` accuracy against Ewald for `N ∈ {1000, 2000, 5000, 10000}` in `pot` and `pot+grad` modes.

**Architecture:** Keep the existing `bench_pbc.cpp` focused on timing. Add a new driver that owns particle generation, DMK invocation, Ewald reference evaluation, and CSV output. Build it as a separate target from the workspace CMake file.

**Tech Stack:** C++20, MPI, OpenMP, DMK public API, self-contained Ewald reference.

---

### Task 1: Add the dedicated accuracy benchmark source

**Files:**
- Create: `/mnt/home/xgao1/work/dmk_benchmark/bench_pbc_accuracy.cpp`

- [ ] Implement a standalone driver that sweeps `N = {1000, 2000, 5000, 10000}` and `mode ∈ {pot, pot+grad}`.
- [ ] Generate deterministic charge-neutral random input data in `[0.01, 0.99)^3`.
- [ ] Reuse the Ewald potential/gradient reference formula from the validated periodic doctest logic.
- [ ] Emit CSV rows to stdout.

### Task 2: Wire the new target into the benchmark workspace build

**Files:**
- Modify: `/mnt/home/xgao1/work/dmk_benchmark/CMakeLists.txt`

- [ ] Add a `bench_pbc_accuracy` executable target linked against `dmk`.
- [ ] Reuse the same include-path pattern already needed by benchmark binaries in this workspace.

### Task 3: Build and run the sweep

**Files:**
- Verify only

- [ ] Build with `cmake --build /mnt/home/xgao1/work/dmk_benchmark/build --target bench_pbc_accuracy -j4`
- [ ] Run the binary and save CSV output under `/mnt/home/xgao1/work/dmk_benchmark/results/2026-04-09-pbc-ewald-accuracy.csv`

### Task 4: Summarize results

**Files:**
- Verify only

- [ ] Read the CSV and summarize the error trends for potential and gradients on sources and targets.
