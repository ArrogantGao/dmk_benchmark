# Laplace Gradient FMM3D Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible benchmark workflow that generates one fixed FMM3D reference for 3D Laplace source and target gradients, then benchmarks DMK over the requested tolerance, MPI-rank, and OpenMP-thread matrix and emits a markdown report.

**Architecture:** Add one benchmark executable that supports two modes: `reference` to generate and store the fixed dataset and FMM3D reference, and `benchmark` to load those artifacts, run DMK, and write raw CSV rows. Add one driver script to run the full matrix and one report script to aggregate the raw results into markdown tables and formulas.

**Tech Stack:** C++20, MPI, OpenMP, DMK C API, local FMM3D C wrappers, Python 3 for report generation, Slurm shell scripting.

---

### Task 1: Add a Smoke-Testable Benchmark Driver Skeleton

**Files:**
- Create: `examples/benchmark_laplace_grad_fmm3d.cpp`
- Modify: `examples/CMakeLists.txt`

**Step 1: Add the executable to CMake**

Modify `examples/CMakeLists.txt` to build a new executable:

```cmake
add_executable(benchmark_laplace_grad_fmm3d benchmark_laplace_grad_fmm3d.cpp)
target_link_libraries(benchmark_laplace_grad_fmm3d dmk ${PAPI_TARGET} ${DMK_MPI_TARGET} ${DMK_OPENMP_TARGET} ${DMK_REFERENCE_TARGET})
```

Keep it gated on `DMK_HAVE_MPI AND DMK_HAVE_OPENMP`.

**Step 2: Write a minimal CLI skeleton**

Create `examples/benchmark_laplace_grad_fmm3d.cpp` with:

- `--mode reference|benchmark`
- `--out-dir`
- `--n-src`
- `--n-trg`
- `--eps`
- `--seed`

The first version should only parse args, initialize/finalize MPI, and print the parsed config on rank 0.

**Step 3: Build it**

Run: `cmake --build build --target benchmark_laplace_grad_fmm3d -j4`

Expected: build succeeds.

**Step 4: Run a tiny smoke command**

Run:

```bash
OMP_NUM_THREADS=1 mpiexec -n 1 ./build/examples/benchmark_laplace_grad_fmm3d --mode benchmark --out-dir results/laplace_grad_fmm3d_smoke --n-src 64 --n-trg 16 --eps 1e-3 --seed 0
```

Expected: the executable starts, prints config, and exits cleanly without doing real work yet.

**Step 5: Commit**

```bash
git add examples/CMakeLists.txt examples/benchmark_laplace_grad_fmm3d.cpp
git commit -m "bench: add laplace gradient benchmark driver skeleton"
```

### Task 2: Add Deterministic Dataset Generation and Artifact I/O

**Files:**
- Modify: `examples/benchmark_laplace_grad_fmm3d.cpp`

**Step 1: Add deterministic dataset generation**

Reuse the compare-driver style generator for:

- `N_src` source points
- `N_trg` target points
- `nd = 1`
- seed `0`

Store:

- source coordinates
- target coordinates
- charges

Use a compact binary format with a small metadata header or separate metadata text file.

**Step 2: Add artifact writers**

Write helpers that save:

- `metadata.yaml`
- `sources.bin`
- `targets.bin`
- `charges.bin`

The metadata must include:

- `n_dim`
- `n_src`
- `n_trg`
- `nd`
- `seed`
- normalization convention

**Step 3: Add artifact readers and metadata validation**

On benchmark mode startup:

- load metadata
- verify `n_dim == 3`
- verify `n_src` and `n_trg` match the requested run

Fail with a clear message if the fixed reference artifacts are missing or inconsistent.

**Step 4: Run a tiny reference-generation smoke**

Run:

```bash
OMP_NUM_THREADS=1 mpiexec -n 1 ./build/examples/benchmark_laplace_grad_fmm3d --mode reference --out-dir results/laplace_grad_fmm3d_smoke --n-src 64 --n-trg 16 --eps 1e-12 --seed 0
```

Expected: the output directory contains metadata plus dataset files.

**Step 5: Commit**

```bash
git add examples/benchmark_laplace_grad_fmm3d.cpp
git commit -m "bench: add deterministic dataset artifacts for laplace gradient runs"
```

### Task 3: Add FMM3D Reference Generation for Source and Target Gradients

**Files:**
- Modify: `examples/benchmark_laplace_grad_fmm3d.cpp`
- Modify: `CMakeLists.txt`

**Step 1: Add a configurable local FMM3D path**

In `CMakeLists.txt`, add a cache path such as:

```cmake
set(DMK_FMM3D_DIR "/mnt/home/xgao1/codes/FMM3D" CACHE PATH "Path to local FMM3D checkout")
```

Add include/library wiring for the FMM3D C headers and built library, without changing the main DMK library target.

**Step 2: Add the FMM3D reference call**

In `examples/benchmark_laplace_grad_fmm3d.cpp`, call the FMM3D Laplace charge routine that returns:

- source gradients
- target gradients

Request source and target potential+gradient outputs, then keep gradients only.

**Step 3: Normalize DMK gradients to FMM3D convention**

Keep the stored FMM3D gradients in their native `1/(4*pi*r)` normalization and divide DMK gradients by `4*pi` during comparison.

**Step 4: Write reference artifacts**

Store:

- `grad_src_ref.bin`
- `grad_trg_ref.bin`

The files should be reusable by all later benchmark runs.

**Step 5: Validate on a tiny case**

Run:

```bash
OMP_NUM_THREADS=2 mpiexec -n 1 ./build/examples/benchmark_laplace_grad_fmm3d --mode reference --out-dir results/laplace_grad_fmm3d_smoke --n-src 128 --n-trg 32 --eps 1e-14 --seed 0
```

Expected: reference files are written and their sizes match `3 * n_points * sizeof(double)`.

**Step 6: Commit**

```bash
git add CMakeLists.txt examples/benchmark_laplace_grad_fmm3d.cpp
git commit -m "bench: add FMM3D gradient reference generation"
```

### Task 4: Add DMK Benchmark Mode and Accuracy Metrics

**Files:**
- Modify: `examples/benchmark_laplace_grad_fmm3d.cpp`

**Step 1: Add DMK evaluation for both source and target gradients**

In benchmark mode:

- build the DMK tree with `kernel = DMK_LAPLACE`
- set `pgh_src = DMK_POTENTIAL_GRAD`
- set `pgh_trg = DMK_POTENTIAL_GRAD`
- pass the fixed sources, charges, and targets

Allocate:

- `grad_src`
- `grad_trg`

**Step 2: Time build and eval separately**

Measure:

- tree build time
- eval time
- total time

Add one warm-up run before the measured loop.

**Step 3: Compute source and target error metrics**

Against the stored reference, compute:

- source relative L2
- target relative L2
- source max relative
- target max relative

Use vector norms per point for the max-relative metric.

**Step 4: Emit one CSV row per measured run**

Include columns:

- `mpi_ranks`
- `omp_threads`
- `eps`
- `n_src`
- `n_trg`
- `build_time`
- `eval_time`
- `total_time`
- `grad_src_rel_l2`
- `grad_trg_rel_l2`
- `grad_src_max_rel`
- `grad_trg_max_rel`
- `pts_per_sec`

Also print a comment header with config metadata.

**Step 5: Run a tiny end-to-end smoke**

Run:

```bash
OMP_NUM_THREADS=2 mpiexec -n 1 ./build/examples/benchmark_laplace_grad_fmm3d --mode benchmark --out-dir results/laplace_grad_fmm3d_smoke --n-src 128 --n-trg 32 --eps 1e-6 --seed 0
```

Expected: benchmark CSV is written and all four gradient error columns are finite.

**Step 6: Commit**

```bash
git add examples/benchmark_laplace_grad_fmm3d.cpp
git commit -m "bench: add DMK laplace gradient benchmark mode"
```

### Task 5: Add Matrix Runner for the Full MPI and OpenMP Sweep

**Files:**
- Create: `scripts/run_laplace_grad_fmm3d_matrix.sh`

**Step 1: Add a reusable shell runner**

Create a script that:

- creates the output directory
- generates the fixed reference if missing
- loops over:
  - `eps in 1e-3 1e-6 1e-9 1e-12`
  - `mpi in 1 2 4`
  - `omp in 1 2 4 8 16 32 64`
- runs the benchmark executable for each combination

**Step 2: Make the runner cluster-friendly**

The script should:

- load `gcc`, `fftw`, and `openmpi` if needed
- use `mpiexec -n "$mpi"` or `srun` if running under Slurm
- export `OMP_NUM_THREADS="$omp"`
- append each run to a raw CSV file per configuration

**Step 3: Add resume behavior**

Skip a configuration if its output CSV already exists and is non-empty.

**Step 4: Smoke-test a tiny subset**

Run:

```bash
bash scripts/run_laplace_grad_fmm3d_matrix.sh --out-dir results/laplace_grad_fmm3d_smoke --n-src 128 --n-trg 32 --eps-list 1e-3 --mpi-list 1 --omp-list 1,2
```

Expected: two configuration outputs are produced.

**Step 5: Commit**

```bash
git add scripts/run_laplace_grad_fmm3d_matrix.sh
git commit -m "bench: add laplace gradient matrix runner"
```

### Task 6: Add Markdown Report Generation

**Files:**
- Create: `scripts/report_laplace_grad_fmm3d.py`
- Create: `docs/reports/2026-03-24-laplace-gradient-fmm3d-benchmark.md`

**Step 1: Add an aggregator script**

Read the per-configuration CSV files and compute medians over measured runs for:

- build time
- eval time
- total time
- source and target relative L2
- source and target max relative
- throughput

**Step 2: Generate markdown tables**

Write tables grouped by:

- digits
- MPI ranks
- OpenMP threads

Include both source and target gradient accuracy in each row.

**Step 3: Add markdown formulas**

Emit formulas for:

- relative L2
- max relative
- speedup
- parallel efficiency

**Step 4: Run the script on smoke results**

Run:

```bash
python3 scripts/report_laplace_grad_fmm3d.py --input-dir results/laplace_grad_fmm3d_smoke --output docs/reports/2026-03-24-laplace-gradient-fmm3d-benchmark.md
```

Expected: markdown report is written and includes at least one timing table and one accuracy table.

**Step 5: Commit**

```bash
git add scripts/report_laplace_grad_fmm3d.py docs/reports/2026-03-24-laplace-gradient-fmm3d-benchmark.md
git commit -m "bench: add laplace gradient benchmark report generator"
```

### Task 7: Run the Full Benchmark and Verify Outputs

**Files:**
- Modify: `docs/reports/2026-03-24-laplace-gradient-fmm3d-benchmark.md`

**Step 1: Build the benchmark executable**

Run:

```bash
cmake --build build --target benchmark_laplace_grad_fmm3d -j4
```

Expected: build succeeds.

**Step 2: Generate the fixed production reference**

Run:

```bash
OMP_NUM_THREADS=64 mpiexec -n 1 ./build/examples/benchmark_laplace_grad_fmm3d --mode reference --out-dir results/laplace_grad_fmm3d --n-src 1000000 --n-trg 10000 --eps 1e-14 --seed 0
```

Expected: reference artifacts are written once.

**Step 3: Run the full matrix**

Run:

```bash
bash scripts/run_laplace_grad_fmm3d_matrix.sh --out-dir results/laplace_grad_fmm3d --n-src 1000000 --n-trg 10000
```

Expected: raw per-configuration benchmark files appear for all requested combinations.

**Step 4: Generate the final report**

Run:

```bash
python3 scripts/report_laplace_grad_fmm3d.py --input-dir results/laplace_grad_fmm3d --output docs/reports/2026-03-24-laplace-gradient-fmm3d-benchmark.md
```

Expected: final report is written with formulas and all benchmark tables.

**Step 5: Sanity-check the result set**

Verify:

- 84 configurations exist (`4 * 3 * 7`)
- source and target errors decrease with requested digits
- timing scales plausibly with MPI ranks and OpenMP threads

**Step 6: Commit**

```bash
git add docs/reports/2026-03-24-laplace-gradient-fmm3d-benchmark.md results/laplace_grad_fmm3d scripts/run_laplace_grad_fmm3d_matrix.sh scripts/report_laplace_grad_fmm3d.py
git commit -m "bench: add laplace gradient FMM3D benchmark results"
```
