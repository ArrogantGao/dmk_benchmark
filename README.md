# DMK Benchmark Workspace

This workspace holds the FMM3D-based Laplace potential and gradient benchmark artifacts that were moved out of the main `FIDMK` repo.

## Layout

- `examples/benchmark_laplace_grad_fmm3d.cpp`: benchmark driver
- `scripts/`: matrix runner, report generator, and Slurm batch script
- `docs/`: benchmark plans and generated reports
- `results/`: copied benchmark datasets and CSVs
- `benchmark_results/`: copied Slurm stdout/stderr logs

## Build

```bash
cmake -S . -B build \
  -DFIDMK_ROOT=/mnt/home/xgao1/codes/FIDMK \
  -DDMK_FMM3D_DIR=/mnt/home/xgao1/codes/FMM3D
cmake --build build --target benchmark_laplace_grad_fmm3d -j4
```

## Run

```bash
bash scripts/run_laplace_grad_fmm3d_matrix.sh --out-dir results/laplace_pot_grad_fmm3d
./scripts/report_laplace_grad_fmm3d.py \
  --input-dir results/laplace_pot_grad_fmm3d \
  --output docs/reports/2026-03-24-laplace-pot-grad-fmm3d-benchmark.md
```
