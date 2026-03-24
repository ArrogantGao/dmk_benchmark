#!/bin/bash
#SBATCH --job-name=lap_grad_fmm3d
#SBATCH --output=benchmark_results/laplace_grad_fmm3d_%j.out
#SBATCH --error=benchmark_results/laplace_grad_fmm3d_%j.err
#SBATCH -p ccm
#SBATCH -C icelake
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -c 64
#SBATCH --time=04:00:00

set -euo pipefail

cd /mnt/home/xgao1/work/dmk_benchmark

module load gcc fftw openmpi

cmake -S . -B build \
  -DFIDMK_ROOT=/mnt/home/xgao1/codes/FIDMK \
  -DDMK_FMM3D_DIR=/mnt/home/xgao1/codes/FMM3D
cmake --build build --target benchmark_laplace_grad_fmm3d -j4

export REFERENCE_OMP_THREADS=64

bash scripts/run_laplace_grad_fmm3d_matrix.sh \
  --out-dir results/laplace_pot_grad_fmm3d \
  --n-src 1000000 \
  --n-trg 10000 \
  --eps-list 1e-3,1e-6,1e-9,1e-12 \
  --mpi-list 1,2,4 \
  --omp-list 1,2,4,8,16,32,64 \
  --n-runs 1 \
  --warmup-runs 0 \
  --n-per-leaf 280

./scripts/report_laplace_grad_fmm3d.py \
  --input-dir results/laplace_pot_grad_fmm3d \
  --output docs/reports/2026-03-24-laplace-pot-grad-fmm3d-benchmark.md
