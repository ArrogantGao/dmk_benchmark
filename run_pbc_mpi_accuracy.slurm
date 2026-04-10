#!/bin/bash
#SBATCH --job-name=pbc_mpi_acc
#SBATCH --output=pbc_mpi_acc.%J.out
#SBATCH --error=pbc_mpi_acc.%J.err
#SBATCH -p ccm
#SBATCH -N1
#SBATCH -n 96
#SBATCH -C genoa

set -euo pipefail

module load gcc fftw openmpi

WORKDIR=/mnt/home/xgao1/work/dmk_benchmark
FIDMK_ROOT=/mnt/home/xgao1/codes/FIDMK
BUILD_DIR="${WORKDIR}/build-mpi-bench"
BIN="${BUILD_DIR}/bench_pbc_mpi"
RESULT_DIR="${WORKDIR}/results"
RESULT="${RESULT_DIR}/$(date +%F)-pbc-mpi-accuracy.txt"

mkdir -p "${BUILD_DIR}" "${RESULT_DIR}"

# Build
cd "${BUILD_DIR}"
cmake "${WORKDIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFIDMK_ROOT="${FIDMK_ROOT}" \
    -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
    -DDMK_HAVE_MPI=ON \
    -DDMK_HAVE_OPENMP=ON
make -j bench_pbc_mpi
cd "${WORKDIR}"

echo "==================================================" | tee "${RESULT}"
echo "PBC MPI accuracy test — $(date)" | tee -a "${RESULT}"
echo "Node: $(hostname)" | tee -a "${RESULT}"
echo "==================================================" | tee -a "${RESULT}"

export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Test MPI rank counts: 1, 2, 4, 8, 16, 32
# For each: OMP_NUM_THREADS = 96 / nranks
for NRANKS in 1 2 4 8 16 32; do
    THREADS=$((96 / NRANKS))
    export OMP_NUM_THREADS="${THREADS}"
    echo "" | tee -a "${RESULT}"
    echo "--- MPI ranks=${NRANKS}, OMP threads/rank=${THREADS} ---" | tee -a "${RESULT}"
    mpirun -np "${NRANKS}" \
        --bind-to none \
        -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
        "${BIN}" --mode accuracy 2>&1 | tee -a "${RESULT}"
done

echo "" | tee -a "${RESULT}"
echo "[done] $(date)" | tee -a "${RESULT}"
