#!/bin/bash
#SBATCH --job-name=pbc_runtime
#SBATCH --output=pbc_runtime.%J.out
#SBATCH --error=pbc_runtime.%J.err
#SBATCH -p ccm
#SBATCH -N1
#SBATCH -n1
#SBATCH -c 96
#SBATCH -C genoa

set -euo pipefail

module load gcc fftw openmpi

WORKDIR=/mnt/home/xgao1/work/dmk_benchmark
FIDMK_ROOT=/mnt/home/xgao1/codes/FIDMK
MPI_CXX=$(command -v mpicxx)
BIN_DIR="${WORKDIR}/build-rusty"
BIN="${BIN_DIR}/bench_pbc_runtime_omp"
RESULT_DIR="${WORKDIR}/results"
RESULT_CSV="${RESULT_DIR}/$(date +%F)-pbc-runtime-rusty-genoa.csv"
TMP_CSV=$(mktemp)

mkdir -p "${BIN_DIR}" "${RESULT_DIR}"

${MPI_CXX} -O3 -fopenmp \
  -I"${FIDMK_ROOT}/include" \
  -I"${FIDMK_ROOT}/extern/nda/c++" \
  "${WORKDIR}/bench_pbc_runtime.cpp" \
  -L"${FIDMK_ROOT}/build" \
  -Wl,-rpath,"${FIDMK_ROOT}/build" \
  -ldmk \
  -o "${BIN}"

export OMP_PROC_BIND=close
export OMP_PLACES=cores

THREADS_LIST=(1 4 8 16 32 64 96)
N_LIST=(500000 600000 700000 800000 900000 1000000)
EPS=1e-3
N_PER_LEAF=500
N_RUNS=3

echo "=================================================="
echo "PBC runtime benchmark on Rusty Genoa — $(date)"
echo "Node: $(hostname)"
echo "Binary: ${BIN}"
echo "Output CSV: ${RESULT_CSV}"
echo "=================================================="

first_write=1
for threads in "${THREADS_LIST[@]}"; do
  export OMP_NUM_THREADS="${threads}"
  echo "[run] OMP_NUM_THREADS=${threads}"
  for n in "${N_LIST[@]}"; do
    echo "[run] N=${n}"
    "${BIN}" --n "${n}" --n-runs "${N_RUNS}" --eps "${EPS}" --n-per-leaf "${N_PER_LEAF}" --pbc both > "${TMP_CSV}"
    if [[ ${first_write} -eq 1 ]]; then
      cat "${TMP_CSV}" > "${RESULT_CSV}"
      first_write=0
    else
      tail -n +2 "${TMP_CSV}" >> "${RESULT_CSV}"
    fi
  done
done

rm -f "${TMP_CSV}"

echo "[done] wrote ${RESULT_CSV}"
