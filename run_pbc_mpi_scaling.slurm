#!/bin/bash
#SBATCH --job-name=pbc_mpi_scale
#SBATCH --output=pbc_mpi_scale.%J.out
#SBATCH --error=pbc_mpi_scale.%J.err
#SBATCH -p ccm
#SBATCH -C genoa

# NOTE: submit with different -N values:
#   sbatch -N1 run_pbc_mpi_scaling.slurm
#   sbatch -N2 run_pbc_mpi_scaling.slurm
#   sbatch -N4 run_pbc_mpi_scaling.slurm

set -euo pipefail

module load gcc fftw openmpi

WORKDIR=/mnt/home/xgao1/work/dmk_benchmark
FIDMK_ROOT=/mnt/home/xgao1/codes/FIDMK
BUILD_DIR="${WORKDIR}/build-mpi-bench"
BIN="${BUILD_DIR}/bench_pbc_mpi"
RESULT_DIR="${WORKDIR}/results"

NNODES=${SLURM_NNODES}
RESULT_CSV="${RESULT_DIR}/$(date +%F)-pbc-mpi-scaling-${NNODES}nodes.csv"

mkdir -p "${BUILD_DIR}" "${RESULT_DIR}"

# Build (skip if binary exists and is newer than source)
if [[ ! -x "${BIN}" || "${WORKDIR}/bench_pbc_mpi.cpp" -nt "${BIN}" ]]; then
    cd "${BUILD_DIR}"
    cmake "${WORKDIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DFIDMK_ROOT="${FIDMK_ROOT}" \
        -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
        -DDMK_HAVE_MPI=ON \
        -DDMK_HAVE_OPENMP=ON
    make -j bench_pbc_mpi
    cd "${WORKDIR}"
fi

echo "=================================================="
echo "PBC MPI scaling benchmark — $(date)"
echo "Nodes: ${NNODES}, Node list: ${SLURM_NODELIST}"
echo "Output CSV: ${RESULT_CSV}"
echo "=================================================="

export OMP_PROC_BIND=close
export OMP_PLACES=cores

N_RUNS=3
EPS_LIST="1e-3 1e-6"
N_LIST="500000 1000000 2000000"

# Rank configurations: 1 rank/node up to 8 ranks/node
RANKS_PER_NODE_LIST="1 2 4 8"

first_write=1

for RANKS_PER_NODE in ${RANKS_PER_NODE_LIST}; do
    TOTAL_RANKS=$((NNODES * RANKS_PER_NODE))
    THREADS=$((96 / RANKS_PER_NODE))
    export OMP_NUM_THREADS="${THREADS}"

    echo "[config] nodes=${NNODES} ranks_per_node=${RANKS_PER_NODE} total_ranks=${TOTAL_RANKS} threads/rank=${THREADS}"

    for EPS in ${EPS_LIST}; do
        for N in ${N_LIST}; do
            echo "[run] N=${N} eps=${EPS} ranks=${TOTAL_RANKS}"
            TMP=$(mktemp)
            mpirun -np "${TOTAL_RANKS}" \
                --bind-to none \
                -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
                "${BIN}" --mode speed --n "${N}" --eps "${EPS}" --n-runs "${N_RUNS}" \
                > "${TMP}" 2>/dev/null

            if [[ ${first_write} -eq 1 ]]; then
                cat "${TMP}" > "${RESULT_CSV}"
                first_write=0
            else
                tail -n +2 "${TMP}" >> "${RESULT_CSV}"
            fi
            rm -f "${TMP}"
        done
    done
done

echo "[done] wrote ${RESULT_CSV}"
