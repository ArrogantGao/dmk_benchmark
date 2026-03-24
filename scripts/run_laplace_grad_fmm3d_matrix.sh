#!/usr/bin/env bash
set -euo pipefail

out_dir="results/laplace_grad_fmm3d"
benchmark_bin="./build/examples/benchmark_laplace_grad_fmm3d"
n_src=1000000
n_trg=10000
seed=0
reference_eps="1e-14"
n_runs=5
warmup_runs=1
n_per_leaf=280
eps_list=("1e-3" "1e-6" "1e-9" "1e-12")
mpi_list=("1" "2" "4")
omp_list=("1" "2" "4" "8" "16" "32" "64")

usage() {
  cat <<'EOF'
Usage: run_laplace_grad_fmm3d_matrix.sh [options]
  --out-dir PATH
  --benchmark-bin PATH
  --n-src N
  --n-trg N
  --seed N
  --reference-eps VALUE
  --n-runs N
  --warmup-runs N
  --n-per-leaf N
  --eps-list "1e-3,1e-6"
  --mpi-list "1,2,4"
  --omp-list "1,2,4,8"
EOF
}

split_csv() {
  local value="$1"
  IFS=',' read -r -a parts <<<"$value"
  printf '%s\n' "${parts[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --benchmark-bin)
      benchmark_bin="$2"
      shift 2
      ;;
    --n-src)
      n_src="$2"
      shift 2
      ;;
    --n-trg)
      n_trg="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --reference-eps)
      reference_eps="$2"
      shift 2
      ;;
    --n-runs)
      n_runs="$2"
      shift 2
      ;;
    --warmup-runs)
      warmup_runs="$2"
      shift 2
      ;;
    --n-per-leaf)
      n_per_leaf="$2"
      shift 2
      ;;
    --eps-list)
      mapfile -t eps_list < <(split_csv "$2")
      shift 2
      ;;
    --mpi-list)
      mapfile -t mpi_list < <(split_csv "$2")
      shift 2
      ;;
    --omp-list)
      mapfile -t omp_list < <(split_csv "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$benchmark_bin" ]]; then
  echo "Benchmark binary not found: $benchmark_bin" >&2
  exit 1
fi

if type module >/dev/null 2>&1; then
  module load gcc fftw openmpi >/dev/null 2>&1 || true
fi

mkdir -p "$out_dir"

ref_threads=1
for omp in "${omp_list[@]}"; do
  if (( omp > ref_threads )); then
    ref_threads="$omp"
  fi
done

if [[ ! -f "$out_dir/metadata.yaml" || ! -f "$out_dir/pot_src_ref.bin" || ! -f "$out_dir/pot_trg_ref.bin" || ! -f "$out_dir/grad_src_ref.bin" || ! -f "$out_dir/grad_trg_ref.bin" ]]; then
  echo "[reference] generating fixed FMM3D reference in $out_dir"
  OMP_NUM_THREADS="${REFERENCE_OMP_THREADS:-$ref_threads}" \
    "$benchmark_bin" \
    --mode reference \
    --out-dir "$out_dir" \
    --n-src "$n_src" \
    --n-trg "$n_trg" \
    --reference-eps "$reference_eps" \
    --seed "$seed"
fi

launcher_base=()
if command -v mpiexec >/dev/null 2>&1; then
  launcher_base=(mpiexec)
elif command -v srun >/dev/null 2>&1; then
  launcher_base=(srun)
else
  echo "Neither mpiexec nor srun is available in PATH" >&2
  exit 1
fi

export OMP_PROC_BIND=spread
export OMP_PLACES=cores

for eps in "${eps_list[@]}"; do
  digits=$(python3 - <<PY
import math
eps = float("$eps")
print(int(round(-math.log10(eps))))
PY
)
  for mpi in "${mpi_list[@]}"; do
    for omp in "${omp_list[@]}"; do
      csv_path="$out_dir/bench_digits${digits}_mpi${mpi}_omp${omp}.csv"
      if [[ -s "$csv_path" ]]; then
        echo "[skip] $csv_path"
        continue
      fi

      echo "[run] eps=$eps mpi=$mpi omp=$omp"
      export OMP_NUM_THREADS="$omp"

      if [[ "${launcher_base[0]}" == "mpiexec" ]]; then
        launch_args=(-n "$mpi" --bind-to core)
        if [[ -n "${SLURM_JOB_NUM_NODES:-}" ]] && (( SLURM_JOB_NUM_NODES >= mpi )); then
          launch_args+=(--map-by "ppr:1:node:PE=${omp}")
        else
          launch_args+=(--map-by "slot:PE=${omp}")
        fi

        "${launcher_base[@]}" "${launch_args[@]}" \
          "$benchmark_bin" \
          --mode benchmark \
          --out-dir "$out_dir" \
          --csv "$csv_path" \
          --n-src "$n_src" \
          --n-trg "$n_trg" \
          --eps "$eps" \
          --n-runs "$n_runs" \
          --warmup-runs "$warmup_runs" \
          --n-per-leaf "$n_per_leaf" \
          --seed "$seed"
      else
        "${launcher_base[@]}" -n "$mpi" \
          "$benchmark_bin" \
          --mode benchmark \
          --out-dir "$out_dir" \
          --csv "$csv_path" \
          --n-src "$n_src" \
          --n-trg "$n_trg" \
          --eps "$eps" \
          --n-runs "$n_runs" \
          --warmup-runs "$warmup_runs" \
          --n-per-leaf "$n_per_leaf" \
          --seed "$seed"
      fi
    done
  done
done
