# PBC Runtime Benchmark Notes

**Local validation**

Runtime driver:
- `/mnt/home/xgao1/work/dmk_benchmark/bench_pbc_runtime.cpp`

OpenMP-enabled local binary used for validation:
- `/mnt/home/xgao1/work/dmk_benchmark/build-mpi3/bench_pbc_runtime_omp`

Validation commands run on the workstation:

```bash
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=1 /mnt/home/xgao1/work/dmk_benchmark/build-mpi3/bench_pbc_runtime_omp --n 200000 --n-runs 2 --eps 1e-3 --n-per-leaf 500 --pbc both > /mnt/home/xgao1/work/dmk_benchmark/results/pbc_runtime_local_omp1.csv
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=4 /mnt/home/xgao1/work/dmk_benchmark/build-mpi3/bench_pbc_runtime_omp --n 200000 --n-runs 2 --eps 1e-3 --n-per-leaf 500 --pbc both > /mnt/home/xgao1/work/dmk_benchmark/results/pbc_runtime_local_omp4.csv
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=8 /mnt/home/xgao1/work/dmk_benchmark/build-mpi3/bench_pbc_runtime_omp --n 200000 --n-runs 2 --eps 1e-3 --n-per-leaf 500 --pbc both > /mnt/home/xgao1/work/dmk_benchmark/results/pbc_runtime_local_omp8.csv
```

Observed thread counts:

- `OMP_NUM_THREADS=1` produced `omp_threads_actual=1`
- `OMP_NUM_THREADS=4` produced `omp_threads_actual=4`
- `OMP_NUM_THREADS=8` produced `omp_threads_actual=8`

Representative validation timings:

- `free`, `1 thread`: `eval_min_seconds=1.644130`
- `free`, `4 threads`: `eval_min_seconds=0.826414`
- `free`, `8 threads`: `eval_min_seconds=0.361135`
- `pbc`, `1 thread`: `eval_min_seconds=2.043243`
- `pbc`, `4 threads`: `eval_min_seconds=0.663264`
- `pbc`, `8 threads`: `eval_min_seconds=0.403950`

These runs confirm that OpenMP thread control is functioning on the workstation.

**Rusty Genoa batch script**

- `/mnt/home/xgao1/work/dmk_benchmark/run_pbc_runtime_rusty_genoa.slurm`

Submit with:

```bash
sbatch /mnt/home/xgao1/work/dmk_benchmark/run_pbc_runtime_rusty_genoa.slurm
```

Expected cluster CSV location:

- `/mnt/home/xgao1/work/dmk_benchmark/results/<date>-pbc-runtime-rusty-genoa.csv`
