# Laplace Pot+Grad Summary Report Design

**Date:** 2026-03-24

**Goal:** Add a separate summary report for the full `2026-03-24-laplace-pot-grad-fmm3d-benchmark.md` results that keeps the full report intact, collapses duplicated accuracy rows, and replaces timing tables with plots.

## Context

The full report under `docs/reports/2026-03-24-laplace-pot-grad-fmm3d-benchmark.md` is complete but long. The timing section is currently a large table for every `(digits, mpi, omp)` combination, and the accuracy section repeats the same row for every MPI/OMP setting. The raw data already lives in `results/laplace_pot_grad_fmm3d`.

## Chosen Approach

Create a new sidecar summary generator that:

- reads the same benchmark CSV files as the full report generator
- writes a separate markdown summary next to the full report
- emits plot images into a sibling plot directory
- leaves the full report and its generator untouched

## Output Shape

The summary report should contain:

- a short setup section
- one merged accuracy table with one row per digits level
- a timing section that embeds generated figures instead of large markdown tables
- a short observations section that calls out the main timing and accuracy trends

The timing figures should show:

- `pot-only` total time versus OMP threads
- `pot+grad` total time versus OMP threads
- overhead ratio versus OMP threads

Each figure should separate MPI ranks clearly, either by subplot or by multiple labeled lines. One figure per digits level is sufficient.

## Non-Goals

- do not rewrite the full report
- do not change the benchmark data format
- do not rerun the benchmark

## Implementation Notes

- put the new generator in `scripts/`
- keep the parsing logic close to the existing report generator to avoid divergence
- use `matplotlib` to emit static PNG plots
- derive the merged accuracy rows by grouping records by `digits` and asserting the accuracy fields are effectively identical across MPI/OMP
