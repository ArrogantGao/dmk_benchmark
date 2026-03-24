# Laplace Pot+Grad Summary Report Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a separate summary markdown report and timing plots for the `2026-03-24` Laplace potential+gradient benchmark, while keeping the full report unchanged.

**Architecture:** A new standalone Python report script will read the existing benchmark CSVs, aggregate timing and accuracy metrics, collapse duplicated accuracy rows to one row per requested digits level, emit timing PNG plots, and write a separate markdown summary beside the full report. The existing full-report script remains the source of the detailed tabular view.

**Tech Stack:** Python 3, `csv`, `statistics`, `pathlib`, `matplotlib`

---

### Task 1: Add the summary report generator

**Files:**
- Create: `scripts/report_laplace_grad_fmm3d_summary.py`
- Reference: `scripts/report_laplace_grad_fmm3d.py`

**Step 1: Write the failing implementation target**

Define a new CLI with:

- `--input-dir`
- `--output`
- `--plot-dir`

and fail when no benchmark CSVs are found.

**Step 2: Run it to verify the empty-path behavior**

Run:

```bash
python3 scripts/report_laplace_grad_fmm3d_summary.py --input-dir /tmp/does-not-exist --output /tmp/out.md --plot-dir /tmp/plots
```

Expected: the script exits with a clear error about missing benchmark CSV files.

**Step 3: Implement CSV parsing and aggregation**

Reuse the existing metadata and median aggregation pattern from `scripts/report_laplace_grad_fmm3d.py`.

**Step 4: Run on the real benchmark directory**

Run:

```bash
python3 scripts/report_laplace_grad_fmm3d_summary.py \
  --input-dir results/laplace_pot_grad_fmm3d \
  --output docs/reports/2026-03-24-laplace-pot-grad-fmm3d-benchmark-summary.md \
  --plot-dir docs/reports/plots/2026-03-24-laplace-pot-grad-fmm3d
```

Expected: the summary markdown and plot files are created.

### Task 2: Collapse accuracy rows by digits

**Files:**
- Modify: `scripts/report_laplace_grad_fmm3d_summary.py`

**Step 1: Group records by digits**

For each digits level, collect all records and extract one representative accuracy row.

**Step 2: Validate duplication assumption**

Add a tolerance check that the accuracy fields match across MPI/OMP records for the same digits level. If they do not match, fail loudly rather than silently merging inconsistent data.

**Step 3: Emit one simple accuracy table**

Write one markdown table with columns:

- `digits`
- `pot src relL2`
- `pot trg relL2`
- `pot src maxRel`
- `pot trg maxRel`
- `grad src relL2`
- `grad trg relL2`
- `grad src maxRel`
- `grad trg maxRel`

**Step 4: Regenerate the summary**

Run the same command from Task 1 Step 4.

Expected: the markdown contains a single merged accuracy table.

### Task 3: Replace timing tables with figures

**Files:**
- Modify: `scripts/report_laplace_grad_fmm3d_summary.py`

**Step 1: Add plot generation**

For each digits level, generate a PNG that shows:

- total time for `pot-only`
- total time for `pot+grad`
- total overhead ratio

Use OMP threads on the x-axis and distinguish MPI ranks clearly.

**Step 2: Save plots into the requested plot directory**

Use deterministic filenames such as:

- `timing_digits3.png`
- `timing_digits6.png`
- `timing_digits9.png`
- `timing_digits12.png`

**Step 3: Embed plot links in markdown**

Reference the generated PNGs from the summary markdown.

**Step 4: Regenerate the summary**

Run the same command from Task 1 Step 4.

Expected: the markdown contains image links instead of timing tables.

### Task 4: Add concise observations and verify output

**Files:**
- Modify: `scripts/report_laplace_grad_fmm3d_summary.py`
- Generate: `docs/reports/2026-03-24-laplace-pot-grad-fmm3d-benchmark-summary.md`

**Step 1: Add short report observations**

Summarize:

- accuracy trend versus digits
- timing overhead trend versus digits
- the main low-precision versus high-precision contrast

**Step 2: Run the generator on the production dataset**

Run:

```bash
python3 scripts/report_laplace_grad_fmm3d_summary.py \
  --input-dir results/laplace_pot_grad_fmm3d \
  --output docs/reports/2026-03-24-laplace-pot-grad-fmm3d-benchmark-summary.md \
  --plot-dir docs/reports/plots/2026-03-24-laplace-pot-grad-fmm3d
```

Expected: success with all plots and the summary markdown generated.

**Step 3: Sanity-check the generated files**

Run:

```bash
ls docs/reports/2026-03-24-laplace-pot-grad-fmm3d-benchmark-summary.md
ls docs/reports/plots/2026-03-24-laplace-pot-grad-fmm3d
```

Expected: the markdown file exists and the plot directory contains one PNG per digits level.
