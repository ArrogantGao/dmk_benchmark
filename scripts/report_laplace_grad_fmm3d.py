#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from pathlib import Path


def parse_header(path: Path):
    meta = {}
    with path.open() as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            text = line[1:].strip()
            if ":" not in text:
                continue
            key, value = text.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta


def parse_rows(path: Path):
    with path.open() as fh:
        reader = csv.DictReader(row for row in fh if not row.startswith("#"))
        return list(reader)


def to_float(row, key):
    return float(row[key])


def aggregate_file(path: Path):
    meta = parse_header(path)
    rows = parse_rows(path)
    if not rows:
        raise RuntimeError(f"No benchmark rows found in {path}")

    eps = float(meta["eps"])
    digits = int(round(-math.log10(eps)))
    numeric_cols = [
        "pot_build_time",
        "pot_eval_time",
        "pot_total_time",
        "pot_eval_pts_per_sec",
        "pot_total_pts_per_sec",
        "potgrad_build_time",
        "potgrad_eval_time",
        "potgrad_total_time",
        "potgrad_eval_pts_per_sec",
        "potgrad_total_pts_per_sec",
        "eval_overhead",
        "total_overhead",
        "pot_src_rel_l2",
        "pot_trg_rel_l2",
        "pot_src_max_rel",
        "pot_trg_max_rel",
        "grad_src_rel_l2",
        "grad_trg_rel_l2",
        "grad_src_max_rel",
        "grad_trg_max_rel",
    ]
    med = {col: statistics.median(to_float(row, col) for row in rows) for col in numeric_cols}
    med["eps"] = eps
    med["digits"] = digits
    med["mpi_ranks"] = int(meta["mpi_ranks"])
    med["omp_threads_per_rank"] = int(meta["omp_threads_per_rank"])
    med["n_src"] = int(meta["n_src"])
    med["n_trg"] = int(meta["n_trg"])
    med["measured_runs"] = int(meta["measured_runs"])
    med["reference_eps"] = float(meta["reference_eps"])
    return med


def format_float(value):
    return f"{value:.3e}"


def format_time(value):
    return f"{value:.3f}"


def make_table(records):
    header = (
        "| MPI | OMP | pot eval(s) | pot total(s) | pot+grad eval(s) | pot+grad total(s) | eval overhead | total overhead |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for rec in records:
        rows.append(
            "| {mpi_ranks} | {omp_threads_per_rank} | {pot_eval} | {pot_total} | {potgrad_eval} | {potgrad_total} | {eval_over}x | {total_over}x |".format(
                mpi_ranks=rec["mpi_ranks"],
                omp_threads_per_rank=rec["omp_threads_per_rank"],
                pot_eval=format_time(rec["pot_eval_time"]),
                pot_total=format_time(rec["pot_total_time"]),
                potgrad_eval=format_time(rec["potgrad_eval_time"]),
                potgrad_total=format_time(rec["potgrad_total_time"]),
                eval_over=f'{rec["eval_overhead"]:.2f}',
                total_over=f'{rec["total_overhead"]:.2f}',
            )
        )
    return header + "\n".join(rows) + "\n"


def make_accuracy_table(records):
    header = (
        "| MPI | OMP | pot src relL2 | pot trg relL2 | pot src maxRel | pot trg maxRel | grad src relL2 | grad trg relL2 | grad src maxRel | grad trg maxRel |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for rec in records:
        rows.append(
            "| {mpi_ranks} | {omp_threads_per_rank} | {pot_src_l2} | {pot_trg_l2} | {pot_src_max} | {pot_trg_max} | {grad_src_l2} | {grad_trg_l2} | {grad_src_max} | {grad_trg_max} |".format(
                mpi_ranks=rec["mpi_ranks"],
                omp_threads_per_rank=rec["omp_threads_per_rank"],
                pot_src_l2=format_float(rec["pot_src_rel_l2"]),
                pot_trg_l2=format_float(rec["pot_trg_rel_l2"]),
                pot_src_max=format_float(rec["pot_src_max_rel"]),
                pot_trg_max=format_float(rec["pot_trg_max_rel"]),
                grad_src_l2=format_float(rec["grad_src_rel_l2"]),
                grad_trg_l2=format_float(rec["grad_trg_rel_l2"]),
                grad_src_max=format_float(rec["grad_src_max_rel"]),
                grad_trg_max=format_float(rec["grad_trg_max_rel"]),
            )
        )
    return header + "\n".join(rows) + "\n"


def generate_report(records):
    if not records:
        raise RuntimeError("No benchmark CSV files were found")

    records = sorted(records, key=lambda r: (r["digits"], r["mpi_ranks"], r["omp_threads_per_rank"]))
    first = records[0]
    lines = []
    lines.append("# Laplace Potential And Gradient FMM3D Benchmark Report")
    lines.append("")
    lines.append("**Date:** 2026-03-24")
    lines.append("**Kernel:** 3D Laplace")
    lines.append(f'**Sources:** {first["n_src"]}')
    lines.append(f'**Targets:** {first["n_trg"]}')
    lines.append(f'**Reference:** FMM3D at eps={first["reference_eps"]:.1e}')
    lines.append("")
    lines.append("## Formulas")
    lines.append("")
    lines.append(r"\[")
    lines.append(r"\mathrm{relL2}(u, u^{ref}) = \sqrt{\frac{\sum_i |u_i - u_i^{ref}|^2}{\sum_i |u_i^{ref}|^2}}, \qquad \mathrm{relL2}(g, g^{ref}) = \sqrt{\frac{\sum_i \|g_i - g_i^{ref}\|_2^2}{\sum_i \|g_i^{ref}\|_2^2}}")
    lines.append(r"\]")
    lines.append("")
    lines.append(r"\[")
    lines.append(r"\mathrm{maxRel}(u, u^{ref}) = \max_i \frac{|u_i - u_i^{ref}|}{|u_i^{ref}|}, \qquad \mathrm{maxRel}(g, g^{ref}) = \max_i \frac{\|g_i - g_i^{ref}\|_2}{\|g_i^{ref}\|_2}")
    lines.append(r"\]")
    lines.append("")
    lines.append(r"\[")
    lines.append(r"S(p, t) = \frac{T_{1,1}}{T_{p,t}}, \qquad E(p, t) = \frac{S(p, t)}{pt}, \qquad H = \frac{T_{\mathrm{pot+grad}}}{T_{\mathrm{pot}}}")
    lines.append(r"\]")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- FMM3D uses the `1/(4*pi*r)` Laplace normalization.")
    lines.append("- DMK potentials and gradients are scaled by `1/(4*pi)` before comparison.")
    lines.append("- `pot-only` timings are the baseline; `pot+grad` timings report the cost of computing both outputs.")
    lines.append("- Times below are medians over the measured runs in each CSV.")
    lines.append("")

    baseline = None
    for rec in records:
        if rec["mpi_ranks"] == 1 and rec["omp_threads_per_rank"] == 1 and rec["digits"] == 3:
            baseline = rec
            break

    if baseline is not None:
        lines.append("## Baseline")
        lines.append("")
        lines.append(
            f'Baseline for `(digits=3, mpi=1, omp=1)`: pot-only total `{baseline["pot_total_time"]:.3f}` s, '
            f'pot+grad total `{baseline["potgrad_total_time"]:.3f}` s, overhead `{baseline["total_overhead"]:.2f}x`.'
        )
        lines.append("")

    grouped = {}
    for rec in records:
        grouped.setdefault(rec["digits"], []).append(rec)

    for digits, group in grouped.items():
        lines.append(f"## {digits} Digits")
        lines.append("")
        lines.append("### Timing")
        lines.append("")
        lines.append(make_table(group))
        lines.append("")
        lines.append("### Accuracy")
        lines.append("")
        lines.append(make_accuracy_table(group))
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("bench_digits*_mpi*_omp*.csv"))
    records = [aggregate_file(path) for path in files]
    report = generate_report(records)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)


if __name__ == "__main__":
    main()
