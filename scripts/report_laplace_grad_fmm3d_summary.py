#!/usr/bin/env python3
import argparse
import csv
import math
import os
import statistics
from pathlib import Path
from xml.sax.saxutils import escape


ACCURACY_FIELDS = [
    "pot_src_rel_l2",
    "pot_trg_rel_l2",
    "pot_src_max_rel",
    "pot_trg_max_rel",
    "grad_src_rel_l2",
    "grad_trg_rel_l2",
    "grad_src_max_rel",
    "grad_trg_max_rel",
]


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
        *ACCURACY_FIELDS,
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


def group_by_digits(records):
    grouped = {}
    for rec in records:
        grouped.setdefault(rec["digits"], []).append(rec)
    for digits in grouped:
        grouped[digits].sort(key=lambda rec: (rec["mpi_ranks"], rec["omp_threads_per_rank"]))
    return grouped


def validate_accuracy_consistency(group, digits):
    for field in ACCURACY_FIELDS:
        values = [rec[field] for rec in group]
        v_min = min(values)
        v_max = max(values)
        scale = max(abs(v_min), abs(v_max), 1.0)
        if (v_max - v_min) > max(1e-14, 1e-8 * scale):
            raise RuntimeError(
                f"Accuracy field {field} varies too much within digits={digits}: "
                f"min={v_min:.16e}, max={v_max:.16e}"
            )


def summarize_accuracy_group(group, digits):
    validate_accuracy_consistency(group, digits)
    summary = {"digits": digits}
    for field in ACCURACY_FIELDS:
        summary[field] = statistics.median(rec[field] for rec in group)
    return summary


def make_accuracy_summary_table(grouped):
    header = (
        "| Digits | pot src relL2 | pot trg relL2 | pot src maxRel | pot trg maxRel | "
        "grad src relL2 | grad trg relL2 | grad src maxRel | grad trg maxRel |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for digits in sorted(grouped):
        group = grouped[digits]
        rec = summarize_accuracy_group(group, digits)
        rows.append(
            "| {digits} | {pot_src_l2} | {pot_trg_l2} | {pot_src_max} | {pot_trg_max} | "
            "{grad_src_l2} | {grad_trg_l2} | {grad_src_max} | {grad_trg_max} |".format(
                digits=digits,
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


def svg_line(points, color):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points), color


def draw_line_panel(lines, x_labels, panel_title, y_label, accessor, out_lines, x0, y0, width, height, colors,
                    y_mode):
    plot_left = x0 + 58
    plot_right = x0 + width - 18
    plot_top = y0 + 18
    plot_bottom = y0 + height - 36
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    values = [accessor(rec) for rec in lines]
    y_min = min(values)
    y_max = max(values)
    if y_mode == "overhead":
        y_low = max(0.0, y_min * 0.85)
        y_high = y_max * 1.10
    else:
        y_low = 0.0
        y_high = y_max * 1.08
    if math.isclose(y_low, y_high):
        y_high = y_low + 1.0

    x_keys = sorted({rec["omp_threads_per_rank"] for rec in lines})
    x_pos = {
        omp: plot_left + idx * plot_width / max(1, len(x_keys) - 1)
        for idx, omp in enumerate(x_keys)
    }

    def map_y(value):
        frac = (value - y_low) / (y_high - y_low)
        return plot_bottom - frac * plot_height

    out_lines.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{width:.2f}" height="{height:.2f}" fill="white" stroke="#d9d9d9"/>')
    out_lines.append(f'<text x="{x0 + width / 2:.2f}" y="{y0 + 12:.2f}" text-anchor="middle" font-size="14" font-weight="bold">{escape(panel_title)}</text>')

    for i in range(5):
        tick_value = y_low + i * (y_high - y_low) / 4
        y = map_y(tick_value)
        out_lines.append(f'<line x1="{plot_left:.2f}" y1="{y:.2f}" x2="{plot_right:.2f}" y2="{y:.2f}" stroke="#ededed"/>')
        out_lines.append(
            f'<text x="{plot_left - 8:.2f}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{tick_value:.2f}</text>'
        )

    out_lines.append(f'<line x1="{plot_left:.2f}" y1="{plot_top:.2f}" x2="{plot_left:.2f}" y2="{plot_bottom:.2f}" stroke="#444"/>')
    out_lines.append(f'<line x1="{plot_left:.2f}" y1="{plot_bottom:.2f}" x2="{plot_right:.2f}" y2="{plot_bottom:.2f}" stroke="#444"/>')

    for omp, x in x_pos.items():
        out_lines.append(f'<line x1="{x:.2f}" y1="{plot_bottom:.2f}" x2="{x:.2f}" y2="{plot_bottom + 5:.2f}" stroke="#444"/>')
        out_lines.append(f'<text x="{x:.2f}" y="{plot_bottom + 18:.2f}" text-anchor="middle" font-size="11">{omp}</text>')

    out_lines.append(
        f'<text x="{plot_left + plot_width / 2:.2f}" y="{y0 + height - 8:.2f}" text-anchor="middle" font-size="12">OMP threads per rank</text>'
    )
    out_lines.append(
        f'<text x="{x0 + 14:.2f}" y="{y0 + height / 2:.2f}" text-anchor="middle" font-size="12" transform="rotate(-90 {x0 + 14:.2f} {y0 + height / 2:.2f})">{escape(y_label)}</text>'
    )

    for mpi in sorted({rec["mpi_ranks"] for rec in lines}):
        points = []
        for omp in x_keys:
            rec = next(r for r in lines if r["mpi_ranks"] == mpi and r["omp_threads_per_rank"] == omp)
            points.append((x_pos[omp], map_y(accessor(rec))))
        point_str, color = svg_line(points, colors[mpi])
        out_lines.append(f'<polyline points="{point_str}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for x, y in points:
            out_lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{color}"/>')


def make_timing_plot_svg(group, digits, output_path):
    colors = {1: "#1f77b4", 2: "#d95f02", 4: "#2ca02c"}
    width = 1080
    height = 860
    legend_y = 34
    panel_x = 34
    panel_width = width - 2 * panel_x
    panel_height = 220
    panel_gap = 34

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.2f}" y="22" text-anchor="middle" font-size="18" font-weight="bold">Laplace timing summary, {digits} digits</text>',
    ]

    legend_x = 50
    lines.append(f'<text x="{legend_x:.2f}" y="{legend_y:.2f}" font-size="12" font-weight="bold">MPI ranks</text>')
    for idx, mpi in enumerate(sorted({rec["mpi_ranks"] for rec in group})):
        x = legend_x + 80 + idx * 110
        color = colors.get(mpi, "#444444")
        lines.append(f'<line x1="{x:.2f}" y1="{legend_y - 4:.2f}" x2="{x + 24:.2f}" y2="{legend_y - 4:.2f}" stroke="{color}" stroke-width="2.2"/>')
        lines.append(f'<circle cx="{x + 12:.2f}" cy="{legend_y - 4:.2f}" r="3.2" fill="{color}"/>')
        lines.append(f'<text x="{x + 34:.2f}" y="{legend_y:.2f}" font-size="12">{mpi}</text>')

    panels = [
        ("pot-only total time", "seconds", lambda rec: rec["pot_total_time"], "time"),
        ("pot+grad total time", "seconds", lambda rec: rec["potgrad_total_time"], "time"),
        ("total overhead", "ratio", lambda rec: rec["total_overhead"], "overhead"),
    ]

    for idx, (title, y_label, accessor, y_mode) in enumerate(panels):
        y = 62 + idx * (panel_height + panel_gap)
        draw_line_panel(group, None, title, y_label, accessor, lines, panel_x, y, panel_width, panel_height, colors, y_mode)

    lines.append("</svg>")
    output_path.write_text("\n".join(lines))


def relative_plot_reference(output_path, plot_path):
    return os.path.relpath(plot_path, output_path.parent)


def make_observations(grouped):
    first_digits = min(grouped)
    last_digits = max(grouped)
    first = grouped[first_digits][0]
    last = grouped[last_digits][0]

    max_total_over = max(rec["total_overhead"] for group in grouped.values() for rec in group)
    max_total_rec = max(
        (rec for group in grouped.values() for rec in group),
        key=lambda rec: rec["total_overhead"],
    )

    high_precision = grouped[last_digits]
    high_min = min(rec["total_overhead"] for rec in high_precision)
    high_max = max(rec["total_overhead"] for rec in high_precision)

    return [
        "Accuracy is invariant across MPI ranks and OMP threads in this dataset; the summary table below keeps one representative row per requested digits level after an explicit consistency check.",
        "Potential and gradient errors improve monotonically with digits. For example, target gradient relL2 drops from "
        f"`{format_float(first['grad_trg_rel_l2'])}` at {first_digits} digits to `{format_float(last['grad_trg_rel_l2'])}` at {last_digits} digits.",
        "Timing overhead is most severe at low precision and low concurrency. The largest total overhead in the matrix is "
        f"`{max_total_over:.2f}x` at `(digits={max_total_rec['digits']}, mpi={max_total_rec['mpi_ranks']}, omp={max_total_rec['omp_threads_per_rank']})`.",
        f"At {last_digits} digits, total overhead is much tighter, staying in the `{high_min:.2f}x` to `{high_max:.2f}x` range across the full MPI/OMP sweep.",
    ]


def generate_summary(records, output_path, plot_dir):
    if not records:
        raise RuntimeError("No benchmark CSV files were found")

    grouped = group_by_digits(records)
    first = records[0]

    lines = []
    lines.append("# Laplace Potential And Gradient FMM3D Benchmark Summary")
    lines.append("")
    lines.append("**Date:** 2026-03-24")
    lines.append("**Kernel:** 3D Laplace")
    lines.append(f'**Sources:** {first["n_src"]}')
    lines.append(f'**Targets:** {first["n_trg"]}')
    lines.append(f'**Reference:** FMM3D at eps={first["reference_eps"]:.1e}')
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This summary is derived from the same CSVs as the full benchmark report.")
    lines.append("- Accuracy rows are merged by digits because the benchmark data shows the same accuracy across MPI/OMP settings for each digits level.")
    lines.append("- Timing is shown as figures instead of large markdown tables.")
    lines.append("")
    lines.append("## Accuracy")
    lines.append("")
    lines.append(make_accuracy_summary_table(grouped))
    lines.append("")
    lines.append("## Timing Figures")
    lines.append("")

    plot_dir.mkdir(parents=True, exist_ok=True)
    for digits in sorted(grouped):
        plot_path = plot_dir / f"timing_digits{digits}.svg"
        make_timing_plot_svg(grouped[digits], digits, plot_path)
        rel_plot = relative_plot_reference(output_path, plot_path)
        lines.append(f"### {digits} Digits")
        lines.append("")
        lines.append(f"![Timing summary for {digits} digits]({rel_plot})")
        lines.append("")

    lines.append("## Observations")
    lines.append("")
    for item in make_observations(grouped):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("bench_digits*_mpi*_omp*.csv"))
    if not files:
        raise RuntimeError(f"No benchmark CSV files were found in {input_dir}")

    records = [aggregate_file(path) for path in files]
    output = Path(args.output)
    plot_dir = Path(args.plot_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = generate_summary(records, output, plot_dir)
    output.write_text(report)


if __name__ == "__main__":
    main()
