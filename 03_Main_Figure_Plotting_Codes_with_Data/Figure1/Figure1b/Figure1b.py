#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 1b for PulseBat-combo: compact protocol table.

The table summarizes the pulse protocol values used in the combo paper:
C-rate, pulse width, and SOC levels.  The visual style follows the existing
PulseBat figure family: Arial text, dark outer frame, thin internal rules,
and restrained scientific colors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def cm_to_in(cm: float) -> float:
    return cm / 2.54


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def draw_table_row(
    ax: plt.Axes,
    labels: Sequence[str],
    row_y0: float,
    row_y1: float,
    x0: float,
    x1: float,
    facecolor: str,
    edgecolor: str,
    textcolors: Sequence[str],
    fontsize: float,
) -> None:
    n = len(labels)
    cell_w = (x1 - x0) / float(n)
    ax.add_patch(Rectangle((x0, row_y0), x1 - x0, row_y1 - row_y0, facecolor=facecolor, edgecolor="none", zorder=0))
    for i in range(1, n):
        x = x0 + i * cell_w
        ax.plot([x, x], [row_y0, row_y1], color=edgecolor, lw=0.42, zorder=1)
    for i, label in enumerate(labels):
        ax.text(
            x0 + (i + 0.5) * cell_w,
            (row_y0 + row_y1) / 2.0,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=textcolors[i],
            zorder=2,
        )


def plot_figure1b(out_dir: Path, fig_w_cm: float = 7.3, fig_h_cm: float = 3.57, dpi: int = 600) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    dark = "#092431"
    rule = "#C7D0D4"
    text = "#111111"
    frame_lw = 0.75

    c_rates = ["+0.5", "-0.5", "+1.0", "-1.0", "+1.5", "-1.5", "+2.0", "-2.0", "+2.5", "-2.5"]
    widths = ["30", "50", "70", "100", "300", "500", "700", "1000", "3000", "5000"]
    socs = [str(v) for v in range(5, 91, 5)]

    c_text_colors = ["#B63A2C" if v.startswith("+") else "#1D5F95" for v in c_rates]
    width_text_colors = ["#111111"] * len(widths)
    soc_text_colors = ["#111111"] * len(socs)

    fig = plt.figure(figsize=(cm_to_in(fig_w_cm), cm_to_in(fig_h_cm)), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left, right = 0.035, 0.985
    bottom, top = 0.13, 0.91
    label_right = 0.235
    row_edges = [top, bottom + 2.0 / 3.0 * (top - bottom), bottom + 1.0 / 3.0 * (top - bottom), bottom]

    ax.add_patch(Rectangle((left, bottom), right - left, top - bottom, facecolor="white", edgecolor=dark, linewidth=frame_lw))
    ax.plot([label_right, label_right], [bottom, top], color=dark, lw=frame_lw)

    for y in row_edges[1:-1]:
        ax.plot([left, right], [y, y], color=rule, lw=0.55)

    for i in range(3):
        y0, y1 = row_edges[i + 1], row_edges[i]
        if i % 2 == 1:
            ax.add_patch(Rectangle((label_right, y0), right - label_right, y1 - y0, facecolor="#FBFCFC", edgecolor="none", zorder=0))

    row_labels = ["C Rate (-)", "Width (ms)", "SOC (%)"]
    for i, label in enumerate(row_labels):
        y0, y1 = row_edges[i + 1], row_edges[i]
        ax.text(
            (left + label_right) / 2.0,
            (y0 + y1) / 2.0,
            label,
            ha="center",
            va="center",
            fontsize=8.2,
            color=text,
        )

    content_x0 = label_right
    content_x1 = right

    draw_table_row(
        ax, c_rates, row_edges[1], row_edges[0], content_x0, content_x1,
        facecolor="#FFF9F4", edgecolor=rule, textcolors=c_text_colors, fontsize=6.2
    )
    draw_table_row(
        ax, widths, row_edges[2], row_edges[1], content_x0, content_x1,
        facecolor="#F8FBFC", edgecolor=rule, textcolors=width_text_colors, fontsize=6.2
    )
    draw_table_row(
        ax, socs, row_edges[3], row_edges[2], content_x0, content_x1,
        facecolor="#FBFCF9", edgecolor=rule, textcolors=soc_text_colors, fontsize=5.5
    )

    png_path = out_dir / "Figure1b.png"
    pdf_path = out_dir / "Figure1b.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw combo Figure1b protocol table.")
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--fig_w_cm", type=float, default=7.3)
    parser.add_argument("--fig_h_cm", type=float, default=3.57)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    plot_figure1b(Path(args.out_dir), fig_w_cm=float(args.fig_w_cm), fig_h_cm=float(args.fig_h_cm), dpi=int(args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
