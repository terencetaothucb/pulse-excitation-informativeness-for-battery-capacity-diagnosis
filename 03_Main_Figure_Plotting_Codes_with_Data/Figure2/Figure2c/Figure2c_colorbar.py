#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone colorbar/legend for PulseBat-combo Figure 2c.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DPI = 600
LEGEND_W_CM = 6.0
LEGEND_H_CM = 0.45

CRATES: Sequence[Tuple[str, str, str]] = (
    ("0.5C", "0.5C", "#CFC3EE"),
    ("1C", "1.0C", "#D07BBB"),
    ("1.5C", "1.5C", "#9A6CC4"),
    ("2C", "2.0C", "#7971C5"),
    ("2.5C", "2.5C", "#49374F"),
)


def cm_to_in(cm: float) -> float:
    return cm / 2.54


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.fontsize": 7,
    })


def plot_colorbar(out_dir: Path, dpi: int) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(cm_to_in(LEGEND_W_CM), cm_to_in(LEGEND_H_CM)), dpi=dpi)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=color,
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=4.2,
            label=label,
        )
        for _, label, color in CRATES
    ]
    fig.legend(
        handles=handles,
        loc="center",
        ncol=len(CRATES),
        frameon=False,
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=0.85,
        borderaxespad=0.0,
    )

    png_path = out_dir / "Figure2c_colorbar.png"
    pdf_path = out_dir / "Figure2c_colorbar.pdf"
    fig.savefig(png_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(pdf_path, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw PulseBat-combo Figure2c colorbar.")
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    plot_colorbar(Path(args.out_dir), int(args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
