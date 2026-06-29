#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5f for PulseBat-combo.

Integrated model ranking using equal metric weights: Accuracy, Stability,
Inference, and Fit each contribute 25%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure5_common import (  # noqa: E402
    DATA_XLSX,
    DPI,
    SCORE_COLUMNS,
    cm_to_in,
    load_sheet_table,
    setup_style,
    write_csv,
)


FIG_W_CM = 7.00
FIG_H_CM = 5.25
OUT_STEM = "Figure5f"
SHEET_NAME = "5e"
COMPOSITE_COLUMN = "Composite_equal_weight_score"
TITLE = "Equal weights"


def display_model_name(model: object) -> str:
    name = str(model)
    if name in {"transformer", "informer"}:
        return name.capitalize()
    return name


def plot(rows: Sequence[Dict[str, object]], out_dir: Path, fig_w_cm: float, fig_h_cm: float, dpi: int) -> None:
    setup_style()
    mpl.rcParams.update({
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "Figure5f_integrated_ranking_equal_weights.csv", rows)

    ranked = sorted(rows, key=lambda r: int(r["rank"]))
    models = [display_model_name(r["model"]) for r in ranked]
    score_values = np.asarray([[float(r[col]) for col, _ in SCORE_COLUMNS] for r in ranked], dtype=float)
    composite = np.asarray([float(r[COMPOSITE_COLUMN]) for r in ranked], dtype=float)

    fig = plt.figure(figsize=(cm_to_in(fig_w_cm), cm_to_in(fig_h_cm)), dpi=dpi)
    ax_heat = fig.add_axes([0.1693, 0.2990, 0.2957, 0.5810])
    cax = fig.add_axes([0.1693, 0.070, 0.2957, 0.026])
    ax_bar = fig.add_axes([0.600, 0.274, 0.350, 0.606])

    cmap = plt.get_cmap("viridis")
    image = ax_heat.imshow(score_values, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    ax_heat.set_title("Multi-metric scores", fontsize=7, pad=2)
    ax_heat.set_xticks(np.arange(len(SCORE_COLUMNS)))
    ax_heat.set_xticklabels(
        [label for _, label in SCORE_COLUMNS],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=7,
    )
    ax_heat.set_yticks(np.arange(len(models)))
    ax_heat.set_yticklabels([])
    ax_heat.tick_params(length=0, pad=1.0)
    for row_idx, model in enumerate(models):
        ax_heat.text(
            -0.045,
            row_idx,
            model,
            transform=ax_heat.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7,
            clip_on=False,
        )
    for row_idx in range(score_values.shape[0]):
        for col_idx in range(score_values.shape[1]):
            value = score_values[row_idx, col_idx]
            ax_heat.text(col_idx, row_idx, f"{value:.0f}", ha="center", va="center", fontsize=7, color="white" if value < 45 else "black")
    for spine in ax_heat.spines.values():
        spine.set_linewidth(0.75)

    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7, length=1.6, width=0.5)
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.outline.set_linewidth(0.55)

    y = np.arange(len(ranked), dtype=float)
    colors = ["#4E79A7"] + ["#C9CED6"] * (len(ranked) - 1)
    ax_bar.barh(y, composite, color=colors, edgecolor="white", linewidth=0.5)
    ax_bar.set_title("Integrated ranking", fontsize=7, pad=2)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([])
    ax_bar.set_xlim(0, 112)
    ax_bar.set_xlabel("Composite score", labelpad=1, fontsize=7)
    ax_bar.grid(True, axis="x", color="#E8E8E8", lw=0.35)
    ax_bar.invert_yaxis()
    ax_bar.tick_params(axis="x", length=2.0, width=0.6, pad=1.0)
    ax_bar.tick_params(axis="y", length=0)
    for yi, score, row in zip(y, composite, ranked):
        label_color = "white" if int(row["rank"]) == 1 else "black"
        ax_bar.text(1.5, yi, f"#{int(row['rank'])}", va="center", ha="left", fontsize=7, color=label_color)
        ax_bar.text(score + 1.0, yi, f"{score:.1f}", va="center", ha="left", fontsize=7)
    for spine in ax_bar.spines.values():
        spine.set_linewidth(0.75)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.text(0.475, 0.985, TITLE, ha="center", va="top", fontsize=7)

    png_path = out_dir / f"{OUT_STEM}.png"
    pdf_path = out_dir / f"{OUT_STEM}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw PulseBat-combo Figure5f.")
    parser.add_argument("--source_xlsx", type=str, default=str(DATA_XLSX))
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--fig_w_cm", type=float, default=FIG_W_CM)
    parser.add_argument("--fig_h_cm", type=float, default=FIG_H_CM)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    rows = load_sheet_table(SHEET_NAME, Path(args.source_xlsx))
    plot(rows, Path(args.out_dir), float(args.fig_w_cm), float(args.fig_h_cm), int(args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
