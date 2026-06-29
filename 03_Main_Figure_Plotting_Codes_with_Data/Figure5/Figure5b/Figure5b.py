#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5b for PulseBat-combo.

Selected-model MAE and compute-time trade-off across combinational feature
strategies.
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
from matplotlib.ticker import FuncFormatter, LogLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure5_common import (  # noqa: E402
    DATA_XLSX,
    DPI,
    FEATURE_COLORS,
    FEATURE_ORDER,
    MATERIAL_ORDER,
    MATERIAL_SHORT,
    MODEL_ORDER,
    cm_to_in,
    load_sheet_table,
    setup_style,
    write_csv,
    zero_trim,
)


FIG_W_CM = 8.72
FIG_H_CM = 5.90
OUT_STEM = "Figure5b"
BAR_WIDTH = 0.22
MODEL_LABELS: Dict[str, str] = {
    "linear": "linear",
    "rf": "rf",
    "xgb": "xgb",
    "gpr": "gpr",
    "mlp": "mlp",
    "transformer": "transformer",
    "informer": "informer",
}
MATERIAL_MARKERS: Dict[str, str] = {
    "20Ah LFP": "o",
    "35Ah LFP": "s",
    "68Ah LFP": "^",
}


def _rows_by_component(rows: Sequence[Dict[str, object]], component: str) -> Sequence[Dict[str, object]]:
    return [r for r in rows if r.get("plot_component") == component]


def _find_row(rows: Sequence[Dict[str, object]], model: str, strategy: str) -> Dict[str, object]:
    for row in rows:
        if row.get("model") == model and row.get("feature_strategy_short") == strategy:
            return row
    raise KeyError((model, strategy))


def plot(rows: Sequence[Dict[str, object]], out_dir: Path, fig_w_cm: float, fig_h_cm: float, dpi: int) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "Figure5b_plotting_data.csv", rows)

    bars = list(_rows_by_component(rows, "mae_bar_summary"))
    material_points = list(_rows_by_component(rows, "material_overlap_points"))
    compute = list(_rows_by_component(rows, "compute_time_summary"))

    fig, (ax_mae, ax_time) = plt.subplots(
        2,
        1,
        figsize=(cm_to_in(fig_w_cm), cm_to_in(fig_h_cm)),
        dpi=dpi,
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 0.95]},
    )
    fig.subplots_adjust(left=0.112, right=0.760, bottom=0.175, top=0.965, hspace=0.15)

    x = np.arange(len(MODEL_ORDER), dtype=float)
    offsets = np.linspace(-BAR_WIDTH, BAR_WIDTH, len(FEATURE_ORDER))
    linear_span = (-0.5, 0.5)
    for ax in (ax_mae, ax_time):
        ax.axvspan(linear_span[0], linear_span[1], color="#F1F1F1", zorder=0)

    for i, strategy in enumerate(FEATURE_ORDER):
        xpos = x + offsets[i]
        means = [float(_find_row(bars, model, strategy)["mae"]) for model in MODEL_ORDER]
        errs = [float(_find_row(bars, model, strategy)["mae_std"]) for model in MODEL_ORDER]
        ax_mae.bar(
            xpos,
            means,
            width=BAR_WIDTH * 0.88,
            color=FEATURE_COLORS[strategy],
            edgecolor="white",
            linewidth=0.45,
            label=strategy,
            zorder=2,
        )
        ax_mae.errorbar(xpos, means, yerr=errs, fmt="none", ecolor="#606060", elinewidth=0.55, capsize=1.5, zorder=3)

        for model_idx, model in enumerate(MODEL_ORDER):
            subset = [
                r for r in material_points
                if r.get("model") == model and r.get("feature_strategy_short") == strategy
            ]
            for material in MATERIAL_ORDER:
                point_rows = [r for r in subset if r.get("material") == material]
                if not point_rows:
                    continue
                row = point_rows[0]
                ax_mae.scatter(
                    xpos[model_idx],
                    float(row["mae"]),
                    s=12,
                    marker=MATERIAL_MARKERS[material],
                    facecolor="white",
                    edgecolor="#5C5C5C",
                    linewidth=0.55,
                    zorder=4,
                )

    ax_mae.set_ylabel("MAE", labelpad=2)
    ax_mae.set_ylim(0.0, 0.055)
    ax_mae.set_yticks([0.00, 0.025, 0.05])
    ax_mae.yaxis.set_major_formatter(FuncFormatter(zero_trim))
    ax_mae.grid(True, axis="y", color="#E8E8E8", lw=0.35, zorder=1)
    ax_mae.tick_params(length=2.0, width=0.6, pad=1.2)

    legend_features = fig.legend(
        handles=[
            mpl.patches.Patch(facecolor=FEATURE_COLORS[s], edgecolor="none", label=s)
            for s in FEATURE_ORDER
        ],
        title="Feature strategy",
        loc="upper left",
        bbox_to_anchor=(0.790, 0.925),
        frameon=False,
        handlelength=0.95,
        handletextpad=0.35,
        labelspacing=0.18,
        borderaxespad=0.0,
        fontsize=6,
        title_fontsize=6,
    )
    legend_features.get_title().set_fontweight("normal")
    material_handles = [
        mpl.lines.Line2D([], [], marker=MATERIAL_MARKERS[m], linestyle="None", markersize=4.4, markerfacecolor="white", markeredgecolor="#5C5C5C", label=MATERIAL_SHORT[m])
        for m in MATERIAL_ORDER
    ]
    legend_material = fig.legend(
        handles=material_handles,
        title="Material points",
        loc="upper left",
        bbox_to_anchor=(0.790, 0.695),
        frameon=False,
        handletextpad=0.45,
        labelspacing=0.18,
        borderaxespad=0.0,
        fontsize=6,
        title_fontsize=6,
    )
    legend_material.get_title().set_fontweight("normal")

    for i, strategy in enumerate(FEATURE_ORDER):
        xpos = x + offsets[i]
        for model_idx, model in enumerate(MODEL_ORDER):
            row = _find_row(compute, model, strategy)
            pred_s = float(row["pred_time_s"])
            fit_s = float(row["fit_time_s"])
            total_s = float(row["total_time_s"])
            color = FEATURE_COLORS[strategy]
            ax_time.vlines(xpos[model_idx], pred_s, fit_s, color=color, lw=0.55, alpha=0.85, zorder=1)
            ax_time.scatter(xpos[model_idx], pred_s, s=13, marker="o", facecolor="white", edgecolor=color, linewidth=0.7, zorder=3)
            ax_time.scatter(xpos[model_idx], fit_s, s=13, marker="s", color=color, linewidth=0, zorder=3)
            ax_time.scatter(xpos[model_idx], total_s, s=24, marker="_", color="#333333", linewidth=0.8, zorder=4)

    ax_time.set_yscale("log")
    ax_time.set_ylim(5e-6, 1.5)
    ax_time.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax_time.set_ylabel("Compute time (s, log-scale)", labelpad=2)
    ax_time.set_xlabel("Model", labelpad=2)
    ax_time.grid(True, axis="y", which="both", color="#E8E8E8", lw=0.35)
    ax_time.tick_params(length=2.0, width=0.6, pad=1.2)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=6.2, rotation=22, ha="center", rotation_mode="default")

    time_handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="None", markersize=4, markerfacecolor="white", markeredgecolor="#666666", label="Prediction"),
        mpl.lines.Line2D([], [], marker="s", linestyle="None", markersize=4, color="#666666", label="Training"),
        mpl.lines.Line2D([], [], marker="_", linestyle="None", markersize=6, color="#333333", label="Total"),
    ]
    legend_time = fig.legend(
        handles=time_handles,
        title="Time components",
        loc="upper left",
        bbox_to_anchor=(0.790, 0.415),
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.45,
        labelspacing=0.18,
        fontsize=6,
        title_fontsize=6,
    )
    legend_time.get_title().set_fontweight("normal")

    for ax in (ax_mae, ax_time):
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)

    png_path = out_dir / f"{OUT_STEM}.png"
    pdf_path = out_dir / f"{OUT_STEM}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw PulseBat-combo Figure5b.")
    parser.add_argument("--source_xlsx", type=str, default=str(DATA_XLSX))
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--fig_w_cm", type=float, default=FIG_W_CM)
    parser.add_argument("--fig_h_cm", type=float, default=FIG_H_CM)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    rows = load_sheet_table("5b", Path(args.source_xlsx))
    plot(rows, Path(args.out_dir), float(args.fig_w_cm), float(args.fig_h_cm), int(args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
