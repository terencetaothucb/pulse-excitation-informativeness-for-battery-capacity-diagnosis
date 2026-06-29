#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5a for PulseBat-combo.

Linear-model MAE distributions across materials and feature strategies, with
the lower bubble heatmap summarizing combinational-feature improvement relative
to the single-feature baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure5_common import (  # noqa: E402
    DATA_XLSX,
    DPI,
    FEATURE_COLORS,
    FEATURE_ORDER,
    MATERIAL_ORDER,
    MATERIAL_SHORT,
    cm_to_in,
    load_figure5a_tables,
    setup_style,
    write_csv,
    zero_trim,
)


FIG_W_CM = 4.96
FIG_H_CM = 5.90
OUT_STEM = "Figure5a"
STRATEGY_ORDER: Sequence[str] = ("Single", "Comb.-C", "Comb.-W", "Comb.-SOC")
Y_LIM = (0.0, 0.061)
Y_TICKS: Sequence[float] = (0.00, 0.03, 0.06)
VIOLIN_FACE_ALPHA = 0.35
VIOLIN_EDGE_COLOR = "#466F6B"
VIOLIN_EDGE_LW = 0.75


def _darken_color(color: str, factor: float = 0.15) -> tuple[float, float, float]:
    rgb = np.asarray(mpl.colors.to_rgb(color), dtype=float)
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


def _values(records: Sequence[Dict[str, object]], material: str, strategy: str) -> List[float]:
    return [
        float(r["mae_test_median"])
        for r in records
        if r.get("material") == material and r.get("feature_strategy_short") == strategy
    ]


def _scatter_points(ax: plt.Axes, x: float, values: Sequence[float], color: str) -> None:
    if not values:
        return
    rng = np.random.default_rng(5)
    sample = np.asarray(values, dtype=float)
    if len(sample) > 550:
        sample = rng.choice(sample, size=550, replace=False)
    jitter = rng.normal(0.0, 0.050, size=len(sample))
    ax.scatter(
        np.full(len(sample), x) + jitter,
        sample,
        s=5.2,
        color=color,
        alpha=0.75,
        edgecolors="#FFFFFF",
        linewidths=0.35,
        rasterized=True,
        zorder=4,
    )


def plot(violin_rows: Sequence[Dict[str, object]], bubble_rows: Sequence[Dict[str, object]], out_dir: Path, fig_w_cm: float, fig_h_cm: float, dpi: int) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "Figure5a_violin_raw_points.csv", violin_rows)
    write_csv(out_dir / "Figure5a_bubble_heatmap_summary.csv", bubble_rows)

    fig = plt.figure(figsize=(cm_to_in(fig_w_cm), cm_to_in(fig_h_cm)), dpi=dpi)
    ax_top = fig.add_axes([0.17, 0.52, 0.80, 0.37])
    ax_bot = fig.add_axes([0.18, 0.235, 0.55, 0.235])

    positions: List[float] = []
    data: List[List[float]] = []
    colors: List[str] = []
    labels: List[str] = []
    centers: List[float] = []
    for material_idx, material in enumerate(MATERIAL_ORDER):
        base = material_idx * 5.0
        centers.append(base + 1.5)
        for strategy_idx, strategy in enumerate(STRATEGY_ORDER):
            x = base + strategy_idx
            vals = _values(violin_rows, material, strategy)
            if not vals:
                continue
            positions.append(x)
            data.append(vals)
            colors.append(FEATURE_COLORS[strategy])
            labels.append(strategy)

    violin = ax_top.violinplot(data, positions=positions, widths=0.78, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_alpha(None)
        body.set_facecolors([mpl.colors.to_rgba(color, VIOLIN_FACE_ALPHA)])
        body.set_edgecolors([mpl.colors.to_rgba(VIOLIN_EDGE_COLOR, 1.0)])
        body.set_linewidth(VIOLIN_EDGE_LW)
        body.set_zorder(5)

    box = ax_top.boxplot(
        data,
        positions=positions,
        widths=0.68,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 0.85, "zorder": 3},
        boxprops={"facecolor": "white", "edgecolor": "#111111", "linewidth": 0.80, "alpha": 0.80, "zorder": 3},
        whiskerprops={"color": "#111111", "linewidth": 0.70, "zorder": 3},
        capprops={"color": "#111111", "linewidth": 0.70, "zorder": 3},
    )
    for patch in box["boxes"]:
        patch.set_zorder(3)

    for x, vals, color in zip(positions, data, colors):
        _scatter_points(ax_top, x, vals, color)

    ax_top.set_ylim(*Y_LIM)
    ax_top.set_yticks(Y_TICKS)
    ax_top.yaxis.set_major_formatter(FuncFormatter(zero_trim))
    ax_top.set_ylabel("MAE", labelpad=0.8, fontsize=7)
    ax_top.set_xticks(centers)
    ax_top.set_xticklabels([MATERIAL_SHORT[m] for m in MATERIAL_ORDER], fontsize=6)
    ax_top.grid(True, axis="y", color="#E6E6E6", lw=0.30)
    ax_top.grid(True, axis="x", color="#EEEEEE", lw=0.22)
    ax_top.tick_params(length=1.5, width=0.50, pad=0.7, labelsize=6)
    for spine in ax_top.spines.values():
        spine.set_linewidth(0.75)

    handles = [
        mpl.patches.Patch(facecolor=FEATURE_COLORS[s], edgecolor="#777777", alpha=0.55, label=s)
        for s in STRATEGY_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.995),
        ncol=4,
        frameon=False,
        handlelength=0.55,
        handleheight=0.55,
        handletextpad=0.20,
        columnspacing=0.28,
        labelspacing=0.05,
        fontsize=6,
    )

    improvements = np.asarray([float(r["mae_improvement_pct"]) for r in bubble_rows], dtype=float)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(25.0, float(np.nanmax(improvements))))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("figure5a_improvement", ["#FBF7C3", "#F07C4A", "#149164"], N=256)
    std_values = np.asarray([float(r["combo_mae_std"]) for r in bubble_rows], dtype=float)
    std_min = float(np.nanmin(std_values))
    std_max = float(np.nanmax(std_values))

    for row in bubble_rows:
        x = FEATURE_ORDER.index(str(row["feature_strategy_short"]))
        y = MATERIAL_ORDER.index(str(row["material"]))
        std = float(row["combo_mae_std"])
        if std_max > std_min:
            size = 22.0 + 110.0 * (std - std_min) / (std_max - std_min)
        else:
            size = 70.0
        improvement = float(row["mae_improvement_pct"])
        ax_bot.scatter(x, y, s=size, color=cmap(norm(improvement)), edgecolor="#666666", linewidth=0.45, zorder=2)
        ax_bot.text(x + 0.27, y - 0.30, f"+{improvement:.1f}", ha="left", va="center", fontsize=5, weight="normal", color="black", zorder=3)

    ax_bot.set_xlim(-0.65, len(FEATURE_ORDER) - 0.02)
    ax_bot.set_ylim(len(MATERIAL_ORDER) - 0.45, -0.55)
    ax_bot.set_xticks(range(len(FEATURE_ORDER)))
    ax_bot.set_xticklabels([])
    ax_bot.set_yticks(range(len(MATERIAL_ORDER)))
    ax_bot.set_yticklabels([MATERIAL_SHORT[m] for m in MATERIAL_ORDER], fontsize=6)
    ax_bot.set_xlabel("")
    ax_bot.set_ylabel("Material", labelpad=0.8, fontsize=7)
    ax_bot.grid(True, color="#EAEAEA", lw=0.30)
    ax_bot.tick_params(length=1.5, width=0.50, pad=1.0)
    for spine in ax_bot.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)

    legend_ax = fig.add_axes([0.765, 0.265, 0.210, 0.205])
    legend_ax.axis("off")
    legend_ax.text(0.02, 0.96, "MAE std", ha="left", va="top", fontsize=6)
    legend_values = np.linspace(std_min, std_max, 3)
    for i, std in enumerate(legend_values):
        y_leg = 0.68 - i * 0.27
        size = 8.0 + 38.0 * (float(std) - std_min) / (std_max - std_min) if std_max > std_min else 26.0
        legend_ax.scatter(0.24, y_leg, s=size, facecolor="#D9D9D9", edgecolor="#777777", linewidth=0.40)
        legend_ax.text(0.48, y_leg, f"{std:.3f}", ha="left", va="center", fontsize=6)
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    fig.text(0.455, 0.180, "Combinational feature strategy", ha="center", va="center", fontsize=7)
    cax = fig.add_axes([0.18, 0.105, 0.55, 0.035])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([0, 12.5, 25])
    cbar.set_ticklabels(["0", "12.5", "25"])
    cbar.set_label("MAE improvement (%)", labelpad=0.2, fontsize=6)
    cbar.ax.tick_params(labelsize=6, length=1.3, width=0.45, pad=0.6)
    cbar.outline.set_linewidth(0.6)

    png_path = out_dir / f"{OUT_STEM}.png"
    pdf_path = out_dir / f"{OUT_STEM}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw PulseBat-combo Figure5a.")
    parser.add_argument("--source_xlsx", type=str, default=str(DATA_XLSX))
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--fig_w_cm", type=float, default=FIG_W_CM)
    parser.add_argument("--fig_h_cm", type=float, default=FIG_H_CM)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    violin_rows, bubble_rows = load_figure5a_tables(Path(args.source_xlsx))
    plot(violin_rows, bubble_rows, Path(args.out_dir), float(args.fig_w_cm), float(args.fig_h_cm), int(args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
