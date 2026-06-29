#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5c-e for PulseBat-combo.

Linear-model scenario-group ranking for C-rate, width, and SOC combinational
feature groups. This script writes three independent panels: Figure5c,
Figure5d, and Figure5e.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure5_common import (  # noqa: E402
    DATA_XLSX,
    DPI,
    cm_to_in,
    load_sheet_table,
    setup_style,
    write_csv,
    zero_trim,
)


FIG_W_CM = 4.15
FIG_H_CM = 3.81
OUT_STEM = "Figure5c"
GROUPS: Sequence[tuple[str, str, str, str]] = (
    ("Figure5c", "c_rate_group", "C-Rate", "#72B7B2"),
    ("Figure5d", "width_group", "Width", "#F28E2B"),
    ("Figure5e", "soc_group", "SOC", "#59A14F"),
)
PANEL_AXES = {
    "Figure5c": (0.32, 0.63),
    "Figure5d": (0.54, 0.41),
    "Figure5e": (0.42, 0.53),
}
AX_BOTTOM = 0.17
AX_HEIGHT = 0.73
RIGHT_MARGIN = 0.05
MAIN_WIDTH_EXTRA_CM = 0.40
OUTER_RIGHT_EXTRA_CM = 0.12

LABELS = {
    "all_crates_05_10_15_20_25": "All C-Rate",
    "high_crates_20_25": "2.0/2.5C",
    "mid_crates_15_20": "1.5/2.0C",
    "mid_crates_10_15": "1.0/1.5C",
    "low_crates_05_10": "0.5/1.0C",
    "all_widths": "All Width",
    "high_widths_1000_3000_5000": "1000/3000/5000 ms",
    "high_widths_1000_3000": "1000/3000 ms",
    "high_widths_3000_5000": "3000/5000 ms",
    "mid_widths_300_500_700": "300/500/700ms",
    "mid_widths_100_300_500": "100/300/500ms",
    "low_widths_50_70_100": "50/70/100ms",
    "low_widths_30_50_70_100": "30/50/70/100ms",
    "low_widths_30_50_70": "30/50/70ms",
    "all_socs": "All SOC",
    "high_socs_55_60_65_70": "55/60/65/70%",
    "high_socs_65_70": "65/70%",
    "mid_socs_35_40_45_50": "35/40/45/50%",
    "mid_socs_35_40": "35/40%",
    "mid_socs_45_50": "45/50%",
    "high_socs_55_60": "55/60%",
    "low_socs_05_30": "5/30%",
    "low_socs_05_20": "5/20%",
    "low_socs_05_10": "5/10%",
}


def _panel_label(row: Dict[str, object]) -> str:
    label = LABELS.get(str(row["scenario_group"]), str(row["scenario_group_label"]))
    label = label.replace("/", "|")
    if row.get("scenario_group_type") == "width_group":
        label = label.replace(" ms", "ms")
    return label


def _panel_width_and_axes(panel_name: str, base_fig_w_cm: float) -> tuple[float, list[float]]:
    left_frac, _ = PANEL_AXES[panel_name]
    avg_main_frac = sum(width for _, width in PANEL_AXES.values()) / len(PANEL_AXES)
    left_cm = left_frac * base_fig_w_cm
    main_cm = avg_main_frac * base_fig_w_cm + MAIN_WIDTH_EXTRA_CM
    right_cm = RIGHT_MARGIN * base_fig_w_cm + OUTER_RIGHT_EXTRA_CM
    fig_w_cm = left_cm + main_cm + right_cm
    return fig_w_cm, [left_cm / fig_w_cm, AX_BOTTOM, main_cm / fig_w_cm, AX_HEIGHT]


def _scale_x_rect(rect: Sequence[float], base_fig_w_cm: float, fig_w_cm: float) -> list[float]:
    scale = base_fig_w_cm / fig_w_cm
    return [rect[0] * scale, rect[1], rect[2] * scale, rect[3]]


def _ordered_subset(rows: Sequence[Dict[str, object]], group_key: str) -> list[Dict[str, object]]:
    subset = sorted(
        [r for r in rows if r.get("scenario_group_type") == group_key],
        key=lambda r: int(r["rank_within_type"]),
    )
    if group_key == "soc_group":
        all_rows = [r for r in subset if r.get("scenario_group") == "all_socs"]
        other_rows = [r for r in subset if r.get("scenario_group") != "all_socs"]
        return all_rows + other_rows
    return subset


def _draw_panel(rows: Sequence[Dict[str, object]], out_dir: Path, panel_name: str, group_key: str, title: str, color: str, fig_w_cm: float, fig_h_cm: float, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = _ordered_subset(rows, group_key)
    write_csv(out_dir / f"{panel_name}_scenario_group_ranking.csv", subset)

    panel_fig_w_cm, ax_rect = _panel_width_and_axes(panel_name, fig_w_cm)
    fig = plt.figure(figsize=(cm_to_in(panel_fig_w_cm), cm_to_in(fig_h_cm)), dpi=dpi)
    ax = fig.add_axes(ax_rect)

    y = np.arange(len(subset), dtype=float)
    maes = np.asarray([float(r["mae"]) for r in subset], dtype=float)
    stds = np.asarray([float(r["mae_std"]) for r in subset], dtype=float)
    left = np.clip(maes - stds, 0.0, None)
    right = maes + stds

    for yi, lo, mid, hi in zip(y, left, maes, right):
        ax.hlines(yi, lo, hi, color=color, lw=0.75, alpha=0.90, zorder=1)
        ax.plot([lo, hi], [yi, yi], linestyle="None", marker="|", markersize=5.0, markeredgewidth=0.85, color="#111111", zorder=2)
        ax.scatter(mid, yi, s=10.0, color=color, edgecolor="#333333", linewidth=0.40, zorder=3)
        ax.text(hi + 0.0016, yi, f"{mid:.3f}", ha="left", va="center", fontsize=7, clip_on=False)

    ax.set_title(title, fontsize=7, pad=3)
    ax.set_yticks(y)
    ax.set_yticklabels([_panel_label(r) for r in subset], fontsize=7)
    ax.set_ylim(len(subset) - 0.45, -0.55)
    ax.set_xlim(0.0, 0.06)
    ax.set_xticks([0.0, 0.03, 0.06])
    ax.xaxis.set_major_formatter(FuncFormatter(zero_trim))
    ax.set_xlabel("MAE", fontsize=7, labelpad=0.2)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color="#E8E8E8", lw=0.35)
    ax.grid(True, axis="y", color="#C0C0C0", lw=0.35, linestyle="-")
    ax.tick_params(axis="x", length=2.0, width=0.6, pad=0.8, labelsize=7)
    ax.tick_params(axis="y", length=0, pad=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_visible(True)

    if group_key == "width_group":
        legend_ax = fig.add_axes(_scale_x_rect([0.055, 0.020, 0.35, 0.130], fig_w_cm, panel_fig_w_cm))
        legend_ax.axis("off")
        legend_ax.scatter(0.10, 0.72, s=10.0, color="#BDBDBD", edgecolor="#333333", linewidth=0.40, zorder=2)
        legend_ax.text(0.22, 0.72, "Mean", ha="left", va="center", fontsize=7)
        legend_ax.plot([0.10], [0.22], linestyle="None", marker="|", markersize=5.0, markeredgewidth=0.85, color="#111111", zorder=2)
        legend_ax.text(0.22, 0.22, "Std", ha="left", va="center", fontsize=7)
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)

    png_path = out_dir / f"{panel_name}.png"
    pdf_path = out_dir / f"{panel_name}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def plot(rows: Sequence[Dict[str, object]], out_dir: Path, fig_w_cm: float, fig_h_cm: float, dpi: int) -> None:
    setup_style()
    matplotlib.rcParams.update({
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "Figure5c_scenario_group_ranking.csv", rows)
    for panel_name, group_key, title, color in GROUPS:
        panel_dir = out_dir if panel_name == OUT_STEM else out_dir.parent / panel_name
        _draw_panel(rows, panel_dir, panel_name, group_key, title, color, fig_w_cm, fig_h_cm, dpi)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw PulseBat-combo Figure5c-e.")
    parser.add_argument("--source_xlsx", type=str, default=str(DATA_XLSX))
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--fig_w_cm", type=float, default=FIG_W_CM)
    parser.add_argument("--fig_h_cm", type=float, default=FIG_H_CM)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    rows = load_sheet_table("5c", Path(args.source_xlsx))
    plot(rows, Path(args.out_dir), float(args.fig_w_cm), float(args.fig_h_cm), int(args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
