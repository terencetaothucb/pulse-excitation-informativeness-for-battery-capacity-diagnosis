#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 3c for PulseBat-combo.

20Ah LFP contour maps of the difference in MAE between combinational-pulse
features and the corresponding mean single-pulse features.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(r"E:\Datasets\PulseBat_combo")
SOURCE_CSV = ROOT / "Figure3" / "Figure3c_old" / "Figure3c_combo_minus_single_mae.csv"
FIG_W_CM = 13.43
FIG_H_CM = 2.85
DPI = 600

CAPACITY = "20Ah"
OUT_STEM = "Figure3c"

WIDTHS_MS: Sequence[int] = (50, 500, 3000, 5000)
SOCS: Sequence[int] = (5, 15, 25, 35, 45, 55, 65, 75)
GROUPS: Sequence[str] = (
    "0.5C|1.0C",
    "1.0C|1.5C",
    "1.5C|2.0C",
    "2.0C|2.5C",
    "0.5C|1.0C|1.5C|2.0C|2.5C",
)
CMAP = LinearSegmentedColormap.from_list(
    "blue_cyan_green_yellow",
    ["#2D2DFF", "#1B8CFF", "#18C6A7", "#7FD34E", "#F2E21D"],
)
LEVEL_COUNT = 17
SAMPLE_GRID_COLOR = "#FFFFFF"
SAMPLE_GRID_LW = 0.28
SAMPLE_GRID_ALPHA = 0.55
SAMPLE_GRID_DASH = (0, (2.0, 2.0))


@dataclass(frozen=True)
class DeltaRecord:
    capacity: str
    width_ms: int
    soc: int
    group: str
    combo_mae: float
    single_mean_mae: float
    delta_mae: float


def cm_to_in(cm: float) -> float:
    return cm / 2.54


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.65,
        "axes.labelsize": 5.8,
        "axes.titlesize": 6.2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.unicode_minus": False,
    })


def load_records(csv_path: Path, capacity: str) -> List[DeltaRecord]:
    records: List[DeltaRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["capacity"].strip() != capacity:
                continue
            records.append(DeltaRecord(
                capacity=row["capacity"].strip(),
                width_ms=int(row["width_ms"]),
                soc=int(row["soc_%"]),
                group=row["group"].strip(),
                combo_mae=float(row["combo_mae_%"]) / 100.0,
                single_mean_mae=float(row["single_mean_mae_%"]) / 100.0,
                delta_mae=float(row["delta_mae_%"]) / 100.0,
            ))
    if not records:
        raise RuntimeError(f"No {capacity} records found in {csv_path}.")
    return records


def write_records(records: Sequence[DeltaRecord], out_dir: Path) -> None:
    out_csv = out_dir / f"{OUT_STEM}_combo_minus_single_mae.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "capacity",
            "width_ms",
            "soc_%",
            "group",
            "combo_mae",
            "single_mean_mae",
            "delta_mae",
        ])
        for r in records:
            writer.writerow([
                r.capacity,
                r.width_ms,
                r.soc,
                r.group,
                r.combo_mae,
                r.single_mean_mae,
                r.delta_mae,
            ])


def data_levels(records: Sequence[DeltaRecord]) -> np.ndarray:
    values = np.asarray([r.delta_mae for r in records], dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 0.001)
        vmin -= pad
        vmax += pad
    return np.linspace(vmin, vmax, LEVEL_COUNT)


def grid_for_group(records: Sequence[DeltaRecord], group: str) -> np.ndarray:
    lookup = {(r.width_ms, r.soc): r.delta_mae for r in records if r.group == group}
    grid = np.full((len(WIDTHS_MS), len(SOCS)), np.nan, dtype=float)
    for i, width_ms in enumerate(WIDTHS_MS):
        for j, soc in enumerate(SOCS):
            grid[i, j] = lookup[(width_ms, soc)]
    return grid


def plot(records: Sequence[DeltaRecord], out_dir: Path, fig_w_cm: float, fig_h_cm: float, dpi: int) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_records(records, out_dir)

    fig = plt.figure(figsize=(cm_to_in(fig_w_cm), cm_to_in(fig_h_cm)), dpi=dpi)
    left = 0.070
    right = 0.987
    gap = 0.018
    panel_w = (right - left - gap * (len(GROUPS) - 1)) / len(GROUPS)
    panel_y = 0.385
    panel_h = 0.435

    cbar_ax = fig.add_axes([0.240, 0.090, 0.520, 0.050])
    axes = [
        fig.add_axes([left + i * (panel_w + gap), panel_y, panel_w, panel_h])
        for i in range(len(GROUPS))
    ]

    levels = data_levels(records)
    norm = BoundaryNorm(levels, ncolors=CMAP.N, clip=True)
    x = np.asarray(SOCS, dtype=float)
    y = np.arange(len(WIDTHS_MS), dtype=float)
    xx, yy = np.meshgrid(x, y)
    mappable = None

    for i, (ax, group) in enumerate(zip(axes, GROUPS)):
        z = grid_for_group(records, group)
        mappable = ax.contourf(xx, yy, z, levels=levels, cmap=CMAP, norm=norm, extend="both")
        ax.contour(xx, yy, z, levels=levels[::2], colors="white", linewidths=0.18, alpha=0.38)
        ax.vlines(x, ymin=-0.08, ymax=len(WIDTHS_MS) - 0.92, colors=SAMPLE_GRID_COLOR,
                  linewidth=SAMPLE_GRID_LW, linestyles=SAMPLE_GRID_DASH, alpha=SAMPLE_GRID_ALPHA, zorder=3)
        ax.hlines(y, xmin=min(SOCS), xmax=max(SOCS), colors=SAMPLE_GRID_COLOR,
                  linewidth=SAMPLE_GRID_LW, linestyles=SAMPLE_GRID_DASH, alpha=SAMPLE_GRID_ALPHA, zorder=3)

        ax.set_xlim(min(SOCS), max(SOCS))
        ax.set_ylim(-0.08, len(WIDTHS_MS) - 0.92)
        ax.set_xticks(SOCS)
        ax.set_yticks(y)
        ax.set_yticklabels([str(w) for w in WIDTHS_MS])
        ax.tick_params(length=1.8, width=0.55, pad=1.0)
        if i == 0:
            ax.set_ylabel("Pulse width (ms)", labelpad=1.2, fontsize=7)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)

    if mappable is None:
        raise RuntimeError("No contour mappable was created.")

    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks(levels[::4])
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.tick_params(labelsize=5.2, length=1.7, width=0.5, pad=0.8)
    cbar.outline.set_linewidth(0.55)

    fig.text(0.535, 0.220, "State of charge (%)", ha="center", va="center", fontsize=7)

    png_path = out_dir / f"{OUT_STEM}.png"
    pdf_path = out_dir / f"{OUT_STEM}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw PulseBat-combo Figure3c contour panel.")
    parser.add_argument("--source_csv", type=str, default=str(SOURCE_CSV))
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--fig_w_cm", type=float, default=FIG_W_CM)
    parser.add_argument("--fig_h_cm", type=float, default=FIG_H_CM)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    records = load_records(Path(args.source_csv), CAPACITY)
    plot(records, Path(args.out_dir), float(args.fig_w_cm), float(args.fig_h_cm), int(args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
