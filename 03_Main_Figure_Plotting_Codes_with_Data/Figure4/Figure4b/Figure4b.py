#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 4b for PulseBat-combo.

35Ah LFP single-pulse MAE versus normalized I sqrt(t) in selected excitation
ranges, shown separately for selected SOC values.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter


SOURCE_CSV = Path(r"E:\Datasets\PulseBat_combo\Figure4\Figure4a\Figure4a_single_pulse_mae_vs_soc.csv")
FIG_W_CM = 7.02
FIG_H_CM = 3.77
DPI = 600
CAPACITY = "LFP 35Ah"
PANEL_LETTER = "b"
OUT_STEM = "Figure4b"

SOCS: Sequence[int] = (25, 50, 75)
RANGES: Sequence[Tuple[str, float]] = (
    ("0-0.3", 0.30),
    ("0-0.6", 0.60),
    ("0-1.0", 1.00),
)
GRADIENT_STOPS: Sequence[Tuple[float, str]] = (
    (0.00, "#EE8575"),
    (0.12, "#FBB482"),
    (0.36, "#CDEBB8"),
    (0.66, "#9CD7BC"),
    (0.90, "#867BB9"),
    (1.00, "#867BB9"),
)

Y_LIM = (0.00, 0.06)
Y_TICKS: Sequence[float] = (0.00, 0.03, 0.06)
MARKER_SIZE = 24.0
MARKER_ALPHA = 1
MARKER_EDGE_COLOR = "#FFFFFF"
MARKER_EDGE_LW = 0.65
COLOR_DARKEN = 0.78
FIT_LINE_WIDTH = 0.85
AXES_XSHIFT = -0.020
AXES_YSHIFT = 0.000
AXES_HEIGHT_SCALE = 1.00


@dataclass(frozen=True)
class MaeRecord:
    capacity: str
    width_ms: int
    soc: int
    crate: str
    d_norm: float
    mae: float


@dataclass(frozen=True)
class FitRecord:
    capacity: str
    soc: int
    range_label: str
    slope: float
    intercept: float
    r2: float
    pearson_r: float
    n: int


def cm_to_in(cm: float) -> float:
    return cm / 2.54


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.75,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.unicode_minus": False,
    })


def zero_trim_formatter(value: float, pos: int) -> str:
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.2f}"


def load_records(path: Path, capacity: str) -> List[MaeRecord]:
    records: List[MaeRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["capacity"] != capacity:
                continue
            records.append(MaeRecord(
                capacity=row["capacity"],
                width_ms=int(row["width_ms"]),
                soc=int(row["soc_%"]),
                crate=row["crate"],
                d_norm=float(row["d_norm"]),
                mae=float(row["mae"]),
            ))
    return records


def linear_fit(records: Sequence[MaeRecord]) -> Tuple[float, float, float, float]:
    x = np.asarray([r.d_norm for r in records], dtype=float)
    y = np.asarray([r.mae for r in records], dtype=float)
    if len(x) < 2 or float(np.nanstd(x)) == 0.0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    pearson_r = float("nan") if float(np.nanstd(y)) == 0.0 else float(np.corrcoef(x, y)[0, 1])
    return float(slope), float(intercept), r2, pearson_r


def write_fit_csv(fits: Sequence[FitRecord], out_dir: Path, out_stem: str) -> None:
    path = out_dir / f"{out_stem}_mae_vs_dnorm_fit_r2.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["capacity", "soc_%", "range", "slope", "intercept", "r2", "pearson_r", "n"])
        for r in fits:
            writer.writerow([r.capacity, r.soc, r.range_label, r.slope, r.intercept, r.r2, r.pearson_r, r.n])


def write_subplot_stats_csv(fits: Sequence[FitRecord], out_dir: Path, out_stem: str) -> None:
    path = out_dir / f"{out_stem}_subplot_N_R2.csv"
    try:
        f = path.open("w", encoding="utf-8-sig", newline="")
    except PermissionError:
        path = out_dir / f"{out_stem}_subplot_N_R2_with_pearson_r.csv"
        f = path.open("w", encoding="utf-8-sig", newline="")
    with f:
        writer = csv.writer(f)
        writer.writerow(["capacity", "soc_%", "range", "N", "R2", "pearson_r"])
        for r in fits:
            writer.writerow([r.capacity, r.soc, r.range_label, r.n, r.r2, r.pearson_r])


def plot(records: Sequence[MaeRecord], out_dir: Path, fig_w_cm: float, fig_h_cm: float, dpi: int) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = LinearSegmentedColormap.from_list("pulsebat_figure2c", GRADIENT_STOPS, N=256)
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(
        len(SOCS),
        len(RANGES),
        figsize=(cm_to_in(fig_w_cm), cm_to_in(fig_h_cm)),
        dpi=dpi,
        sharey=True,
    )
    fig.subplots_adjust(left=0.145, right=0.805, bottom=0.180, top=0.955, wspace=0.32, hspace=0.28)
    for ax in axes.ravel():
        pos = ax.get_position()
        new_height = pos.height * AXES_HEIGHT_SCALE
        ax.set_position([pos.x0 + AXES_XSHIFT, pos.y0 + AXES_YSHIFT, pos.width, new_height])

    fits: List[FitRecord] = []
    for row, soc in enumerate(SOCS):
        for col, (range_label, upper) in enumerate(RANGES):
            ax = axes[row, col]
            subset = [
                r for r in records
                if r.soc == soc and 0.0 <= r.d_norm <= upper + 1e-12
            ]
            x = np.asarray([r.d_norm for r in subset], dtype=float)
            y = np.asarray([r.mae for r in subset], dtype=float)
            colors = cmap(norm(x))
            colors[:, :3] *= COLOR_DARKEN
            ax.scatter(
                x,
                y,
                s=MARKER_SIZE,
                c=colors,
                alpha=MARKER_ALPHA,
                edgecolors=MARKER_EDGE_COLOR,
                linewidths=MARKER_EDGE_LW,
                rasterized=True,
                zorder=2,
            )

            slope, intercept, r2, pearson_r = linear_fit(subset)
            fits.append(FitRecord(CAPACITY, soc, range_label, slope, intercept, r2, pearson_r, len(subset)))
            if np.isfinite(slope) and np.isfinite(intercept):
                xx = np.linspace(0.0, upper, 100)
                ax.plot(xx, slope * xx + intercept, color="black", lw=FIT_LINE_WIDTH, ls=(0, (3, 2)), zorder=3)

            ax.set_xlim(0.0, upper)
            ax.set_ylim(*Y_LIM)
            ax.set_xticks([0.0, upper / 2.0, upper])
            ax.set_yticks(Y_TICKS)
            ax.xaxis.set_major_formatter(FuncFormatter(zero_trim_formatter))
            ax.yaxis.set_major_formatter(FuncFormatter(zero_trim_formatter))
            ax.tick_params(length=2.0, width=0.6, pad=1.1)
            if row < len(SOCS) - 1:
                ax.tick_params(labelbottom=False)
            if col > 0:
                ax.tick_params(labelleft=False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.75)

            if col == len(RANGES) - 1:
                row_pos = ax.get_position()
                fig.text(
                    0.965,
                    row_pos.y0 + 0.5 * row_pos.height,
                    f"{soc}% SOC",
                    ha="right",
                    va="center",
                    fontsize=7,
                )

    write_fit_csv(fits, out_dir, OUT_STEM)
    write_subplot_stats_csv(fits, out_dir, OUT_STEM)
    fig.text(0.022, 0.570, "MAE", rotation=90, ha="center", va="center", fontsize=7)
    fig.text(0.500, 0.060, r"Single pulse excitation range indicated by $I\sqrt{t}$", ha="center", va="center", fontsize=7)

    png_path = out_dir / f"{OUT_STEM}.png"
    pdf_path = out_dir / f"{OUT_STEM}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Draw PulseBat-combo {OUT_STEM}.")
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
