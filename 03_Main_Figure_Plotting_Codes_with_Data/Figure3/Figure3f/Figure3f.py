#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 3f for PulseBat-combo.

Feature-importance distributions for the selected C-rate at SOC=25%, 50%,
and 75% under 5000 ms combo-pulse conditions.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


ROOT = Path(r"E:\Datasets\PulseBat_combo")
SOURCE_CSV = ROOT / "Figure3" / "Figure3e" / "Figure3e_linear_featimp_points.csv"
FIG_W_CM = 4.55
FIG_H_CM = 3.62
DPI = 600

SELECTED_CRATE = "2.5"
CRATES: Sequence[str] = ("0.5", "1.0", "1.5", "2.0", "2.5")
SOCS: Sequence[int] = (25, 50, 75)
CAPACITIES: Sequence[str] = ("20Ah", "68Ah")
SOC_COLORS: Dict[int, str] = {
    25: "#5BAEFF",
    50: "#4C9390",
    75: "#D65D48",
}

LINE_LW = 0.85
FILL_ALPHA = 0.22


@dataclass(frozen=True)
class ImportanceRecord:
    capacity: str
    soc: int
    crate: str
    seed: int
    importance: float


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
        "legend.fontsize": 5.8,
        "axes.unicode_minus": False,
    })


def normalize_crate(crate: str) -> str:
    value = crate.strip().replace("C", "").replace("c", "")
    if value == "1":
        return "1.0"
    if value == "2":
        return "2.0"
    return value


def crate_token(crate: str) -> str:
    return f"{normalize_crate(crate).replace('.', 'p')}C"


def output_stem_for_crate(crate: str) -> str:
    normalized = normalize_crate(crate)
    if normalized == SELECTED_CRATE:
        return "Figure3f"
    return f"Figure3f_{crate_token(normalized)}"


def load_records(csv_path: Path, selected_crate: str) -> List[ImportanceRecord]:
    selected_crate = normalize_crate(selected_crate)
    records: List[ImportanceRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["crate"].strip() != selected_crate:
                continue
            records.append(ImportanceRecord(
                capacity=row["capacity"].strip(),
                soc=int(row["soc_%"]),
                crate=row["crate"].strip(),
                seed=int(row["seed"]),
                importance=float(row["importance"]),
            ))
    if not records:
        raise RuntimeError(f"No records found for C-rate {selected_crate} in {csv_path}.")
    return records


def write_selected_csv(records: Sequence[ImportanceRecord], out_dir: Path, output_stem: str) -> None:
    if output_stem == "Figure3f":
        out_csv = out_dir / "Figure3f_linear_featimp_distribution_points.csv"
    else:
        out_csv = out_dir / f"{output_stem}_linear_featimp_distribution_points.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["capacity", "soc_%", "crate", "seed", "importance"])
        for r in records:
            writer.writerow([r.capacity, r.soc, r.crate, r.seed, r.importance])


def kde_density(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros_like(x_grid)
    if values.size == 1 or np.nanstd(values) <= 0:
        center = float(values[0])
        bandwidth = max(float(np.nanmax(x_grid) - np.nanmin(x_grid)) / 120.0, 1e-4)
    else:
        std = float(np.nanstd(values, ddof=1))
        bandwidth = 1.06 * std * values.size ** (-1.0 / 5.0)
        bandwidth = max(bandwidth, 1e-4)
        center = 0.0
    z = (x_grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).sum(axis=1) / (values.size * bandwidth * np.sqrt(2.0 * np.pi))
    if values.size == 1:
        density = np.exp(-0.5 * ((x_grid - center) / bandwidth) ** 2) / (bandwidth * np.sqrt(2.0 * np.pi))
    return density


def rounded_xlim(values: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    vmax = float(np.nanmax(values))
    return 0.0, max(0.01, np.ceil((vmax + 0.003) / 0.025) * 0.025)


def feature_tick_formatter(value: float, pos: int) -> str:
    if abs(value) < 5e-7:
        return "0"
    return f"{value:.3f}"


def density_tick_formatter(value: float, pos: int) -> str:
    if abs(value) < 1e-9:
        return "0"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def rounded_density_top(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    raw_top = value * 1.08
    order = 10 ** math.floor(math.log10(raw_top))
    step = order / 2.0
    return max(step, math.ceil(raw_top / step) * step)


def plot(
    records: Sequence[ImportanceRecord],
    out_dir: Path,
    fig_w_cm: float,
    fig_h_cm: float,
    dpi: int,
    output_stem: str,
) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_selected_csv(records, out_dir, output_stem)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(cm_to_in(fig_w_cm), cm_to_in(fig_h_cm)),
        dpi=dpi,
        sharex=False,
        sharey=False,
    )
    fig.subplots_adjust(left=0.195, right=0.900, bottom=0.240, top=0.920, hspace=0.38)

    densities: Dict[tuple[str, int], np.ndarray] = {}
    x_grids: Dict[str, np.ndarray] = {}
    x_limits: Dict[str, tuple[float, float]] = {}
    for capacity in CAPACITIES:
        capacity_values = [
            r.importance for r in records if r.capacity == capacity
        ]
        xmin, xmax = rounded_xlim(capacity_values)
        x_grid = np.linspace(xmin, xmax, 500)
        x_grids[capacity] = x_grid
        x_limits[capacity] = (xmin, xmax)
        for soc in SOCS:
            values = np.asarray([
                r.importance for r in records if r.capacity == capacity and r.soc == soc
            ], dtype=float)
            density = kde_density(values, x_grid)
            densities[(capacity, soc)] = density

    for ax, capacity in zip(axes, CAPACITIES):
        capacity_max_density = 0.0
        x_grid = x_grids[capacity]
        xmin, xmax = x_limits[capacity]
        for soc in SOCS:
            density = densities[(capacity, soc)]
            capacity_max_density = max(capacity_max_density, float(np.nanmax(density)))
            color = SOC_COLORS[soc]
            ax.fill_between(x_grid, density, color=color, alpha=FILL_ALPHA, linewidth=0.0)
            ax.plot(x_grid, density, color=color, lw=LINE_LW, label=f"{soc}% SOC")
        ax.text(0.96, 0.92, capacity, transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_xlim(xmin, xmax)
        y_top = rounded_density_top(capacity_max_density)
        ax.set_ylim(0.0, y_top)
        ax.set_yticks([0.0, y_top / 2.0, y_top])
        ax.yaxis.set_major_formatter(FuncFormatter(density_tick_formatter))
        ax.set_xticks([xmin, (xmin + xmax) / 2.0, xmax])
        ax.xaxis.set_major_formatter(FuncFormatter(feature_tick_formatter))
        ax.tick_params(length=2.0, width=0.6, pad=1.2, labelsize=7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)

    handles = [
        Line2D([0], [0], color=SOC_COLORS[soc], lw=LINE_LW, label=f"{soc}% SOC")
        for soc in SOCS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(SOCS),
        frameon=False,
        handlelength=0.75,
        handletextpad=0.35,
        columnspacing=0.80,
        borderaxespad=0.0,
        bbox_to_anchor=(0.5, -0.080),
    )
    fig.text(0.040, 0.565, "Density", rotation=90, ha="center", va="center", fontsize=7)
    axes[-1].set_xlabel("Feature importance", labelpad=2, fontsize=7)

    png_path = out_dir / f"{output_stem}.png"
    pdf_path = out_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw PulseBat-combo Figure3f.")
    parser.add_argument("--source_csv", type=str, default=str(SOURCE_CSV))
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--crate", type=str, default="all", help="C-rate to draw, or 'all'.")
    parser.add_argument("--fig_w_cm", type=float, default=FIG_W_CM)
    parser.add_argument("--fig_h_cm", type=float, default=FIG_H_CM)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    if str(args.crate).strip().lower() == "all":
        crates = CRATES
    else:
        crates = (normalize_crate(str(args.crate)),)

    for crate in crates:
        records = load_records(Path(args.source_csv), crate)
        plot(
            records,
            Path(args.out_dir),
            float(args.fig_w_cm),
            float(args.fig_h_cm),
            int(args.dpi),
            output_stem_for_crate(crate),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
