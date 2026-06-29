#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 4d for PulseBat-combo.

Combo-pulse linear-model MAE versus SOC for LFP 20Ah, 35Ah, and 68Ah.
Each line corresponds to one exact (combo group, pulse width) pair and is
colored by normalized d = mean(C-rate in the combo) * sqrt(width in seconds).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FormatStrFormatter, FuncFormatter


RESULTS_ROOT = Path(r"G:\Experiment_Output\gpu_combo_all\Results_exp1_soc_width")
FIG_W_CM = 13.92
FIG_H_CM = 2.04
DPI = 600
MODEL = "linear"

CAPACITIES: Sequence[Tuple[str, str]] = (
    ("20Ah_LFP", "LFP 20Ah"),
    ("35Ah_LFP", "LFP 35Ah"),
    ("68Ah_LFP", "LFP 68Ah"),
)
SOCS: Sequence[int] = tuple(range(5, 80, 5))
PLOT_SOCS: Sequence[int] = tuple(range(5, 80, 10))
WIDTHS_MS: Sequence[int] = (30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000)
GROUPS: Sequence[Tuple[str, str, Tuple[float, ...]]] = (
    ("low_crates_05_10", "0.5C|1.0C", (0.5, 1.0)),
    ("mid_crates_10_15", "1.0C|1.5C", (1.0, 1.5)),
    ("mid_crates_15_20", "1.5C|2.0C", (1.5, 2.0)),
    ("high_crates_20_25", "2.0C|2.5C", (2.0, 2.5)),
    ("all_crates_05_10_15_20_25", "0.5C|1.0C|1.5C|2.0C|2.5C", (0.5, 1.0, 1.5, 2.0, 2.5)),
)
GRADIENT_STOPS: Sequence[Tuple[float, str]] = (
    (0.00, "#0c0786"),
    (0.08, "#4b03a1"),
    (0.18, "#8204a8"),
    (0.32, "#b12a90"),
    (0.48, "#db5278"),
    (0.62, "#f67b4a"),
    (0.76, "#fca936"),
    (0.88, "#f0f921"),
    (1.00, "#f0f921"),
)
LINE_WIDTH = 1.15
MARKER_SIZE = 4.0
MARKER_EDGE_LW = 0.35
MAIN_AXES_XSHIFT = 0.010
CBAR_XSHIFT = -0.006
Y_LIM = (0.00, 0.06)
Y_TICKS: Sequence[float] = (0.00, 0.03, 0.06)


@dataclass(frozen=True)
class MaeRecord:
    capacity_key: str
    capacity_label: str
    width_ms: int
    soc: int
    group_key: str
    group_label: str
    mean_crate: float
    d_raw: float
    d_norm: float
    mae: float


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


def mean_crate(group_key: str) -> float:
    for key, _, crates in GROUPS:
        if key == group_key:
            return float(np.mean(crates))
    raise KeyError(group_key)


def d_value(group_key: str, width_ms: int) -> float:
    return mean_crate(group_key) * math.sqrt(width_ms / 1000.0)


def build_result_index(results_root: Path) -> Dict[Tuple[str, int, int, str], Path]:
    pattern = re.compile(
        r"^\d{8}-\d{6}__(?P<cap>\d+Ah_LFP)__exp1__SOC(?P<soc>\d+)_W(?P<width>\d+)_(?P<group>.+)$"
    )
    expected_groups = {group_key for group_key, _, _ in GROUPS}
    index: Dict[Tuple[str, int, int, str], Path] = {}
    for p in results_root.iterdir():
        if not p.is_dir():
            continue
        match = pattern.match(p.name)
        if match is None:
            continue
        group_key = match.group("group")
        if group_key not in expected_groups:
            continue
        key = (
            match.group("cap"),
            int(match.group("width")),
            int(match.group("soc")),
            group_key,
        )
        if key in index:
            raise RuntimeError(f"Duplicate result directory for {key}: {index[key]} and {p}")
        index[key] = p
    return index


def linear_mae(metrics_csv: Path) -> float:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"].strip().lower() == MODEL:
                return float(row["mae_test_median"])
    raise RuntimeError(f"No {MODEL} row in {metrics_csv}")


def load_records(results_root: Path) -> List[MaeRecord]:
    index = build_result_index(results_root)
    all_d_values = [d_value(group_key, width_ms) for group_key, _, _ in GROUPS for width_ms in WIDTHS_MS]
    d_min = min(all_d_values)
    d_max = max(all_d_values)

    records: List[MaeRecord] = []
    for capacity_key, capacity_label in CAPACITIES:
        for width_ms in WIDTHS_MS:
            for soc in SOCS:
                for group_key, group_label, _ in GROUPS:
                    key = (capacity_key, width_ms, soc, group_key)
                    if key not in index:
                        raise RuntimeError(f"Missing result directory for {key}")
                    raw_d = d_value(group_key, width_ms)
                    records.append(MaeRecord(
                        capacity_key=capacity_key,
                        capacity_label=capacity_label,
                        width_ms=width_ms,
                        soc=soc,
                        group_key=group_key,
                        group_label=group_label,
                        mean_crate=mean_crate(group_key),
                        d_raw=raw_d,
                        d_norm=(raw_d - d_min) / (d_max - d_min),
                        mae=linear_mae(index[key] / "metrics_summary.csv"),
                    ))
    return records


def write_records_csv(records: Sequence[MaeRecord], out_dir: Path) -> None:
    path = out_dir / "Figure4d_combo_pulse_mae_vs_soc.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["capacity", "width_ms", "soc_%", "group", "mean_crate", "d_raw", "d_norm", "mae"])
        for r in records:
            writer.writerow([r.capacity_label, r.width_ms, r.soc, r.group_label, r.mean_crate, r.d_raw, r.d_norm, r.mae])


def y_limits(records: Sequence[MaeRecord]) -> Tuple[float, float, List[float]]:
    values = np.asarray([r.mae for r in records], dtype=float)
    ymin = max(0.0, math.floor((float(np.nanmin(values)) - 0.001) / 0.001) * 0.001)
    ymax = math.ceil((float(np.nanmax(values)) + 0.001) / 0.001) * 0.001
    return ymin, ymax, [ymin, 0.5 * (ymin + ymax), ymax]


def plot(records: Sequence[MaeRecord], out_dir: Path, fig_w_cm: float, fig_h_cm: float, dpi: int) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_records_csv(records, out_dir)

    fig = plt.figure(figsize=(cm_to_in(fig_w_cm), cm_to_in(fig_h_cm)), dpi=dpi)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.07], wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    fig.subplots_adjust(left=0.075, right=0.94, bottom=0.34, top=0.94)
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 + MAIN_AXES_XSHIFT, pos.y0, pos.width, pos.height])
    cpos = cax.get_position()
    cax.set_position([cpos.x0 + CBAR_XSHIFT, cpos.y0, cpos.width, cpos.height])

    cmap = LinearSegmentedColormap.from_list("pulsebat_figure2c", GRADIENT_STOPS, N=256)
    norm = Normalize(vmin=0.0, vmax=1.0)
    ymin, ymax = Y_LIM

    for i, (ax, (capacity_key, capacity_label)) in enumerate(zip(axes, CAPACITIES)):
        subset_capacity = [r for r in records if r.capacity_key == capacity_key]
        for width_ms in WIDTHS_MS:
            for group_key, _, _ in GROUPS:
                line_records = sorted(
                    [
                        r for r in subset_capacity
                        if r.width_ms == width_ms and r.group_key == group_key and r.soc in PLOT_SOCS
                    ],
                    key=lambda r: r.soc,
                )
                if not line_records:
                    continue
                color = cmap(norm(line_records[0].d_norm))
                ax.plot(
                    [r.soc for r in line_records],
                    [r.mae for r in line_records],
                    color=color,
                    lw=LINE_WIDTH,
                    marker="o",
                    ms=MARKER_SIZE,
                    mec="0.55",
                    mew=MARKER_EDGE_LW,
                    mfc=color,
                    zorder=2,
                )

        ax.set_xlim(min(PLOT_SOCS) - 1.8, max(PLOT_SOCS) + 1.8)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(PLOT_SOCS)
        ax.set_yticks(Y_TICKS)
        ax.yaxis.set_major_formatter(FuncFormatter(zero_trim_formatter))
        ax.grid(True, color="#E6E6E6", lw=0.28, alpha=0.65)
        ax.tick_params(length=1.9, width=0.55, pad=1.2)
        ax.tick_params(axis="x", labelrotation=0, pad=1.6)
        for label in ax.get_xticklabels():
            label.set_ha("center")
            label.set_rotation_mode("default")
        if i == 0:
            ax.set_ylabel("MAE", labelpad=2, fontsize=7)
        else:
            ax.tick_params(labelleft=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)

    axes[1].set_xlabel("State of charge (%)", labelpad=2, fontsize=7)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.ax.tick_params(labelsize=7, length=1.8, width=0.55, pad=1.0)
    cbar.outline.set_linewidth(0.7)

    png_path = out_dir / "Figure4d.png"
    pdf_path = out_dir / "Figure4d.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw PulseBat-combo Figure4d.")
    parser.add_argument("--results_root", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--fig_w_cm", type=float, default=FIG_W_CM)
    parser.add_argument("--fig_h_cm", type=float, default=FIG_H_CM)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    records = load_records(Path(args.results_root))
    plot(records, Path(args.out_dir), float(args.fig_w_cm), float(args.fig_h_cm), int(args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
