#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 3b for PulseBat-combo.

68Ah LFP combo-pulse linear-model MAE versus SOC.
Panels are pulse widths of 50, 500, 3000, and 5000 ms.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


RESULTS_ROOT = Path(r"G:\Experiment_Output\gpu_combo_all\Results_exp1_soc_width")
FIG_W_CM = 13.65
FIG_H_CM = 2.0
DPI = 600

CAPACITY = "68Ah_LFP"
MODEL = "linear"

WIDTHS_MS: Sequence[int] = (50, 500, 3000, 5000)
SOCS: Sequence[int] = (5, 15, 25, 35, 45, 55, 65, 75)
GROUPS: Sequence[Tuple[str, str, str]] = (
    ("low_crates_05_10", "0.5C|1.0C", "#A7E8E2"),
    ("mid_crates_10_15", "1.0C|1.5C", "#84BEC8"),
    ("mid_crates_15_20", "1.5C|2.0C", "#659CB2"),
    ("high_crates_20_25", "2.0C|2.5C", "#478C7A"),
    ("all_crates_05_10_15_20_25", "0.5C|1.0C|1.5C|2.0C|2.5C", "#185F59"),
)

LINE_LW = 1.15
MARKER_SIZE = 4.0
MARKER_EDGE_LW = 0.35
LEGEND_W_CM = 9.0
LEGEND_H_CM = 0.45


@dataclass(frozen=True)
class MaeRecord:
    width_ms: int
    soc: int
    group_key: str
    group_label: str
    color: str
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
        "legend.fontsize": 7,
        "axes.unicode_minus": False,
    })


def build_result_index(results_root: Path) -> Dict[Tuple[int, int, str], Path]:
    pattern = re.compile(
        rf"^\d{{8}}-\d{{6}}__{CAPACITY}__exp1__SOC(?P<soc>\d+)_W(?P<width>\d+)"
        rf"_(?P<group>.+)$"
    )
    expected_groups = {group_key for group_key, _, _ in GROUPS}
    index: Dict[Tuple[int, int, str], Path] = {}
    duplicates: Dict[Tuple[int, int, str], List[Path]] = {}
    for p in results_root.iterdir():
        if not p.is_dir():
            continue
        match = pattern.match(p.name)
        if match is None:
            continue
        group_key = match.group("group")
        if group_key not in expected_groups:
            continue
        key = (int(match.group("width")), int(match.group("soc")), group_key)
        if key in index:
            duplicates.setdefault(key, [index[key]]).append(p)
        else:
            index[key] = p
    if duplicates:
        raise RuntimeError(f"Duplicate result directories found: {duplicates}")
    return index


def result_dir(index: Dict[Tuple[int, int, str], Path], width_ms: int, soc: int, group_key: str) -> Path:
    key = (width_ms, soc, group_key)
    if key not in index:
        raise RuntimeError(
            f"Expected 1 result directory for {CAPACITY}, W{width_ms}, SOC{soc}, {group_key}; "
            "found 0"
        )
    return index[key]


def linear_mae(metrics_csv: Path) -> float:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"].strip().lower() == MODEL:
                return float(row["mae_test_median"])
    raise RuntimeError(f"No {MODEL} row found in {metrics_csv}")


def load_records(results_root: Path) -> List[MaeRecord]:
    records: List[MaeRecord] = []
    index = build_result_index(results_root)
    for width_ms in WIDTHS_MS:
        for soc in SOCS:
            for group_key, group_label, color in GROUPS:
                d = result_dir(index, width_ms, soc, group_key)
                records.append(MaeRecord(
                    width_ms=width_ms,
                    soc=soc,
                    group_key=group_key,
                    group_label=group_label,
                    color=color,
                    mae=linear_mae(d / "metrics_summary.csv"),
                ))
    return records


def load_cached_records(csv_path: Path) -> List[MaeRecord]:
    color_by_label = {label: color for _, label, color in GROUPS}
    key_by_label = {label: key for key, label, _ in GROUPS}
    records: List[MaeRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        mae_col = "mae" if "mae" in fieldnames else "mae_%"
        for row in reader:
            group_label = row["group"].strip()
            mae = float(row[mae_col])
            if mae_col == "mae_%":
                mae /= 100.0
            records.append(MaeRecord(
                width_ms=int(row["width_ms"]),
                soc=int(row["soc_%"]),
                group_key=key_by_label[group_label],
                group_label=group_label,
                color=color_by_label[group_label],
                mae=mae,
            ))
    if not records:
        raise RuntimeError(f"No cached MAE records found in {csv_path}.")
    return records


def write_mae_csv(records: Sequence[MaeRecord], out_dir: Path) -> None:
    out_csv = out_dir / "Figure3b_linear_combo_mae_vs_soc.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["width_ms", "soc_%", "group", "mae"])
        for r in records:
            writer.writerow([r.width_ms, r.soc, r.group_label, r.mae])


def rounded_ymax(records: Sequence[MaeRecord]) -> float:
    ymax = max(r.mae for r in records)
    return max(0.005, math.ceil((ymax + 0.0015) / 0.005) * 0.005)


def mae_tick_formatter(value: float, pos: int) -> str:
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.2f}"


def plot(records: Sequence[MaeRecord], out_dir: Path, fig_w_cm: float, fig_h_cm: float, dpi: int) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_mae_csv(records, out_dir)

    fig, axes = plt.subplots(
        1,
        len(WIDTHS_MS),
        figsize=(cm_to_in(fig_w_cm), cm_to_in(fig_h_cm)),
        dpi=dpi,
        sharey=True,
    )
    fig.subplots_adjust(left=0.080, right=0.985, bottom=0.34, top=0.92, wspace=0.09)

    y_top = rounded_ymax(records)
    y_ticks = [0.0, y_top / 2.0, y_top]

    for i, (ax, width_ms) in enumerate(zip(axes, WIDTHS_MS)):
        for group_key, group_label, color in GROUPS:
            y = [
                r.mae
                for soc in SOCS
                for r in records
                if r.width_ms == width_ms and r.soc == soc and r.group_key == group_key
            ]
            ax.plot(
                SOCS,
                y,
                color=color,
                lw=LINE_LW,
                marker="o",
                ms=MARKER_SIZE,
                mec="0.55",
                mew=MARKER_EDGE_LW,
                mfc=color,
                zorder=2,
            )

        ax.set_xlim(min(SOCS) - 2.0, max(SOCS) + 2.0)
        ax.set_ylim(0.0, y_top)
        ax.set_xticks(SOCS)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(mae_tick_formatter))
        ax.tick_params(length=2.0, width=0.6, pad=1.3)
        ax.tick_params(axis="x", labelrotation=0, pad=2.0)
        for label in ax.get_xticklabels():
            label.set_ha("center")
        if i == 0:
            ax.set_ylabel("MAE", labelpad=2)
        else:
            ax.tick_params(labelleft=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)

    fig.supxlabel("State of charge (%)", fontsize=7, y=0.005)

    png_path = out_dir / "Figure3b.png"
    pdf_path = out_dir / "Figure3b.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


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
        for _, label, color in GROUPS
    ]
    fig.legend(
        handles=handles,
        loc="center",
        ncol=len(GROUPS),
        frameon=False,
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )

    png_path = out_dir / "Figure3b_colorbar.png"
    pdf_path = out_dir / "Figure3b_colorbar.pdf"
    fig.savefig(png_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(pdf_path, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw PulseBat-combo Figure3b.")
    parser.add_argument("--results_root", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--fig_w_cm", type=float, default=FIG_W_CM)
    parser.add_argument("--fig_h_cm", type=float, default=FIG_H_CM)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_root = Path(args.results_root)
    if results_root.exists():
        records = load_records(results_root)
    else:
        records = load_cached_records(out_dir / "Figure3b_linear_combo_mae_vs_soc.csv")
    plot(records, out_dir, float(args.fig_w_cm), float(args.fig_h_cm), int(args.dpi))
    plot_colorbar(out_dir, int(args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
