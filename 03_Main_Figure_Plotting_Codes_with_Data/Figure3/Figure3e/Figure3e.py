#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 3e for PulseBat-combo.

Linear-model feature importance for all-rate combo pulses at SOC=25%, 50%,
and 75% under 5000 ms, plotting all 100 random-seed experiments for 20Ah
and 68Ah LFP.
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
from matplotlib.ticker import FuncFormatter


RESULTS_ROOT = Path(r"G:\Experiment_Output\gpu_combo_all\Results_exp1_soc_width")
FIG_W_CM = 9.03
FIG_H_CM = 3.93
DPI = 600

MODEL = "linear"
WIDTH_MS = 5000
SOCS: Sequence[int] = (25, 50, 75)
GROUP_KEY = "all_crates_05_10_15_20_25"
CAPACITIES: Sequence[Tuple[str, str]] = (
    ("20Ah_LFP", "20Ah"),
    ("68Ah_LFP", "68Ah"),
)
CRATES: Sequence[Tuple[str, str, str]] = (
    ("0.5C", "0.5", "#EEA552"),
    ("1C", "1.0", "#D1C454"),
    ("1.5C", "1.5", "#D7826B"),
    ("2C", "2.0", "#964D50"),
    ("2.5C", "2.5", "#6F4043"),
)

POINT_SIZE = 11.0
POINT_ALPHA = 0.85
POINT_EDGE_COLOR = "#FFFFFF"
POINT_EDGE_LW = 0.55
JITTER_WIDTH = 0.17
BOX_WIDTH = 0.62
BOX_EDGE_COLOR = "#111111"
BOX_LINE_LW = 0.75
MEDIAN_COLOR = "#D4031F"
MEDIAN_LINE_LW = 0.90


@dataclass(frozen=True)
class ImportanceRecord:
    capacity_key: str
    capacity_label: str
    soc: int
    crate_key: str
    crate_label: str
    color: str
    seed: int
    importance: float
    signed_importance: float


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
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6,
        "axes.unicode_minus": False,
    })


def result_dir(results_root: Path, capacity_key: str, soc: int) -> Path:
    pattern = re.compile(
        rf"^\d{{8}}-\d{{6}}__{capacity_key}__exp1__SOC{soc}_W{WIDTH_MS}_{GROUP_KEY}$"
    )
    matches = sorted(p for p in results_root.iterdir() if p.is_dir() and pattern.match(p.name))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected 1 result directory for {capacity_key}, SOC{soc}, found {len(matches)}: {matches}"
        )
    return matches[0]


def crate_from_feature(feature: str) -> str:
    match = re.match(r"^fai_irrev_(?P<crate>.+)__W\d+__SOC\d+$", feature)
    if match is None:
        raise RuntimeError(f"Unexpected feature name: {feature}")
    return match.group("crate")


def load_capacity_records(results_root: Path, capacity_key: str, capacity_label: str, soc: int) -> List[ImportanceRecord]:
    path = result_dir(results_root, capacity_key, soc) / "featimp_all.csv"
    crate_meta: Dict[str, Tuple[str, str]] = {
        crate_key: (crate_label, color)
        for crate_key, crate_label, color in CRATES
    }
    records: List[ImportanceRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"].strip().lower() != MODEL:
                continue
            crate_key = crate_from_feature(row["feature"])
            if crate_key not in crate_meta:
                continue
            crate_label, color = crate_meta[crate_key]
            records.append(ImportanceRecord(
                capacity_key=capacity_key,
                capacity_label=capacity_label,
                soc=soc,
                crate_key=crate_key,
                crate_label=crate_label,
                color=color,
                seed=int(row["seed"]),
                importance=float(row["importance"]),
                signed_importance=float(row["signed"]),
            ))
    if not records:
        raise RuntimeError(f"No {MODEL} feature-importance records found in {path}")
    return records


def load_records(results_root: Path) -> List[ImportanceRecord]:
    records: List[ImportanceRecord] = []
    for soc in SOCS:
        for capacity_key, capacity_label in CAPACITIES:
            records.extend(load_capacity_records(results_root, capacity_key, capacity_label, soc))
    return records


def load_cached_records(csv_path: Path) -> List[ImportanceRecord]:
    capacity_key_by_label = {capacity_label: capacity_key for capacity_key, capacity_label in CAPACITIES}
    crate_key_by_label = {crate_label: crate_key for crate_key, crate_label, _ in CRATES}
    color_by_label = {crate_label: color for _, crate_label, color in CRATES}
    records: List[ImportanceRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"capacity", "soc_%", "crate", "seed", "importance", "signed_importance"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{csv_path} missing columns: {sorted(missing)}")

        for row in reader:
            capacity_label = row["capacity"].strip()
            crate_label = row["crate"].strip()
            records.append(ImportanceRecord(
                capacity_key=capacity_key_by_label[capacity_label],
                capacity_label=capacity_label,
                soc=int(row["soc_%"]),
                crate_key=crate_key_by_label[crate_label],
                crate_label=crate_label,
                color=color_by_label[crate_label],
                seed=int(row["seed"]),
                importance=float(row["importance"]),
                signed_importance=float(row["signed_importance"]),
            ))
    if not records:
        raise RuntimeError(f"No cached feature-importance records found in {csv_path}.")
    return records


def write_points_csv(records: Sequence[ImportanceRecord], out_dir: Path) -> None:
    out_csv = out_dir / "Figure3e_linear_featimp_points.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["capacity", "soc_%", "crate", "seed", "importance", "signed_importance"])
        for r in records:
            writer.writerow([r.capacity_label, r.soc, r.crate_label, r.seed, r.importance, r.signed_importance])


def rounded_ymax(records: Sequence[ImportanceRecord]) -> float:
    ymax = max(r.importance for r in records)
    if ymax <= 0:
        return 1.0
    step = 0.005
    return math.ceil((ymax + 0.001) / step) * step


def importance_tick_formatter(value: float, pos: int) -> str:
    if abs(value) < 5e-7:
        return "0"
    return f"{value:.3f}"


def plot(records: Sequence[ImportanceRecord], out_dir: Path, fig_w_cm: float, fig_h_cm: float, dpi: int) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_points_csv(records, out_dir)

    fig, axes = plt.subplots(
        2,
        len(SOCS),
        figsize=(cm_to_in(fig_w_cm), cm_to_in(fig_h_cm)),
        dpi=dpi,
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.135, right=0.990, bottom=0.215, top=0.885, hspace=0.38, wspace=0.25)

    crate_x = {crate_key: i for i, (crate_key, _, _) in enumerate(CRATES)}
    y_top = rounded_ymax(records)
    rng = np.random.default_rng(20260502)

    for row_idx, (capacity_key, capacity_label) in enumerate(CAPACITIES):
        for col_idx, soc in enumerate(SOCS):
            ax = axes[row_idx, col_idx]
            subset_panel = [r for r in records if r.capacity_key == capacity_key and r.soc == soc]
            box_data = [
                [r.importance for r in subset_panel if r.crate_key == crate_key]
                for crate_key, _, _ in CRATES
            ]
            box = ax.boxplot(
                box_data,
                positions=list(range(len(CRATES))),
                widths=BOX_WIDTH,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": MEDIAN_COLOR, "linewidth": MEDIAN_LINE_LW},
                whiskerprops={"color": BOX_EDGE_COLOR, "linewidth": BOX_LINE_LW},
                capprops={"color": BOX_EDGE_COLOR, "linewidth": BOX_LINE_LW},
                boxprops={"edgecolor": BOX_EDGE_COLOR, "linewidth": BOX_LINE_LW},
                zorder=3,
            )
            for patch, (_, _, color) in zip(box["boxes"], CRATES):
                patch.set_facecolor("none")
                patch.set_edgecolor(BOX_EDGE_COLOR)
                patch.set_alpha(1.0)

            medians = [float(np.median(values)) if values else np.nan for values in box_data]
            ax.plot(
                range(len(CRATES)),
                medians,
                color=MEDIAN_COLOR,
                lw=MEDIAN_LINE_LW,
                marker="none",
                zorder=4,
            )

            for crate_key, crate_label, color in CRATES:
                subset = [r for r in subset_panel if r.crate_key == crate_key]
                x0 = crate_x[crate_key]
                jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, len(subset))
                x = x0 + jitter
                y = np.asarray([r.importance for r in subset], dtype=float)
                ax.scatter(
                    x,
                    y,
                    s=POINT_SIZE,
                    color=color,
                    alpha=POINT_ALPHA,
                    edgecolors=POINT_EDGE_COLOR,
                    linewidths=POINT_EDGE_LW,
                    rasterized=True,
                    zorder=1,
                )
            if row_idx == 0:
                ax.set_title(f"{soc}% SOC", pad=2.0, fontsize=7)
            ax.text(0.05, 0.96, capacity_label, transform=ax.transAxes, ha="left", va="top", fontsize=7)
            ax.set_xlim(-0.55, len(CRATES) - 0.45)
            ax.set_ylim(0.0, y_top)
            ax.set_yticks([0.0, y_top / 2.0, y_top])
            ax.yaxis.set_major_formatter(FuncFormatter(importance_tick_formatter))
            ax.tick_params(length=2.0, width=0.6, pad=1.3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.75)

    fig.text(0.045, 0.555, "Feature importance", rotation=90, ha="center", va="center", fontsize=7)
    fig.supxlabel("C-Rate", fontsize=7, y=0.040)
    for ax in axes[-1, :]:
        ax.set_xticks(range(len(CRATES)))
        ax.set_xticklabels([label for _, label, _ in CRATES])

    png_path = out_dir / "Figure3e.png"
    pdf_path = out_dir / "Figure3e.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw PulseBat-combo Figure3e.")
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
        records = load_cached_records(out_dir / "Figure3e_linear_featimp_points.csv")
    plot(records, out_dir, float(args.fig_w_cm), float(args.fig_h_cm), int(args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
