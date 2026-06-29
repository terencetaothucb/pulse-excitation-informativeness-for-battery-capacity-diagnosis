#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 2a for PulseBat-combo.

20Ah LFP, SOC=50%, pulse width=5000 ms, single-pulse features.
Only the linear model test predictions from the 100 random seeds are plotted.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


RESULTS_ROOT = Path(r"G:\gpu_combo_all_single\Results_single_feature")
FIG_W_CM = 13.25
FIG_H_CM = 5.09
DPI = 600

CAPACITY = "20Ah_LFP"
WIDTH_MS = 5000
SOC = 50
MODEL = "linear"
SPLIT = "test"

CRATES: Sequence[Tuple[str, str, str]] = (
    ("0.5C", "0.5C", "#B5AED5"),
    ("1C", "1.0C", "#B2E6FD"),
    ("1.5C", "1.5C", "#B8D2CC"),
    ("2C", "2.0C", "#E8B2A7"),
    ("2.5C", "2.5C", "#FEEBB9"),
)

SOH_CMAP_COLORS = ("#3F8585", "#6ABBA5", "#66A931", "#8EBA35", "#D8C957")
SOH_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "figure2a_soh_teal_green_yellow_blue",
    SOH_CMAP_COLORS,
)

POINT_SIZE = 17.0
POINT_ALPHA = 0.75
POINT_EDGE_COLOR = "#FFFFFF"
POINT_EDGE_LW = 0.65
HIST_FACE_COLOR = "#B8B8B8"
HIST_EDGE_COLOR = "#111111"
HIST_ALPHA = 0.78
HIST_TOP = 800.0
GAUSS_LINE_COLOR = "#111111"
GAUSS_LINE_LW = 1.15


@dataclass(frozen=True)
class PredictionRecord:
    crate_key: str
    crate_label: str
    color: str
    seed: int
    sample_idx: int
    y_true_pct: float
    y_pred_pct: float
    err_pct: float


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
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 7,
        "axes.unicode_minus": False,
    })


def result_dir_for_crate(results_root: Path, crate_key: str) -> Path:
    pattern = re.compile(
        rf"^\d{{8}}-\d{{6}}__{CAPACITY}__W{WIDTH_MS}__SOC{SOC}"
        rf"__single__fai_irrev_{re.escape(crate_key)}$"
    )
    matches = sorted(p for p in results_root.iterdir() if p.is_dir() and pattern.match(p.name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected 1 result directory for {crate_key}, found {len(matches)}: {matches}")
    return matches[0]


def load_prediction_records(results_root: Path) -> List[PredictionRecord]:
    records: List[PredictionRecord] = []
    for crate_key, crate_label, color in CRATES:
        csv_path = result_dir_for_crate(results_root, crate_key) / "predictions_all.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"model", "split", "seed", "sample_idx", "y_true", "y_pred", "err"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise RuntimeError(f"{csv_path} missing columns: {sorted(missing)}")

            for row in reader:
                if row["model"].strip().lower() != MODEL or row["split"].strip().lower() != SPLIT:
                    continue
                records.append(PredictionRecord(
                    crate_key=crate_key,
                    crate_label=crate_label,
                    color=color,
                    seed=int(row["seed"]),
                    sample_idx=int(row["sample_idx"]),
                    y_true_pct=float(row["y_true"]) * 100.0,
                    y_pred_pct=float(row["y_pred"]) * 100.0,
                    err_pct=float(row["err"]) * 100.0,
                ))
    if not records:
        raise RuntimeError("No prediction records found.")
    return records


def load_cached_prediction_records(csv_path: Path) -> List[PredictionRecord]:
    color_by_label = {label: color for _, label, color in CRATES}
    key_by_label = {label: key for key, label, _ in CRATES}
    records: List[PredictionRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"crate", "seed", "sample_idx", "true_soh_%", "predicted_soh_%", "error_%"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{csv_path} missing columns: {sorted(missing)}")

        for row in reader:
            crate_label = row["crate"].strip()
            if crate_label not in key_by_label:
                raise RuntimeError(f"Unexpected C-rate label in {csv_path}: {crate_label!r}")
            records.append(PredictionRecord(
                crate_key=key_by_label[crate_label],
                crate_label=crate_label,
                color=color_by_label[crate_label],
                seed=int(row["seed"]),
                sample_idx=int(row["sample_idx"]),
                y_true_pct=float(row["true_soh_%"]),
                y_pred_pct=float(row["predicted_soh_%"]),
                err_pct=float(row["error_%"]),
            ))
    if not records:
        raise RuntimeError(f"No cached prediction records found in {csv_path}.")
    return records


def write_points_csv(records: Sequence[PredictionRecord], out_dir: Path) -> None:
    out_csv = out_dir / "Figure2a_linear_test_points.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["crate", "seed", "sample_idx", "true_soh_%", "predicted_soh_%", "error_%"])
        for r in records:
            writer.writerow([r.crate_label, r.seed, r.sample_idx, r.y_true_pct, r.y_pred_pct, r.err_pct])


def write_panel_annotation_csv(records: Sequence[PredictionRecord], out_dir: Path) -> None:
    path = out_dir / "Figure2a_panel_annotations.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "panel_col",
            "crate",
            "scatter_colormap",
            "scatter_annotation",
            "error_mean",
            "error_std",
            "error_unit",
            "n_points",
        ])
        for idx, (_, crate_label, color) in enumerate(CRATES, start=1):
            err = np.asarray([r.err_pct / 100.0 for r in records if r.crate_label == crate_label], dtype=float)
            writer.writerow([
                idx,
                crate_label,
                "-".join(SOH_CMAP_COLORS),
                crate_label,
                float(np.mean(err)) if err.size else float("nan"),
                float(np.std(err, ddof=1)) if err.size > 1 else float("nan"),
                "fraction",
                int(err.size),
            ])


def round_down(x: float, step: float) -> float:
    return math.floor(x / step) * step


def round_up(x: float, step: float) -> float:
    return math.ceil(x / step) * step


def axis_limits(records: Sequence[PredictionRecord]) -> Tuple[float, float]:
    values = np.asarray(
        [v for r in records for v in (r.y_true_pct, r.y_pred_pct)],
        dtype=float,
    )
    vmin = round_down(float(np.nanmin(values)) - 0.4, 5.0)
    vmax = round_up(float(np.nanmax(values)) + 0.4, 5.0)
    return vmin, vmax


def error_bins(records: Sequence[PredictionRecord]) -> np.ndarray:
    values = np.asarray([r.err_pct / 100.0 for r in records], dtype=float)
    left = round_down(float(np.nanmin(values)) - 0.003, 0.01)
    right = round_up(float(np.nanmax(values)) + 0.003, 0.01)
    return np.arange(left, right + 0.005, 0.01)


def soh_color_limits(records: Sequence[PredictionRecord]) -> Tuple[float, float]:
    return 65.0, 85.0


def colorbar_ticks(vmin: float, vmax: float) -> List[float]:
    left = int(round(vmin))
    right = int(round(vmax))
    return [float(t) for t in np.rint(np.linspace(left, right, 5))]


def three_ticks(vmin: float, vmax: float) -> List[float]:
    return [float(vmin), 0.5 * (float(vmin) + float(vmax)), float(vmax)]


def error_ticks(vmin: float, vmax: float) -> List[float]:
    ticks = [float(vmin), 0.0, float(vmax)]
    out: List[float] = []
    for tick in ticks:
        if not any(abs(tick - existing) < 1e-9 for existing in out):
            out.append(tick)
    return out


def error_tick_labels(vmin: float, vmax: float, panel_idx: int, panel_count: int) -> List[str]:
    return [f"{vmin:.2f}", "0", f"{vmax:.2f}"]


def hist_ymax(records: Sequence[PredictionRecord], bins: np.ndarray) -> float:
    ymax = 0
    for crate_key, _, _ in CRATES:
        err = np.asarray([r.err_pct / 100.0 for r in records if r.crate_key == crate_key], dtype=float)
        counts, _ = np.histogram(err, bins=bins)
        if counts.size:
            ymax = max(ymax, int(np.max(counts)))
    return float(max(1, int(math.ceil(ymax / 100.0) * 100)))


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if not np.isfinite(std) or std <= 0:
        return np.full_like(x, np.nan, dtype=float)
    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


def style_shared_axes(ax: mpl.axes.Axes, show_y: bool) -> None:
    ax.tick_params(length=2.0, width=0.6, pad=1.2)
    ax.tick_params(axis="y", left=show_y, labelleft=show_y)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)


def plot(records: Sequence[PredictionRecord], out_dir: Path, fig_w_cm: float, fig_h_cm: float, dpi: int) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_points_csv(records, out_dir)
    write_panel_annotation_csv(records, out_dir)

    fig = plt.figure(figsize=(cm_to_in(fig_w_cm), cm_to_in(fig_h_cm)), dpi=dpi)

    scatter_axes: List[mpl.axes.Axes] = []
    hist_axes: List[mpl.axes.Axes] = []
    xy_min, xy_max = axis_limits(records)
    y_min = 65.0
    soh_vmin, soh_vmax = soh_color_limits(records)
    soh_norm = mpl.colors.Normalize(vmin=soh_vmin, vmax=soh_vmax)
    bins = error_bins(records)
    hist_top = HIST_TOP
    bin_width = float(np.mean(np.diff(bins)))

    panel_h = 0.305
    left = 0.060
    right = 0.985
    col_gap = 0.035
    panel_w = (right - left - (len(CRATES) - 1) * col_gap) / len(CRATES)
    top_row_y = 0.675
    bottom_row_y = 0.215

    for idx, (crate_key, crate_label, color) in enumerate(CRATES):
        x0 = left + idx * (panel_w + col_gap)
        ax_scatter = fig.add_axes([x0, top_row_y, panel_w, panel_h])
        ax_hist = fig.add_axes([x0, bottom_row_y, panel_w, panel_h])
        scatter_axes.append(ax_scatter)
        hist_axes.append(ax_hist)

        subset = [r for r in records if r.crate_key == crate_key]
        x = np.asarray([r.y_pred_pct for r in subset], dtype=float)
        y = np.asarray([r.y_true_pct for r in subset], dtype=float)
        err = np.asarray([r.err_pct / 100.0 for r in subset], dtype=float)

        ax_scatter.plot(
            [xy_min, xy_max],
            [xy_min, xy_max],
            color="#111111",
            lw=1.15,
            ls=(0, (3.0, 2.0)),
            zorder=0,
        )
        ax_scatter.scatter(
            x,
            y,
            s=POINT_SIZE,
            c=y,
            cmap=SOH_CMAP,
            norm=soh_norm,
            alpha=POINT_ALPHA,
            edgecolors=POINT_EDGE_COLOR,
            linewidths=POINT_EDGE_LW,
            rasterized=True,
            zorder=3,
        )
        ax_scatter.set_xlim(xy_min, xy_max)
        ax_scatter.set_ylim(y_min, xy_max)
        ax_scatter.set_xticks(three_ticks(xy_min, xy_max))
        ax_scatter.set_yticks(three_ticks(y_min, xy_max))
        ax_scatter.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax_scatter.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        style_shared_axes(ax_scatter, show_y=idx == 0)

        ax_hist.axvline(
            0.0,
            color="#B8B8B8",
            lw=0.80,
            ls=(0, (3.0, 2.0)),
            zorder=0,
        )
        ax_hist.hist(
            err,
            bins=bins,
            color=HIST_FACE_COLOR,
            alpha=HIST_ALPHA,
            edgecolor=HIST_EDGE_COLOR,
            linewidth=0.45,
            label=crate_label,
            zorder=2,
        )
        err_mean = float(np.mean(err)) if err.size else float("nan")
        err_std = float(np.std(err, ddof=1)) if err.size > 1 else float("nan")
        x_fit = np.linspace(float(bins[0]), float(bins[-1]), 300)
        y_fit = normal_pdf(x_fit, err_mean, err_std) * float(err.size) * bin_width
        ax_hist.plot(
            x_fit,
            y_fit,
            color=GAUSS_LINE_COLOR,
            lw=GAUSS_LINE_LW,
            solid_capstyle="round",
            zorder=4,
        )
        ax_hist.set_xlim(float(bins[0]), float(bins[-1]))
        ax_hist.set_ylim(0.0, hist_top)
        ax_hist.set_xticks(error_ticks(float(bins[0]), float(bins[-1])))
        ax_hist.set_yticks(three_ticks(0.0, hist_top))
        ax_hist.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax_hist.set_xticklabels(error_tick_labels(float(bins[0]), float(bins[-1]), idx, len(CRATES)))
        hist_tick_labels = ax_hist.get_xticklabels()
        if idx == 0 and hist_tick_labels:
            hist_tick_labels[0].set_ha("left")
        if idx == len(CRATES) - 1 and hist_tick_labels:
            hist_tick_labels[-1].set_ha("right")
        style_shared_axes(ax_hist, show_y=idx == 0)

    fig.text(0.017, top_row_y + panel_h / 2.0, "True SOH (%)", ha="center", va="center", rotation=90, fontsize=7)
    fig.text(0.017, bottom_row_y + panel_h / 2.0, "Count", ha="center", va="center", rotation=90, fontsize=7)
    fig.text(0.535, 0.590, "Predicted SOH (%)", ha="center", va="center", fontsize=7)
    fig.text(0.535, 0.132, "Error", ha="center", va="center", fontsize=7)

    cax = fig.add_axes([left, 0.070, 0.5 * (right - left), 0.030])
    sm = mpl.cm.ScalarMappable(norm=soh_norm, cmap=SOH_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks(colorbar_ticks(soh_vmin, soh_vmax))
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    cbar.ax.tick_params(length=2.0, width=0.6, pad=1.0, labelsize=5.5)
    cbar.outline.set_linewidth(0.6)

    png_path = out_dir / "Figure2a.png"
    pdf_path = out_dir / "Figure2a.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw PulseBat-combo Figure2a.")
    parser.add_argument("--results_root", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--fig_w_cm", type=float, default=FIG_W_CM)
    parser.add_argument("--fig_h_cm", type=float, default=FIG_H_CM)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_root = Path(args.results_root)
    if results_root.exists():
        records = load_prediction_records(results_root)
    else:
        cached_csv = out_dir / "Figure2a_linear_test_points.csv"
        records = load_cached_prediction_records(cached_csv)
    plot(records, out_dir, float(args.fig_w_cm), float(args.fig_h_cm), int(args.dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
