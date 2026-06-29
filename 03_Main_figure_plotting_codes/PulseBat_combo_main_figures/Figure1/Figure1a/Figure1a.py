#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Figure 1a for PulseBat-combo: SOH histograms for 20, 35, and 68 Ah LFP cells.

SOH is calculated with the same Step1-consistent definition used in the
PulseBat Figure1a script:
    Q   = -df.values[3, 16]
    Qn  = int(filename.split("_")[2])
    SOH = Q / Qn

Only SOC_5-x and Part_1-1 / Part_1-2 files are kept, matching the prior
Figure1a filtering protocol.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from matplotlib.colors import to_rgb


plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.linewidth": 0.75,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


RE_DUP_SUFFIX_COPY = re.compile(r"[锛?]\s*\d+\s*[锛?]\s*$")
RE_SOC_5_X = re.compile(r"(?i)(?:^|_)soc_5-(55|60|65|70|75|80|85|90)(?:_|\.|$)")
RE_PART_1 = re.compile(r"(?i)(?:^|_)part_(1-1|1-2)(?:_|\.|$)")


@dataclass(frozen=True)
class LfpGroup:
    folder_name: str
    label: str
    color: str

    @property
    def capacity_ah(self) -> int:
        return int(self.folder_name.split("Ah", 1)[0])


def extract_soc_high_from_name(name: str) -> Optional[int]:
    match = RE_SOC_5_X.search(name)
    if match is None:
        return None
    return int(match.group(1))


def extract_part_tag_from_name(name: str) -> Optional[str]:
    match = RE_PART_1.search(name)
    if match is None:
        return None
    return str(match.group(1))


def should_keep_xlsx_file(fp: Path) -> Tuple[bool, str, Optional[int], Optional[str]]:
    name = fp.name
    stem = fp.stem.strip()

    if name.startswith("~$"):
        return False, "excel_temp", None, None
    if RE_DUP_SUFFIX_COPY.search(stem) is not None:
        return False, "duplicate_copy_(n)", None, None

    soc_high = extract_soc_high_from_name(name)
    if soc_high is None:
        return False, "drop_non_SOC_5-x", None, None

    part_tag = extract_part_tag_from_name(name)
    if part_tag is None:
        return False, "drop_non_Part_1-1_or_1-2", soc_high, None

    return True, f"keep_SOC_5-{soc_high}_Part_{part_tag}", soc_high, part_tag


def parse_qn_from_filename(filename: str) -> Optional[int]:
    parts = filename.split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[2])
    except Exception:
        return None


def compute_soh_from_step1_xlsx(xlsx_path: Path) -> Optional[Tuple[float, float, int]]:
    try:
        import openpyxl

        wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
        ws = wb.worksheets[0]
        # pandas.read_excel(...).values[3, 16] corresponds to Excel cell Q5
        # when the first row is interpreted as the header.
        raw_q = ws.cell(row=5, column=17).value
        wb.close()
    except Exception:
        return None

    try:
        q = -float(raw_q)
    except Exception:
        return None

    qn = parse_qn_from_filename(xlsx_path.name)
    if qn is None or qn == 0:
        return None

    soh = q / float(qn)
    if not np.isfinite(soh):
        return None

    return float(soh), float(q), int(qn)


def collect_soh_for_group(group_folder: Path, group: LfpGroup) -> Tuple[List[float], pd.DataFrame, Dict[str, int]]:
    raw_files = sorted([
        p for p in group_folder.rglob("*.xlsx")
        if p.is_file() and not p.name.startswith("~$")
    ])

    stats = {
        "n_xlsx_total_raw": len(raw_files),
        "n_after_name_filter": 0,
        "n_filtered_out_name": 0,
        "n_invalid_soh_read": 0,
        "n_used": 0,
    }

    rows = []
    sohs: List[float] = []

    for fp in raw_files:
        keep, keep_reason, soc_high, part_tag = should_keep_xlsx_file(fp)
        if not keep:
            stats["n_filtered_out_name"] += 1
            continue

        stats["n_after_name_filter"] += 1
        out = compute_soh_from_step1_xlsx(fp)
        if out is None:
            stats["n_invalid_soh_read"] += 1
            continue

        soh, q, qn = out
        sohs.append(float(soh))
        stats["n_used"] += 1
        rows.append({
            "group": group.folder_name,
            "label": group.label,
            "file_name": fp.name,
            "file_path": str(fp),
            "SOC_low": 5,
            "SOC_high": int(soc_high) if soc_high is not None else np.nan,
            "SOC_window": f"5-{soc_high}" if soc_high is not None else "",
            "Part": str(part_tag) if part_tag is not None else "",
            "keep_reason": keep_reason,
            "SOH": float(soh),
            "SOH_percent": float(soh * 100.0),
            "Qn_Ah": int(qn),
            "Q_Ah": float(q),
        })

    return sohs, pd.DataFrame(rows), stats


def _round_down(x: float, step: float) -> float:
    return math.floor(x / step) * step


def _round_up(x: float, step: float) -> float:
    return math.ceil(x / step) * step


def build_bins(all_vals_percent: np.ndarray, bin_w_percent: float = 5.0) -> Tuple[np.ndarray, float, float]:
    vals = np.asarray(all_vals_percent, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise RuntimeError("No valid SOH values found.")

    x_left = min(60.0, _round_down(float(np.min(vals)), bin_w_percent))
    x_right = max(100.0, _round_up(float(np.max(vals)), bin_w_percent))
    bins = np.arange(x_left, x_right + bin_w_percent * 0.5, bin_w_percent)
    return bins, x_left, x_right


def compute_ymax(group_sohs_percent: List[np.ndarray], bins: np.ndarray) -> int:
    ymax = 0
    for vals in group_sohs_percent:
        counts, _ = np.histogram(vals, bins=bins)
        if counts.size:
            ymax = max(ymax, int(np.max(counts)))
    return max(1, int(math.ceil(ymax / 5.0) * 5))


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if not np.isfinite(std) or std <= 0:
        return np.full_like(x, np.nan, dtype=float)
    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


def lighten_color(color: str, amount: float = 0.18) -> Tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple(rgb + (1.0 - rgb) * float(amount))


def write_group_counts(
    out_dir: Path,
    groups: List[LfpGroup],
    group_sohs: List[List[float]],
    stats_list: List[Dict[str, int]],
    bins_percent: np.ndarray,
) -> None:
    def fmt_edge(x: float) -> str:
        return f"{x:.0f}" if abs(x - round(x)) < 1e-8 else f"{x:.1f}"

    bin_labels = [
        f"{fmt_edge(float(bins_percent[i]))}-{fmt_edge(float(bins_percent[i + 1]))}%"
        for i in range(len(bins_percent) - 1)
    ]
    rows = []

    for group, sohs, stats in zip(groups, group_sohs, stats_list):
        arr = np.asarray(sohs, dtype=float)
        arr_pct = arr * 100.0
        counts, _ = np.histogram(arr_pct, bins=bins_percent)
        soh_mean = float(np.mean(arr)) if arr.size else np.nan
        soh_std = float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan
        row = {
            "group": group.folder_name,
            "label": group.label,
            "N": int(stats["n_used"]),
            "n_xlsx_total_raw": int(stats["n_xlsx_total_raw"]),
            "n_after_name_filter": int(stats["n_after_name_filter"]),
            "n_filtered_out_name": int(stats["n_filtered_out_name"]),
            "n_invalid_soh_read": int(stats["n_invalid_soh_read"]),
            "n_used": int(stats["n_used"]),
            "soh_min": float(np.min(arr)) if arr.size else np.nan,
            "soh_mean": soh_mean,
            "soh_std": soh_std,
            "soh_max": float(np.max(arr)) if arr.size else np.nan,
            "soh_percent_mean": soh_mean * 100.0 if np.isfinite(soh_mean) else np.nan,
            "soh_percent_std": soh_std * 100.0 if np.isfinite(soh_std) else np.nan,
        }
        for label, count in zip(bin_labels, counts):
            row[f"bin_{label}"] = int(count)
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / "Figure1a_lfp_group_counts.csv", index=False, encoding="utf-8-sig")

    stat_rows = []
    for group, sohs, stats in zip(groups, group_sohs, stats_list):
        arr = np.asarray(sohs, dtype=float)
        arr_pct = arr * 100.0
        stat_rows.append({
            "panel_order": int(groups.index(group) + 1),
            "group": group.folder_name,
            "label": group.label,
            "N": int(stats["n_used"]),
            "mean_soh_percent": float(np.mean(arr_pct)) if arr_pct.size else np.nan,
            "std_soh_percent": float(np.std(arr_pct, ddof=1)) if arr_pct.size > 1 else np.nan,
            "min_soh_percent": float(np.min(arr_pct)) if arr_pct.size else np.nan,
            "max_soh_percent": float(np.max(arr_pct)) if arr_pct.size else np.nan,
        })
    pd.DataFrame(stat_rows).to_csv(out_dir / "Figure1a_lfp_panel_stats.csv", index=False, encoding="utf-8-sig")


def load_soh_from_summary(
    summary_csv: Path,
    group_counts_csv: Path,
    groups: List[LfpGroup],
) -> Tuple[List[List[float]], List[Dict[str, int]]]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    counts_df = pd.read_csv(group_counts_csv) if group_counts_csv.exists() else pd.DataFrame()
    group_sohs: List[List[float]] = []
    stats_list: List[Dict[str, int]] = []

    for group in groups:
        d = df[df["group"].eq(group.folder_name)].copy()
        if d.empty:
            raise RuntimeError(f"No rows found in summary CSV for {group.folder_name}")

        sohs = d["SOH"].astype(float).dropna().tolist()
        group_sohs.append(sohs)

        stats = {
            "n_xlsx_total_raw": len(sohs),
            "n_after_name_filter": len(sohs),
            "n_filtered_out_name": 0,
            "n_invalid_soh_read": 0,
            "n_used": len(sohs),
        }
        if not counts_df.empty:
            matched = counts_df[counts_df["group"].eq(group.folder_name)]
            if not matched.empty:
                row = matched.iloc[0]
                for key in stats:
                    if key in row and pd.notna(row[key]):
                        stats[key] = int(row[key])
        stats_list.append(stats)

    return group_sohs, stats_list


def plot_lfp_soh_histograms(
    groups: List[LfpGroup],
    group_sohs: List[List[float]],
    stats_list: List[Dict[str, int]],
    out_dir: Path,
    fig_w_cm: float = 14.18,
    fig_h_cm: float = 3.26,
    alpha: float = 0.68,
    bin_w_percent: float = 2.5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    group_sohs_percent = [np.asarray(vals, dtype=float) * 100.0 for vals in group_sohs]
    all_vals_percent = np.concatenate(group_sohs_percent)
    bins, x_left, x_right = build_bins(all_vals_percent, bin_w_percent=bin_w_percent)
    y_max = compute_ymax(group_sohs_percent, bins)
    y_mid = int(math.ceil((y_max / 2.0) / 5.0) * 5)

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(fig_w_cm / 2.54, fig_h_cm / 2.54),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.29, top=0.95, wspace=0.18)

    for ax, group, vals in zip(axes, groups, group_sohs_percent):
        vals = vals[np.isfinite(vals)]
        counts, _ = np.histogram(vals, bins=bins)
        ax.bar(
            bins[:-1],
            counts,
            width=bin_w_percent,
            align="edge",
            color=lighten_color(group.color, amount=0.18),
            edgecolor=group.color,
            linewidth=0.45,
            alpha=alpha,
        )

        n = int(vals.size)
        mean = float(np.mean(vals)) if n else np.nan
        std = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        if np.isfinite(std) and std > 0:
            x_fit = np.linspace(x_left, x_right, 400)
            y_fit = normal_pdf(x_fit, mean, std) * n * bin_w_percent
            ax.plot(x_fit, y_fit, color=group.color, lw=0.75, alpha=0.95)

        ax.set_xlim(x_left, x_right + 1.0)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("State of health (%)", fontsize=8, labelpad=2)
        ax.set_ylabel("")

        ax.xaxis.set_major_locator(FixedLocator([x for x in [60, 70, 80, 90, 100] if x_left <= x <= x_right]))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.yaxis.set_major_locator(FixedLocator([0, y_mid, y_max]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

        ax.tick_params(length=2.2, width=0.6, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)

    png_path = out_dir / "Figure1a_lfp_soh_hist.png"
    pdf_path = out_dir / "Figure1a_lfp_soh_hist.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)

    write_group_counts(out_dir, groups, group_sohs, stats_list, bins)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def default_groups() -> List[LfpGroup]:
    return [
        LfpGroup("20Ah LFP", "20 Ah", "#077FFF"),
        LfpGroup("35Ah LFP", "35 Ah", "#FF972F"),
        LfpGroup("68Ah LFP", "68 Ah", "#50D541"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot combo Figure1a: SOH histograms for LFP cells.")
    parser.add_argument(
        "--step1_root",
        type=str,
        default=r"E:\Datasets\PulseBat\PulseBat-all-raw\ProcessingData All\step_1_extract workstep sheet",
        help="Root folder containing Step1 group subfolders.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Output directory.",
    )
    parser.add_argument("--fig_w_cm", type=float, default=13.46, help="Figure width in cm.")
    parser.add_argument("--fig_h_cm", type=float, default=3.26, help="Figure height in cm.")
    parser.add_argument("--alpha", type=float, default=0.68, help="Histogram transparency.")
    parser.add_argument("--bin_w_percent", type=float, default=2.5, help="Histogram bin width in SOH percent.")
    parser.add_argument(
        "--from_summary",
        action="store_true",
        help="Use the existing Figure1a_lfp_soh_summary.csv instead of re-reading Step1 xlsx files.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "Figure1a_lfp_soh_summary.csv"),
        help="Existing SOH summary CSV used with --from_summary.",
    )
    args = parser.parse_args()

    step1_root = Path(args.step1_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = default_groups()
    if args.from_summary:
        group_sohs, stats_list = load_soh_from_summary(
            summary_csv=Path(args.summary_csv),
            group_counts_csv=out_dir / "Figure1a_lfp_group_counts.csv",
            groups=groups,
        )
        for group, sohs, stats in zip(groups, group_sohs, stats_list):
            print(
                f"[INFO] {group.folder_name}: from_summary={len(sohs)}, "
                f"raw={stats['n_xlsx_total_raw']}, used={stats['n_used']}"
            )
    else:
        if not step1_root.exists():
            raise FileNotFoundError(f"Step1 root folder not found: {step1_root}")

        group_sohs = []
        summary_frames: List[pd.DataFrame] = []
        stats_list = []

        for group in groups:
            folder = step1_root / group.folder_name
            if not folder.exists():
                raise FileNotFoundError(f"Group folder not found: {folder}")

            sohs, df_group, stats = collect_soh_for_group(folder, group)
            if len(sohs) == 0:
                raise RuntimeError(f"No valid SOH values found for {group.folder_name}")

            group_sohs.append(sohs)
            summary_frames.append(df_group)
            stats_list.append(stats)
            print(
                f"[INFO] {group.folder_name}: raw={stats['n_xlsx_total_raw']}, "
                f"after_filter={stats['n_after_name_filter']}, "
                f"invalid={stats['n_invalid_soh_read']}, used={stats['n_used']}"
            )

        df_all = pd.concat(summary_frames, ignore_index=True)
        summary_path = out_dir / "Figure1a_lfp_soh_summary.csv"
        df_all.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved: {summary_path}")

    plot_lfp_soh_histograms(
        groups=groups,
        group_sohs=group_sohs,
        stats_list=stats_list,
        out_dir=out_dir,
        fig_w_cm=float(args.fig_w_cm),
        fig_h_cm=float(args.fig_h_cm),
        alpha=float(args.alpha),
        bin_w_percent=float(args.bin_w_percent),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
