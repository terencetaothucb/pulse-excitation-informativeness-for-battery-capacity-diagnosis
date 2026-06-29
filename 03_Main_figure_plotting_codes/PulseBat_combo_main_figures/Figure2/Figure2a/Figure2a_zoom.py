#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Zoomed view of the first +0.5C/-0.5C pulse pair in Figure2a."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


INPUT_CSV = Path(
    r"E:\Datasets\PulseBat\PulseBat_Fig\Figure2\Figure2data"
    r"\LFP_C_68_B_89_SOC_5-70_Part_1-1_ID_02HCB68111211Y86M0002366_5000_only.csv"
)
OUT_DIR = Path(__file__).resolve().parent

CAPACITY_AH = 68.0
ANCHOR_CSV_ROW_INCLUDING_HEADER = 1183
DT_S = 0.01

# Display the complete first 0.5C pulse pair and all subsequent rest data,
# stopping one sample before the +1C pulse begins at 160.04 s.
DATA_T_START_S = -2.0
DATA_T_END_S = 160.03
DATA_FRACTION_OF_PANEL = 0.60

# Calibrated for the original script's tight-bounding-box export so that the
# final 600 dpi PNG is exactly 3303 x 951 px, matching Figure2a.png.
FIG_W_CM = 12.9178
FIG_H_CM = 2.9905
DPI = 600

VOLT_COLOR = "#2DAFF5"
CRATE_COLOR = "#FF7F0E"


def cm_to_in(value_cm: float) -> float:
    return value_cm / 2.54


def main() -> None:
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    voltage = pd.to_numeric(df["实际电压(V)"], errors="coerce").to_numpy(float)
    current = pd.to_numeric(df["实际电流(A)"], errors="coerce").to_numpy(float)

    anchor_index = ANCHOR_CSV_ROW_INCLUDING_HEADER - 2
    time_s = (np.arange(len(df), dtype=float) - anchor_index) * DT_S
    crate = current / CAPACITY_AH

    selected = (
        np.isfinite(voltage)
        & np.isfinite(crate)
        & (time_s >= DATA_T_START_S)
        & (time_s <= DATA_T_END_S)
    )
    t = time_s[selected]
    v = voltage[selected]
    c = crate[selected]

    # Extending the x-axis after the plotted data leaves
    # approximately 40% of each panel blank on the right.
    x_max = DATA_T_START_S + (
        (DATA_T_END_S - DATA_T_START_S) / DATA_FRACTION_OF_PANEL
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    fig = plt.figure(figsize=(cm_to_in(FIG_W_CM), cm_to_in(FIG_H_CM)))
    grid = fig.add_gridspec(2, 1, height_ratios=(1, 1), hspace=0.25)
    ax_v = fig.add_subplot(grid[0, 0])
    ax_c = fig.add_subplot(grid[1, 0], sharex=ax_v)

    ax_v.plot(t, v, color=VOLT_COLOR, linewidth=0.8)
    ax_v.set_ylabel("Voltage (V)", labelpad=0)
    ax_v.tick_params(axis="x", labelbottom=False)
    ax_v.set_ylim(3.290, 3.385)
    ax_v.set_yticks([3.29, 3.34, 3.38])
    ax_v.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax_c.plot(t, c, color=CRATE_COLOR, linewidth=0.8)
    ax_c.set_ylabel("C-Rate", labelpad=0)
    ax_c.set_xlabel("Time (s)", labelpad=0)
    ax_c.set_ylim(-0.60, 0.60)
    ax_c.set_yticks([-0.5, 0.0, 0.5])
    ax_c.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_c.set_xticks([0, 40, 80, 120, 160])
    ax_c.tick_params(axis="x", pad=1)

    for axis in (ax_v, ax_c):
        axis.set_xlim(DATA_T_START_S, x_max)
        axis.spines["top"].set_visible(True)
        axis.spines["right"].set_visible(True)
        axis.tick_params(width=0.8, length=4)

    fig.subplots_adjust(left=0, right=1, top=1.1, bottom=0)

    out_png = OUT_DIR / "Figure2a_zoom.png"
    out_pdf = OUT_DIR / "Figure2a_zoom.pdf"
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"PNG: {out_png}")
    print(f"PDF: {out_pdf}")
    print(f"Data window: {t.min():.2f} to {t.max():.2f} s")
    print(f"Panel x-limits: {DATA_T_START_S:.2f} to {x_max:.2f} s")


if __name__ == "__main__":
    main()
