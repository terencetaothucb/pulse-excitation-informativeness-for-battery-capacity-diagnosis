#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extra Figure 4 panels for 20Ah LFP using the Figure 4b and 4e layouts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Figure4.Figure4b import Figure4b as figure4b
from Figure4.Figure4e import Figure4e as figure4e


OUT_DIR = Path(__file__).resolve().parent
CAPACITY = "LFP 20Ah"
GRADIENT_STOPS = (
    (0.00, "#ABA3B0"),
    (0.33, "#817990"),
    (0.66, "#77785E"),
    (1.00, "#58412F"),
)


def draw_single(out_dir: Path) -> None:
    figure4b.CAPACITY = CAPACITY
    figure4b.OUT_STEM = "Figure4extra_b_20Ah"
    figure4b.GRADIENT_STOPS = GRADIENT_STOPS
    records = figure4b.load_records(figure4b.SOURCE_CSV, CAPACITY)
    figure4b.plot(records, out_dir, figure4b.FIG_W_CM, figure4b.FIG_H_CM, figure4b.DPI)


def draw_combo(out_dir: Path) -> None:
    figure4e.CAPACITY = CAPACITY
    figure4e.OUT_STEM = "Figure4extra_e_20Ah"
    figure4e.GRADIENT_STOPS = GRADIENT_STOPS
    records = figure4e.load_records(figure4e.SOURCE_CSV, CAPACITY)
    figure4e.plot(records, out_dir, figure4e.FIG_W_CM, figure4e.FIG_H_CM, figure4e.DPI)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw PulseBat-combo Figure4 extra 20Ah panels.")
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_single(out_dir)
    draw_combo(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
