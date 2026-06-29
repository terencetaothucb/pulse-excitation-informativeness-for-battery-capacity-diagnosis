#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run all PulseBat-combo Figure 5 panel scripts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PANELS = ("Figure5a", "Figure5b", "Figure5c", "Figure5f", "Figure5g")


def main() -> int:
    for panel in PANELS:
        script = ROOT / panel / f"{panel}.py"
        print(f"[RUN] {script}")
        subprocess.run([sys.executable, str(script)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
