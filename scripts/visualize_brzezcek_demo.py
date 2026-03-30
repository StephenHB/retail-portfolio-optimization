#!/usr/bin/env python3
"""
Write Brzęczek demo figures under figures/ (synthetic data + retail_portfolio outputs).

Usage (from repo root):
  PYTHONPATH=src python scripts/visualize_brzezcek_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retail_portfolio.viz.brzezcek_demo import save_brzezcek_demo  # noqa: E402


def main() -> None:
    out = ROOT / "figures" / "brzezcek_demo.png"
    path = save_brzezcek_demo(out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
