#!/usr/bin/env python3
"""
Write Brzęczek demo figures under figures/.

**Default:** Kaggle *Retail Insights* CSV — place `data/raw/data.csv` after download
(see README). Numeric notes: `figures/brzezcek_demo_summary.md`

Usage (from repo root):

  PYTHONPATH=src python scripts/visualize_brzezcek_demo.py
  PYTHONPATH=src python scripts/visualize_brzezcek_demo.py --synthetic
  PYTHONPATH=src python scripts/visualize_brzezcek_demo.py --data /path/to/data.csv
"""
from __future__ import annotations

import argparse
import os
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
    parser = argparse.ArgumentParser(description="Brzęczek-style demo figure")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic 6-category RNG demo instead of Kaggle CSV",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to retail insights data.csv (default: data/raw/data.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figures" / "brzezcek_demo.png",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    save_brzezcek_demo(
        args.out,
        data_csv=args.data,
        use_synthetic=args.synthetic,
    )
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
