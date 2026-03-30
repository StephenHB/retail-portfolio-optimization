"""Smoke test for optional matplotlib demo (skipped if matplotlib missing)."""

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

from retail_portfolio.viz.brzezcek_demo import (
    make_brzezcek_demo_figure,
    save_brzezcek_demo,
    synthetic_demo_inputs,
)


def test_synthetic_demo_inputs_shapes() -> None:
    y, v, corr, labels = synthetic_demo_inputs()
    assert y.shape == (6,) and v.shape == (6, 6) and corr.shape == (6, 6)
    assert len(labels) == 6


def test_make_figure_runs(tmp_path: Path) -> None:
    y, v, corr, labels = synthetic_demo_inputs()
    fig = make_brzezcek_demo_figure(y, v, corr, labels)
    out = tmp_path / "demo.png"
    fig.savefig(out)
    assert out.stat().st_size > 1000


def test_save_brzezcek_demo(tmp_path: Path) -> None:
    path = save_brzezcek_demo(tmp_path / "x.png")
    assert path.exists() and path.stat().st_size > 1000
