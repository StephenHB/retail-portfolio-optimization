"""Tests for Kaggle retail insights → Brzęczek inputs."""

from pathlib import Path

import numpy as np
import pytest

from retail_portfolio.data.kaggle_retail_insights import (
    brzezcek_inputs_from_retail_insights_csv,
    load_retail_insights_csv,
    monthly_category_revenue,
    parse_currency_series,
    wide_monthly_revenue,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "retail_insights_sample.csv"


def test_parse_currency_series() -> None:
    import pandas as pd

    s = parse_currency_series(pd.Series(["$1,234.56", "$0.00"]))
    assert np.isclose(s.iloc[0], 1234.56)
    assert np.isclose(s.iloc[1], 0.0)


def test_load_and_aggregate_pipeline() -> None:
    raw = load_retail_insights_csv(FIXTURE)
    assert len(raw) == 30
    long_df = monthly_category_revenue(raw)
    wide = wide_monthly_revenue(long_df)
    assert wide.shape[0] == 10
    assert set(wide.columns) == {"A", "B", "C"}


def test_brzezcek_inputs_from_fixture_csv() -> None:
    y_star, v, corr, labels, meta = brzezcek_inputs_from_retail_insights_csv(FIXTURE)
    assert set(labels) == {"A", "B", "C"}
    assert y_star.shape == (3,)
    assert v.shape == (3, 3)
    assert corr.shape == (3, 3)
    assert meta["n_categories"] == 3
    assert meta["n_months"] == 10
    assert np.allclose(np.diag(corr), 1.0)
