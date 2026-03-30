"""Tests for pluggable residual panel -> (mu, Sigma)."""

import numpy as np
import pandas as pd
import pytest

from retail_portfolio.data.forecast_inputs import (
    mu_sigma_from_residual_matrix,
    wide_residual_matrix_from_long,
)


def test_wide_residual_matrix_from_long_orders_categories() -> None:
    df = pd.DataFrame(
        {
            "period": [1, 1, 2, 2],
            "cat": ["a", "b", "a", "b"],
            "resid": [1.0, 2.0, -1.0, 0.0],
        }
    )
    mat, cats = wide_residual_matrix_from_long(
        df, period_col="period", category_col="cat", residual_col="resid"
    )
    assert cats == sorted(["a", "b"])  # pivot column order is sorted by default
    assert mat.shape == (2, 2)


def test_mu_sigma_from_residual_matrix_diagonal_independent() -> None:
    rng = np.random.default_rng(1)
    n_obs, n_cat = 500, 3
    r = rng.standard_normal((n_obs, n_cat))
    mu = np.array([1.0, 2.0, 3.0])
    y, sigma = mu_sigma_from_residual_matrix(mu, r, ddof=1)
    assert np.allclose(y, mu)
    sample = np.cov(r, rowvar=False, ddof=1)
    assert np.allclose(sigma, sample)


def test_mu_sigma_from_residual_matrix_shape_errors() -> None:
    with pytest.raises(ValueError, match="2-D"):
        mu_sigma_from_residual_matrix(np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="forecast_means"):
        mu_sigma_from_residual_matrix(np.ones(3), np.ones((10, 2)))
