"""
Build category-level forecast means and residual covariance V from generic tables.

Callers supply column names; no dataset-specific paths or Kaggle column assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wide_residual_matrix_from_long(
    df: pd.DataFrame,
    *,
    period_col: str,
    category_col: str,
    residual_col: str,
) -> tuple[np.ndarray, list]:
    """
    Pivot long panel to a matrix (n_periods, n_categories) with aligned periods.

    Rows with missing residual in any category are dropped after pivot (complete periods).
    Returns (matrix, category_order) consistent with column order.
    """
    sub = df[[period_col, category_col, residual_col]].copy()
    sub = sub.dropna(subset=[residual_col])
    wide = sub.pivot(index=period_col, columns=category_col, values=residual_col)
    wide = wide.dropna(how="any")
    categories = list(wide.columns)
    return wide.to_numpy(dtype=float), categories


def mu_sigma_from_residual_matrix(
    forecast_means: np.ndarray,
    residuals: np.ndarray,
    *,
    ddof: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    forecast_means: shape (n_cat,) — point forecasts y*_i for the decision horizon.
    residuals: shape (n_obs, n_cat) — columns are categories, rows are time periods.

    Sigma is the sample covariance of residual columns (matches V from empirical
    residuals when correlations/stds are estimated jointly).
    """
    y = np.asarray(forecast_means, dtype=float).reshape(-1)
    r = np.asarray(residuals, dtype=float)
    if r.ndim != 2:
        raise ValueError("residuals must be 2-D (n_obs, n_categories)")
    if r.shape[1] != y.shape[0]:
        raise ValueError("residuals.shape[1] must equal len(forecast_means)")
    if r.shape[0] <= ddof:
        raise ValueError("not enough rows to estimate covariance with given ddof")
    sigma = np.cov(r, rowvar=False, ddof=ddof)
    return y, sigma
