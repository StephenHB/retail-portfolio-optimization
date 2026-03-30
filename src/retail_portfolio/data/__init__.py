"""Pluggable loaders: map panels to Brzęczek inputs (mu / V) without source-specific paths."""

from retail_portfolio.data.forecast_inputs import (
    correlation_from_covariance,
    mu_sigma_from_residual_matrix,
    wide_residual_matrix_from_long,
)
from retail_portfolio.data.kaggle_retail_insights import (
    brzezcek_inputs_from_retail_insights_csv,
    default_retail_insights_paths,
    load_retail_insights_csv,
    resolve_retail_insights_csv,
)

__all__ = [
    "brzezcek_inputs_from_retail_insights_csv",
    "correlation_from_covariance",
    "default_retail_insights_paths",
    "load_retail_insights_csv",
    "mu_sigma_from_residual_matrix",
    "resolve_retail_insights_csv",
    "wide_residual_matrix_from_long",
]
