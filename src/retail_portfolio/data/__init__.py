"""Pluggable loaders: map panels to Brzęczek inputs (mu / V) without source-specific paths."""

from retail_portfolio.data.forecast_inputs import (
    mu_sigma_from_residual_matrix,
    wide_residual_matrix_from_long,
)

__all__ = [
    "mu_sigma_from_residual_matrix",
    "wide_residual_matrix_from_long",
]
