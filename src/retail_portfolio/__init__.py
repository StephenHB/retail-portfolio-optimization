"""Retail portfolio optimization helpers (Brzęczek-style forecast risk, loaders)."""

from retail_portfolio.brzezcek_portfolio import (
    best_marginal_add,
    covariance_from_std_correlation,
    flip_category,
    meta_objective_profit_safety_stock,
    portfolio_forecast_total,
    portfolio_residual_std,
    portfolio_residual_variance,
    relative_forecast_error_risk,
    score_marginal_adds,
)
from retail_portfolio.forecasters import (
    Forecaster,
    ForecastResult,
    NaiveForecaster,
    get_forecaster,
)

__all__ = [
    "best_marginal_add",
    "covariance_from_std_correlation",
    "flip_category",
    "meta_objective_profit_safety_stock",
    "portfolio_forecast_total",
    "portfolio_residual_std",
    "portfolio_residual_variance",
    "relative_forecast_error_risk",
    "score_marginal_adds",
    "Forecaster",
    "ForecastResult",
    "NaiveForecaster",
    "get_forecaster",
]
