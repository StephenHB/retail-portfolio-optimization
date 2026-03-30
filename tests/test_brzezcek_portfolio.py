"""Tests for Brzęczek (2020) portfolio quadratic risk and marginal moves."""

import numpy as np
import pytest

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
    t_critical_student,
)


def test_portfolio_residual_variance_quadratic_form_symmetric() -> None:
    rng = np.random.default_rng(0)
    n = 5
    a = rng.standard_normal((n, n))
    v = (a + a.T) / 2.0
    x = rng.standard_normal(n)
    manual = float(x @ v @ x)
    assert np.isclose(portfolio_residual_variance(x, v), manual)


def test_portfolio_residual_variance_matches_sum_variance_uncorrelated() -> None:
    """Diagonal V: Var(sum x_i eps_i) = sum x_i^2 sigma_i^2 for independent residuals."""
    sig = np.array([2.0, 3.0, 1.0])
    v = np.diag(sig**2)
    x = np.array([1.0, 1.0, 0.0])
    expected = float(np.sum((x**2) * (sig**2)))
    assert np.isclose(portfolio_residual_variance(x, v), expected)
    assert np.isclose(portfolio_residual_std(x, v), np.sqrt(expected))


def test_covariance_from_std_correlation_matches_outer() -> None:
    std = np.array([2.0, 3.0])
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    v = covariance_from_std_correlation(std, corr)
    assert np.allclose(v, np.outer(std, std) * corr)
    assert np.isclose(v[0, 0], 4.0)
    assert np.isclose(v[1, 1], 9.0)


def test_marginal_add_increases_nominal_risk_uncorrelated() -> None:
    """With nonnegative correlations (here zero), adding a category strictly increases sqrt(x'Vx)."""
    v = np.diag([4.0, 9.0, 1.0])
    y_star = np.array([10.0, 20.0, 5.0])
    x0 = np.array([1, 0, 1], dtype=int)
    base = portfolio_residual_std(x0.astype(float), v)
    for j in range(3):
        if x0[j] == 0:
            y = flip_category(x0, j, 1).astype(float)
            assert portfolio_residual_std(y, v) > base


def test_relative_risk_infinite_when_zero_sales_forecast() -> None:
    y_star = np.array([1.0, 0.0])
    x = np.array([0.0, 1.0])
    v = np.eye(2)
    assert relative_forecast_error_risk(x, y_star, v) == float("inf")


def test_best_marginal_add_returns_highest_scoring_inactive() -> None:
    y_star = np.array([10.0, 20.0, 5.0])
    v = np.eye(3) * 0.01
    x0 = np.array([1, 0, 1], dtype=int)
    scores = score_marginal_adds(x0, y_star, v)
    j, s = best_marginal_add(x0, y_star, v)
    assert j == 1
    assert s == scores[j]


def test_meta_objective_profit_safety_stock_linear_in_w() -> None:
    x = np.array([1.0, 1.0])
    y_star = np.array([10.0, 10.0])
    v = np.eye(2) * 4.0
    t = 2.0
    m0 = meta_objective_profit_safety_stock(x, y_star, v, w=0.0, t_critical=t)
    m1 = meta_objective_profit_safety_stock(x, y_star, v, w=1.0, t_critical=t)
    std = portfolio_residual_std(x, v)
    assert np.isclose(m0 - m1, t * std)


def test_t_critical_two_sided_exceeds_one_sided_for_same_alpha() -> None:
    df = 20.0
    alpha = 0.05
    one = t_critical_student(df, alpha, one_sided_upper=True)
    two = t_critical_student(df, alpha, one_sided_upper=False)
    assert two > one


def test_covariance_from_std_correlation_shape_error() -> None:
    with pytest.raises(ValueError, match="square"):
        covariance_from_std_correlation(np.ones(3), np.eye(2))
