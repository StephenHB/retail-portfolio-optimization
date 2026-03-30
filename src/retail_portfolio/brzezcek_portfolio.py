"""
Brzęczek (2020) portfolio constructions on category forecasts and residual sales.

Notation aligned with the paper (Sect. 3, Table 1):
- x: binary inclusion per category; total forecasted sales in the portfolio is x' y*_t.
- V: variance–covariance matrix of *residual* (forecast error) sales per category,
  with v_ij = rho_ij * s_i * s_j.
- Nominal forecast-error risk (models 1–2): sqrt(x' V x), an estimate of (or lower
  bound on) expected error of *portfolio* sales; under Gaussian residuals, x' V x is
  Var(sum_i x_i epsilon_i), i.e. the variance of the sum of included categories'
  residual sales.
- Relative risk (models 3–4): sqrt(x' V x) / (x' y*_t).
- Meta-objective (models 5–6): x' y*_t - w * t_{alpha, N-K} * sqrt(x' V x).

See research/papers/Brzezcek_2020_summary_and_algorithms.md for citations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def covariance_from_std_correlation(
    std: np.ndarray,
    corr: np.ndarray,
) -> np.ndarray:
    """
    Build V with v_ij = rho_ij * s_i * s_j (paper Eq. parameter block for V).

    Parameters
    ----------
    std :
        Per-category standard deviations of residual sales, shape (n,).
    corr :
        Symmetric correlation matrix of residual sales, shape (n, n), diag 1.
    """
    std = np.asarray(std, dtype=float).reshape(-1)
    corr = np.asarray(corr, dtype=float)
    if corr.shape != (std.shape[0], std.shape[0]):
        raise ValueError("corr must be square with len(std) rows/columns")
    outer = np.outer(std, std)
    return corr * outer


def portfolio_forecast_total(x: np.ndarray, y_star: np.ndarray) -> float:
    """Total forecasted sales x' y* (paper y*_t vector for the active decision set)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y_star = np.asarray(y_star, dtype=float).reshape(-1)
    if x.shape != y_star.shape:
        raise ValueError("x and y_star must have the same shape")
    return float(x @ y_star)


def portfolio_residual_variance(x: np.ndarray, v: np.ndarray) -> float:
    """Quadratic form x' V x for residual covariance V."""
    x = np.asarray(x, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float)
    if v.shape != (x.shape[0], x.shape[0]):
        raise ValueError("v must be (n, n) matching len(x)")
    return float(x @ v @ x)


def portfolio_residual_std(x: np.ndarray, v: np.ndarray) -> float:
    """sqrt(x' V x), nonnegative; paper nominal risk measure (models 1–2)."""
    var = portfolio_residual_variance(x, v)
    return float(np.sqrt(max(var, 0.0)))


def relative_forecast_error_risk(
    x: np.ndarray,
    y_star: np.ndarray,
    v: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """
    sqrt(x' V x) / (x' y*), paper models (3)–(4); inf if denominator <= eps.
    """
    denom = portfolio_forecast_total(x, y_star)
    if denom <= eps:
        return float("inf")
    return portfolio_residual_std(x, v) / denom


def t_critical_student(df: float, alpha: float, *, one_sided_upper: bool = True) -> float:
    """
    t critical value as in Brzęczek (2020): right-tail at level alpha by default;
    set one_sided_upper=False for two-sided symmetric alpha (each tail alpha/2).
    """
    if df <= 0 or not np.isfinite(df):
        raise ValueError("df must be positive finite")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if one_sided_upper:
        return float(stats.t.ppf(1.0 - alpha, df))
    return float(stats.t.ppf(1.0 - alpha / 2.0, df))


def meta_objective_profit_safety_stock(
    x: np.ndarray,
    y_star: np.ndarray,
    v: np.ndarray,
    *,
    w: float,
    t_critical: float,
) -> float:
    """
    x' y* - w * t * sqrt(x' V x), paper (5)–(6); pass t from t_critical_student(..., df=N-K).
    """
    return float(
        portfolio_forecast_total(x, y_star) - w * t_critical * portfolio_residual_std(x, v)
    )


def flip_category(x: np.ndarray, index: int, value: int) -> np.ndarray:
    y = np.asarray(x, dtype=int).copy()
    y[index] = value
    return y


def score_marginal_adds(
    x: np.ndarray,
    y_star: np.ndarray,
    v: np.ndarray,
    *,
    sales_weight: float = 1.0,
    risk_weight: float = 1.0,
    risk_use_std: bool = True,
) -> np.ndarray:
    """
    For each j with x_j == 0, score adding j: sales_weight * Δsales - risk_weight * Δrisk.
    Inactive slots get -inf.
    """
    x = np.asarray(x, dtype=int).reshape(-1)
    n = x.shape[0]
    base_sales = portfolio_forecast_total(x.astype(float), y_star)

    def risk_fn(xx: np.ndarray) -> float:
        if risk_use_std:
            return portfolio_residual_std(xx, v)
        return portfolio_residual_variance(xx, v)

    base_r = risk_fn(x.astype(float))
    scores = np.full(n, -np.inf, dtype=float)
    for j in range(n):
        if x[j] != 0:
            continue
        y = flip_category(x, j, 1).astype(float)
        ds = portfolio_forecast_total(y, y_star) - base_sales
        dr = risk_fn(y) - base_r
        scores[j] = sales_weight * ds - risk_weight * dr
    return scores


def best_marginal_add(
    x: np.ndarray,
    y_star: np.ndarray,
    v: np.ndarray,
    **kwargs: Any,
) -> tuple[int | None, float]:
    scores = score_marginal_adds(x, y_star, v, **kwargs)
    j = int(np.argmax(scores))
    if not np.isfinite(scores[j]) or scores[j] == -np.inf:
        return None, float("-inf")
    return j, float(scores[j])
