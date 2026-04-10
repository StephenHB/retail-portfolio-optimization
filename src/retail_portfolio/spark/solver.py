"""
Portfolio evaluation and search strategies that run per-worker on numpy arrays.

These functions are designed to be called inside PySpark UDFs / map operations.
They accept plain numpy arrays (broadcast-friendly) and return dicts or arrays,
avoiding any Spark dependency so they remain unit-testable without a cluster.

Scaling approach
----------------
The search space for n portfolios × m products is exponential (2^m per portfolio).
Rather than exhaustively enumerating all 2^m masks, this module provides three
strategies that partition work across Spark executors:

1. **Batch evaluation** — evaluate a chunk of pre-generated binary masks in
   vectorized numpy (one call per partition).
2. **Greedy forward selection** — O(m²) per portfolio; trivially parallelised
   across the n portfolios.
3. **Random search** — sample k random masks per portfolio, evaluate, keep the
   best; embarrassingly parallel.

All three accept the same (y_star, V) inputs and return standardised result dicts.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Core vectorized evaluation (no loops over portfolios)
# ---------------------------------------------------------------------------

def _portfolio_metrics(
    x: np.ndarray,
    y_star: np.ndarray,
    v: np.ndarray,
    w: float,
    t_critical: float,
) -> dict[str, float]:
    """Compute all Brzęczek metrics for a single binary mask x."""
    sales = float(x @ y_star)
    variance = float(x @ v @ x)
    std = float(np.sqrt(max(variance, 0.0)))
    rel_risk = std / sales if sales > 1e-12 else float("inf")
    meta_obj = sales - w * t_critical * std
    return {
        "sales": sales,
        "variance": variance,
        "std": std,
        "relative_risk": rel_risk,
        "meta_objective": meta_obj,
        "n_included": int(x.sum()),
    }


def evaluate_portfolio_batch(
    masks: np.ndarray,
    y_star: np.ndarray,
    v: np.ndarray,
    w: float = 1.0,
    t_critical: float = 1.96,
) -> list[dict[str, Any]]:
    """
    Evaluate a batch of binary masks against shared (y_star, V).

    Parameters
    ----------
    masks : shape (k, m)
        Each row is a binary inclusion vector.
    y_star : shape (m,)
        Forecast sales per product.
    v : shape (m, m)
        Residual covariance matrix.
    w : float
        Safety-stock cost weight.
    t_critical : float
        Student-t critical value.

    Returns
    -------
    List of dicts, one per mask row, with keys:
        mask, sales, variance, std, relative_risk, meta_objective, n_included.
    """
    masks = np.asarray(masks, dtype=float)
    y_star = np.asarray(y_star, dtype=float).ravel()
    v = np.asarray(v, dtype=float)

    # Vectorized: sales = masks @ y_star  (k,)
    sales = masks @ y_star

    # Vectorized: variance_i = x_i' V x_i = sum of (masks @ V) * masks row-wise
    mv = masks @ v  # (k, m)
    variances = np.einsum("ij,ij->i", mv, masks)  # (k,)

    stds = np.sqrt(np.maximum(variances, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_risks = np.where(sales > 1e-12, stds / sales, np.inf)
    meta_objs = sales - w * t_critical * stds
    n_included = masks.sum(axis=1).astype(int)

    results = []
    for i in range(masks.shape[0]):
        results.append({
            "mask": masks[i].astype(int).tolist(),
            "sales": float(sales[i]),
            "variance": float(variances[i]),
            "std": float(stds[i]),
            "relative_risk": float(rel_risks[i]),
            "meta_objective": float(meta_objs[i]),
            "n_included": int(n_included[i]),
        })
    return results


# ---------------------------------------------------------------------------
# Greedy forward selection (per-portfolio, O(m²))
# ---------------------------------------------------------------------------

def greedy_forward_selection(
    y_star: np.ndarray,
    v: np.ndarray,
    *,
    w: float = 1.0,
    t_critical: float = 1.96,
    max_products: int | None = None,
    objective: str = "meta",
) -> list[dict[str, Any]]:
    """
    Build a portfolio one product at a time, greedily adding the product that
    most improves the chosen objective.

    Parameters
    ----------
    objective : {"meta", "relative_risk"}
        "meta" maximises the meta-objective (sales - w·t·std).
        "relative_risk" minimises relative forecast-error risk.
    max_products : int or None
        Stop after this many products are added (None = try all m).

    Returns
    -------
    List of step dicts recording each addition (product_index, metrics after add).
    """
    y_star = np.asarray(y_star, dtype=float).ravel()
    v = np.asarray(v, dtype=float)
    m = y_star.shape[0]
    if max_products is None:
        max_products = m

    x = np.zeros(m, dtype=float)
    steps: list[dict[str, Any]] = []

    for _ in range(min(max_products, m)):
        best_idx = -1
        best_score = -np.inf if objective == "meta" else np.inf
        best_metrics: dict[str, Any] = {}

        inactive = np.where(x == 0)[0]
        if len(inactive) == 0:
            break

        for j in inactive:
            candidate = x.copy()
            candidate[j] = 1.0
            met = _portfolio_metrics(candidate, y_star, v, w, t_critical)

            if objective == "meta":
                score = met["meta_objective"]
                if score > best_score:
                    best_score = score
                    best_idx = int(j)
                    best_metrics = met
            else:  # relative_risk
                score = met["relative_risk"]
                if score < best_score:
                    best_score = score
                    best_idx = int(j)
                    best_metrics = met

        if best_idx < 0:
            break

        x[best_idx] = 1.0
        steps.append({
            "step": len(steps) + 1,
            "product_added": best_idx,
            "mask": x.astype(int).tolist(),
            **best_metrics,
        })

    return steps


# ---------------------------------------------------------------------------
# Random search (per-portfolio, embarrassingly parallel)
# ---------------------------------------------------------------------------

def random_search(
    y_star: np.ndarray,
    v: np.ndarray,
    *,
    n_samples: int = 10_000,
    w: float = 1.0,
    t_critical: float = 1.96,
    min_products: int = 1,
    max_products: int | None = None,
    seed: int | None = None,
    objective: str = "meta",
) -> dict[str, Any]:
    """
    Sample random binary masks and return the best under the chosen objective.

    Parameters
    ----------
    n_samples : int
        How many random masks to draw.
    min_products / max_products : int
        Constrain how many products each random mask includes.
    seed : int or None
        For reproducibility.
    objective : {"meta", "relative_risk"}
        What to optimize.

    Returns
    -------
    Dict with 'best' (best result dict) and 'n_evaluated'.
    """
    y_star = np.asarray(y_star, dtype=float).ravel()
    v = np.asarray(v, dtype=float)
    m = y_star.shape[0]
    if max_products is None:
        max_products = m

    rng = np.random.default_rng(seed)

    # Generate random masks with controlled sparsity
    masks = np.zeros((n_samples, m), dtype=float)
    for i in range(n_samples):
        k = rng.integers(min_products, max_products + 1)
        chosen = rng.choice(m, size=k, replace=False)
        masks[i, chosen] = 1.0

    results = evaluate_portfolio_batch(masks, y_star, v, w, t_critical)

    if objective == "meta":
        best = max(results, key=lambda r: r["meta_objective"])
    else:
        best = min(results, key=lambda r: r["relative_risk"])

    return {"best": best, "n_evaluated": n_samples}
