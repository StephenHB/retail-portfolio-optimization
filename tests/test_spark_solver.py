"""
Tests for the solver module (pure numpy, no Spark dependency).

These validate the batch evaluator, greedy selection, and random search
against the original single-mask functions in brzezcek_portfolio.
"""

import numpy as np
import pytest

from retail_portfolio.brzezcek_portfolio import (
    meta_objective_profit_safety_stock,
    portfolio_forecast_total,
    portfolio_residual_std,
    relative_forecast_error_risk,
)
from retail_portfolio.spark.solver import (
    evaluate_portfolio_batch,
    greedy_forward_selection,
    random_search,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def small_inputs():
    """3-product inputs matching the original test fixtures."""
    y_star = np.array([10.0, 20.0, 5.0])
    v = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 9.0, 0.5],
        [0.0, 0.5, 1.0],
    ])
    return y_star, v


@pytest.fixture
def six_product_inputs():
    """6-product inputs for testing at moderate scale."""
    rng = np.random.default_rng(123)
    m = 6
    y_star = rng.uniform(5, 50, size=m)
    # Build a valid PSD covariance
    a = rng.normal(size=(m, m))
    v = a.T @ a / m + np.eye(m) * 0.1
    return y_star, v


# ── Batch evaluation tests ────────────────────────────────────────────────

class TestEvaluatePortfolioBatch:
    def test_single_mask_matches_original(self, small_inputs):
        y_star, v = small_inputs
        mask = np.array([[1, 1, 0]])
        w, t_crit = 1.0, 1.96
        results = evaluate_portfolio_batch(mask, y_star, v, w, t_crit)
        assert len(results) == 1
        r = results[0]

        x = np.array([1.0, 1.0, 0.0])
        assert r["sales"] == pytest.approx(portfolio_forecast_total(x, y_star))
        assert r["std"] == pytest.approx(portfolio_residual_std(x, v))
        assert r["relative_risk"] == pytest.approx(relative_forecast_error_risk(x, y_star, v))
        assert r["meta_objective"] == pytest.approx(
            meta_objective_profit_safety_stock(x, y_star, v, w=w, t_critical=t_crit)
        )

    def test_multiple_masks(self, small_inputs):
        y_star, v = small_inputs
        masks = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ])
        results = evaluate_portfolio_batch(masks, y_star, v)
        assert len(results) == 4
        # All-in portfolio should have highest sales
        sales = [r["sales"] for r in results]
        assert sales[3] == max(sales)

    def test_empty_mask_zero_sales(self, small_inputs):
        y_star, v = small_inputs
        masks = np.array([[0, 0, 0]])
        results = evaluate_portfolio_batch(masks, y_star, v)
        assert results[0]["sales"] == 0.0
        assert results[0]["relative_risk"] == float("inf")

    def test_vectorized_matches_loop(self, six_product_inputs):
        """Batch evaluation should match per-mask evaluation."""
        y_star, v = six_product_inputs
        m = len(y_star)
        rng = np.random.default_rng(99)
        masks = (rng.random((20, m)) > 0.5).astype(float)
        w, t_crit = 0.5, 2.0

        batch_results = evaluate_portfolio_batch(masks, y_star, v, w, t_crit)
        for i, r in enumerate(batch_results):
            x = masks[i]
            assert r["sales"] == pytest.approx(portfolio_forecast_total(x, y_star), abs=1e-10)
            assert r["std"] == pytest.approx(portfolio_residual_std(x, v), abs=1e-10)
            expected_meta = meta_objective_profit_safety_stock(
                x, y_star, v, w=w, t_critical=t_crit
            )
            assert r["meta_objective"] == pytest.approx(expected_meta, abs=1e-10)


# ── Greedy forward selection tests ────────────────────────────────────────

class TestGreedyForwardSelection:
    def test_greedy_adds_all_products(self, small_inputs):
        y_star, v = small_inputs
        steps = greedy_forward_selection(y_star, v, w=1.0, t_critical=1.96)
        # Should add all 3 products
        assert len(steps) == 3
        added = {s["product_added"] for s in steps}
        assert added == {0, 1, 2}

    def test_greedy_monotone_products(self, small_inputs):
        y_star, v = small_inputs
        steps = greedy_forward_selection(y_star, v)
        for i, s in enumerate(steps):
            assert s["n_included"] == i + 1

    def test_greedy_max_products(self, six_product_inputs):
        y_star, v = six_product_inputs
        steps = greedy_forward_selection(y_star, v, max_products=3)
        assert len(steps) == 3
        assert steps[-1]["n_included"] == 3

    def test_greedy_relative_risk_objective(self, small_inputs):
        y_star, v = small_inputs
        steps = greedy_forward_selection(y_star, v, objective="relative_risk")
        assert len(steps) > 0
        # Final relative risk should be finite
        assert np.isfinite(steps[-1]["relative_risk"])

    def test_greedy_first_pick_is_best_single(self, small_inputs):
        """First greedy pick should be the single-product with best meta-obj."""
        y_star, v = small_inputs
        w, t_crit = 1.0, 1.96
        steps = greedy_forward_selection(y_star, v, w=w, t_critical=t_crit)

        single_metas = []
        for j in range(len(y_star)):
            x = np.zeros(len(y_star))
            x[j] = 1.0
            single_metas.append(
                meta_objective_profit_safety_stock(x, y_star, v, w=w, t_critical=t_crit)
            )
        best_single = int(np.argmax(single_metas))
        assert steps[0]["product_added"] == best_single


# ── Random search tests ───────────────────────────────────────────────────

class TestRandomSearch:
    def test_random_returns_valid_result(self, small_inputs):
        y_star, v = small_inputs
        result = random_search(y_star, v, n_samples=100, seed=42)
        assert "best" in result
        assert "n_evaluated" in result
        assert result["n_evaluated"] == 100
        best = result["best"]
        assert best["n_included"] >= 1

    def test_random_seed_reproducible(self, small_inputs):
        y_star, v = small_inputs
        r1 = random_search(y_star, v, n_samples=500, seed=7)
        r2 = random_search(y_star, v, n_samples=500, seed=7)
        assert r1["best"]["mask"] == r2["best"]["mask"]
        assert r1["best"]["meta_objective"] == r2["best"]["meta_objective"]

    def test_random_respects_min_max_products(self, six_product_inputs):
        y_star, v = six_product_inputs
        result = random_search(
            y_star, v, n_samples=200, min_products=2, max_products=4, seed=1
        )
        n = result["best"]["n_included"]
        assert 2 <= n <= 4

    def test_random_beats_trivial(self, small_inputs):
        """Random search with enough samples should beat a single-product portfolio."""
        y_star, v = small_inputs
        w, t_crit = 1.0, 1.96
        result = random_search(y_star, v, n_samples=1000, w=w, t_critical=t_crit, seed=42)
        # Worst single-product meta-objective
        worst_single = min(
            meta_objective_profit_safety_stock(
                np.eye(1, len(y_star), j).ravel(), y_star, v, w=w, t_critical=t_crit
            )
            for j in range(len(y_star))
        )
        assert result["best"]["meta_objective"] >= worst_single
