"""
Tests for the Forecaster abstraction (NaiveForecaster, LGBMForecaster, registry).

LGBM tests are guarded by ``pytest.importorskip("lightgbm")``.
"""

import numpy as np
import pandas as pd
import pytest

from retail_portfolio.forecasters import (
    Forecaster,
    ForecastResult,
    NaiveForecaster,
    get_forecaster,
)
from retail_portfolio.data.kaggle_retail_insights import (
    brzezcek_inputs_from_monthly_revenue_wide,
    load_retail_insights_csv,
    monthly_category_revenue,
    wide_monthly_revenue,
)
from retail_portfolio.spark.solver import greedy_forward_selection


# ── Fixtures ──────────────────────────────────────────────────────────────

FIXTURE_CSV = "tests/fixtures/retail_insights_sample.csv"


@pytest.fixture
def wide_from_fixture() -> pd.DataFrame:
    """Wide monthly revenue panel from the test fixture CSV (10 months, 3 cats)."""
    raw = load_retail_insights_csv(FIXTURE_CSV)
    long = monthly_category_revenue(raw)
    return wide_monthly_revenue(long)


@pytest.fixture
def wide_synthetic_24m() -> pd.DataFrame:
    """Synthetic wide panel: 24 months, 3 products. Enough for LGBM."""
    rng = np.random.default_rng(42)
    periods = pd.date_range("2020-01-01", periods=24, freq="MS")
    data = {
        "Prod_A": 100 + np.cumsum(rng.normal(0, 5, 24)),
        "Prod_B": 200 + np.cumsum(rng.normal(0, 8, 24)),
        "Prod_C": 50 + np.cumsum(rng.normal(0, 3, 24)),
    }
    return pd.DataFrame(data, index=periods)


# ── NaiveForecaster ───────────────────────────────────────────────────────

class TestNaiveForecaster:
    def test_matches_existing_function(self, wide_from_fixture):
        """NaiveForecaster should produce the same y_star and V as the original."""
        y_old, v_old, _, _, _ = brzezcek_inputs_from_monthly_revenue_wide(
            wide_from_fixture
        )
        result = NaiveForecaster().fit(wide_from_fixture).predict()
        np.testing.assert_allclose(result.y_star, y_old)
        np.testing.assert_allclose(result.V, v_old)

    def test_y_star_is_last_row(self, wide_from_fixture):
        result = NaiveForecaster().fit(wide_from_fixture).predict()
        expected = wide_from_fixture.iloc[-1].to_numpy(dtype=float)
        np.testing.assert_array_equal(result.y_star, expected)

    def test_residuals_shape(self, wide_from_fixture):
        result = NaiveForecaster().fit(wide_from_fixture).predict()
        T, m = wide_from_fixture.shape
        assert result.residuals.shape == (T - 1, m)

    def test_labels_match_columns(self, wide_from_fixture):
        result = NaiveForecaster().fit(wide_from_fixture).predict()
        assert result.labels == [str(c) for c in wide_from_fixture.columns]

    def test_V_is_psd(self, wide_from_fixture):
        result = NaiveForecaster().fit(wide_from_fixture).predict()
        eigenvalues = np.linalg.eigvalsh(result.V)
        assert np.all(eigenvalues >= -1e-10)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            NaiveForecaster().predict()

    def test_meta_has_method(self, wide_from_fixture):
        result = NaiveForecaster().fit(wide_from_fixture).predict()
        assert result.meta["method"] == "naive"


# ── LGBMForecaster ────────────────────────────────────────────────────────

class TestLGBMForecaster:
    def test_basic_shape(self, wide_synthetic_24m):
        lgb = pytest.importorskip("lightgbm")
        from retail_portfolio.forecasters import LGBMForecaster

        f = LGBMForecaster(n_lags=3, rolling_windows=(3,), min_train_size=6)
        result = f.fit(wide_synthetic_24m).predict()
        m = wide_synthetic_24m.shape[1]
        assert result.y_star.shape == (m,)
        assert result.V.shape == (m, m)
        assert len(result.labels) == m

    def test_V_is_psd(self, wide_synthetic_24m):
        pytest.importorskip("lightgbm")
        from retail_portfolio.forecasters import LGBMForecaster

        result = LGBMForecaster().fit(wide_synthetic_24m).predict()
        eigenvalues = np.linalg.eigvalsh(result.V)
        assert np.all(eigenvalues >= -1e-10)

    def test_residuals_are_honest(self, wide_synthetic_24m):
        """CV residuals should have fewer rows than the full series."""
        pytest.importorskip("lightgbm")
        from retail_portfolio.forecasters import LGBMForecaster

        result = LGBMForecaster(min_train_size=6).fit(wide_synthetic_24m).predict()
        T = wide_synthetic_24m.shape[0]
        # Residual rows < T (some used for warmup + initial training)
        assert result.residuals.shape[0] < T

    def test_fallback_all_naive(self, wide_synthetic_24m):
        """With min_train_size > T, all products should fall back to naive."""
        pytest.importorskip("lightgbm")
        from retail_portfolio.forecasters import LGBMForecaster

        T = wide_synthetic_24m.shape[0]
        m = wide_synthetic_24m.shape[1]
        result = LGBMForecaster(min_train_size=T + 100).fit(wide_synthetic_24m).predict()
        assert len(result.meta["fallback_products"]) == m

    def test_predict_before_fit_raises(self):
        pytest.importorskip("lightgbm")
        from retail_portfolio.forecasters import LGBMForecaster

        with pytest.raises(RuntimeError, match="fit"):
            LGBMForecaster().predict()

    def test_meta_has_method(self, wide_synthetic_24m):
        pytest.importorskip("lightgbm")
        from retail_portfolio.forecasters import LGBMForecaster

        result = LGBMForecaster().fit(wide_synthetic_24m).predict()
        assert result.meta["method"] == "lgbm"

    def test_custom_params(self, wide_synthetic_24m):
        pytest.importorskip("lightgbm")
        from retail_portfolio.forecasters import LGBMForecaster

        custom = {"n_estimators": 50, "max_depth": 2}
        result = LGBMForecaster(lgbm_params=custom).fit(wide_synthetic_24m).predict()
        assert result.meta["lgbm_params"]["n_estimators"] == 50
        assert result.meta["lgbm_params"]["max_depth"] == 2


# ── Protocol compliance ───────────────────────────────────────────────────

class TestProtocol:
    def test_naive_satisfies_protocol(self):
        assert isinstance(NaiveForecaster(), Forecaster)

    def test_lgbm_satisfies_protocol(self):
        pytest.importorskip("lightgbm")
        from retail_portfolio.forecasters import LGBMForecaster

        assert isinstance(LGBMForecaster(), Forecaster)


# ── Registry ──────────────────────────────────────────────────────────────

class TestRegistry:
    def test_get_naive(self):
        f = get_forecaster("naive")
        assert isinstance(f, NaiveForecaster)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown forecaster"):
            get_forecaster("nonexistent")

    def test_get_lgbm(self):
        pytest.importorskip("lightgbm")
        from retail_portfolio.forecasters import LGBMForecaster

        f = get_forecaster("lgbm")
        assert isinstance(f, LGBMForecaster)

    def test_get_with_kwargs(self):
        f = get_forecaster("naive", ddof=0)
        assert f._ddof == 0


# ── Integration: forecaster → solver ──────────────────────────────────────

class TestForecasterToSolver:
    def test_naive_to_greedy(self, wide_from_fixture):
        result = NaiveForecaster().fit(wide_from_fixture).predict()
        steps = greedy_forward_selection(result.y_star, result.V)
        assert len(steps) > 0
        assert steps[-1]["sales"] > 0
        assert np.isfinite(steps[-1]["meta_objective"])

    def test_lgbm_to_greedy(self, wide_synthetic_24m):
        pytest.importorskip("lightgbm")
        from retail_portfolio.forecasters import LGBMForecaster

        result = LGBMForecaster().fit(wide_synthetic_24m).predict()
        steps = greedy_forward_selection(result.y_star, result.V)
        assert len(steps) > 0
        assert np.isfinite(steps[-1]["meta_objective"])
