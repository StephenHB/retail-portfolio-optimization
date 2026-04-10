"""
Pluggable forecaster objects for producing (y_star, V) from revenue panels.

Two-phase interface (fit / predict) following sklearn convention:

    >>> forecaster = NaiveForecaster()
    >>> result = forecaster.fit(wide_revenue).predict()
    >>> result.y_star   # shape (m,)
    >>> result.V        # shape (m, m)

Built-in implementations:

- **NaiveForecaster** — last observed value (random walk). Always available.
- **LGBMForecaster** — per-product LightGBM with expanding-window CV residuals.
  Requires ``lightgbm`` (install with ``pip install retail-portfolio-optimization[ml]``).

Registry shortcut:

    >>> from retail_portfolio.forecasters import get_forecaster
    >>> f = get_forecaster("naive")         # always works
    >>> f = get_forecaster("lgbm", n_lags=6)  # needs lightgbm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from retail_portfolio.data.forecast_inputs import mu_sigma_from_residual_matrix


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForecastResult:
    """Output of any Forecaster."""

    y_star: np.ndarray  # (m,) point forecasts
    V: np.ndarray  # (m, m) residual covariance
    labels: list[str]  # product / category names
    residuals: np.ndarray  # (n_obs, m) residuals used to estimate V
    meta: dict[str, Any]  # method name, diagnostics, fallback info


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Forecaster(Protocol):
    """
    Structural interface for forecast models.

    ``wide_revenue`` passed to ``fit`` is a pandas DataFrame where:
    - Rows are time periods (sorted chronologically).
    - Columns are products / categories.
    - Values are revenue (float).
    """

    def fit(self, wide_revenue: pd.DataFrame) -> Forecaster: ...
    def predict(self) -> ForecastResult: ...


# ---------------------------------------------------------------------------
# NaiveForecaster
# ---------------------------------------------------------------------------

class NaiveForecaster:
    """Random-walk forecast: y* = last observed value, residuals = first differences."""

    def __init__(self, *, ddof: int = 1) -> None:
        self._ddof = ddof
        self._result: ForecastResult | None = None

    def fit(self, wide_revenue: pd.DataFrame) -> NaiveForecaster:
        labels = [str(c) for c in wide_revenue.columns]
        y_star = wide_revenue.iloc[-1].to_numpy(dtype=float)
        resid = wide_revenue.diff().iloc[1:].to_numpy(dtype=float)
        _, V = mu_sigma_from_residual_matrix(y_star, resid, ddof=self._ddof)
        self._result = ForecastResult(
            y_star=y_star,
            V=V,
            labels=labels,
            residuals=resid,
            meta={
                "method": "naive",
                "n_periods": int(wide_revenue.shape[0]),
                "residual_rows": int(resid.shape[0]),
            },
        )
        return self

    def predict(self) -> ForecastResult:
        if self._result is None:
            raise RuntimeError("Must call fit() before predict()")
        return self._result


# ---------------------------------------------------------------------------
# LGBMForecaster
# ---------------------------------------------------------------------------

_DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "min_child_samples": 3,
    "verbose": -1,
    "n_jobs": 1,
}


class LGBMForecaster:
    """
    Per-product LightGBM forecast with expanding-window CV residuals.

    For each product, trains a LightGBM regressor on lag + rolling features
    derived from that product's revenue history.  Residuals are collected via
    expanding-window cross-validation (train on [0..t-1], predict t) so that
    V is estimated from honest out-of-sample errors.

    Products with fewer than ``min_train_size + 1`` valid feature rows
    fall back to the naive forecast for that product.
    """

    def __init__(
        self,
        *,
        n_lags: int = 3,
        rolling_windows: tuple[int, ...] = (3,),
        lgbm_params: dict[str, Any] | None = None,
        min_train_size: int = 6,
        ddof: int = 1,
    ) -> None:
        try:
            import lightgbm as lgb  # noqa: F811
        except ImportError:
            raise ImportError(
                "LGBMForecaster requires lightgbm. "
                "Install with: pip install retail-portfolio-optimization[ml]"
            ) from None
        self._lgb = lgb
        self._n_lags = n_lags
        self._rolling_windows = rolling_windows
        self._lgbm_params = {**_DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
        self._min_train_size = min_train_size
        self._ddof = ddof
        self._result: ForecastResult | None = None

    # -- Feature engineering ------------------------------------------------

    @staticmethod
    def _build_feature_frame(
        series: np.ndarray,
        n_lags: int,
        rolling_windows: tuple[int, ...],
    ) -> pd.DataFrame:
        """
        Build feature DataFrame for a single product's revenue series.

        Returns a DataFrame with columns: target, lag_1..lag_n,
        roll_mean_w, roll_std_w for each window w.
        Rows with any NaN are dropped.
        """
        s = pd.Series(series, dtype=float)
        frame: dict[str, pd.Series] = {"target": s}
        for k in range(1, n_lags + 1):
            frame[f"lag_{k}"] = s.shift(k)
        for w in rolling_windows:
            rolled = s.shift(1).rolling(window=w)
            frame[f"roll_mean_{w}"] = rolled.mean()
            frame[f"roll_std_{w}"] = rolled.std(ddof=1)
        df = pd.DataFrame(frame)
        df = df.dropna()
        return df

    @staticmethod
    def _feature_cols(n_lags: int, rolling_windows: tuple[int, ...]) -> list[str]:
        cols = [f"lag_{k}" for k in range(1, n_lags + 1)]
        for w in rolling_windows:
            cols.append(f"roll_mean_{w}")
            cols.append(f"roll_std_{w}")
        return cols

    def _build_next_features(self, series: np.ndarray) -> np.ndarray:
        """Build the feature vector for predicting the next (unobserved) period."""
        feats: list[float] = []
        for k in range(1, self._n_lags + 1):
            feats.append(float(series[-k]))
        for w in self._rolling_windows:
            window = series[-w:]
            feats.append(float(np.mean(window)))
            feats.append(float(np.std(window, ddof=1)) if len(window) > 1 else 0.0)
        return np.array(feats, dtype=float)

    # -- Fit ----------------------------------------------------------------

    def fit(self, wide_revenue: pd.DataFrame) -> LGBMForecaster:
        labels = [str(c) for c in wide_revenue.columns]
        m = len(labels)
        arr = wide_revenue.to_numpy(dtype=float)  # (T, m)

        feat_cols = self._feature_cols(self._n_lags, self._rolling_windows)
        y_star = np.empty(m, dtype=float)
        fallback_products: list[int] = []

        # First pass: determine the CV residual length
        # All products share the same T, so the feature frame lengths are identical.
        probe = self._build_feature_frame(arr[:, 0], self._n_lags, self._rolling_windows)
        n_valid = len(probe)
        n_cv = max(0, n_valid - self._min_train_size)

        # Collect residuals aligned across products
        all_resid_cols: list[np.ndarray] = []

        for j in range(m):
            series_j = arr[:, j]
            fdf = self._build_feature_frame(series_j, self._n_lags, self._rolling_windows)
            X = fdf[feat_cols].to_numpy(dtype=float)
            y = fdf["target"].to_numpy(dtype=float)
            n_valid_j = len(X)

            if n_valid_j < self._min_train_size + 1:
                # Fallback: naive
                y_star[j] = series_j[-1]
                fallback_products.append(j)
                # Naive residuals at the CV indices
                naive_diffs = np.diff(series_j)
                # Align to the same tail indices as LGBM CV
                if n_cv > 0 and len(naive_diffs) >= n_cv:
                    all_resid_cols.append(naive_diffs[-n_cv:])
                elif len(naive_diffs) > 0:
                    all_resid_cols.append(naive_diffs)
                else:
                    all_resid_cols.append(np.zeros(max(n_cv, 1)))
                continue

            # Expanding-window CV
            cv_residuals: list[float] = []
            for t in range(self._min_train_size, n_valid_j):
                model = self._lgb.LGBMRegressor(**self._lgbm_params)
                model.fit(X[:t], y[:t])
                pred = model.predict(X[t : t + 1])[0]
                cv_residuals.append(y[t] - pred)

            all_resid_cols.append(np.array(cv_residuals, dtype=float))

            # Final model for y_star prediction
            final_model = self._lgb.LGBMRegressor(**self._lgbm_params)
            final_model.fit(X, y)
            next_feat = self._build_next_features(series_j).reshape(1, -1)
            y_star[j] = float(final_model.predict(next_feat)[0])

        # Align residual lengths (trim to shortest)
        min_len = min(len(col) for col in all_resid_cols)
        if min_len <= self._ddof:
            raise ValueError(
                f"Only {min_len} CV residual rows available, need > {self._ddof} "
                f"for covariance estimation. Increase data length or reduce min_train_size."
            )
        resid_matrix = np.column_stack([col[-min_len:] for col in all_resid_cols])

        _, V = mu_sigma_from_residual_matrix(y_star, resid_matrix, ddof=self._ddof)

        self._result = ForecastResult(
            y_star=y_star,
            V=V,
            labels=labels,
            residuals=resid_matrix,
            meta={
                "method": "lgbm",
                "n_periods": int(arr.shape[0]),
                "residual_rows": int(resid_matrix.shape[0]),
                "fallback_products": fallback_products,
                "lgbm_params": self._lgbm_params,
            },
        )
        return self

    def predict(self) -> ForecastResult:
        if self._result is None:
            raise RuntimeError("Must call fit() before predict()")
        return self._result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FORECASTER_REGISTRY: dict[str, Callable[..., Forecaster]] = {
    "naive": NaiveForecaster,
}


def _register_lgbm() -> None:
    try:
        import lightgbm  # noqa: F401

        FORECASTER_REGISTRY["lgbm"] = LGBMForecaster
    except ImportError:
        pass


_register_lgbm()


def get_forecaster(name: str, **kwargs: Any) -> Forecaster:
    """Look up a forecaster by name. Raises KeyError if not found."""
    if name not in FORECASTER_REGISTRY:
        available = ", ".join(sorted(FORECASTER_REGISTRY))
        raise KeyError(f"Unknown forecaster {name!r}; available: {available}")
    return FORECASTER_REGISTRY[name](**kwargs)
