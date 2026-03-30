"""
Adapter for Kaggle *Retail Insights: A Comprehensive Sales Dataset* (rajneesh231).

Expected file: `data.csv` (or `.zip` containing it) under `data/raw/` after download.
Schema: `Order Date`, `Product Category`, `Order Total` (currency strings).

Core optimization code stays source-agnostic; this module is the dataset-specific
mapping to category–time panels and Brzęczek inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_portfolio.data.forecast_inputs import (
    correlation_from_covariance,
    mu_sigma_from_residual_matrix,
)


@dataclass
class RetailInsightsColumns:
    """Column names for the rajneesh231 CSV / Excel export."""

    order_date: str = "Order Date"
    product_category: str = "Product Category"
    order_total: str = "Order Total"


def parse_currency_series(series: pd.Series) -> pd.Series:
    """Strip $ and thousands separators; coerce to float."""
    s = series.astype(str).str.replace(r"[$,\s]", "", regex=True)
    s = s.replace({"": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")


def load_retail_insights_csv(
    path: str | Path,
    *,
    columns: RetailInsightsColumns | None = None,
    dayfirst: bool = True,
) -> pd.DataFrame:
    """
    Read transaction-level CSV; parse dates and money.

    Parameters
    ----------
    path :
        Path to `data.csv` (plain CSV). If you only have the Kaggle zip, unzip first.
    dayfirst :
        Australian-style dates in this dataset (e.g. 02-09-2014 = 2 Sep 2014).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cols = columns or RetailInsightsColumns()
    df = pd.read_csv(path)
    for c in (cols.order_date, cols.product_category, cols.order_total):
        if c not in df.columns:
            raise ValueError(f"Missing column {c!r}; have {list(df.columns)}")
    out = df[[cols.order_date, cols.product_category, cols.order_total]].copy()
    out = out.rename(
        columns={
            cols.order_date: "order_date",
            cols.product_category: "category",
            cols.order_total: "revenue",
        }
    )
    out["revenue"] = parse_currency_series(out["revenue"])
    out["order_date"] = pd.to_datetime(out["order_date"], dayfirst=dayfirst, errors="coerce")
    out = out.dropna(subset=["order_date", "category", "revenue"])
    return out


def monthly_category_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate revenue by calendar month and product category.

    Returns long frame with columns `period` (Timestamp), `category`, `revenue`.
    """
    d = df.copy()
    d["period"] = d["order_date"].dt.to_period("M").dt.to_timestamp(how="start")
    g = d.groupby(["period", "category"], as_index=False)["revenue"].sum()
    return g


def wide_monthly_revenue(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to wide: index = period (sorted), columns = category, values = revenue."""
    wide = long_df.pivot(index="period", columns="category", values="revenue")
    wide = wide.sort_index().fillna(0.0)
    return wide


def brzezcek_inputs_from_monthly_revenue_wide(
    wide: pd.DataFrame,
    *,
    max_categories: int | None = None,
    min_months: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """
    Build y*, V, and correlation matrix for the demo from monthly revenue wide panel.

    Uses a random-walk-style one-step innovation: residual_itc = rev_itc - rev_{i,t-1}.
    Sample covariance of innovations is V. Point forecast y* is the **last** month's
    revenue per category (naive level forecast).

    Parameters
    ----------
    max_categories :
        Keep top-N categories by total revenue (None = all).
    min_months :
        Minimum number of monthly rows required after differencing.
    """
    if wide.shape[0] < min_months + 1:
        raise ValueError(
            f"Need at least {min_months + 1} monthly rows; got {wide.shape[0]}"
        )
    w = wide.copy()
    totals = w.sum(axis=0).sort_values(ascending=False)
    if max_categories is not None:
        keep = list(totals.index[:max_categories])
        w = w[keep]
    labels = [str(c) for c in w.columns]
    y_star = w.iloc[-1].to_numpy(dtype=float)
    resid = w.diff().iloc[1:]
    if resid.shape[0] <= resid.shape[1]:
        raise ValueError(
            "Not enough differenced rows relative to categories for stable covariance"
        )
    _, v = mu_sigma_from_residual_matrix(y_star, resid.to_numpy(dtype=float), ddof=1)
    corr = correlation_from_covariance(v)
    meta: dict[str, Any] = {
        "n_months": int(w.shape[0]),
        "n_categories": len(labels),
        "period_start": str(w.index.min().date()),
        "period_end": str(w.index.max().date()),
        "residual_rows": int(resid.shape[0]),
    }
    return y_star, v, corr, labels, meta


def brzezcek_inputs_from_retail_insights_csv(
    path: str | Path,
    *,
    columns: RetailInsightsColumns | None = None,
    max_categories: int | None = None,
    min_months: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """End-to-end: CSV path → (y_star, V, corr, labels, meta)."""
    raw = load_retail_insights_csv(path, columns=columns)
    meta_pre: dict[str, Any] = {"n_transactions": len(raw)}
    long_df = monthly_category_revenue(raw)
    wide = wide_monthly_revenue(long_df)
    y_star, v, corr, labels, meta = brzezcek_inputs_from_monthly_revenue_wide(
        wide,
        max_categories=max_categories,
        min_months=min_months,
    )
    meta = {**meta_pre, **meta, "source_csv": str(Path(path).resolve())}
    return y_star, v, corr, labels, meta


def default_retail_insights_paths() -> list[Path]:
    """Typical locations after Kaggle download (repo-relative)."""
    return [
        Path("data/raw/data.csv"),
        Path("data/raw/Data.csv"),
    ]


def resolve_retail_insights_csv(explicit: str | Path | None = None) -> Path:
    """
    Return first existing path from *explicit* or `default_retail_insights_paths()`.
    Raises FileNotFoundError with install hint if none exist.
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates.extend(default_retail_insights_paths())
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "No retail insights CSV found. Download from "
        "https://www.kaggle.com/datasets/rajneesh231/retail-insights-a-comprehensive-sales-dataset "
        "and place `data.csv` in data/raw/ (unzip if needed), or pass --data PATH."
    )
