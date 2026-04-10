"""
PySpark data preparation: transaction-level data → Brzęczek inputs (y*, V).

Supports two workflows:

1. **Single portfolio** — all transactions belong to one portfolio.
   ``brzezcek_inputs_from_spark()`` returns a single (y*, V, labels).

2. **Per-store portfolios** — transactions have a ``store_id`` column.
   Each store carries a different assortment of products (different m).
   ``brzezcek_inputs_per_store()`` returns a list of ``make_portfolio_config``
   dicts ready for ``optimize_n_portfolios()``.

The covariance matrix V is m_i × m_i per store and always collected to the
driver (products per store are typically O(10²–10³), well within memory).

Usage on Databricks
-------------------
>>> from pyspark.sql import SparkSession
>>> spark = SparkSession.builder.getOrCreate()
>>> sdf = spark.read.csv("dbfs:/data/transactions.csv", header=True, inferSchema=True)
>>> # Single portfolio:
>>> y_star, v, labels = brzezcek_inputs_from_spark(sdf)
>>> # Per-store portfolios:
>>> configs = brzezcek_inputs_per_store(sdf, store_col="store_id")
"""

from __future__ import annotations

import numpy as np
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window


def monthly_category_revenue_spark(
    sdf: DataFrame,
    *,
    date_col: str = "order_date",
    category_col: str = "category",
    revenue_col: str = "revenue",
) -> DataFrame:
    """
    Aggregate transaction-level Spark DataFrame to monthly category revenue.

    Returns DataFrame with columns: period (string 'YYYY-MM'), category, revenue.
    """
    return (
        sdf
        .withColumn("period", F.date_format(F.col(date_col), "yyyy-MM"))
        .groupBy("period", F.col(category_col).alias("category"))
        .agg(F.sum(F.col(revenue_col)).alias("revenue"))
        .orderBy("period", "category")
    )


def pivot_monthly_revenue_spark(monthly_sdf: DataFrame) -> DataFrame:
    """
    Pivot long monthly revenue to wide: one row per period, one column per category.
    Missing combinations filled with 0.
    """
    categories = sorted(
        [row["category"] for row in monthly_sdf.select("category").distinct().collect()]
    )
    wide = (
        monthly_sdf
        .groupBy("period")
        .pivot("category", categories)
        .agg(F.sum("revenue"))
        .fillna(0.0)
        .orderBy("period")
    )
    return wide


def compute_residual_covariance_spark(
    wide_sdf: DataFrame,
    category_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute Brzęczek inputs from a wide monthly revenue Spark DataFrame.

    Uses month-over-month differencing (random walk residuals) computed via
    Spark window functions, then collects the residual matrix to the driver
    for covariance estimation.

    Parameters
    ----------
    wide_sdf : DataFrame
        Columns: period, plus one column per category (revenue values).
    category_cols : list[str]
        Names of the category columns in wide_sdf.

    Returns
    -------
    y_star : ndarray, shape (m,)
        Last month's revenue per category (naive forecast).
    v : ndarray, shape (m, m)
        Sample covariance of month-to-month revenue innovations.
    meta : dict
        Metadata (n_months, n_categories, etc.).
    """
    m = len(category_cols)
    w = Window.orderBy("period")

    # Compute month-over-month differences via lag window
    diff_sdf = wide_sdf
    for col_name in category_cols:
        prev_col = f"_prev_{col_name}"
        diff_col = f"_diff_{col_name}"
        diff_sdf = (
            diff_sdf
            .withColumn(prev_col, F.lag(col_name).over(w))
            .withColumn(diff_col, F.col(col_name) - F.col(prev_col))
            .drop(prev_col)
        )

    # Drop first row (null from lag) and collect diff columns
    diff_cols = [f"_diff_{c}" for c in category_cols]
    resid_sdf = diff_sdf.select(diff_cols).dropna()

    # Collect to driver — this is m columns × (T-1) rows, always fits in memory
    resid_rows = resid_sdf.collect()
    resid_matrix = np.array([[row[c] for c in diff_cols] for row in resid_rows], dtype=float)

    # Last row of wide for y* (naive level forecast)
    last_row = wide_sdf.orderBy(F.col("period").desc()).limit(1).collect()[0]
    y_star = np.array([float(last_row[c]) for c in category_cols], dtype=float)

    n_rows = resid_matrix.shape[0]
    if n_rows <= 1:
        raise ValueError(f"Need at least 2 residual rows; got {n_rows}")

    v = np.cov(resid_matrix, rowvar=False, ddof=1)
    # Ensure 2-D even for m=1
    if v.ndim == 0:
        v = v.reshape(1, 1)

    n_months = wide_sdf.count()
    meta = {
        "n_months": int(n_months),
        "n_categories": m,
        "residual_rows": int(n_rows),
    }
    return y_star, v, meta


def brzezcek_inputs_from_spark(
    sdf: DataFrame,
    *,
    date_col: str = "order_date",
    category_col: str = "category",
    revenue_col: str = "revenue",
    max_categories: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    End-to-end: raw transaction Spark DataFrame → (y_star, V, category_labels).

    Parameters
    ----------
    sdf : DataFrame
        Transaction-level data with date, category, and revenue columns.
    max_categories : int or None
        Keep top-N categories by total revenue (None = all).

    Returns
    -------
    y_star : ndarray (m,)
    v : ndarray (m, m)
    labels : list[str]
    """
    monthly = monthly_category_revenue_spark(
        sdf, date_col=date_col, category_col=category_col, revenue_col=revenue_col
    )

    # Optionally limit to top-N categories by total revenue
    if max_categories is not None:
        top_cats = (
            monthly
            .groupBy("category")
            .agg(F.sum("revenue").alias("total"))
            .orderBy(F.col("total").desc())
            .limit(max_categories)
            .select("category")
        )
        monthly = monthly.join(top_cats, on="category", how="inner")

    wide = pivot_monthly_revenue_spark(monthly)
    labels = sorted([c for c in wide.columns if c != "period"])

    y_star, v, _ = compute_residual_covariance_spark(wide, labels)
    return y_star, v, labels


def brzezcek_inputs_per_store(
    sdf: DataFrame,
    *,
    store_col: str = "store_id",
    date_col: str = "order_date",
    category_col: str = "category",
    revenue_col: str = "revenue",
    max_categories: int | None = None,
    min_months: int = 6,
    w: float = 1.0,
    t_critical: float = 1.96,
    forecaster: object | None = None,
) -> list[dict]:
    """
    Build one portfolio config per store from a transactions table.

    Each store may carry a different set of products, so the resulting configs
    have variable-length y_star and V.  Stores with too few months of data
    (< ``min_months + 1``) are silently skipped.

    Parameters
    ----------
    sdf : DataFrame
        Transaction-level data with store, date, category, and revenue columns.
    store_col : str
        Column identifying the store.
    max_categories : int or None
        Per-store cap on number of products (top-N by revenue).
    min_months : int
        Minimum monthly rows required after differencing.
    w, t_critical : float
        Passed through to ``make_portfolio_config()``.
    forecaster : Forecaster or None
        If provided, the wide panel is collected to pandas and passed to
        ``forecaster.fit(wide_pdf).predict()`` to obtain (y_star, V).
        If None (default), the built-in naive forecast is used.

    Returns
    -------
    List of dicts compatible with ``optimize_n_portfolios()``.
    """
    from retail_portfolio.spark.distributed import make_portfolio_config

    store_ids = sorted(
        row[store_col]
        for row in sdf.select(store_col).distinct().collect()
    )

    configs = []
    for sid in store_ids:
        store_sdf = sdf.filter(F.col(store_col) == sid)
        monthly = monthly_category_revenue_spark(
            store_sdf, date_col=date_col, category_col=category_col,
            revenue_col=revenue_col,
        )

        # Optionally limit to top-N products by total revenue for this store
        if max_categories is not None:
            top_cats = (
                monthly
                .groupBy("category")
                .agg(F.sum("revenue").alias("total"))
                .orderBy(F.col("total").desc())
                .limit(max_categories)
                .select("category")
            )
            monthly = monthly.join(top_cats, on="category", how="inner")

        wide = pivot_monthly_revenue_spark(monthly)
        labels = sorted([c for c in wide.columns if c != "period"])

        if len(labels) == 0:
            continue
        if wide.count() < min_months + 1:
            continue

        try:
            if forecaster is not None:
                # Collect to pandas and use the pluggable forecaster
                wide_pdf = wide.toPandas().sort_values("period").set_index("period")
                wide_pdf = wide_pdf[labels]
                result = forecaster.fit(wide_pdf).predict()
                y_star, v = result.y_star, result.V
            else:
                y_star, v, _ = compute_residual_covariance_spark(wide, labels)
        except ValueError:
            continue

        configs.append(make_portfolio_config(
            portfolio_id=str(sid),
            y_star=y_star,
            v=v,
            w=w,
            t_critical=t_critical,
            labels=labels,
        ))

    return configs
