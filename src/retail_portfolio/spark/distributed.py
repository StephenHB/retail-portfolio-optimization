"""
Distributed portfolio optimization using PySpark.

Each portfolio represents a **store** (or region / scenario).  Different stores
carry different numbers of products, so each portfolio config has its own
m_i-dimensional y_star (m_i,) and covariance V (m_i, m_i).

The core pattern:

    1. Build one config per store via ``make_portfolio_config()``.
       Each config is self-contained: (portfolio_id, y_star, V, w, t_critical, labels).
    2. **Parallelize** configs across the cluster (one RDD element per store).
    3. **Map** a per-store solver (greedy / random / exhaustive) on each executor.
       Because each config carries its own arrays, variable product counts are
       handled naturally — no shared broadcast is needed.
    4. **Collect** the best mask per store back to the driver.

Deployment on Databricks
------------------------
>>> from pyspark.sql import SparkSession
>>> spark = SparkSession.builder.getOrCreate()
>>> from retail_portfolio.spark.distributed import optimize_n_portfolios
>>> results_df = optimize_n_portfolios(spark, portfolio_configs, strategy="greedy")
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyspark.sql import DataFrame, SparkSession, Row
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from retail_portfolio.spark.solver import (
    evaluate_portfolio_batch,
    greedy_forward_selection,
    random_search,
)


# ---------------------------------------------------------------------------
# Result schema (shared across strategies)
# ---------------------------------------------------------------------------

RESULT_SCHEMA = StructType([
    StructField("portfolio_id", StringType(), False),
    StructField("strategy", StringType(), False),
    StructField("sales", DoubleType(), False),
    StructField("std", DoubleType(), False),
    StructField("relative_risk", DoubleType(), False),
    StructField("meta_objective", DoubleType(), False),
    StructField("n_included", IntegerType(), False),
    StructField("mask", ArrayType(IntegerType()), False),
])


# ---------------------------------------------------------------------------
# Portfolio configuration
# ---------------------------------------------------------------------------

def make_portfolio_config(
    portfolio_id: str,
    y_star: np.ndarray,
    v: np.ndarray,
    *,
    w: float = 1.0,
    t_critical: float = 1.96,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a config dict for one store's portfolio.

    Each store may carry a different number of products (m_i), so y_star and V
    are sized per-store.  The config is fully self-contained and serialisable
    (plain Python lists) so it can be shipped to any Spark executor.

    Parameters
    ----------
    portfolio_id : str
        Unique identifier (e.g. "store_42", "region_west").
    y_star : ndarray (m_i,)
        Forecast sales per product **for this store**.
    v : ndarray (m_i, m_i)
        Residual covariance matrix **for this store's products**.
    w : float
        Safety-stock cost weight.
    t_critical : float
        Student-t critical value.
    labels : list[str] or None
        Product names carried by this store (length m_i).
    """
    return {
        "portfolio_id": portfolio_id,
        "y_star": np.asarray(y_star, dtype=float).ravel().tolist(),
        "v": np.asarray(v, dtype=float).tolist(),
        "w": float(w),
        "t_critical": float(t_critical),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Strategy: Greedy (n portfolios in parallel)
# ---------------------------------------------------------------------------

def _run_greedy(config: dict[str, Any]) -> Row:
    """Map function: run greedy forward selection for one portfolio config."""
    y_star = np.array(config["y_star"], dtype=float)
    v = np.array(config["v"], dtype=float)
    steps = greedy_forward_selection(
        y_star, v,
        w=config["w"],
        t_critical=config["t_critical"],
        objective="meta",
    )
    if not steps:
        return Row(
            portfolio_id=config["portfolio_id"],
            strategy="greedy",
            sales=0.0,
            std=0.0,
            relative_risk=float("inf"),
            meta_objective=float("-inf"),
            n_included=0,
            mask=[0] * len(config["y_star"]),
        )
    best = steps[-1]  # final step has all greedily-added products
    return Row(
        portfolio_id=config["portfolio_id"],
        strategy="greedy",
        sales=best["sales"],
        std=best["std"],
        relative_risk=best["relative_risk"],
        meta_objective=best["meta_objective"],
        n_included=best["n_included"],
        mask=best["mask"],
    )


# ---------------------------------------------------------------------------
# Strategy: Random search (n portfolios in parallel)
# ---------------------------------------------------------------------------

def _run_random(config: dict[str, Any], n_samples: int, seed: int | None) -> Row:
    """Map function: run random search for one portfolio config."""
    y_star = np.array(config["y_star"], dtype=float)
    v = np.array(config["v"], dtype=float)
    per_seed = None if seed is None else seed + hash(config["portfolio_id"]) % (2**31)
    result = random_search(
        y_star, v,
        n_samples=n_samples,
        w=config["w"],
        t_critical=config["t_critical"],
        seed=per_seed,
        objective="meta",
    )
    best = result["best"]
    return Row(
        portfolio_id=config["portfolio_id"],
        strategy="random",
        sales=best["sales"],
        std=best["std"],
        relative_risk=best["relative_risk"],
        meta_objective=best["meta_objective"],
        n_included=best["n_included"],
        mask=best["mask"],
    )


# ---------------------------------------------------------------------------
# Strategy: Exhaustive batch (small m only; partitioned across cluster)
# ---------------------------------------------------------------------------

def _generate_all_masks(m: int) -> np.ndarray:
    """Generate all 2^m - 1 non-empty binary masks for m products."""
    n_combos = (1 << m) - 1
    masks = np.zeros((n_combos, m), dtype=float)
    for i in range(n_combos):
        bits = i + 1  # skip the all-zeros mask
        for j in range(m):
            if bits & (1 << j):
                masks[i, j] = 1.0
    return masks


def _run_exhaustive_partition(iterator, y_star_bc, v_bc, w, t_critical):
    """
    Map-partitions function: evaluate a chunk of (portfolio_id, mask_start, mask_end)
    assignments for exhaustive search.
    """
    y_star = y_star_bc.value
    v = v_bc.value
    for row in iterator:
        pid = row["portfolio_id"]
        masks_list = np.array(row["masks"], dtype=float)
        results = evaluate_portfolio_batch(masks_list, y_star, v, w, t_critical)
        best = max(results, key=lambda r: r["meta_objective"])
        yield Row(
            portfolio_id=pid,
            strategy="exhaustive",
            sales=best["sales"],
            std=best["std"],
            relative_risk=best["relative_risk"],
            meta_objective=best["meta_objective"],
            n_included=best["n_included"],
            mask=best["mask"],
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def optimize_n_portfolios(
    spark: SparkSession,
    portfolio_configs: list[dict[str, Any]],
    *,
    strategy: str = "greedy",
    n_random_samples: int = 10_000,
    random_seed: int | None = 42,
    num_partitions: int | None = None,
) -> DataFrame:
    """
    Solve n stores in parallel on a Spark cluster.

    Each store carries its own product set (possibly different sizes m_i), so
    each config is self-contained with its own y_star (m_i,) and V (m_i, m_i).

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    portfolio_configs : list[dict]
        List of dicts from ``make_portfolio_config()``.
        Each represents one store with its own y* and V (variable m per store).
    strategy : {"greedy", "random", "exhaustive"}
        - "greedy": O(m_i²) forward selection per store, parallelised across n stores.
        - "random": Sample n_random_samples masks per store, parallelised across n.
        - "exhaustive": Evaluate all 2^m_i - 1 masks per store.
          Only practical when every store's m_i ≤ ~20.
    num_partitions : int or None
        Number of Spark partitions. Defaults to len(portfolio_configs).

    Returns
    -------
    Spark DataFrame with schema RESULT_SCHEMA: one row per store with the
    best mask found by the chosen strategy.  The ``mask`` column length varies
    per store (matching that store's m_i).

    Example (Databricks)
    --------------------
    >>> configs = [make_portfolio_config(f"store_{i}", y_star_i, v_i) for i, ...]
    >>> results = optimize_n_portfolios(spark, configs, strategy="greedy")
    >>> results.show()
    """
    sc = spark.sparkContext
    n = len(portfolio_configs)

    if num_partitions is None:
        num_partitions = max(1, n)

    if strategy == "greedy":
        configs_rdd = sc.parallelize(portfolio_configs, numSlices=num_partitions)
        results_rdd = configs_rdd.map(_run_greedy)
        return spark.createDataFrame(results_rdd, schema=RESULT_SCHEMA)

    elif strategy == "random":
        configs_rdd = sc.parallelize(portfolio_configs, numSlices=num_partitions)
        results_rdd = configs_rdd.map(
            lambda cfg: _run_random(cfg, n_random_samples, random_seed)
        )
        return spark.createDataFrame(results_rdd, schema=RESULT_SCHEMA)

    elif strategy == "exhaustive":
        # Each store can have a different m — generate masks per-config
        for cfg in portfolio_configs:
            mi = len(cfg["y_star"])
            if mi > 20:
                raise ValueError(
                    f"Portfolio {cfg['portfolio_id']!r} has m={mi} products; "
                    f"exhaustive search would generate {2**mi - 1} masks. "
                    f"Use 'greedy' or 'random' strategy for large m."
                )

        rows = []
        for cfg in portfolio_configs:
            mi = len(cfg["y_star"])
            rows.append({
                "portfolio_id": cfg["portfolio_id"],
                "masks": _generate_all_masks(mi).tolist(),
                "y_star": cfg["y_star"],
                "v": cfg["v"],
                "w": cfg["w"],
                "t_critical": cfg["t_critical"],
            })

        work_rdd = sc.parallelize(rows, numSlices=num_partitions)

        def _eval_exhaustive(row):
            y_star = np.array(row["y_star"], dtype=float)
            v = np.array(row["v"], dtype=float)
            masks_arr = np.array(row["masks"], dtype=float)
            results = evaluate_portfolio_batch(
                masks_arr, y_star, v, row["w"], row["t_critical"]
            )
            best = max(results, key=lambda r: r["meta_objective"])
            return Row(
                portfolio_id=row["portfolio_id"],
                strategy="exhaustive",
                sales=best["sales"],
                std=best["std"],
                relative_risk=best["relative_risk"],
                meta_objective=best["meta_objective"],
                n_included=best["n_included"],
                mask=best["mask"],
            )

        results_rdd = work_rdd.map(_eval_exhaustive)
        return spark.createDataFrame(results_rdd, schema=RESULT_SCHEMA)

    else:
        raise ValueError(f"Unknown strategy {strategy!r}; use 'greedy', 'random', or 'exhaustive'")


# ---------------------------------------------------------------------------
# Large-m helper: partition the mask space across the cluster
# ---------------------------------------------------------------------------

def optimize_single_portfolio_distributed(
    spark: SparkSession,
    y_star: np.ndarray,
    v: np.ndarray,
    *,
    w: float = 1.0,
    t_critical: float = 1.96,
    n_samples_per_partition: int = 100_000,
    num_partitions: int = 100,
    min_products: int = 1,
    max_products: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Distributed random search for a single portfolio with many products (large m).

    Distributes random mask generation and evaluation across partitions.
    Each partition generates and evaluates its own batch of random masks,
    then the driver collects only the per-partition best.

    Parameters
    ----------
    n_samples_per_partition : int
        Each partition generates this many random masks.
    num_partitions : int
        Total partitions; total samples = n_samples_per_partition × num_partitions.

    Returns
    -------
    Dict with 'best' result and 'total_evaluated'.
    """
    sc = spark.sparkContext
    y_star = np.asarray(y_star, dtype=float).ravel()
    v = np.asarray(v, dtype=float)
    m = y_star.shape[0]
    if max_products is None:
        max_products = m

    # Broadcast the shared arrays
    y_star_bc = sc.broadcast(y_star.tolist())
    v_bc = sc.broadcast(v.tolist())

    partition_seeds = list(range(seed, seed + num_partitions))
    seeds_rdd = sc.parallelize(partition_seeds, numSlices=num_partitions)

    def _search_partition(part_seed):
        y = np.array(y_star_bc.value, dtype=float)
        cov = np.array(v_bc.value, dtype=float)
        result = random_search(
            y, cov,
            n_samples=n_samples_per_partition,
            w=w,
            t_critical=t_critical,
            min_products=min_products,
            max_products=max_products,
            seed=part_seed,
            objective="meta",
        )
        return result["best"]

    partition_bests = seeds_rdd.map(_search_partition).collect()

    # Find global best across all partitions
    global_best = max(partition_bests, key=lambda r: r["meta_objective"])

    # Clean up broadcasts
    y_star_bc.unpersist()
    v_bc.unpersist()

    return {
        "best": global_best,
        "total_evaluated": n_samples_per_partition * num_partitions,
    }
