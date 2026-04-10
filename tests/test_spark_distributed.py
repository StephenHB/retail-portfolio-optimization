"""
Integration tests for PySpark distributed optimization.

These tests use a local SparkSession. They are marked with @pytest.mark.spark
so they can be skipped in environments without PySpark installed:

    pytest -m "not spark"   # skip Spark tests
    pytest -m spark          # run only Spark tests
"""

import numpy as np
import pytest

try:
    from pyspark.sql import SparkSession

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

if PYSPARK_AVAILABLE:
    from retail_portfolio.spark.distributed import (
        make_portfolio_config,
        optimize_n_portfolios,
        optimize_single_portfolio_distributed,
    )

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def spark():
    if not PYSPARK_AVAILABLE:
        pytest.skip("pyspark not installed")
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("test-retail-portfolio")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "localhost")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def three_product_configs():
    """Three portfolio configs with 3 products each (e.g. 3 stores)."""
    rng = np.random.default_rng(42)
    configs = []
    for i in range(3):
        m = 3
        y_star = rng.uniform(5, 30, size=m)
        a = rng.normal(size=(m, m))
        v = a.T @ a / m + np.eye(m) * 0.5
        configs.append(
            make_portfolio_config(f"store_{i}", y_star, v, w=1.0, t_critical=1.96)
        )
    return configs


@pytest.fixture
def variable_m_configs():
    """Four stores carrying different numbers of products (m = 2, 5, 3, 4)."""
    rng = np.random.default_rng(55)
    product_counts = [2, 5, 3, 4]
    configs = []
    for i, m in enumerate(product_counts):
        y_star = rng.uniform(5, 40, size=m)
        a = rng.normal(size=(m, m))
        v = a.T @ a / m + np.eye(m) * 0.3
        configs.append(
            make_portfolio_config(
                f"store_{i}", y_star, v,
                w=1.0, t_critical=1.96,
                labels=[f"prod_{j}" for j in range(m)],
            )
        )
    return configs


@pytest.fixture
def six_product_config():
    """Single portfolio config with 6 products."""
    rng = np.random.default_rng(77)
    m = 6
    y_star = rng.uniform(10, 50, size=m)
    a = rng.normal(size=(m, m))
    v = a.T @ a / m + np.eye(m) * 0.3
    return y_star, v


class TestOptimizeNPortfolios:
    def test_greedy_returns_all_portfolios(self, spark, three_product_configs):
        df = optimize_n_portfolios(spark, three_product_configs, strategy="greedy")
        assert df.count() == 3
        ids = {row["portfolio_id"] for row in df.collect()}
        assert ids == {"store_0", "store_1", "store_2"}

    def test_greedy_results_have_valid_metrics(self, spark, three_product_configs):
        df = optimize_n_portfolios(spark, three_product_configs, strategy="greedy")
        for row in df.collect():
            assert row["sales"] > 0
            assert row["std"] >= 0
            assert row["n_included"] >= 1
            assert np.isfinite(row["meta_objective"])

    def test_random_returns_all_portfolios(self, spark, three_product_configs):
        df = optimize_n_portfolios(
            spark, three_product_configs,
            strategy="random", n_random_samples=500, random_seed=1
        )
        assert df.count() == 3

    def test_exhaustive_returns_all_portfolios(self, spark, three_product_configs):
        df = optimize_n_portfolios(spark, three_product_configs, strategy="exhaustive")
        assert df.count() == 3
        for row in df.collect():
            assert row["strategy"] == "exhaustive"

    def test_exhaustive_finds_global_optimum(self, spark, three_product_configs):
        """Exhaustive should find a meta_objective >= greedy for each portfolio."""
        df_exh = optimize_n_portfolios(spark, three_product_configs, strategy="exhaustive")
        df_gre = optimize_n_portfolios(spark, three_product_configs, strategy="greedy")

        exh_map = {r["portfolio_id"]: r["meta_objective"] for r in df_exh.collect()}
        gre_map = {r["portfolio_id"]: r["meta_objective"] for r in df_gre.collect()}

        for pid in exh_map:
            assert exh_map[pid] >= gre_map[pid] - 1e-10

    def test_invalid_strategy_raises(self, spark, three_product_configs):
        with pytest.raises(ValueError, match="Unknown strategy"):
            optimize_n_portfolios(spark, three_product_configs, strategy="bogus")


class TestOptimizeSingleDistributed:
    def test_distributed_random_returns_result(self, spark, six_product_config):
        y_star, v = six_product_config
        result = optimize_single_portfolio_distributed(
            spark, y_star, v,
            n_samples_per_partition=500,
            num_partitions=4,
            seed=10,
        )
        assert "best" in result
        assert result["total_evaluated"] == 500 * 4
        assert result["best"]["n_included"] >= 1

    def test_distributed_random_beats_single_product(self, spark, six_product_config):
        y_star, v = six_product_config
        w, t_crit = 1.0, 1.96
        result = optimize_single_portfolio_distributed(
            spark, y_star, v,
            w=w, t_critical=t_crit,
            n_samples_per_partition=1000,
            num_partitions=4,
            seed=20,
        )
        from retail_portfolio.brzezcek_portfolio import meta_objective_profit_safety_stock
        worst_single = min(
            meta_objective_profit_safety_stock(
                np.eye(1, len(y_star), j).ravel(), y_star, v, w=w, t_critical=t_crit
            )
            for j in range(len(y_star))
        )
        assert result["best"]["meta_objective"] >= worst_single


class TestVariableProductCounts:
    """Stores carry different numbers of products (variable m)."""

    def test_greedy_variable_m(self, spark, variable_m_configs):
        df = optimize_n_portfolios(spark, variable_m_configs, strategy="greedy")
        assert df.count() == 4
        for row in df.collect():
            assert row["n_included"] >= 1
            assert row["sales"] > 0

    def test_greedy_mask_lengths_match_store(self, spark, variable_m_configs):
        """Each store's mask length should equal its product count."""
        df = optimize_n_portfolios(spark, variable_m_configs, strategy="greedy")
        expected_m = {f"store_{i}": m for i, m in enumerate([2, 5, 3, 4])}
        for row in df.collect():
            assert len(row["mask"]) == expected_m[row["portfolio_id"]]

    def test_random_variable_m(self, spark, variable_m_configs):
        df = optimize_n_portfolios(
            spark, variable_m_configs,
            strategy="random", n_random_samples=300, random_seed=99
        )
        assert df.count() == 4
        expected_m = {f"store_{i}": m for i, m in enumerate([2, 5, 3, 4])}
        for row in df.collect():
            assert len(row["mask"]) == expected_m[row["portfolio_id"]]

    def test_exhaustive_variable_m(self, spark, variable_m_configs):
        df = optimize_n_portfolios(spark, variable_m_configs, strategy="exhaustive")
        assert df.count() == 4
        expected_m = {f"store_{i}": m for i, m in enumerate([2, 5, 3, 4])}
        for row in df.collect():
            assert len(row["mask"]) == expected_m[row["portfolio_id"]]
            assert row["strategy"] == "exhaustive"

    def test_exhaustive_beats_greedy_variable_m(self, spark, variable_m_configs):
        df_exh = optimize_n_portfolios(spark, variable_m_configs, strategy="exhaustive")
        df_gre = optimize_n_portfolios(spark, variable_m_configs, strategy="greedy")
        exh = {r["portfolio_id"]: r["meta_objective"] for r in df_exh.collect()}
        gre = {r["portfolio_id"]: r["meta_objective"] for r in df_gre.collect()}
        for pid in exh:
            assert exh[pid] >= gre[pid] - 1e-10


class TestMakePortfolioConfig:
    def test_roundtrip(self):
        y = np.array([1.0, 2.0, 3.0])
        v = np.eye(3)
        cfg = make_portfolio_config("test", y, v, w=0.5, t_critical=2.0, labels=["a", "b", "c"])
        assert cfg["portfolio_id"] == "test"
        assert cfg["w"] == 0.5
        assert len(cfg["y_star"]) == 3
        assert len(cfg["v"]) == 3
        assert cfg["labels"] == ["a", "b", "c"]

    def test_variable_sizes(self):
        """Configs with different m should be independently valid."""
        cfg2 = make_portfolio_config("s1", np.array([1.0, 2.0]), np.eye(2))
        cfg5 = make_portfolio_config("s2", np.arange(1, 6, dtype=float), np.eye(5))
        assert len(cfg2["y_star"]) == 2
        assert len(cfg2["v"]) == 2
        assert len(cfg5["y_star"]) == 5
        assert len(cfg5["v"]) == 5
