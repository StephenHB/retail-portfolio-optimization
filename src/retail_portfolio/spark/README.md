# PySpark distributed portfolio optimization

Distributes Brzęczek (2020) portfolio optimization across a Spark cluster. Designed for **n stores**, each carrying a **different number of products** (variable $m_i$).

## Problem

Each store $i$ has:

- $\mathbf{y}^{\ast}_i \in \mathbb{R}^{m_i}$ — forecast sales per product (the products this store carries).
- $V_i \in \mathbb{R}^{m_i \times m_i}$ — residual covariance of those products at this store.
- Decision: binary mask $\mathbf{x}_i \in \{0,1\}^{m_i}$ (include/exclude each product).

The objective per store is the Brzęczek meta-objective:

$$
\max_{\mathbf{x}_i} \quad \mathbf{x}_i^\top \mathbf{y}^{\ast}_i \;-\; w \cdot t_{\alpha} \cdot \sqrt{\mathbf{x}_i^\top V_i \,\mathbf{x}_i}
$$

With $n$ stores and variable $m_i$, exhaustive search ($2^{m_i}$ per store) is infeasible for large $m_i$. This package provides three strategies that parallelise across the cluster.

## Module layout

```
src/retail_portfolio/spark/
├── __init__.py          # Package exports (solver functions)
├── solver.py            # Pure-numpy solvers (no Spark dependency)
├── data_prep.py         # PySpark data loading → per-store (y*, V)
├── distributed.py       # PySpark orchestration across n stores
└── README.md            # This file
```

### `solver.py` — per-executor numpy operations

No Spark dependency. Runs inside each executor's Python process.

| Function | Complexity | Description |
|----------|-----------|-------------|
| `evaluate_portfolio_batch(masks, y_star, v)` | O(k·m²) | Vectorized evaluation of k binary masks against (y\*, V). Uses `einsum` for the quadratic form. |
| `greedy_forward_selection(y_star, v)` | O(m²) | Builds a portfolio one product at a time, greedily maximising the meta-objective (or minimising relative risk). |
| `random_search(y_star, v, n_samples=...)` | O(k·m²) | Samples k random masks with controlled sparsity, evaluates all, returns the best. |

### `data_prep.py` — PySpark data loading

Transforms a transaction-level Spark DataFrame into Brzęczek inputs.

| Function | Input | Output |
|----------|-------|--------|
| `brzezcek_inputs_from_spark(sdf)` | Single-portfolio transactions | `(y_star, V, labels)` |
| `brzezcek_inputs_per_store(sdf, store_col=...)` | Multi-store transactions | List of `make_portfolio_config()` dicts |

`brzezcek_inputs_per_store` groups by `store_col`, computes per-store monthly revenue, and derives each store's (y\*, V) independently. Stores with different product assortments produce configs with different $m_i$. Stores with insufficient history are silently skipped.

### `distributed.py` — cluster orchestration

| Function | Description |
|----------|-------------|
| `make_portfolio_config(portfolio_id, y_star, v, ...)` | Builds one self-contained config dict (serialisable to any executor). |
| `optimize_n_portfolios(spark, configs, strategy=...)` | Main entry: solves n stores in parallel. Returns a Spark DataFrame with one row per store. |
| `optimize_single_portfolio_distributed(spark, y_star, v, ...)` | For a single store with very large $m$: distributes random search across partitions. |

## Strategies

| Strategy | When to use | How it scales |
|----------|------------|---------------|
| `"greedy"` | Default for any $m$. Deterministic, O($m_i^2$) per store. | Parallelised across $n$ stores. Each store runs independently on one executor. |
| `"random"` | Large $m_i$ where greedy may miss the global optimum. | Parallelised across $n$ stores. Configurable sample count per store. |
| `"exhaustive"` | Small $m_i$ (guarded to $m_i \le 20$). Finds the true optimum. | Parallelised across $n$ stores. Each store evaluates all $2^{m_i}-1$ masks. |

For a **single store with very large $m$**, use `optimize_single_portfolio_distributed()` which partitions the random search itself across the cluster (total samples = `n_samples_per_partition * num_partitions`).

## Architecture

```
Driver                              Executors
──────                              ─────────
brzezcek_inputs_per_store(sdf)
  → configs: [{store_0, y*(3,), V(3,3)},
               {store_1, y*(7,), V(7,7)},
               {store_2, y*(5,), V(5,5)},
               ...]

optimize_n_portfolios(spark, configs)
  sc.parallelize(configs)    ──→    ├── executor 1: greedy(y*₀, V₀)  m=3
                                    ├── executor 2: greedy(y*₁, V₁)  m=7
                                    ├── executor 3: greedy(y*₂, V₂)  m=5
                                    └── ...
               ←── collect ────────┘

Result DataFrame:
┌──────────────┬──────────┬───────┬──────┬────────────────┬────────┐
│ portfolio_id │ strategy │ sales │ std  │ meta_objective │ mask   │
├──────────────┼──────────┼───────┼──────┼────────────────┼────────┤
│ store_0      │ greedy   │ 35.2  │ 4.1  │ 27.2           │ [1,1,0]│
│ store_1      │ greedy   │ 120.5 │ 12.3 │ 96.4           │ [1,0,…]│
│ store_2      │ greedy   │ 68.0  │ 7.8  │ 52.7           │ [1,1,…]│
└──────────────┴──────────┴───────┴──────┴────────────────┴────────┘
```

Key design: each config carries its own arrays, so **variable product counts are handled naturally** — no shared broadcast needed, no assumption that all stores share the same $m$.

## Quick start (Databricks)

```python
from pyspark.sql import SparkSession
from retail_portfolio.spark.data_prep import brzezcek_inputs_per_store
from retail_portfolio.spark.distributed import optimize_n_portfolios

spark = SparkSession.builder.getOrCreate()

# Load transactions (must have store_id, order_date, category, revenue)
sdf = spark.read.csv("dbfs:/data/transactions.csv", header=True, inferSchema=True)

# Build per-store configs (each store gets its own y*, V based on its products)
configs = brzezcek_inputs_per_store(sdf, store_col="store_id")

# Optimize all stores in parallel
results = optimize_n_portfolios(spark, configs, strategy="greedy")
results.show()
```

## Quick start (local / testing)

```python
import numpy as np
from retail_portfolio.spark.solver import greedy_forward_selection

# No Spark needed for the solver layer
y_star = np.array([10.0, 20.0, 5.0, 15.0])
V = np.diag([4.0, 9.0, 1.0, 6.0])

steps = greedy_forward_selection(y_star, V, w=1.0, t_critical=1.96)
for s in steps:
    print(f"Step {s['step']}: add product {s['product_added']}, "
          f"meta_obj={s['meta_objective']:.2f}")
```

## Installation

```bash
pip install -e ".[spark]"     # includes pyspark>=3.4
# or, if PySpark is already on your Databricks cluster:
pip install -e .
```

## Tests

```bash
# Solver tests (pure numpy, no Spark needed) — always run:
PYTHONPATH=src python -m pytest tests/test_spark_solver.py -v

# Spark integration tests (needs pyspark installed):
PYTHONPATH=src python -m pytest tests/test_spark_distributed.py -v

# Skip Spark tests in CI without pyspark:
PYTHONPATH=src python -m pytest -m "not spark"
```

## Relationship to the original single-machine code

The original `brzezcek_portfolio.py` computes the same quadratic forms (`x'y*`, `x'Vx`) and meta-objective for a single portfolio. This spark package:

1. Reuses the same math (validated in `test_spark_solver.py` against the original functions).
2. Adds scalable search strategies (greedy, random) that avoid 2^m enumeration.
3. Wraps them in PySpark RDD operations for cluster-level parallelism across n stores.
4. Adds a per-store data prep pipeline that handles variable product assortments.
