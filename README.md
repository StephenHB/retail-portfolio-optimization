# retail-portfolio-optimization

Retail portfolio optimization (demo data + research-driven methods). Includes Brzęczek (2020)-style **category portfolio** helpers (forecast risk, marginal assortment moves), a **Kaggle retail demo** adapter, tests, and research notes.

## Quick start

```bash
# Tests (no Kaggle file required)
PYTHONPATH=src python -m pytest

# Brzęczek demo figure — needs data/raw/data.csv (see data/README.md)
PYTHONPATH=src python scripts/visualize_brzezcek_demo.py

# Same figure without data
PYTHONPATH=src python scripts/visualize_brzezcek_demo.py --synthetic
```

Python package lives under **`src/retail_portfolio/`** (see `pyproject.toml` for dependencies).

## PySpark distributed optimization

For scaling to **n stores × variable m products** on Databricks / any Spark cluster, see the [`spark/` subpackage](src/retail_portfolio/spark/README.md). Key idea: each store carries a different product set, so each gets its own `(y*, V)` of size `m_i`. Three strategies (greedy, random, exhaustive) parallelise across stores.

```bash
pip install -e ".[spark]"   # adds pyspark>=3.4
```

```python
from retail_portfolio.spark.data_prep import brzezcek_inputs_per_store
from retail_portfolio.spark.distributed import optimize_n_portfolios

configs = brzezcek_inputs_per_store(sdf, store_col="store_id")
results = optimize_n_portfolios(spark, configs, strategy="greedy")
```

## Cursor agents and rules

- **`AGENTS.md`** — how to use manager, research, coding, validation, and GitHub-oriented workflows in chat.
- **`.cursor/rules/`** — always-on project goals, multi-agent role switching, and pluggable data-loading conventions (see `data-loading-extensible.mdc` when editing Python).
- **`.cursor/skills/`** — per-role `SKILL.md` files (research, coding, manager, validation, git).

## Demo data (Kaggle)

**Dataset:** [Retail Insights: A Comprehensive Sales Dataset](https://www.kaggle.com/datasets/rajneesh231/retail-insights-a-comprehensive-sales-dataset). Put `data.csv` in **`data/raw/`** (gitignored).  

**Overview, columns, and download steps:** [`data/README.md`](data/README.md).

**Brzęczek demo:** [`scripts/visualize_brzezcek_demo.py`](scripts/visualize_brzezcek_demo.py) → [`figures/brzezcek_demo.png`](figures/brzezcek_demo.png); numbers for the last run: [`figures/brzezcek_demo_summary.md`](figures/brzezcek_demo_summary.md).

## Research PDFs

Put papers in **`research/papers/`**. **Index, citations, and summaries:** [`research/papers/README.md`](research/papers/README.md) (includes Brzęczek 2020). **Theory + algorithm note:** [`research/papers/Brzezcek_2020_summary_and_algorithms.md`](research/papers/Brzezcek_2020_summary_and_algorithms.md).