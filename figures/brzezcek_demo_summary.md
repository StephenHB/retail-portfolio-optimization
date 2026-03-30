# Brzęczek demo: data summary and numeric results

This document covers two ways to build [`brzezcek_demo.png`](brzezcek_demo.png):

1. **Kaggle (default for `scripts/visualize_brzezcek_demo.py`)** — [*Retail Insights: A Comprehensive Sales Dataset*](https://www.kaggle.com/datasets/rajneesh231/retail-insights-a-comprehensive-sales-dataset). Put `data.csv` in `data/raw/` (unzip if needed). The adapter lives in [`src/retail_portfolio/data/kaggle_retail_insights.py`](../src/retail_portfolio/data/kaggle_retail_insights.py).
2. **Synthetic** — pass `--synthetic` to the script; six categories from `numpy.random.Generator(42)` as before.

---

## Kaggle pipeline (default)

- **Columns used:** `Order Date`, `Product Category`, `Order Total` (currency strings with `$` / commas).
- **Dates:** parsed with `dayfirst=True` (dataset is AU-style).
- **Panel:** sum `Order Total` by calendar month × category → wide matrix of monthly revenue.
- **y\* (point “forecast”):** revenue in the **last** month of the panel per category (carry-forward / random-walk level).
- **Residuals for V:** month-to-month first difference per category (innovations of a naïve random-walk forecast); **V** = sample covariance of those innovations across time (`mu_sigma_from_residual_matrix`).
- **Correlation heatmap:** `correlation_from_covariance(V)`.

Run: `PYTHONPATH=src python scripts/visualize_brzezcek_demo.py`  
Synthetic: add `--synthetic`. Custom path: `--data /path/to/data.csv`.

---

## 1. Synthetic mode — data summary

### 1.1 Categories

Six categories: **Cat 1 … Cat 6**, with one vector of point forecasts **y\*** (same units as sales, e.g. currency per horizon) and one residual standard deviation **s_i** per category (from the paper’s residual-sales layer).

| Category | Forecast **y\***_i | Residual std **s**_i |
|----------|-------------------:|---------------------:|
| Cat 1    | 420.0              | 120.0                |
| Cat 2    | 380.0              | 95.0                 |
| Cat 3    | 290.0              | 80.0                 |
| Cat 4    | 510.0              | 110.0                |
| Cat 5    | 240.0              | 70.0                 |
| Cat 6    | 180.0              | 55.0                 |

### 1.2 Correlation matrix **ρ** (residual sales)

Built from a random symmetric core (`numpy.random.Generator(42)`), then eigenvalue-clipped to stay PSD, then re-normalized to unit diagonal. **V** follows the paper: **v_ij = ρ_ij s_i s_j** (`covariance_from_std_correlation`).

Rounded to three decimals:

|        | C1   | C2   | C3   | C4   | C5   | C6   |
|--------|------|------|------|------|------|------|
| **C1** | 1.000 | 0.330 | 0.375 | 0.379 | 0.281 | 0.408 |
| **C2** | 0.330 | 1.000 | 0.293 | 0.312 | 0.235 | 0.434 |
| **C3** | 0.375 | 0.293 | 1.000 | 0.298 | 0.303 | 0.208 |
| **C4** | 0.379 | 0.312 | 0.298 | 1.000 | 0.302 | 0.340 |
| **C5** | 0.281 | 0.235 | 0.303 | 0.302 | 1.000 | 0.323 |
| **C6** | 0.408 | 0.434 | 0.208 | 0.340 | 0.323 | 1.000 |

Off-diagonals are **positive** (complementarity-style co-movement in residuals), consistent with the paper’s discussion of related categories.

### 1.3 Meta-objective parameters (plot defaults)

| Parameter | Value | Role |
|-----------|------:|------|
| **w** | 0.35 | Safety-stock cost weight on forecast error scale (paper’s *w*) |
| **α** | 0.05 | Level for Student-*t* critical value |
| **df** | 24 | Degrees of freedom **N − K** (illustrative) |
| **t** | ≈ **1.711** | `t_critical_student(df, α, one_sided_upper=True)` |

Meta-objective: **x′y\* − w · t · √(x′Vx)** (`meta_objective_profit_safety_stock`).

---

## 2. Synthetic mode — demo results

### 2.1 Baseline portfolio (marginal-add panel)

Binary mask **x = [1, 1, 1, 0, 0, 0]** (Cats 1–3 on).

| Metric | Value |
|--------|------:|
| Forecast total **x′y\*** | **1090.0** |
| Nominal risk **√(x′Vx)** | **≈ 221.37** |
| Relative risk **√(x′Vx)/(x′y\*)** | **≈ 0.2031** |

### 2.2 Marginal add scores (Δsales − Δrisk, unit weights)

Adding one inactive category to that baseline:

| Add | Score |
|-----|------:|
| **Cat 4** | **≈ 443.49** |
| Cat 5 | ≈ 206.14 |
| Cat 6 | ≈ 148.81 |

**Best single add:** **Cat 4** (same as `best_marginal_add`).

### 2.3 All 63 non-empty binary portfolios

| Quantity | Min | Max |
|----------|----:|----:|
| **x′y\*** | 180.0 | **2020.0** (all six on) |
| **√(x′Vx)** | 55.0 | **≈ 353.61** |
| Relative risk (finite) | **≈ 0.1721** | **≈ 0.3056** |
| Meta-objective | **≈ 147.07** | **≈ 1808.26** |

**Extremes (illustrative):**

- **Highest forecast sales** and **highest meta-objective:** all categories on (**x′y\* = 2020**, meta **≈ 1808.26**).
- **Lowest relative risk** among finite values: mask **[0, 1, 1, 1, 1, 1]** (Cat 1 off), **x′y\* = 1600**, relative **≈ 0.1721**.

### 2.4 Bottom-right panel: top 12 portfolios by **x′y\***

Bars are ordered by descending forecast sales. Labels **P*k*** correspond to portfolio id **k = 1 … 63** in the demo enumerator (integer bitmask loop in `brzezcek_demo.py`: row **k−1** in the stacked mask array).

| Rank | Portfolio id **k** | **x′y\*** | **√(x′Vx)** | Relative risk | Meta-objective |
|-----:|---------------------:|----------:|------------:|--------------:|---------------:|
| 1 | 63 | 2020.0 | 353.61 | 0.1751 | 1808.26 |
| 2 | 31 | 1840.0 | 321.97 | 0.1750 | 1647.20 |
| 3 | 47 | 1780.0 | 319.00 | 0.1792 | 1588.98 |
| 4 | 59 | 1730.0 | 310.67 | 0.1796 | 1543.97 |
| 5 | 61 | 1640.0 | 299.76 | 0.1828 | 1460.50 |
| 6 | 62 | 1600.0 | 275.33 | 0.1721 | 1435.13 |
| 7 | 15 | 1600.0 | 287.87 | 0.1799 | 1427.62 |
| 8 | 27 | 1550.0 | 277.44 | 0.1790 | 1383.87 |
| 9 | 55 | 1510.0 | 287.07 | 0.1901 | 1338.10 |
| 10 | 43 | 1490.0 | 276.82 | 0.1858 | 1324.24 |
| 11 | 29 | 1460.0 | 270.21 | 0.1851 | 1298.19 |
| 12 | 30 | 1420.0 | 244.57 | 0.1722 | 1273.55 |

---

## 3. How to reproduce

```bash
# Kaggle CSV at data/raw/data.csv (default)
PYTHONPATH=src python scripts/visualize_brzezcek_demo.py

# Synthetic six-category demo
PYTHONPATH=src python scripts/visualize_brzezcek_demo.py --synthetic
```

To recompute **synthetic** tables in Python, import `synthetic_demo_inputs` from `retail_portfolio.viz.brzezcek_demo`. For **Kaggle** inputs, use `brzezcek_inputs_from_retail_insights_csv` from `retail_portfolio.data.kaggle_retail_insights`.

---

## 4. Related code

| Piece | Location |
|-------|----------|
| Synthetic + figure | `src/retail_portfolio/viz/brzezcek_demo.py` |
| Kaggle adapter | `src/retail_portfolio/data/kaggle_retail_insights.py` |
| Portfolio math | `src/retail_portfolio/brzezcek_portfolio.py` |
| Script | `scripts/visualize_brzezcek_demo.py` |
