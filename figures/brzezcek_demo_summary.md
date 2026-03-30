# Brzęczek demo: data summary and numeric results

This file documents [`brzezcek_demo.png`](brzezcek_demo.png). **Regenerated: 2026-03-30.**

**Dataset overview:** [`../data/README.md`](../data/README.md).

The committed PNG is built from whatever sits at **`data/raw/data.csv`** when you run the script (Kaggle *Retail Insights* schema). If that file is missing, use `--synthetic` or download the dataset (see [README](../README.md)). **All Kaggle numbers below reflect the current local CSV** (5,000 rows → 49 monthly buckets in this run).

---

## Current PNG — Kaggle mode

### Source and panel

| Field | Value |
|-------|--------|
| **File** | `data/raw/data.csv` |
| **Transactions** | 5,000 |
| **Calendar months** | 49 (`2013-02-01` → `2017-02-01`) |
| **Differenced rows for V** | 48 |
| **Categories** | 3 — Furniture, Office Supplies, Technology |

### **y\*** (last month revenue per category, random-walk level)

| Category | **y\*** (last month total) |
|----------|---------------------------:|
| Furniture | 158.00 |
| Office Supplies | 18,294.29 |
| Technology | 8,498.48 |

### Correlation **ρ** of month-to-month revenue innovations (from **V**)

Rounded to three decimals (same order as labels above):

|  | Furniture | Office Supplies | Technology |
|--|----------:|------------------:|-----------:|
| **Furniture** | 1.000 | 0.228 | 0.064 |
| **Office Supplies** | 0.228 | 1.000 | 0.426 |
| **Technology** | 0.064 | 0.426 | 1.000 |

### Plot parameters (unchanged)

| Parameter | Value |
|-----------|------:|
| **w** | 0.35 |
| **α** | 0.05 |
| **df** | 24 |
| **t** (one-sided) | ≈ 1.711 |

Meta-objective: **x′y\* − w · t · √(x′Vx)**. On this scale, **meta can be negative** when the safety-stock term dominates forecast sales (still valid for ranking portfolios).

### Baseline mask (marginal-add panel)

**x = [1, 1, 0]** → Furniture and Office Supplies on, Technology off.

| Metric | Value |
|--------|------:|
| **x′y\*** | 18,452.29 |
| **√(x′Vx)** | ≈ 42,004.66 |
| Relative risk | ≈ 2.276 |

### Marginal add (Δsales − Δrisk, unit weights)

Only **Technology** is off; adding it:

| Add | Score |
|-----|------:|
| Technology | ≈ **−3,033.63** |

So under unit weights the marginal move **raises** forecast total but **raises** nominal risk by more (best score is still Technology, as the only candidate).

### All **2³ − 1 = 7** non-empty portfolios

| Quantity | Min | Max |
|----------|----:|----:|
| **x′y\*** | 158.0 | 26,950.77 |
| **√(x′Vx)** | ≈ 2,700.66 | ≈ 53,536.77 |
| Relative risk (finite) | ≈ 1.975 | ≈ 17.093 |
| Meta-objective | ≈ −6,700.47 | ≈ −1,459.18 |

**Extremes:**  
- **Max forecast sales:** **[1,1,1]**, **x′y\* = 26,950.77**.  
- **Lowest relative risk:** **[0,1,1]** (Office + Technology only), relative **≈ 1.975**, **x′y\* = 26,792.77**.

### Bottom-right panel — top portfolios by **x′y\***

Only seven bars (all non-empty masks). **P*k*** = portfolio index **k** in `1 … 2^n−1` (row **k−1** in the enumerator).

| Rank | **k** | **x′y\*** | **√(x′Vx)** | Relative risk | Meta-objective |
|-----:|------:|----------:|------------:|---------------:|---------------:|
| 1 | 7 | 26,950.77 | 53,536.77 | 1.9865 | −5,107.51 |
| 2 | 6 | 26,792.77 | 52,925.82 | 1.9754 | −4,899.67 |
| 3 | 3 | 18,452.29 | 42,004.66 | 2.2764 | −6,700.47 |
| 4 | 2 | 18,294.29 | 41,305.80 | 2.2579 | −6,439.98 |
| 5 | 5 | 8,656.48 | 20,223.10 | 2.3362 | −3,453.29 |
| 6 | 4 | 8,498.48 | 19,871.05 | 2.3382 | −3,400.48 |
| 7 | 1 | 158.00 | 2,700.66 | 17.0928 | −1,459.18 |

---

## Kaggle pipeline (reference)

- **Columns:** `Order Date`, `Product Category`, `Order Total`.
- **Dates:** `dayfirst=True`.
- **Panel:** monthly sum of `Order Total` by category.
- **y\*:** last month level; **V:** sample covariance of month-to-month **differences**.
- **Code:** [`src/retail_portfolio/data/kaggle_retail_insights.py`](../src/retail_portfolio/data/kaggle_retail_insights.py).

---

## Synthetic mode (`--synthetic`)

Six RNG categories (seed **42**); **not** what the default PNG uses. Summary tables for regression testing and comparison.

### Categories (y\* and residual std **s_i** in synthetic generator)

| Category | **y\***_i | **s**_i |
|----------|----------:|--------:|
| Cat 1 | 420.0 | 120.0 |
| Cat 2 | 380.0 | 95.0 |
| Cat 3 | 290.0 | 80.0 |
| Cat 4 | 510.0 | 110.0 |
| Cat 5 | 240.0 | 70.0 |
| Cat 6 | 180.0 | 55.0 |

### **ρ** (synthetic, rounded)

|  | C1 | C2 | C3 | C4 | C5 | C6 |
|--|-----|-----|-----|-----|-----|-----|
| **C1** | 1.000 | 0.330 | 0.375 | 0.379 | 0.281 | 0.408 |
| **C2** | 0.330 | 1.000 | 0.293 | 0.312 | 0.235 | 0.434 |
| **C3** | 0.375 | 0.293 | 1.000 | 0.298 | 0.303 | 0.208 |
| **C4** | 0.379 | 0.312 | 0.298 | 1.000 | 0.302 | 0.340 |
| **C5** | 0.281 | 0.235 | 0.303 | 0.302 | 1.000 | 0.323 |
| **C6** | 0.408 | 0.434 | 0.208 | 0.340 | 0.323 | 1.000 |

### Baseline **x = [1,1,1,0,0,0]**

| Metric | Value |
|--------|------:|
| **x′y\*** | 1,090.0 |
| **√(x′Vx)** | ≈ 221.37 |
| Relative risk | ≈ 0.2031 |

### Marginal adds (synthetic)

| Add | Score |
|-----|------:|
| Cat 4 | ≈ 443.49 |
| Cat 5 | ≈ 206.14 |
| Cat 6 | ≈ 148.81 |

### 63 portfolios — ranges

| Quantity | Min | Max |
|----------|----:|----:|
| **x′y\*** | 180.0 | 2,020.0 |
| **√(x′Vx)** | 55.0 | ≈ 353.61 |
| Relative risk | ≈ 0.1721 | ≈ 0.3056 |
| Meta-objective | ≈ 147.07 | ≈ 1,808.26 |

### Synthetic top 12 by **x′y\***

| Rank | **k** | **x′y\*** | **√(x′Vx)** | Relative risk | Meta-objective |
|-----:|------:|----------:|------------:|---------------:|---------------:|
| 1 | 63 | 2,020.0 | 353.61 | 0.1751 | 1,808.26 |
| 2 | 31 | 1,840.0 | 321.97 | 0.1750 | 1,647.20 |
| 3 | 47 | 1,780.0 | 319.00 | 0.1792 | 1,588.98 |
| 4 | 59 | 1,730.0 | 310.67 | 0.1796 | 1,543.97 |
| 5 | 61 | 1,640.0 | 299.76 | 0.1828 | 1,460.50 |
| 6 | 62 | 1,600.0 | 275.33 | 0.1721 | 1,435.13 |
| 7 | 15 | 1,600.0 | 287.87 | 0.1799 | 1,427.62 |
| 8 | 27 | 1,550.0 | 277.44 | 0.1790 | 1,383.87 |
| 9 | 55 | 1,510.0 | 287.07 | 0.1901 | 1,338.10 |
| 10 | 43 | 1,490.0 | 276.82 | 0.1858 | 1,324.24 |
| 11 | 29 | 1,460.0 | 270.21 | 0.1851 | 1,298.19 |
| 12 | 30 | 1,420.0 | 244.57 | 0.1722 | 1,273.55 |

---

## How to reproduce

```bash
# Refresh PNG + recompute numbers from data/raw/data.csv
cd /path/to/retail-portfolio-optimization
PYTHONPATH=src python scripts/visualize_brzezcek_demo.py

# Synthetic figure only
PYTHONPATH=src python scripts/visualize_brzezcek_demo.py --synthetic
```

---

## Related code

| Piece | Location |
|-------|----------|
| Figure + synthetic helper | `src/retail_portfolio/viz/brzezcek_demo.py` |
| Kaggle adapter | `src/retail_portfolio/data/kaggle_retail_insights.py` |
| Portfolio math | `src/retail_portfolio/brzezcek_portfolio.py` |
| CLI | `scripts/visualize_brzezcek_demo.py` |
