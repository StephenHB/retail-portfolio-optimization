# Brzęczek (2020): summary and algorithm sketches

**Paper**: *Optimisation of product portfolio sales and their risk subject to product width and diversity*  
**Author**: Tomasz Brzęczek  
**Venue**: *Review of Managerial Science* 14(5), 1009–1027 (2020)  
**DOI**: [10.1007/s11846-018-0315-y](https://doi.org/10.1007/s11846-018-0315-y)

**Local PDF (in repo)**: [Brzezcek_2020_RMS_portfolio_sales_risk_width_diversity.pdf](Brzezcek_2020_RMS_portfolio_sales_risk_width_diversity.pdf). This note also draws on the publisher abstract ([IDEAS/RePEc](https://ideas.repec.org/a/spr/rvmgts/v14y2020i5d10.1007_s11846-018-0315-y.html)), the project index in [`README.md`](README.md), and standard portfolio–forecast notation; prefer the PDF for exact equations and aggregation details.

---

## 1. One-paragraph gist

The paper treats **product categories** as portfolio assets: decisions are about **widening or narrowing** the assortment (in the empirical setup, **by one category at a time**). **Risk** is not financial return volatility but **sales forecast error** (prediction error from time-series forecasting). Several **decision models** differ by how expected prediction error enters the objective (e.g. nominal vs relative measures, and profit-style objectives that trade off expected sales against a **safety-stock–style** cost driven by forecast uncertainty). Empirical work uses a wholesaler’s quarterly sales and forecasts; a recurring empirical pattern reported in the abstract is that **expansion** tends to increase **forecasted sales** and **nominal** risk while **lowering relative** risk.

---

## 2. Formal problem (conceptual)

**Indices**: categories \(i = 1,\ldots,n\). **Decision**: binary inclusion \(x_i \in \{0,1\}\) (category \(i\) in the portfolio or not). Let \(\mathbf{x} \in \{0,1\}^n\).

**Forecast level**: \(\mu_i\) = expected / forecast sales (or contribution) for category \(i\) over the relevant horizon; \(\boldsymbol{\mu} = (\mu_1,\ldots,\mu_n)^\top\).

**Forecast-error risk**: let \(\varepsilon_i\) be the forecast error for category \(i\). The paper works from the **covariance of forecast errors** \(\Sigma = \mathrm{Cov}(\boldsymbol{\varepsilon})\) (estimated from residuals after fitting forecasts). For a fixed portfolio \(\mathbf{x}\), a **Markowitz-style** aggregate uncertainty for linearly combined exposure is the quadratic form \(\mathbf{x}^\top \Sigma \mathbf{x}\) (interpretation depends on whether errors are modeled as additive across categories at the total-sales level or combined with fixed weights; the PDF should be checked for the exact aggregation).

**Portfolio sales (mean)**:

\[
S(\mathbf{x}) = \sum_i x_i \mu_i = \mathbf{x}^\top \boldsymbol{\mu}.
\]

**Nominal vs relative risk (typical reading)**:  
- *Nominal* risk can be taken as a monotone function of \(\mathbf{x}^\top \Sigma \mathbf{x}\) (e.g. its square root, a portfolio standard deviation of combined error).  
- *Relative* risk often scales nominal risk by the level of sales, e.g. \(\sqrt{\mathbf{x}^\top \Sigma \mathbf{x}} / S(\mathbf{x})\) (with a small constant to avoid division by zero), capturing “risk per unit of expected sales.”

**Marginal width change (scope emphasized in this repo’s index)**: compare a baseline set \(B = \{i : x_i = 1\}\) to \(B \cup \{j\}\) (add one category) or \(B \setminus \{k\}\) (remove one). Full combinatorial optimization over all \(2^n\) subsets is **not** the paper’s empirical focus; heuristics or MILP would be needed for global optima.

**Profit-style / safety-stock–style objective (as indexed in this project)**: maximize an expression of the form **expected sales minus a cost proportional to forecast uncertainty**, where the cost may use a **\(t\)-distribution critical value** (e.g. for a given confidence level and degrees of freedom) times a measure derived from \(\mathbf{x}^\top \Sigma \mathbf{x}\). Exact formula and units should be copied from the PDF.

---

## 3. Methods / algorithm (bullets)

1. **Forecast** each category’s sales for the decision horizon; collect point forecasts into \(\boldsymbol{\mu}\).
2. **Residuals** from those forecasts \(\Rightarrow\) estimate **\(\Sigma\)** (sample covariance of errors; consider shrinkage or factor structure if \(n\) is large relative to history).
3. **Define objectives** (several models in the paper): minimize prediction-error measure, or maximize sales minus risk penalty, under **binary** \(\mathbf{x}\) and constraints reflecting **width** / **diversity** (see paper).
4. **Evaluate marginal moves**: for “add \(j\)” / “remove \(k\)”, compute \(S(\mathbf{x})\) and the chosen risk functional for \(\mathbf{x}\) before and after the flip.
5. **Empirical split**: reserve part of the time series for **parameter estimation** and part for **out-of-sample forecast testing** (as described in the abstract).

**Complexity**: evaluating one marginal flip is \(O(n^2)\) if \(\Sigma\) is dense (quadratic form); repeated flips are cheap compared to full subset enumeration.

---

## 4. Data needs vs. this repository

| Need | Demo / generic mapping |
|------|-------------------------|
| Panel of sales by category and time | Map Kaggle (or other) sales to **category** grain |
| Out-of-sample forecasts per category | Any time-series or ML forecaster; residuals \(\Rightarrow \Sigma\) |
| Binary assortment decision | Category in / out; not store-SKU mix unless aggregated |

Core optimization code should stay **source-agnostic**: accept `mu`, `Sigma`, and masks \(\mathbf{x}\), not hard-coded column names.

---

## 5. Reference implementations (illustrative Python)

Dependencies: **NumPy** (and **SciPy** only for the \(t\) quantile example). These are **teaching sketches**, not a full replication of every model in the paper.

### 5.1 Portfolio sales and forecast-error variance

```python
import numpy as np


def portfolio_sales(x: np.ndarray, mu: np.ndarray) -> float:
    """x, mu: shape (n,), x binary in {0,1}."""
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return float(x @ mu)


def portfolio_error_variance(x: np.ndarray, sigma: np.ndarray) -> float:
    """Quadratic form x^T Sigma x for forecast-error covariance Sigma."""
    x = np.asarray(x, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return float(x @ sigma @ x)


def portfolio_error_std(x: np.ndarray, sigma: np.ndarray) -> float:
    v = portfolio_error_variance(x, sigma)
    return float(np.sqrt(max(v, 0.0)))
```

### 5.2 Nominal and relative risk (simple definitions)

```python
def nominal_risk_std(x: np.ndarray, sigma: np.ndarray) -> float:
    return portfolio_error_std(x, sigma)


def relative_risk_std_per_sales(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    eps: float = 1e-9,
) -> float:
    sales = portfolio_sales(x, mu)
    if sales <= eps:
        return float("inf")
    return nominal_risk_std(x, sigma) / sales
```

### 5.3 Marginal add / remove one category

```python
def flip_category(x: np.ndarray, index: int, value: int) -> np.ndarray:
    y = np.asarray(x, dtype=int).copy()
    y[index] = value
    return y


def score_marginal_adds(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    sales_weight: float = 1.0,
    risk_weight: float = 1.0,
    risk_use_std: bool = True,
) -> np.ndarray:
    """
    For each category currently out (x_j==0), score adding j:
    higher is better: sales_weight * delta_sales - risk_weight * delta_risk.
    """
    x = np.asarray(x, dtype=int)
    n = x.shape[0]
    base_sales = portfolio_sales(x, mu)

    def risk_fn(xx):
        return (
            portfolio_error_std(xx, sigma)
            if risk_use_std
            else portfolio_error_variance(xx, sigma)
        )

    base_r = risk_fn(x.astype(float))
    scores = np.full(n, -np.inf, dtype=float)
    for j in range(n):
        if x[j] != 0:
            continue
        y = flip_category(x, j, 1).astype(float)
        ds = portfolio_sales(y, mu) - base_sales
        dr = risk_fn(y) - base_r
        scores[j] = sales_weight * ds - risk_weight * dr
    return scores


def best_marginal_add(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    **kwargs,
):
    scores = score_marginal_adds(x, mu, sigma, **kwargs)
    j = int(np.argmax(scores))
    if not np.isfinite(scores[j]) or scores[j] == -np.inf:
        return None, float("-inf")
    return j, float(scores[j])  # (best_index_or_None, score)
```

### 5.4 Meta-objective: sales minus \(t\)-scaled uncertainty (template)

Matches the *idea* described in the project README (confirm coefficients with the PDF).

```python
from scipy import stats


def meta_objective_sales_minus_t_buffer(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    cost_per_unit_sigma: float = 1.0,
    df: float = 30.0,
    alpha: float = 0.05,
) -> float:
    """
    Example: sales - cost * t_{df,1-alpha/2} * sqrt(x^T Sigma x).
    Two-sided critical value; adjust to the paper's one-sided choice if needed.
    """
    x = np.asarray(x, dtype=float)
    sales = portfolio_sales(x, mu)
    err_std = portfolio_error_std(x, sigma)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    buffer = cost_per_unit_sigma * t_crit * err_std
    return float(sales - buffer)
```

### 5.5 Tiny sanity check (optional)

```python
if __name__ == "__main__":
    mu = np.array([10.0, 20.0, 5.0])
    sigma = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 9.0, 0.5],
            [0.0, 0.5, 1.0],
        ]
    )
    x0 = np.array([1, 0, 1], dtype=int)
    j, s = best_marginal_add(x0, mu, sigma)
    print("best add index:", j, "score:", s)
    print("meta objective x0:", meta_objective_sales_minus_t_buffer(x0.astype(float), mu, sigma))
```

---

## 6. Suggested next steps (coding / validation)

1. Add the PDF locally and **reconcile** notation (especially how \(\Sigma\) is built and how errors aggregate to total sales).
2. Move stable functions from the blocks above into `src/` with **tests** (e.g. quadratic form symmetry, marginal add monotonicity on toy \(\Sigma\)).
3. Wire \(\boldsymbol{\mu}\) and \(\Sigma\) from the project’s **pluggable loaders** after category-level panels exist.

---

## 7. Keywords (from publisher listing)

Sales forecasting; Portfolio theory; Product diversity decision; Product category; Risk analysis.
