# Brzęczek (2020): summary and algorithm sketches

**Paper**: *Optimisation of product portfolio sales and their risk subject to product width and diversity*  
**Author**: Tomasz Brzęczek  
**Venue**: *Review of Managerial Science* 14(5), 1009–1027 (2020)  
**DOI**: [10.1007/s11846-018-0315-y](https://doi.org/10.1007/s11846-018-0315-y)

**Local PDF (in repo)**: [Brzezcek_2020_RMS_portfolio_sales_risk_width_diversity.pdf](Brzezcek_2020_RMS_portfolio_sales_risk_width_diversity.pdf). This note also draws on the publisher abstract ([IDEAS/RePEc](https://ideas.repec.org/a/spr/rvmgts/v14y2020i5d10.1007_s11846-018-0315-y.html)), the project index in [`README.md`](README.md), and standard portfolio–forecast notation; prefer the PDF for exact equations and aggregation details.

**Canonical Python**: [`src/retail_portfolio/`](../../src/retail_portfolio/) (import after `pip install -e .` or `PYTHONPATH=src`). Section 5 code blocks below are kept for reading; the package is the maintained source.

**Math in this file:** inline `$…$` and display `$$ … $$` (works on **github.com** and in **VS Code / Cursor** if math preview is on). In Cursor/VS Code, enable **Settings → search “markdown math” → Markdown › Math: Enabled** (`markdown.math.enabled`), or use a preview extension such as “Markdown+Math”. Commands like `\mathbf`, `\mathrm`, `\top` are standard LaTeX inside those delimiters.

---

## 1. One-paragraph gist

The paper treats **product categories** as portfolio assets: decisions are about **widening or narrowing** the assortment (in the empirical setup, **by one category at a time**). **Risk** is not financial return volatility but **sales forecast error** (prediction error from time-series forecasting). Several **decision models** differ by how expected prediction error enters the objective (e.g. nominal vs relative measures, and profit-style objectives that trade off expected sales against a **safety-stock–style** cost driven by forecast uncertainty). Empirical work uses a wholesaler’s quarterly sales and forecasts; a recurring empirical pattern reported in the abstract is that **expansion** tends to increase **forecasted sales** and **nominal** risk while **lowering relative** risk.

---

## 2. Formal problem (reconciled with Brzęczek Sect. 3 / Table 1)

**Indices**: categories $i = 1,\ldots,n$. **Decision**: binary inclusion $x_i \in \{0,1\}$. The paper splits **reduction** (vector $\mathbf{x}_1$ over incumbent categories) vs **expansion** (stack $[\mathbf{x}_1^\top, \mathbf{x}_2^\top]^\top$ with candidates in $\mathbf{x}_2$); code uses a single $\mathbf{x}$ over whichever index set is active, with $V$ sized accordingly ($I\times I$ or $(I+m)\times(I+m)$).

**Forecast level**: $y_{it}^*$ is the forecast expected sales of category $i$ in period $t$; $\mathbf{y}^*_t$ is the column vector of those forecasts for the relevant set. **Portfolio forecasted sales**: $\mathbf{x}^\top \mathbf{y}^*_t$.

**Residual sales and $V$**: $s_i$ is the standard deviation of **residual** (forecast error) sales for category $i$; $\rho_{ij}$ is the correlation between residuals of $i$ and $j$. The paper defines the variance–covariance matrix $V$ by $v_{ij} = \rho_{ij} s_i s_j$. In code, the same matrix can be built with `covariance_from_std_correlation` or estimated directly as a sample covariance of aligned residual series via `mu_sigma_from_residual_matrix`.

**How errors aggregate to portfolio / total sales**: Let $\varepsilon_i$ denote residual sales for category $i$ (random forecast error). For portfolio $\mathbf{x}$, **total portfolio residual** is $\sum_i x_i \varepsilon_i$. With $\mathrm{Cov}(\boldsymbol{\varepsilon}) = V$,

$$
\mathrm{Var}\Big(\sum_i x_i \varepsilon_i\Big) = \mathbf{x}^\top V \mathbf{x}.
$$

**Nominal risk (models (1)–(2))**: $\sqrt{\mathbf{x}^\top V \mathbf{x}}$. The paper states these objectives estimate the expected error of **product portfolio sales** (or a lower bound thereof).

**Relative risk (models (3)–(4))**: $\sqrt{\mathbf{x}^\top V \mathbf{x}} \big/ (\mathbf{x}^\top \mathbf{y}^*_t)$ — relative error of **total sales forecasts**, comparable across portfolios with different variety and forecast levels.

**Marginal width**: constraints $\lVert \mathbf{x}_1 \rVert_1 \ge I-1$ or $\lVert \mathbf{x}_1 \rVert_1 = I$ for reduction and $\lVert \mathbf{x}_2 \rVert_1 \le 1$ for expansion (at most one category flipped). Compare before/after flips; full $2^n$ search is outside the paper’s empirical scope.

**Profit meta-objective (models (5)–(6))**: $\mathbf{x}^\top \mathbf{y}^*_t - w \, t_{\alpha,\,N-K} \sqrt{\mathbf{x}^\top V \mathbf{x}}$, where $w$ is the average safety-stock cost per currency unit of sales forecast error at service level tied to $1-\alpha/2$ (see paper), and $t_{\alpha,\,N-K}$ is a Student $t$ critical value with $N-K$ d.f. (paper: right-tail at significance $\alpha$; `t_critical_student(..., one_sided_upper=True)` matches that reading; use `one_sided_upper=False` for a two-sided symmetric $\alpha$ critical value).

---

## 3. Methods / algorithm (bullets)

1. **Forecast** each category’s sales for the decision horizon; collect point forecasts into $\mathbf{y}^*_t$.
2. **Residuals** from those forecasts $\Rightarrow$ estimate **$V$** via $v_{ij} = \rho_{ij}\, s_i\, s_j$ or sample covariance of aligned residual series (shrinkage if $n$ is large relative to history).
3. **Define objectives** (several models in the paper): minimize prediction-error measure, or maximize sales minus risk penalty, under **binary** $\mathbf{x}$ and constraints reflecting **width** / **diversity** (see paper).
4. **Evaluate marginal moves**: for “add $j$” / “remove $k$”, compute $S(\mathbf{x})$ and the chosen risk functional for $\mathbf{x}$ before and after the flip.
5. **Empirical split**: reserve part of the time series for **parameter estimation** and part for **out-of-sample forecast testing** (as described in the abstract).

**Complexity**: evaluating one marginal flip is $O(n^2)$ if $V$ is dense (quadratic form); repeated flips are cheap compared to full subset enumeration.

---

## 4. Data needs vs. this repository

| Need | Demo / generic mapping |
|------|-------------------------|
| Panel of sales by category and time | Map Kaggle (or other) sales to **category** grain |
| Out-of-sample forecasts per category | Any time-series or ML forecaster; residuals $\Rightarrow V$ |
| Binary assortment decision | Category in / out; not store-SKU mix unless aggregated |

Core optimization code should stay **source-agnostic**: accept forecast levels (e.g. **`y_star`** / $\mathbf{y}^*$) and covariance **$V$**, plus binary masks $\mathbf{x}$, not hard-coded column names.

---

## 5. Reference implementations (illustrative Python)

Dependencies: **NumPy** (and **SciPy** only for the $t$ quantile example). Prefer **`retail_portfolio`** in `src/` for real use; the blocks below are **teaching sketches** aligned with earlier drafts (names like `mu` / `sigma` map to $\mathbf{y}^*$ and $V$ in the paper).

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

### 5.4 Meta-objective: sales minus $t$-scaled uncertainty (template)

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

## 6. Implementation status (done)

1. **Notation** — Reconciled with the PDF in §2 and in `brzezcek_portfolio.py` (including $V$ from $\rho_{ij}, s_i$, and $\mathrm{Var}(\sum_i x_i \varepsilon_i)=\mathbf{x}^\top V \mathbf{x}$).
2. **Code & tests** — Package [`src/retail_portfolio/`](../../src/retail_portfolio/); tests in `tests/test_brzezcek_portfolio.py` (quadratic form, $V$ from std/corr, **marginal add increases nominal risk** under diagonal $V$, relative risk edge cases, meta-objective, $t$ tails) and `tests/test_forecast_inputs.py` (long $\to$ wide residuals, sample covariance matrix). Run: `PYTHONPATH=src python -m pytest` (or `pip install -e .` with a current pip/setuptools).
3. **Pluggable loaders** — [`forecast_inputs.py`](../../src/retail_portfolio/data/forecast_inputs.py): `wide_residual_matrix_from_long(...)` takes **column names** only; `mu_sigma_from_residual_matrix(forecast_means, residual_matrix)` pairs $\mathbf{y}^*$ with sample $V$. Wire any category–period panel by mapping adapters to those arguments (no Kaggle-specific fields in core logic).

---

## 7. Keywords (from publisher listing)

Sales forecasting; Portfolio theory; Product diversity decision; Product category; Risk analysis.
