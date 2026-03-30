# Research papers (local)

PDFs in this folder are project references. Prefer citing the canonical copy here: `*.pdf`.

## 1. Brzęczek (2020) — product portfolio sales and forecast risk

- **File**: [Brzezcek_2020_RMS_portfolio_sales_risk_width_diversity.pdf](Brzezcek_2020_RMS_portfolio_sales_risk_width_diversity.pdf) (tracked in git; open from the repo or GitHub file view)  
- **Title**: *Optimisation of product portfolio sales and their risk subject to product width and diversity*  
- **Venue**: *Review of Managerial Science* 14, 1009–1027 (2020)  
- **DOI**: [10.1007/s11846-018-0315-y](https://doi.org/10.1007/s11846-018-0315-y)  
- **Setting**: Wholesaler / multi-category retail-style assortment; empirical quarterly sales and forecasts.

**Core idea**: Extend Markowitz-style portfolio thinking to **product categories** (binary include/exclude decisions), with **risk** defined via **forecast error** (residuals from time-series models), not financial returns. Models minimise nominal or **relative** forecast error, or maximise a **profit-style meta-objective** (forecasted sales minus a safety-stock cost linked to forecast error and a \(t\)-distribution critical value).

**Relevance to this repo**: Pairs naturally with **category-level** sales panels (e.g. Kaggle retail demo mapped to categories); implementation needs **forecasts + residual covariance** per category and horizon, then small **binary quadratic** problems (expand/contract by one category at a time in the paper). **Code:** `src/retail_portfolio/`; **demo data notes:** [`data/README.md`](../../data/README.md).

**Limitation called out in paper**: Results are for **marginal** width changes; full combinatorial assortment search would need extra structure or heuristics.
