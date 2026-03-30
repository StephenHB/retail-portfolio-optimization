"""
Build a multi-panel figure from synthetic category forecasts and residual covariance V.

Uses the same primitives as `retail_portfolio.brzezcek_portfolio` so the plot reflects
actual library outputs (not hand-tuned curves).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from retail_portfolio.brzezcek_portfolio import (
    covariance_from_std_correlation,
    meta_objective_profit_safety_stock,
    portfolio_forecast_total,
    portfolio_residual_std,
    relative_forecast_error_risk,
    score_marginal_adds,
    t_critical_student,
)


def synthetic_demo_inputs(
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Return (y_star, V, corr, labels) for a small illustrative category system.

    Correlation has mild complementarity (positive off-diagonal) like the paper’s
    discussion of related categories.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = 6
    labels = [f"Cat {i + 1}" for i in range(n)]
    std = np.array([120.0, 95.0, 80.0, 110.0, 70.0, 55.0])
    base = rng.uniform(0.15, 0.45, size=(n, n))
    corr = (base + base.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    # ensure PSD-ish for demo
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 0.05)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.clip(np.diag(corr), 1e-9, None))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    v = covariance_from_std_correlation(std, corr)
    y_star = np.array([420.0, 380.0, 290.0, 510.0, 240.0, 180.0])
    return y_star, v, corr, labels


def _all_nonempty_portfolios(n: int) -> np.ndarray:
    """Shape (2**n - 1, n) float 0/1 masks."""
    out = []
    for k in range(1, 2**n):
        bits = np.array([float((k >> i) & 1) for i in range(n)])
        out.append(bits)
    return np.stack(out, axis=0)


def make_brzezcek_demo_figure(
    y_star: np.ndarray,
    v: np.ndarray,
    corr: np.ndarray,
    labels: list[str],
    *,
    baseline_x: np.ndarray | None = None,
    w_safety: float = 0.35,
    alpha_t: float = 0.05,
    df_t: float = 24.0,
):
    """
    Create a 2x2 matplotlib figure: correlation heatmap, sales vs risk scatter,
    marginal-add scores, relative risk bars for top portfolios by forecast sales.

    Parameters
    ----------
    baseline_x :
        Binary mask for marginal-add panel; default first three categories on.
    """
    import matplotlib.pyplot as plt

    n = len(labels)
    if baseline_x is None:
        baseline_x = np.array([1, 1, 1, 0, 0, 0][:n], dtype=int)
    baseline_x = np.asarray(baseline_x, dtype=int).reshape(-1)
    if baseline_x.shape[0] != n:
        raise ValueError("baseline_x length must match labels")

    masks = _all_nonempty_portfolios(n)
    sales = np.array([portfolio_forecast_total(x, y_star) for x in masks])
    risk = np.array([portfolio_residual_std(x, v) for x in masks])
    k_in = masks.sum(axis=1)
    rel = np.array(
        [
            relative_forecast_error_risk(x, y_star, v)
            if sales[i] > 1e-9
            else np.nan
            for i, x in enumerate(masks)
        ]
    )
    t_crit = t_critical_student(df_t, alpha_t, one_sided_upper=True)
    meta = np.array(
        [
            meta_objective_profit_safety_stock(
                x, y_star, v, w=w_safety, t_critical=t_crit
            )
            for x in masks
        ]
    )

    scores = score_marginal_adds(baseline_x, y_star, v)
    inactive = baseline_x == 0
    idx_inactive = np.where(inactive)[0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        "Brzęczek-style demo: library outputs on synthetic categories",
        fontsize=12,
    )

    ax = axes[0, 0]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Residual sales correlation (ρ)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    sc = ax.scatter(
        sales,
        risk,
        c=k_in,
        cmap="viridis",
        alpha=0.85,
        s=36,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.set_xlabel("Forecast portfolio sales (x′y*)")
    ax.set_ylabel("Nominal risk √(x′Vx)")
    ax.set_title("All non-empty binary portfolios (color = # categories on)")
    fig.colorbar(sc, ax=ax, label="|x|", fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    bar_labels = [labels[j] for j in idx_inactive]
    bar_vals = [scores[j] for j in idx_inactive]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in bar_vals]
    ax.bar(bar_labels, bar_vals, color=colors, edgecolor="k", linewidth=0.4)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_ylabel("Score (Δsales − Δrisk)")
    ax.set_title(
        "Marginal add scores from baseline "
        + str(baseline_x.tolist())
        + "\n(higher is better for sales-weight=1, risk-weight=1)"
    )
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 1]
    order = np.argsort(-sales)
    top = order[: min(12, len(order))]
    xlabs = [f"P{k+1}" for k in top]
    rel_top = rel[top]
    m_top = meta[top]
    xpos = np.arange(len(top))
    wbar = 0.35
    ax.bar(
        xpos - wbar / 2,
        rel_top,
        width=wbar,
        label="Relative risk √(x′Vx) / (x′y*)",
        color="#1f77b4",
        alpha=0.85,
    )
    ax2 = ax.twinx()
    ax2.bar(
        xpos + wbar / 2,
        m_top,
        width=wbar,
        label=f"Meta-obj (w={w_safety}, t={t_crit:.2f})",
        color="#ff7f0e",
        alpha=0.85,
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabs, rotation=0)
    ax.set_ylabel("Relative risk")
    ax2.set_ylabel("Meta-objective (currency units)")
    ax.set_title("Top portfolios by forecast sales: relative risk vs meta-objective")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def save_brzezcek_demo(
    out_path: str | Path,
    *,
    dpi: int = 120,
    rng: np.random.Generator | None = None,
) -> Path:
    """Generate synthetic inputs, build figure, save to *out_path* (PNG or PDF)."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_star, v, corr, labels = synthetic_demo_inputs(rng=rng)
    fig = make_brzezcek_demo_figure(y_star, v, corr, labels)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
