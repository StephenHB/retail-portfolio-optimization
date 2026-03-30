# retail-portfolio-optimization

Retail portfolio optimization (demo data + research-driven methods). This branch adds Cursor agent rules/skills, documentation, and the first reference paper.

## Cursor agents and rules

- **`AGENTS.md`** — how to use manager, research, coding, validation, and GitHub-oriented workflows in chat.
- **`.cursor/rules/`** — always-on project goals, multi-agent role switching, and pluggable data-loading conventions (see `data-loading-extensible.mdc` when editing Python).
- **`.cursor/skills/`** — per-role `SKILL.md` files (research, coding, manager, validation, git).

## Demo data (Kaggle)

Primary demo dataset: **[Retail Insights: A Comprehensive Sales Dataset](https://www.kaggle.com/datasets/rajneesh231/retail-insights-a-comprehensive-sales-dataset)** (author: `rajneesh231` on Kaggle). Download via the [Kaggle API](https://www.kaggle.com/docs/api) or the dataset page and extract `data.csv` under `data/raw/` (that folder is gitignored except `.gitkeep`). The Brzęczek demo figure then defaults to this file: `PYTHONPATH=src python scripts/visualize_brzezcek_demo.py` (use `--synthetic` for the RNG-only demo).

## Research PDFs

Put papers in **`research/papers/`**. **Index, citations, and summaries**: [`research/papers/README.md`](research/papers/README.md) (currently includes Brzęczek 2020 on portfolio width, forecast risk, and safety-stock-style objectives).