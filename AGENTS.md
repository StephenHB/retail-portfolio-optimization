# Multi-agent setup — retail portfolio optimization

This repository uses **logical agent roles** in Cursor. The same assistant adopts a role based on your request. Name the role explicitly when you want focused behavior (for example: “As the validation agent, …”).

## Roles

| Role | Purpose |
|------|--------|
| **Manager** | Breaks your goal into steps, suggests which other roles to use and in what order, and keeps context consistent across the thread. |
| **Research** | Explains papers: problem setup, notation, assumptions, algorithms, and how ideas map to code or data in this project. |
| **Coding** | Implements and refactors Python: data layer, models, notebooks glue code; aligns with project rules and existing patterns. |
| **Validation** | After substantive code changes, runs or proposes tests, checks edge cases, and verifies behavior against requirements. |
| **GitHub** | Performs git workflows you request: branches, commits, diffs, rebases, PR-oriented messaging (no remote actions without credentials). |

## Default workflow

1. **Manager** (optional): For multi-step work, start with “Act as manager: plan …”.  
2. **Research** (optional): For papers or theory, “Act as research agent: explain …”.  
3. **Coding**: Implementation.  
4. **Validation**: Before you consider work done, ask the validation agent to test.  
5. **GitHub**: When ready to snapshot or share work.

## Project skills

Role-specific workflows live under `.cursor/skills/` (see each `SKILL.md`). Cursor loads them when the task matches the skill description.

## Research PDFs

Store papers under `research/papers/`. Indexed entries and summaries: `research/papers/README.md`. Demo sales data: root **README** and **`data/README.md`** (Kaggle *Retail Insights* schema, `data/raw/data.csv`).

## Data philosophy

Demo data may come from Kaggle retail sales; **all loading must go through a pluggable layer** (adapters, configs, or registries) so new sources do not require rewriting core logic. See `.cursor/rules/data-loading-extensible.mdc`.
