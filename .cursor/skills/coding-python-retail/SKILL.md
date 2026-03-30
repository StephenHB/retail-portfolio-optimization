---
name: coding-python-retail
description: Implements and refactors Python for retail portfolio optimization—loaders, models, utilities, notebook imports. Use when writing features, fixing bugs, structuring src/, or integrating research ideas into code.
---

# Coding agent — Python implementation

## Behavior

- **Scope**: Touch only files and functions needed for the task; follow existing naming and layout.
- **Data**: Route I/O through adapters or shared loader entry points (see project data-loading rule); no Kaggle-only paths in core logic.
- **Style**: PEP 8; type hints where the repo already uses them; vectorized pandas/numpy for heavy loops.
- **Collaboration**: If research agent output exists in-thread, align variable names and module boundaries with that plan.

## Deliverables

- Code edits with a short summary of behavior change.
- If new public functions: minimal docstring (args, returns, side effects).
- Point validation agent at new behavior (edge cases, invariants).

## Anti-patterns

- Large copy-paste into notebooks instead of importable modules.
- Silent exception swallowing; bare `except:`.
