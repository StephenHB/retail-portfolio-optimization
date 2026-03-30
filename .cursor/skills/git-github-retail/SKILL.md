---
name: git-github-retail
description: Performs git workflows for this repository—branches, status, diff, add, commit, log, rebase guidance, PR-ready summaries. Use when the user asks for commits, branches, history, or GitHub-oriented prep.
---

# GitHub agent — version control

## Behavior

- **Execute** git commands the user requests (status, diff, branch, add, commit, merge, rebase) using the project repo; explain each step briefly.
- **Commits**: Imperative subject line (~50 chars); body explains why if non-obvious. Group related changes.
- **Branches**: Use descriptive names (`feature/…`, `fix/…`, `initialization`, etc.).
- **Remote**: Do not push, set remotes, or use credentials unless the user explicitly asks; if push fails, report the error and options.
- **Destructive**: No `reset --hard`, `push --force`, or branch deletion unless the user clearly requests it.

## PR helper

When asked for a PR description: summarize scope, testing done, and breaking changes.

## Anti-patterns

- Large unrelated changes in one commit without user consent.
- Force-pushing shared branches without explicit approval.
