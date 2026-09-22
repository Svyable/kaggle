# Research workflow

The repository is optimized for fast, falsifiable competition research.

1. Start from `main` and keep the change narrowly tied to one experiment or infrastructure improvement.
2. State the hypothesis and pass/fail metric before running an expensive experiment.
3. Add or update tests for controller invariants and serialization/data contracts.
4. Record the result in `RESULTS.md`, including failures and regressions.
5. Do not spend a Kaggle submission merely to debug packaging; local/offline validation must pass first.
6. Prefer objective CI/test gates and reversible changes. Human review is useful when it adds technical value, not as a mandatory process blocker.

Never commit Kaggle credentials, private competition data, proprietary model weights, or secrets.
