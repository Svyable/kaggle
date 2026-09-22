# Svyable Kaggle Lab

Competition research repository for **ARC Prize 2026 — ARC-AGI-3**.

The immediate goal is not to prompt-engineer a larger language model. It is to test whether a compact, non-autoregressive **action-consequence world model** can learn unfamiliar game mechanics online and spend environment actions more efficiently than an autoregressive controller.

## Operating principles

- **Main stays runnable.** Experimental work lands through focused branches/PRs with tests.
- **Decisions, not strings.** Prefer `state + candidate action -> predicted consequences` over generating long reasoning traces before every move.
- **Hold out whole game families.** Do not validate on random transitions from games seen in training.
- **Leaderboard is an external check, not a training signal.**
- **One expensive submission deserves many cheap local tests.**
- **Record falsification, not narrative certainty.** Persist verified transitions and contradictions; avoid free-form memories that can preserve a wrong theory.
- **Reproducibility is part of the solution.** Any serious candidate should run offline inside the Kaggle notebook constraints.

## Current research ladder

1. **E0** — deterministic online system-identification harness.
2. **E1** — grid-native action-conditioned dynamics model.
3. **E2** — Laya-style typed-decision comparison on identical splits.
4. **E3** — batch counterfactual scoring + confidence gate.
5. **E4** — procedural meta-training across hidden mechanics.
6. **E5** — uncertainty-triggered open-weight System 2.

See `EXPERIMENTS.md` as the source of truth once the Phase-0 branch lands.

## Competition constraints that shape the design

ARC-AGI-3 exposes a small discrete action space, 2-D grid observations, hidden mechanics, interactive multi-step tasks, and a score that rewards completing levels with few environment actions. The Kaggle submission environment is offline and runtime constrained, so the architecture must be both adaptive and operationally efficient.

## Repository workflow

Research changes should include:
- a test or measurable experiment,
- the metric being optimized,
- the held-out split used,
- the result (including negative results),
- any effect on runtime, VRAM, deaths/resets, or action efficiency.

We will prefer objective CI/test gates and mergeable changes over process-heavy review requirements.
