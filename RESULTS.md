# Experiment results

This file is the permanent experiment ledger. Append results; do not rewrite history to make a branch look cleaner.

| ID | Date | Commit | Train families | Held-out families | Primary metric | Result | Runtime / VRAM | Decision |
|---|---|---|---|---|---|---|---|---|
| E0a | 2026-09-22 | `9593e8c` | none | 25 public ARC environments | behavioral harness validity | **failed**: process completed but controller looped on RESET; 0 levels | CPU; 80-action cap/game | revise |
| E0b | 2026-09-22 | `9780fcc` | none | 25 public ARC environments | behavioral harness validity | **pass**: 25/25 executed non-RESET actions; 2,025 actions; 1 level completed; aggregate 0.0557484568 | CPU; no model VRAM; 80-action cap/game | proceed to data/E1 |
| D0 | 2026-09-22 | `fcd8ee5` | 22 public game families | sc25, sk48, tu93 | action-aligned data validity + persistence baseline | **pass**: 2,012 transitions; held-out persistence 99.30% cells / 0% changed cells / 12.86% exact grids | CPU collection/audit; no training VRAM | proceed to first real E1 train run |
| E1a | 2026-09-22 | `ada0ef0` | 22 public game families | sc25, sk48, tu93 | next-state vs persistence | **fail**: 22.01% changed cells, but 0% exact grids and 98.40% cells vs persistence 0% / 12.86% / 99.30% | CPU; width32 depth2; 6 epochs | revise loss toward sparse edits; do not start Laya |

## Result template

### EX — short name

- **Hypothesis:**
- **Commit:**
- **Data / split:**
- **Baseline:**
- **Primary metric:**
- **Secondary metrics:**
- **Result:**
- **Failure analysis:**
- **Decision:** proceed / revise / stop
- **Artifacts:**


### E0a — false-green reset loop

- **Hypothesis:** The Phase-0 controller can execute safely across the official public ARC environment set.
- **Commit:** `9593e8ce81599e7ad2797efbf51571c3d0e6d479`
- **Data / split:** all 25 public environments exposed by the official starter on 2026-09-22.
- **Primary metric:** behavioral harness validity.
- **Result:** process-level run completed, but every game emitted RESET repeatedly to the action cap and completed 0 levels.
- **Failure analysis:** `FrameData.full_reset=True` was incorrectly interpreted as a request to issue RESET. Upstream semantics show it describes the RESET that produced the current observation. This created a self-sustaining reset loop.
- **Decision:** revise; do not count process exit as E0 success.
- **Artifacts:** GitHub Actions run https://github.com/Svyable/kaggle/actions/runs/35682521461

### E0b — corrected official-starter all-public gate

- **Hypothesis:** After correcting `full_reset` semantics, the deterministic controller can execute legal non-RESET actions across every public environment without integration failures.
- **Commit:** `9780fcca1ea02756715a86916601030fccae371a`
- **Data / split:** all 25 public environments exposed by the official starter; no training.
- **Baseline:** E0 heuristic only.
- **Primary metric:** 25/25 games must execute at least one ACTION1..ACTION7 and finish the bounded runner without exceptions or illegal ACTION6 payloads.
- **Secondary metrics:** total actions, levels completed, aggregate score.
- **Result:** **pass**. 25/25 games passed the behavioral log validator; 2,025 actions were executed; `lp85` completed 1 level; aggregate score was 0.055748456790123455. The two-game official smoke also passed the same validator (102 actions, 0 levels).
- **Failure analysis:** no integration failure in the corrected run. Strength remains intentionally weak; E0 is plumbing, not a competitive solver.
- **Decision:** proceed to trajectory collection and E1 evaluation. Do not treat this score as evidence for the System-One hypothesis.
- **Artifacts:** full run https://github.com/Svyable/kaggle/actions/runs/35720668766 ; smoke run https://github.com/Svyable/kaggle/actions/runs/35720668929


### D0 — action-aligned public trajectory dataset

- **Hypothesis:** We can collect trustworthy action-aligned supervision directly from current public ARC environments without relying on the framework recorder's missing `action_input`.
- **Commit:** `fcd8ee5108bda6fbe88facd82cde93ea30bf2d1b`
- **Data / split:** 2,012 non-RESET transitions across all 25 public game families. Deterministic family split `arc3-systemone-v1`: 22 train families / 1,771 transitions; 3 held-out families (`sc25`, `sk48`, `tu93`) / 241 transitions.
- **Action distribution:** ACTION1 219; ACTION2 207; ACTION3 188; ACTION4 185; ACTION5 91; ACTION6 1,104; ACTION7 18.
- **Outcomes:** 58,434 changed cells, 13 death transitions, 1 progress transition.
- **Baseline:** persistence/identity next-frame.
- **Held-out persistence:** cell accuracy 0.9929867819631742; changed-cell accuracy 0.0; exact-grid accuracy 0.12863070539419086.
- **Interpretation:** overall cell accuracy is dominated by unchanged cells and is not a useful success metric by itself. Changed-cell and exact-grid metrics remain mandatory for E1.
- **Result:** **pass**. The full artifact passed the repository's actual E1 JSONL loader and deterministic family-split audit with no train/validation family leakage.
- **Decision:** proceed to the first real E1 training run. This is a data/plumbing result, not evidence that the learned world model beats persistence.
- **Artifacts:** full collection https://github.com/Svyable/kaggle/actions/runs/35721678147 ; E1 loader/audit https://github.com/Svyable/kaggle/actions/runs/35721938631


### E1a — first real grid-native dynamics training run

- **Hypothesis:** A small action-conditioned grid model trained with dense next-grid cross-entropy can beat persistence on held-out next-state prediction.
- **Commit:** `ada0ef0f455c4a197e54e9ddf5569fe231c8e954`
- **Data / split:** D0; 1,771 train transitions from 22 families; 241 held-out transitions from `sc25`, `sk48`, `tu93`.
- **Configuration:** width 32; depth 2; conditioning dim 64; 6 epochs; batch 32; AdamW 3e-4; seed 7; CPU.
- **Baseline:** held-out persistence: cell accuracy 0.9929867819631742; changed-cell accuracy 0.0; exact-grid accuracy 0.12863070539419086.
- **Model result:** validation loss 0.3608991142625136; cell accuracy 0.9840234780212656; changed-cell accuracy 0.22013577928643652; exact-grid accuracy 0.0.
- **Scalar heads:** change Brier 0.08379098027944565 / ECE 0.07751662284135818; death Brier 0.008286096155643463 / ECE 0.006728747859597206. Validation has no positive progress examples, so the tiny progress Brier/ECE does not demonstrate useful progress prediction.
- **Edit diagnostic:** held-out target has 6,923 changed cells. The model predicted 12,825 cells as edits; 10,372 were false edits. Edit-location precision was 0.19126705653021442; edit-location recall 0.35432615917954646; 1,524 changed cells were predicted with the correct new color (0.22013577928643652 recall).
- **Training behavior:** changed-cell accuracy was already 0.24656940632673696 after epoch 1 while exact-grid accuracy stayed 0.0 at every epoch; optimizing the existing validation loss did not solve false edits.
- **Result:** **fail** for the provisional next-state gate. The model learned nontrivial dynamics that persistence cannot (22% correct changed cells), but it over-edited unchanged cells badly enough to lose whole-grid fidelity and overall cell accuracy.
- **Failure analysis:** only about 0.71% of D0 grid cells change even though most transitions contain some change. Dense CE therefore mixes a sparse edit-prediction problem with a dominant identity-copy problem. The observed error pattern is specifically low edit precision, not absence of learned dynamics.
- **Decision:** keep the same split and model size for E1b. Change the objective first: explicitly supervise sparse per-cell change and penalize unnecessary edits while retaining next-color CE. Do not increase capacity or begin E2/Laya yet.
- **Artifacts:** training run https://github.com/Svyable/kaggle/actions/runs/35722484655 ; artifact `e1-first-real-train`.
