# Experiment results

This file is the permanent experiment ledger. Append results; do not rewrite history to make a branch look cleaner.

| ID | Date | Commit | Train families | Held-out families | Primary metric | Result | Runtime / VRAM | Decision |
|---|---|---|---|---|---|---|---|---|
| E0a | 2026-09-22 | `9593e8c` | none | 25 public ARC environments | behavioral harness validity | **failed**: process completed but controller looped on RESET; 0 levels | CPU; 80-action cap/game | revise |
| E0b | 2026-09-22 | `9780fcc` | none | 25 public ARC environments | behavioral harness validity | **pass**: 25/25 executed non-RESET actions; 2,025 actions; 1 level completed; aggregate 0.0557484568 | CPU; no model VRAM; 80-action cap/game | proceed to data/E1 |
| D0 | 2026-09-22 | `fcd8ee5` | 22 public game families | sc25, sk48, tu93 | action-aligned data validity + persistence baseline | **pass**: 2,012 transitions; held-out persistence 99.30% cells / 0% changed cells / 12.86% exact grids | CPU collection/audit; no training VRAM | proceed to first real E1 train run |

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
