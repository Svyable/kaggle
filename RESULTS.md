# Experiment results

This file is the permanent experiment ledger. Append results; do not rewrite history to make a branch look cleaner.

| ID | Date | Commit | Train families | Held-out families | Primary metric | Result | Runtime / VRAM | Decision |
|---|---|---|---|---|---|---|---|---|
| E0a | 2026-09-22 | `9593e8c` | none | 25 public ARC environments | behavioral harness validity | **failed**: process completed but controller looped on RESET; 0 levels | CPU; 80-action cap/game | revise |
| E0b | 2026-09-22 | `9780fcc` | none | 25 public ARC environments | behavioral harness validity | **pass**: 25/25 executed non-RESET actions; 2,025 actions; 1 level completed; aggregate 0.0557484568 | CPU; no model VRAM; 80-action cap/game | proceed to data/E1 |

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
