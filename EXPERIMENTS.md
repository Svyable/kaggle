# ARC-AGI-3 System-One experiment ladder

The purpose of this branch is to test a single claim in increasingly expensive steps:

> A compact, non-autoregressive action-consequence model can spend ARC environment actions more efficiently than an autoregressive LLM controller, while a slower reasoner is invoked only when uncertainty is genuinely high.

## Metric hierarchy

Do not optimize public leaderboard score first. Record, in order:

1. levels completed per game;
2. RHAE-compatible action efficiency on completed levels;
3. environment actions per newly learned mechanic;
4. repeated exact-state no-ops;
5. deaths/resets;
6. wall-clock seconds per environment action;
7. calibration of `p_change`, `p_progress`, and `p_death`;
8. held-out-game score.

The public leaderboard is a final external check, not the training signal.

## E0 — valid deterministic harness

Use `agent/my_agent.py` unchanged in the official Kaggle starter.

Pass conditions:
- no exceptions on every locally available environment;
- no illegal ACTION6 coordinates;
- every played game executes at least one non-RESET action after initialization;
- proven exact-state no-ops are not repeated;
- identical starting observations produce identical first decisions;
- `full_reset` clears stale evidence without causing a RESET loop;
- logs contain the top candidate scores and the observed consequence of the previous action.

The official-starter CI validates behavior from the run log; a process that merely reaches MAX_ACTIONS while looping on RESET is a failure.

This establishes the plumbing. It is not expected to be a strong solver.

## E1 — consequence-model sanity test

Train `research/arc_decision_model.py` on trajectories from a subset of environments.
Hold out complete *game families*, not random transitions.

Required targets:
- next grid;
- changed/not-changed;
- level progress;
- game-over.

Go/no-go criteria on held-out game families:
- next-state prediction beats an identity-next-frame baseline;
- change/death probabilities are measurably calibrated after temperature scaling;
- ranking candidate actions by model utility beats the E0 heuristic on completed levels or action efficiency without increasing deaths materially.

If E1 fails, do not spend time porting Laya.

## E2 — Laya representation test

Fine-tune Laya on the same transition split, but serialize each state into compact structured text/JSON and ask typed questions about candidate actions.

Compare against E1 on:
- held-out games;
- latency;
- VRAM;
- calibration;
- action efficiency.

Proceed with Laya only if it materially beats the small grid-native model. The default assumption is that the grid-native inductive bias should win.

## E3 — counterfactual batch scoring

For every legal action, and every structurally generated ACTION6 coordinate, score all candidates in one GPU batch.

Add a confidence gate:
- high confidence: act immediately;
- low confidence: invoke the slow planner/reasoner;
- disagreement between dynamics and planner: prefer a low-risk information-gathering action.

Measure the fraction of turns requiring System 2. Target: <=15% after early exploration on a level.

## E4 — procedural meta-training

Generate game families with randomized visual surfaces and hidden mechanics. Hold out entire mechanic compositions.

Curriculum:
1. navigation and collision;
2. key/lock and button/door causal chains;
3. collect/transform/place;
4. toggles and reversible state;
5. hazards and delayed consequences;
6. coordinate selection;
7. partial observability / memory;
8. multi-level rule reuse;
9. compositional mixtures of the above.

Training objective is system identification, not game-ID classification. Randomize colors, layouts, sprite shapes, and action-to-effect mappings aggressively.

## E5 — System 2 integration

Only after E1/E3 are positive, add an offline open-weight reasoner. Give it symbolic summaries, action predictions, contradictions, and trajectory memory rather than raw full chat history.

System 2 may:
- propose goal hypotheses;
- identify discriminating experiments;
- propose short plans;
- update mechanic beliefs.

System 2 should not directly emit an unchecked environment action. The System-One controller validates and executes.

## Submission gate

A candidate consumes the single daily Kaggle submission only if:
- local all-game run completes;
- runtime projects below the notebook cap with margin;
- no dependency needs internet;
- all weights are bundled/accessible offline with compatible licensing;
- deterministic replay test passes;
- held-out regression suite does not regress materially.
