# Architecture: ARC System One

## Hypothesis

A controller that predicts **consequences for each legal action** can use ARC-AGI-3 environment interactions more efficiently than a controller that generates free-form reasoning before every action.

The stable interface is:

`(history, state, candidate action) -> consequence distribution -> calibrated utility`

The implementation may change behind that interface without rewriting the competition harness.

## Layers

### 1. Deterministic perception

Input is the exact discrete ARC grid plus recent transitions. Derive cheap structural features such as cell deltas, connected components, candidate click coordinates, repeated states, and legal actions.

### 2. Verified transition ledger

After each environment action, record only observed facts: whether the grid changed, how many cells changed, whether a level advanced, whether the game terminated, and the exact state/action signature. A narrative hypothesis is not promoted to fact merely because a reasoner generated it.

### 3. System-One consequence model

E0 uses smoothed empirical statistics. E1 replaces the scoring function with a grid-native dynamics model that predicts next-grid logits plus change/progress/death probabilities. All legal actions should be scored in one batch when possible.

### 4. Confidence and contradiction gate

Fast execution is appropriate only when the consequence model is sufficiently calibrated. High uncertainty, repeated contradiction, or planner/model disagreement should trigger additional internal computation rather than gratuitous environment actions.

### 5. System Two (later)

An open-weight offline reasoner may propose goals, discriminating experiments, and short plans. It should consume compressed symbolic evidence and model predictions. It should not directly emit an unchecked environment action.

## Why Laya/Jev is a comparison, not a commitment

The useful abstraction is non-autoregressive typed decision-making. Laya is an E2 representation test. Jev motivates the interface. ARC has unusually strong 2-D discrete structure, so the default architectural prior is that a native grid encoder will eventually outperform a text serialization if trained and evaluated fairly.

## Anti-overfitting rules

- Hold out complete games/mechanic families, not random transitions.
- Do not tune utility weights to public-leaderboard noise.
- Log negative experiments.
- Compare against identity-next-frame, heuristic E0, and a strong autoregressive baseline.
- Track deaths, resets, no-ops, action efficiency, wall-clock, VRAM, and calibration—not only completions.
