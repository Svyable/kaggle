# Upstream compatibility

This repository deliberately separates our agent logic from the ARC competition plumbing, but we still track the public upstream contracts we depend on.

## Snapshot checked on 2026-09-21

- Kaggle starter: https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter
  - observed main commit: `eeb1535404f321d280a8f9194bbc1d7aca5f05fc`
  - `agent/my_agent.py` contract: class name `MyAgent`, subclass `agents.agent.Agent`, implement `is_done` and `choose_action`.
  - starter setup currently uses Python 3.12 and `arc-agi>=0.9.6`.
- Agent framework: https://github.com/arcprize/ARC-AGI-3-Agents
  - observed main commit: `4743e7d0aaae0ded0d98a89a7e282e63564cd58b`
  - `FrameData.available_actions` arrives as integer action IDs.
  - `GameAction.from_id(id)` is part of the tested framework API.
  - `ACTION6` is the complex coordinate action.
  - current framework agents explicitly react to `FrameData.full_reset`.

## CI policy

The stubbed unit tests are intentionally fast, but they are not sufficient evidence of compatibility. The `upstream-contract` CI job installs the current public `arc-agi` runtime under Python 3.12 and verifies the API assumptions our controller relies on.

If upstream changes this contract, CI should fail before we spend a Kaggle submission.

## Submission-count discrepancy

The public starter README currently says ARC-AGI-3 allows five official submissions per day. The competition-specific rules supplied from Kaggle state a maximum of one submission per day. Treat the **live Kaggle competition rules/UI as authoritative** and verify the active quota before spending a submission; do not encode either number into agent logic.


## `FrameData.full_reset` semantic note

`full_reset=True` is **descriptive output metadata**: it means the RESET that produced the current observation created a fresh game. It is not a command asking the agent to RESET again.

Controller rule:
- clear within-game learned evidence when a received frame has `full_reset=True`;
- if that frame is already `NOT_FINISHED`, continue by choosing a normal legal environment action;
- only issue RESET for states such as `NOT_PLAYED` or `GAME_OVER` that actually require it.

This distinction is regression-tested because treating `full_reset` as a reset request creates an infinite RESET loop that can still terminate cleanly at the action cap.


## Recorder `action_input` mismatch

As checked on 2026-09-22, the current public `ARC-AGI-3-Agents` implementation records converted `FrameData`, but `Agent._convert_raw_frame_data()` does not copy `raw.action_input` into that `FrameData`. The public recordings documentation shows `action_input` in JSONL records, so the implementation and documented schema are currently inconsistent.

Research rule:
- do not assume current framework Recorder output is action-aligned supervision;
- for E1 data, use `research.public_collector`, which captures the chosen action and ACTION6 coordinates before stepping the environment and writes `Transition` rows directly;
- `research.recording_converter` remains valid for recordings that actually contain the documented `action_input` field.
