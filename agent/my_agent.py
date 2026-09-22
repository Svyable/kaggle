"""ARC-AGI-3 System-One scaffold.

Drop this file into the official ARC-AGI-3 Kaggle Starter as agent/my_agent.py.

Purpose of this phase:
- keep a persistent, compact world-model ledger;
- treat actions as typed decisions rather than generated text;
- learn immediate action consequences online;
- block proven no-ops in identical states;
- generate structured ACTION6 coordinate candidates;
- expose a stable scoring interface that can later be replaced by a learned
  Laya/Jev-style decision model or a grid-native dynamics model.

This is intentionally a *research baseline*, not a claim of a strong final solver.
It uses only the Python standard library plus the ARC competition runtime.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent


Grid = list[list[int]]


@dataclass(frozen=True)
class DecisionKey:
    action_id: int
    x: Optional[int] = None
    y: Optional[int] = None


@dataclass
class ConsequenceStats:
    trials: int = 0
    changed: int = 0
    progress: int = 0
    deaths: int = 0
    total_changed_cells: int = 0

    def update(self, *, changed_cells: int, progressed: bool, died: bool) -> None:
        self.trials += 1
        self.changed += int(changed_cells > 0)
        self.progress += int(progressed)
        self.deaths += int(died)
        self.total_changed_cells += changed_cells

    @property
    def p_change(self) -> float:
        # Beta(1,1) smoothing.
        return (self.changed + 1.0) / (self.trials + 2.0)

    @property
    def p_progress(self) -> float:
        # Sparse event: slightly skeptical prior.
        return (self.progress + 0.25) / (self.trials + 1.0)

    @property
    def p_death(self) -> float:
        return (self.deaths + 0.25) / (self.trials + 1.0)

    @property
    def mean_changed_cells(self) -> float:
        return self.total_changed_cells / max(1, self.trials)


@dataclass
class Outcome:
    state_sig: str
    changed_cells: int
    progressed: bool
    died: bool


class MyAgent(Agent):
    """Online system-identification controller with a pluggable decision scorer."""

    MAX_ACTIONS = 80
    ACTION6_CANDIDATE_LIMIT = 14
    RECENT_DECISIONS = 12

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Global-within-game mechanics evidence.
        self.action_stats: dict[DecisionKey, ConsequenceStats] = defaultdict(ConsequenceStats)
        self.action_id_stats: dict[int, ConsequenceStats] = defaultdict(ConsequenceStats)

        # State-local evidence: exact no-ops and repeated trials.
        self.state_action_trials: Counter[tuple[str, DecisionKey]] = Counter()
        self.proven_noops: set[tuple[str, DecisionKey]] = set()

        # Online transition bookkeeping.
        self.previous_grid: Optional[Grid] = None
        self.previous_levels = 0
        self.previous_state_sig: Optional[str] = None
        self.last_decision: Optional[DecisionKey] = None
        self.last_outcome: Optional[Outcome] = None
        self.recent: deque[DecisionKey] = deque(maxlen=self.RECENT_DECISIONS)

    @property
    def name(self) -> str:
        return f"{super().name}.systemone-v0"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    # ------------------------------------------------------------------ main loop
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if getattr(latest_frame, "full_reset", False):
            self._reset_online_state()
            action = GameAction.RESET
            action.reasoning = {
                "controller": "systemone-v0",
                "why": "framework requested full reset",
            }
            return action

        current_grid = self._last_grid(latest_frame)
        current_sig = self._state_signature(current_grid, latest_frame.levels_completed)

        # The current observation is the result of the action returned on the
        # previous choose_action() call. Learn from that transition first.
        self._learn_from_latest_observation(latest_frame, current_grid, current_sig)

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._clear_transition_anchor(current_grid, latest_frame.levels_completed, current_sig)
            action = GameAction.RESET
            action.reasoning = {
                "controller": "systemone-v0",
                "why": "required reset for NOT_PLAYED/GAME_OVER",
            }
            self.last_decision = None
            return action

        legal_ids = self._legal_action_ids(latest_frame)
        if not legal_ids:
            # Defensive fallback; ARC normally supplies available_actions.
            legal_ids = [1, 2, 3, 4, 5, 6, 7]

        candidates: list[DecisionKey] = []
        for action_id in legal_ids:
            if action_id == 6:
                for x, y in self._action6_candidates(current_grid):
                    candidates.append(DecisionKey(6, x=x, y=y))
            else:
                candidates.append(DecisionKey(action_id))

        if not candidates:
            # RESET should normally not be needed here, but stay valid rather
            # than emitting an illegal fabricated action.
            action = GameAction.RESET
            action.reasoning = {"controller": "systemone-v0", "why": "no legal candidates"}
            self.last_decision = None
            return action

        ranked = sorted(
            ((self._score_candidate(current_sig, c), c) for c in candidates),
            key=lambda item: (-item[0], self._tie_break_key(item[1])),
        )
        score, chosen = ranked[0]

        action = GameAction.from_id(chosen.action_id)
        if chosen.action_id == 6:
            action.set_data({"x": int(chosen.x), "y": int(chosen.y)})

        action.reasoning = {
            "controller": "systemone-v0",
            "decision": {
                "action_id": chosen.action_id,
                "x": chosen.x,
                "y": chosen.y,
                "score": round(score, 4),
            },
            "top_candidates": [
                {
                    "action_id": c.action_id,
                    "x": c.x,
                    "y": c.y,
                    "score": round(s, 4),
                }
                for s, c in ranked[:5]
            ],
            "last_outcome": None
            if self.last_outcome is None
            else {
                "changed_cells": self.last_outcome.changed_cells,
                "progressed": self.last_outcome.progressed,
                "died": self.last_outcome.died,
            },
        }

        # Anchor the observation that this decision acts on. The next call will
        # compare its returned state against this one.
        self.previous_grid = current_grid
        self.previous_levels = latest_frame.levels_completed
        self.previous_state_sig = current_sig
        self.last_decision = chosen
        self.recent.append(chosen)
        self.state_action_trials[(current_sig, chosen)] += 1
        return action

    # ---------------------------------------------------------- learned controller
    def _score_candidate(self, state_sig: str, candidate: DecisionKey) -> float:
        """Return a utility estimate for one legal environment intervention.

        This function is deliberately isolated. Phase 1 replaces this heuristic
        with a learned non-autoregressive consequence model while preserving the
        exact same controller and action interface.
        """
        exact = self.action_stats[candidate]
        broad = self.action_id_stats[candidate.action_id]
        local_trials = self.state_action_trials[(state_sig, candidate)]
        noop = (state_sig, candidate) in self.proven_noops

        # Back off from exact coordinate/action evidence to action-family evidence.
        if exact.trials:
            p_progress = exact.p_progress
            p_change = exact.p_change
            p_death = exact.p_death
        elif broad.trials:
            p_progress = broad.p_progress
            p_change = broad.p_change
            p_death = broad.p_death
        else:
            p_progress, p_change, p_death = 0.25, 0.50, 0.25

        novelty = 1.0 / math.sqrt(1.0 + local_trials)
        repeat_count = sum(1 for d in self.recent if d == candidate)

        # Scoring philosophy:
        #   progress >> informative state change > novelty
        #   death/no-op/repetition are expensive under RHAE.
        value = (
            5.0 * p_progress
            + 0.80 * p_change
            + 0.55 * novelty
            - 4.0 * p_death
            - 2.5 * float(noop)
            - 0.22 * repeat_count
        )

        # Undo is useful but should not be a default exploratory action.
        if candidate.action_id == 7:
            value -= 0.20

        # Coordinate search explodes combinatorially; prefer structurally-derived
        # points but give unseen candidates modest information value.
        if candidate.action_id == 6 and exact.trials == 0:
            value += 0.10

        return value

    # ------------------------------------------------------------- online learning
    def _learn_from_latest_observation(
        self, latest_frame: FrameData, current_grid: Grid, current_sig: str
    ) -> None:
        if self.last_decision is None or self.previous_grid is None or self.previous_state_sig is None:
            self._clear_transition_anchor(current_grid, latest_frame.levels_completed, current_sig)
            return

        changed_cells = self._grid_diff_count(self.previous_grid, current_grid)
        progressed = latest_frame.levels_completed > self.previous_levels
        died = latest_frame.state is GameState.GAME_OVER

        outcome = Outcome(
            state_sig=current_sig,
            changed_cells=changed_cells,
            progressed=progressed,
            died=died,
        )
        self.last_outcome = outcome

        self.action_stats[self.last_decision].update(
            changed_cells=changed_cells, progressed=progressed, died=died
        )
        self.action_id_stats[self.last_decision.action_id].update(
            changed_cells=changed_cells, progressed=progressed, died=died
        )

        if changed_cells == 0 and not progressed and not died:
            self.proven_noops.add((self.previous_state_sig, self.last_decision))

        # Keep the new state as the next transition anchor. choose_action() will
        # overwrite the decision after selecting the next action.
        self.previous_grid = current_grid
        self.previous_levels = latest_frame.levels_completed
        self.previous_state_sig = current_sig

    def _reset_online_state(self) -> None:
        """Clear all within-game evidence after a framework-level full reset."""
        self.action_stats.clear()
        self.action_id_stats.clear()
        self.state_action_trials.clear()
        self.proven_noops.clear()
        self.previous_grid = None
        self.previous_levels = 0
        self.previous_state_sig = None
        self.last_decision = None
        self.last_outcome = None
        self.recent.clear()

    def _clear_transition_anchor(self, grid: Grid, levels: int, sig: str) -> None:
        self.previous_grid = grid
        self.previous_levels = levels
        self.previous_state_sig = sig

    # -------------------------------------------------------------- perception
    @staticmethod
    def _last_grid(frame: FrameData) -> Grid:
        if not frame.frame:
            return []
        return frame.frame[-1]

    @staticmethod
    def _grid_diff_count(a: Grid, b: Grid) -> int:
        h = max(len(a), len(b))
        changed = 0
        for y in range(h):
            ra = a[y] if y < len(a) else []
            rb = b[y] if y < len(b) else []
            w = max(len(ra), len(rb))
            for x in range(w):
                va = ra[x] if x < len(ra) else -1
                vb = rb[x] if x < len(rb) else -1
                changed += int(va != vb)
        return changed

    @staticmethod
    def _state_signature(grid: Grid, levels_completed: int) -> str:
        h = hashlib.blake2b(digest_size=12)
        h.update(bytes([levels_completed & 0xFF]))
        h.update(bytes([len(grid) & 0xFF]))
        h.update(bytes([(len(grid[0]) if grid else 0) & 0xFF]))
        for row in grid:
            h.update(bytes((int(v) & 0x0F) for v in row))
        return h.hexdigest()

    @staticmethod
    def _legal_action_ids(frame: FrameData) -> list[int]:
        out: list[int] = []
        for raw in getattr(frame, "available_actions", []) or []:
            if isinstance(raw, int):
                aid = raw
            elif hasattr(raw, "value"):
                try:
                    aid = int(raw.value)
                except (TypeError, ValueError):
                    continue
            elif hasattr(raw, "id"):
                try:
                    aid = int(raw.id)
                except (TypeError, ValueError):
                    continue
            else:
                continue
            if 1 <= aid <= 7:
                out.append(aid)
        return sorted(set(out))

    def _action6_candidates(self, grid: Grid) -> list[tuple[int, int]]:
        if not grid or not grid[0]:
            return [(32, 32)]

        height, width = len(grid), len(grid[0])
        points: list[tuple[int, int]] = []

        # 1) Changed region since prior state is often the highest-value click target.
        if self.previous_grid:
            changed: list[tuple[int, int]] = []
            for y in range(min(height, len(self.previous_grid))):
                prow = self.previous_grid[y]
                for x in range(min(width, len(prow))):
                    if grid[y][x] != prow[x]:
                        changed.append((x, y))
            if changed:
                points.extend(self._representative_points(changed))

        # 2) Same-color connected components provide object-like click candidates.
        components = self._components_by_color(grid)
        components.sort(key=lambda comp: (-len(comp), comp[0][1], comp[0][0]))
        for comp in components[:10]:
            points.extend(self._representative_points(comp)[:2])

        # 3) Safe structural priors.
        points.extend(
            [
                (width // 2, height // 2),
                (0, 0),
                (max(0, width - 1), 0),
                (0, max(0, height - 1)),
                (max(0, width - 1), max(0, height - 1)),
            ]
        )

        deduped: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for x, y in points:
            p = (min(63, max(0, int(x))), min(63, max(0, int(y))))
            if p not in seen:
                seen.add(p)
                deduped.append(p)
            if len(deduped) >= self.ACTION6_CANDIDATE_LIMIT:
                break
        return deduped or [(min(63, width // 2), min(63, height // 2))]

    @staticmethod
    def _representative_points(points: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        pts = list(points)
        if not pts:
            return []
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = round(sum(xs) / len(xs))
        cy = round(sum(ys) / len(ys))
        # centroid + bounding-box center; these differ on irregular shapes.
        bx = (min(xs) + max(xs)) // 2
        by = (min(ys) + max(ys)) // 2
        return [(cx, cy), (bx, by)]

    @staticmethod
    def _components_by_color(grid: Grid) -> list[list[tuple[int, int]]]:
        if not grid or not grid[0]:
            return []
        height, width = len(grid), len(grid[0])
        seen: set[tuple[int, int]] = set()
        comps: list[list[tuple[int, int]]] = []
        for y in range(height):
            for x in range(width):
                if (x, y) in seen:
                    continue
                color = grid[y][x]
                stack = [(x, y)]
                seen.add((x, y))
                comp: list[tuple[int, int]] = []
                while stack:
                    px, py = stack.pop()
                    comp.append((px, py))
                    for nx, ny in ((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)):
                        if not (0 <= nx < width and 0 <= ny < height):
                            continue
                        if (nx, ny) in seen or grid[ny][nx] != color:
                            continue
                        seen.add((nx, ny))
                        stack.append((nx, ny))
                comps.append(comp)
        return comps

    @staticmethod
    def _tie_break_key(decision: DecisionKey) -> tuple[int, int, int]:
        # Stable deterministic ordering for reproducible ablations.
        return (
            decision.action_id,
            -1 if decision.y is None else decision.y,
            -1 if decision.x is None else decision.x,
        )
