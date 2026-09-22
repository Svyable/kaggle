"""Dataset utilities for E1 action-consequence experiments.

ARC colors are 0..15. Value 16 is reserved internally as a padding token so
variable-size grids can share a minibatch without turning padded black cells
into training targets.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Iterator, Sequence

import torch
from torch.utils.data import Dataset

from .recording_converter import assign_family_split
from .trajectory_schema import Transition

PAD_COLOR = 16
NO_CLICK = 64


def _validate_grid(grid: list[list[int]], *, name: str) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise ValueError(f"{name} must be a non-empty rectangular grid")
    width = len(grid[0])
    if width > 64 or len(grid) > 64:
        raise ValueError(f"{name} exceeds ARC 64x64 bounds")
    for row in grid:
        if len(row) != width:
            raise ValueError(f"{name} must be rectangular")
        for value in row:
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 15:
                raise ValueError(f"{name} contains invalid ARC color {value!r}")
    return len(grid), width


def validate_transition(t: Transition) -> None:
    _validate_grid(t.previous_grid, name="previous_grid")
    _validate_grid(t.current_grid, name="current_grid")
    _validate_grid(t.next_grid, name="next_grid")
    if not 1 <= int(t.action_id) <= 7:
        raise ValueError(f"action_id must be 1..7, got {t.action_id}")
    if t.action_id == 6:
        if t.click_x is None or t.click_y is None:
            raise ValueError("ACTION6 requires click_x/click_y")
        if not 0 <= int(t.click_x) <= 63 or not 0 <= int(t.click_y) <= 63:
            raise ValueError("ACTION6 coordinates must be 0..63")
    elif t.click_x is not None or t.click_y is not None:
        raise ValueError("non-ACTION6 transition must not carry click coordinates")


def transition_from_dict(row: dict) -> Transition:
    required = {
        "game_family",
        "level_index",
        "current_grid",
        "previous_grid",
        "action_id",
        "click_x",
        "click_y",
        "next_grid",
        "progressed",
        "died",
        "changed_cells",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"transition missing fields: {missing}")
    t = Transition(**{key: row[key] for key in required})
    validate_transition(t)
    return t


def load_transition_jsonl(paths: str | Path | Sequence[str | Path]) -> list[Transition]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: list[Transition] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"invalid JSON at {path}:{lineno}: {e}") from e
                if not isinstance(row, dict):
                    raise ValueError(f"transition at {path}:{lineno} is not an object")
                try:
                    out.append(transition_from_dict(row))
                except (TypeError, ValueError) as e:
                    raise ValueError(f"invalid transition at {path}:{lineno}: {e}") from e
    return out


def split_by_family(
    transitions: Iterable[Transition],
    *,
    validation_percent: int = 20,
    seed: str = "arc3-systemone-v1",
) -> tuple[list[Transition], list[Transition]]:
    train: list[Transition] = []
    validation: list[Transition] = []
    for t in transitions:
        target = (
            validation
            if assign_family_split(
                t.game_family, validation_percent=validation_percent, seed=seed
            )
            == "validation"
            else train
        )
        target.append(t)
    return train, validation


class TransitionDataset(Dataset[Transition]):
    def __init__(self, transitions: Sequence[Transition]) -> None:
        self.transitions = list(transitions)
        for t in self.transitions:
            validate_transition(t)

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> Transition:
        return self.transitions[index]


@dataclass
class TransitionBatch:
    previous: torch.Tensor
    current: torch.Tensor
    next_grid: torch.Tensor
    valid_mask: torch.Tensor
    next_valid_mask: torch.Tensor
    action_id: torch.Tensor
    click_x: torch.Tensor
    click_y: torch.Tensor
    changed: torch.Tensor
    progressed: torch.Tensor
    died: torch.Tensor
    changed_cells: torch.Tensor
    families: tuple[str, ...]

    def to(self, device: torch.device | str) -> "TransitionBatch":
        kwargs = {}
        for field_name in (
            "previous",
            "current",
            "next_grid",
            "valid_mask",
            "next_valid_mask",
            "action_id",
            "click_x",
            "click_y",
            "changed",
            "progressed",
            "died",
            "changed_cells",
        ):
            kwargs[field_name] = getattr(self, field_name).to(device)
        kwargs["families"] = self.families
        return TransitionBatch(**kwargs)


def _place(canvas: torch.Tensor, grid: list[list[int]]) -> None:
    h, w = len(grid), len(grid[0])
    canvas[:h, :w] = torch.tensor(grid, dtype=torch.long)


def collate_transitions(items: Sequence[Transition]) -> TransitionBatch:
    if not items:
        raise ValueError("cannot collate an empty batch")
    for t in items:
        validate_transition(t)

    max_h = max(
        max(len(t.previous_grid), len(t.current_grid), len(t.next_grid)) for t in items
    )
    max_w = max(
        max(len(t.previous_grid[0]), len(t.current_grid[0]), len(t.next_grid[0]))
        for t in items
    )
    b = len(items)
    previous = torch.full((b, max_h, max_w), PAD_COLOR, dtype=torch.long)
    current = torch.full((b, max_h, max_w), PAD_COLOR, dtype=torch.long)
    next_grid = torch.full((b, max_h, max_w), PAD_COLOR, dtype=torch.long)

    for i, t in enumerate(items):
        _place(previous[i], t.previous_grid)
        _place(current[i], t.current_grid)
        _place(next_grid[i], t.next_grid)

    valid_mask = current != PAD_COLOR
    next_valid_mask = next_grid != PAD_COLOR
    action_id = torch.tensor([t.action_id for t in items], dtype=torch.long)
    click_x = torch.tensor(
        [NO_CLICK if t.click_x is None else t.click_x for t in items], dtype=torch.long
    )
    click_y = torch.tensor(
        [NO_CLICK if t.click_y is None else t.click_y for t in items], dtype=torch.long
    )

    return TransitionBatch(
        previous=previous,
        current=current,
        next_grid=next_grid,
        valid_mask=valid_mask,
        next_valid_mask=next_valid_mask,
        action_id=action_id,
        click_x=click_x,
        click_y=click_y,
        changed=torch.tensor([t.changed_cells > 0 for t in items], dtype=torch.float32),
        progressed=torch.tensor([t.progressed for t in items], dtype=torch.float32),
        died=torch.tensor([t.died for t in items], dtype=torch.float32),
        changed_cells=torch.tensor([t.changed_cells for t in items], dtype=torch.long),
        families=tuple(t.game_family for t in items),
    )


def shuffled_indices(n: int, *, seed: int) -> list[int]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    return indices
