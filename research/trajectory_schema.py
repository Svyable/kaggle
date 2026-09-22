"""Training-record schema for the ARC System-One experiments.

Keep the schema simple enough that trajectories can come from:
- human or strong-agent replays,
- the official ARC local runner recordings,
- procedurally generated training environments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional


@dataclass
class Transition:
    game_family: str
    level_index: int
    current_grid: list[list[int]]
    previous_grid: list[list[int]]
    action_id: int
    click_x: Optional[int]
    click_y: Optional[int]
    next_grid: list[list[int]]
    progressed: bool
    died: bool
    changed_cells: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def write_jsonl(path: str | Path, transitions: list[Transition]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in transitions:
            f.write(t.to_json() + "\n")
