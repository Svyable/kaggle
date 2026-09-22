"""Command-line E1 dataset audit and persistence-baseline report.

Usage:
    python -m research.e1_evaluate path/to/transitions.jsonl
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from .e1_dataset import load_transition_jsonl, split_by_family
from .trajectory_schema import Transition


@dataclass(frozen=True)
class PersistenceReport:
    transitions: int
    families: int
    cells: int
    cell_accuracy: float
    exact_grid_accuracy: float
    changed_cells: int
    changed_cell_accuracy: float
    progressed: int
    died: int
    action_counts: dict[int, int]


def _cell(grid: list[list[int]], y: int, x: int) -> int | None:
    if y >= len(grid) or x >= len(grid[y]):
        return None
    return grid[y][x]


def persistence_report(transitions: Sequence[Transition]) -> PersistenceReport:
    cells = 0
    correct = 0
    exact = 0
    changed_cells = 0
    changed_correct = 0
    action_counts: dict[int, int] = {}

    for t in transitions:
        action_counts[t.action_id] = action_counts.get(t.action_id, 0) + 1
        transition_exact = True
        for y, row in enumerate(t.next_grid):
            for x, target in enumerate(row):
                current = _cell(t.current_grid, y, x)
                cells += 1
                is_correct = current == target
                correct += int(is_correct)
                transition_exact = transition_exact and is_correct
                if current != target:
                    changed_cells += 1
                    changed_correct += int(is_correct)
        exact += int(transition_exact)

    n = len(transitions)
    return PersistenceReport(
        transitions=n,
        families=len({t.game_family for t in transitions}),
        cells=cells,
        cell_accuracy=(correct / cells) if cells else 0.0,
        exact_grid_accuracy=(exact / n) if n else 0.0,
        changed_cells=changed_cells,
        # Persistence cannot predict a genuinely changed cell correctly by
        # definition, but keep the computed form explicit for auditability.
        changed_cell_accuracy=(changed_correct / changed_cells) if changed_cells else 1.0,
        progressed=sum(int(t.progressed) for t in transitions),
        died=sum(int(t.died) for t in transitions),
        action_counts=dict(sorted(action_counts.items())),
    )


def build_report(
    paths: Sequence[str | Path],
    *,
    validation_percent: int = 20,
    split_seed: str = "arc3-systemone-v1",
) -> dict:
    transitions = load_transition_jsonl(list(paths))
    train, validation = split_by_family(
        transitions,
        validation_percent=validation_percent,
        seed=split_seed,
    )
    return {
        "split_seed": split_seed,
        "validation_percent": validation_percent,
        "all": asdict(persistence_report(transitions)),
        "train": asdict(persistence_report(train)),
        "validation": asdict(persistence_report(validation)),
        "train_families": sorted({t.game_family for t in train}),
        "validation_families": sorted({t.game_family for t in validation}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("transitions", nargs="+", type=Path)
    parser.add_argument("--validation-percent", type=int, default=20)
    parser.add_argument("--split-seed", default="arc3-systemone-v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(
        args.transitions,
        validation_percent=args.validation_percent,
        split_seed=args.split_seed,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
