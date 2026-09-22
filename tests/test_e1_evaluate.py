import json
from pathlib import Path

from research.e1_evaluate import build_report, persistence_report
from research.trajectory_schema import Transition


def tr(family, current, nxt, action_id=1):
    changed = sum(
        int(current[y][x] != nxt[y][x])
        for y in range(len(nxt))
        for x in range(len(nxt[y]))
    )
    return Transition(
        game_family=family,
        level_index=0,
        current_grid=current,
        previous_grid=current,
        action_id=action_id,
        click_x=None,
        click_y=None,
        next_grid=nxt,
        progressed=False,
        died=False,
        changed_cells=changed,
    )


def test_persistence_report_does_not_hide_changed_cells():
    r = persistence_report(
        [
            tr("a", [[0, 0]], [[0, 1]]),
            tr("b", [[2, 2]], [[2, 2]], action_id=2),
        ]
    )
    assert r.cell_accuracy == 0.75
    assert r.exact_grid_accuracy == 0.5
    assert r.changed_cells == 1
    assert r.changed_cell_accuracy == 0.0
    assert r.action_counts == {1: 1, 2: 1}


def test_build_report_keeps_families_disjoint(tmp_path: Path):
    rows = [
        tr("alpha", [[0]], [[1]]),
        tr("alpha", [[1]], [[1]]),
        tr("beta", [[0]], [[0]]),
        tr("gamma", [[0]], [[1]]),
    ]
    path = tmp_path / "transitions.jsonl"
    path.write_text("\n".join(x.to_json() for x in rows) + "\n")
    report = build_report([path], validation_percent=50, split_seed="fixed")
    assert set(report["train_families"]).isdisjoint(report["validation_families"])
    assert set(report["train_families"]) | set(report["validation_families"]) == {
        "alpha",
        "beta",
        "gamma",
    }
    assert report["all"]["transitions"] == 4
