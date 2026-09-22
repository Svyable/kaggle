import json
from pathlib import Path

import pytest

from research.recording_converter import (
    assign_family_split,
    convert_frame_events,
    convert_recording,
    load_frame_events,
)


def fd(grid, *, action_id=0, data=None, levels=0, state="NOT_FINISHED", game_id="demo-abc"):
    return {
        "game_id": game_id,
        "frame": [grid],
        "state": state,
        "levels_completed": levels,
        "win_levels": 3,
        "action_input": {"id": action_id, "data": data or {}, "reasoning": None},
        "guid": "g",
        "full_reset": False,
        "available_actions": [1, 2, 6, 7],
    }


def test_pairing_preserves_action6_and_labels_progress_death():
    frames = [
        fd([[0, 0], [0, 0]], action_id=0),
        fd([[0, 1], [0, 0]], action_id=6, data={"x": 1, "y": 0}),
        fd([[0, 1], [2, 0]], action_id=2, levels=1),
        fd([[0, 1], [2, 0]], action_id=1, levels=1, state="GAME_OVER"),
    ]
    t = convert_frame_events(frames)
    assert len(t) == 3
    assert (t[0].action_id, t[0].click_x, t[0].click_y) == (6, 1, 0)
    assert t[0].changed_cells == 1
    assert t[1].progressed is True
    assert t[2].died is True
    assert all(x.game_family == "demo" for x in t)


def test_reset_is_dropped_by_default_and_can_be_included():
    frames = [fd([[0]]), fd([[1]], action_id=0), fd([[2]], action_id=1)]
    assert [t.action_id for t in convert_frame_events(frames)] == [1]
    assert [t.action_id for t in convert_frame_events(frames, include_reset=True)] == [0, 1]


def test_converter_ignores_non_frame_events_and_writes_manifest(tmp_path: Path):
    source = tmp_path / "sample.recording.jsonl"
    events = [
        {"timestamp": "t0", "data": fd([[0]], action_id=0)},
        {"timestamp": "t1", "data": {"score": 0.5}},
        {"timestamp": "t2", "data": fd([[1]], action_id=1)},
    ]
    source.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    out = tmp_path / "transitions.jsonl"
    manifest = tmp_path / "manifest.json"
    m = convert_recording(source, out, manifest)
    assert m.source_events == 3
    assert m.frame_events == 2
    assert m.transitions == 1
    assert len(m.source_sha256) == 64
    row = json.loads(out.read_text().strip())
    assert row["action_id"] == 1
    assert json.loads(manifest.read_text())["arc_agents_commit"]


def test_invalid_json_reports_line_number(tmp_path: Path):
    source = tmp_path / "bad.jsonl"
    source.write_text('{"ok": 1}\nnot-json\n')
    with pytest.raises(ValueError, match=r":2:"):
        load_frame_events(source)


def test_family_split_is_deterministic_and_family_level():
    a = assign_family_split("ls20", validation_percent=50, seed="x")
    assert a == assign_family_split("ls20", validation_percent=50, seed="x")
    assert a in {"train", "validation"}
    with pytest.raises(ValueError):
        assign_family_split("ls20", validation_percent=101)
