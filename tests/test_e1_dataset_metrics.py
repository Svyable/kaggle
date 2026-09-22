import json
from pathlib import Path

import pytest
import torch

from research.arc_decision_model import ARCDecisionDynamics, training_loss
from research.e1_dataset import (
    NO_CLICK,
    PAD_COLOR,
    TransitionDataset,
    collate_transitions,
    load_transition_jsonl,
    split_by_family,
)
from research.e1_metrics import (
    binary_brier,
    expected_calibration_error,
    persistence_baseline,
)
from research.trajectory_schema import Transition


def tr(
    family: str,
    current,
    nxt,
    *,
    previous=None,
    action_id=1,
    click_x=None,
    click_y=None,
    progressed=False,
    died=False,
):
    previous = current if previous is None else previous
    changed = 0
    h = max(len(current), len(nxt))
    for y in range(h):
        a = current[y] if y < len(current) else []
        b = nxt[y] if y < len(nxt) else []
        for x in range(max(len(a), len(b))):
            av = a[x] if x < len(a) else -1
            bv = b[x] if x < len(b) else -1
            changed += int(av != bv)
    return Transition(
        game_family=family,
        level_index=0,
        current_grid=current,
        previous_grid=previous,
        action_id=action_id,
        click_x=click_x,
        click_y=click_y,
        next_grid=nxt,
        progressed=progressed,
        died=died,
        changed_cells=changed,
    )


def test_variable_grids_collate_with_non_arc_padding():
    a = tr("a", [[0, 1], [2, 3]], [[0, 1], [2, 4]])
    b = tr(
        "b",
        [[5, 6, 7]],
        [[5, 6, 8]],
        action_id=6,
        click_x=2,
        click_y=0,
    )
    batch = collate_transitions([a, b])
    assert batch.current.shape == (2, 2, 3)
    assert batch.current[0, 0, 2].item() == PAD_COLOR
    assert batch.current[1, 1, 0].item() == PAD_COLOR
    assert batch.valid_mask.sum().item() == 7
    assert batch.click_x.tolist() == [NO_CLICK, 2]
    assert batch.click_y.tolist() == [NO_CLICK, 0]


def test_padding_is_ignored_by_model_loss():
    a = tr("a", [[0, 1], [2, 3]], [[0, 1], [2, 4]])
    b = tr("b", [[5, 6, 7]], [[5, 6, 8]])
    batch = collate_transitions([a, b])
    model = ARCDecisionDynamics(width=32, depth=1, cond_dim=48)
    out = model(
        batch.current,
        batch.previous,
        batch.action_id,
        batch.click_x,
        batch.click_y,
    )
    assert out.valid_mask is not None
    assert torch.equal(out.valid_mask, batch.valid_mask)
    loss, parts = training_loss(
        out,
        batch.next_grid,
        batch.changed,
        batch.progressed,
        batch.died,
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v) for v in parts.values())


def test_persistence_baseline_exposes_changed_cells():
    batch = collate_transitions(
        [tr("a", [[0, 1]], [[0, 2]]), tr("b", [[3, 4]], [[3, 4]])]
    )
    prediction, m = persistence_baseline(batch)
    assert torch.equal(prediction, batch.current)
    assert m.cell_accuracy == pytest.approx(0.75)
    assert m.exact_grid_accuracy == pytest.approx(0.5)
    assert m.changed_cell_count == 1
    assert m.changed_cell_accuracy == pytest.approx(0.0)


def test_family_split_never_leaks_a_family_across_sets():
    transitions = [
        tr("alpha", [[0]], [[1]]),
        tr("alpha", [[1]], [[2]]),
        tr("beta", [[0]], [[0]]),
        tr("gamma", [[0]], [[1]]),
    ]
    train, validation = split_by_family(transitions, validation_percent=50, seed="fixed")
    train_families = {t.game_family for t in train}
    val_families = {t.game_family for t in validation}
    assert train_families.isdisjoint(val_families)
    assert train_families | val_families == {"alpha", "beta", "gamma"}


def test_jsonl_loader_rejects_invalid_action6(tmp_path: Path):
    bad = tr("a", [[0]], [[1]])
    row = json.loads(bad.to_json())
    row["action_id"] = 6
    row["click_x"] = None
    row["click_y"] = None
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="ACTION6 requires"):
        load_transition_jsonl(path)


def test_binary_calibration_metrics():
    logits = torch.tensor([0.0, 2.0, -2.0])
    target = torch.tensor([0.0, 1.0, 0.0])
    brier = binary_brier(logits, target)
    ece = expected_calibration_error(logits, target, bins=5)
    assert 0.0 <= brier <= 1.0
    assert 0.0 <= ece <= 1.0
