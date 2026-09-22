import math

import pytest

from research.e1_train import (
    TrainConfig,
    evaluate_model,
    train_model,
)
from research.trajectory_schema import Transition


def transition(family, current, nxt, action=1, progressed=False, died=False):
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
        action_id=action,
        click_x=None,
        click_y=None,
        next_grid=nxt,
        progressed=progressed,
        died=died,
        changed_cells=changed,
    )


def test_train_harness_runs_on_disjoint_tiny_dataset():
    train = [
        transition("train", [[0, 0], [0, 0]], [[0, 1], [0, 0]], action=1),
        transition("train", [[1, 0], [0, 0]], [[1, 1], [0, 0]], action=1),
        transition("train", [[2, 0], [0, 0]], [[2, 1], [0, 0]], action=1),
        transition("train", [[3, 0], [0, 0]], [[3, 1], [0, 0]], action=1),
    ]
    validation = [
        transition("validation", [[4, 0], [0, 0]], [[4, 1], [0, 0]], action=1),
        transition("validation", [[5, 0], [0, 0]], [[5, 1], [0, 0]], action=1),
    ]
    config = TrainConfig(
        epochs=1,
        batch_size=2,
        width=16,
        depth=1,
        cond_dim=32,
        seed=123,
    )
    model, history = train_model(train, validation, config, device="cpu")
    assert len(history) == 1
    metrics = evaluate_model(model, validation, batch_size=2, device="cpu")
    assert math.isfinite(metrics["loss"])
    assert 0.0 <= metrics["cell_accuracy"] <= 1.0
    assert 0.0 <= metrics["changed_cell_accuracy"] <= 1.0
    assert 0.0 <= metrics["change_brier"] <= 1.0


def test_train_harness_rejects_family_leakage():
    t = transition("same", [[0]], [[1]])
    with pytest.raises(ValueError, match="family leakage"):
        train_model(
            [t],
            [t],
            TrainConfig(epochs=1, batch_size=1, width=16, depth=1, cond_dim=32),
        )
