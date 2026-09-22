"""Train/evaluate the E1 grid-native action-consequence model.

This harness is intentionally conservative:
- train/validation families must be disjoint before training starts;
- padding is ignored by the dense grid loss;
- model selection uses held-out validation loss, never leaderboard score;
- evaluation always reports changed-cell quality in addition to overall accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Sequence

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .arc_decision_model import ARCDecisionDynamics, training_loss
from .e1_dataset import (
    TransitionBatch,
    TransitionDataset,
    collate_transitions,
    load_transition_jsonl,
    split_by_family,
)
from .e1_evaluate import persistence_report
from .e1_metrics import binary_brier, expected_calibration_error
from .trajectory_schema import Transition


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    width: int = 96
    depth: int = 6
    cond_dim: int = 128
    seed: int = 7


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    transitions: Sequence[Transition],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TransitionDataset(transitions),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_transitions,
        num_workers=0,
        drop_last=False,
    )


def _assert_family_disjoint(
    train: Sequence[Transition], validation: Sequence[Transition]
) -> None:
    train_families = {t.game_family for t in train}
    validation_families = {t.game_family for t in validation}
    overlap = train_families & validation_families
    if overlap:
        raise ValueError(f"train/validation family leakage: {sorted(overlap)}")
    if not train:
        raise ValueError("training split is empty")
    if not validation:
        raise ValueError("validation split is empty")


def _grid_counts(
    prediction: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, int]:
    valid = valid_mask.bool()
    correct = (prediction == target) & valid
    exact = (((prediction == target) | ~valid).flatten(1)).all(dim=1)
    changed = valid & (current != target)
    return {
        "correct": int(correct.sum().item()),
        "cells": int(valid.sum().item()),
        "exact": int(exact.sum().item()),
        "examples": int(prediction.shape[0]),
        "changed_correct": int(((prediction == target) & changed).sum().item()),
        "changed_cells": int(changed.sum().item()),
    }


@torch.no_grad()
def evaluate_model(
    model: ARCDecisionDynamics,
    transitions: Sequence[Transition],
    *,
    batch_size: int,
    device: torch.device | str,
) -> dict:
    if not transitions:
        raise ValueError("cannot evaluate an empty split")
    model.eval()
    loader = _loader(
        transitions, batch_size=batch_size, shuffle=False, seed=0
    )

    total_loss = 0.0
    total_examples = 0
    counts = {
        "correct": 0,
        "cells": 0,
        "exact": 0,
        "examples": 0,
        "changed_correct": 0,
        "changed_cells": 0,
    }
    change_logits: list[torch.Tensor] = []
    progress_logits: list[torch.Tensor] = []
    death_logits: list[torch.Tensor] = []
    change_targets: list[torch.Tensor] = []
    progress_targets: list[torch.Tensor] = []
    death_targets: list[torch.Tensor] = []

    for batch in loader:
        assert isinstance(batch, TransitionBatch)
        batch = batch.to(device)
        out = model(
            batch.current,
            batch.previous,
            batch.action_id,
            batch.click_x,
            batch.click_y,
        )
        loss, _ = training_loss(
            out,
            batch.next_grid,
            batch.changed,
            batch.progressed,
            batch.died,
        )
        b = int(batch.current.shape[0])
        total_loss += float(loss.item()) * b
        total_examples += b

        prediction = out.next_grid_logits.argmax(dim=1)
        bc = _grid_counts(
            prediction, batch.current, batch.next_grid, batch.next_valid_mask
        )
        for key, value in bc.items():
            counts[key] += value

        change_logits.append(out.p_change_logit.detach().cpu())
        progress_logits.append(out.p_progress_logit.detach().cpu())
        death_logits.append(out.p_death_logit.detach().cpu())
        change_targets.append(batch.changed.detach().cpu())
        progress_targets.append(batch.progressed.detach().cpu())
        death_targets.append(batch.died.detach().cpu())

    change_l = torch.cat(change_logits)
    progress_l = torch.cat(progress_logits)
    death_l = torch.cat(death_logits)
    change_t = torch.cat(change_targets)
    progress_t = torch.cat(progress_targets)
    death_t = torch.cat(death_targets)

    changed_cells = counts["changed_cells"]
    return {
        "loss": total_loss / total_examples,
        "cell_accuracy": counts["correct"] / max(1, counts["cells"]),
        "exact_grid_accuracy": counts["exact"] / max(1, counts["examples"]),
        "changed_cell_accuracy": (
            counts["changed_correct"] / changed_cells if changed_cells else 1.0
        ),
        "changed_cells": changed_cells,
        "change_brier": binary_brier(change_l, change_t),
        "progress_brier": binary_brier(progress_l, progress_t),
        "death_brier": binary_brier(death_l, death_t),
        "change_ece": expected_calibration_error(change_l, change_t),
        "progress_ece": expected_calibration_error(progress_l, progress_t),
        "death_ece": expected_calibration_error(death_l, death_t),
    }


def train_model(
    train: Sequence[Transition],
    validation: Sequence[Transition],
    config: TrainConfig,
    *,
    device: torch.device | str = "cpu",
) -> tuple[ARCDecisionDynamics, list[dict]]:
    _assert_family_disjoint(train, validation)
    if config.epochs < 1:
        raise ValueError("epochs must be >=1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >=1")

    seed_everything(config.seed)
    device = torch.device(device)
    model = ARCDecisionDynamics(
        width=config.width, depth=config.depth, cond_dim=config.cond_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = _loader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_examples = 0

        for batch in loader:
            assert isinstance(batch, TransitionBatch)
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(
                batch.current,
                batch.previous,
                batch.action_id,
                batch.click_x,
                batch.click_y,
            )
            loss, _ = training_loss(
                out,
                batch.next_grid,
                batch.changed,
                batch.progressed,
                batch.died,
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            b = int(batch.current.shape[0])
            train_loss += float(loss.item()) * b
            train_examples += b

        validation_metrics = evaluate_model(
            model,
            validation,
            batch_size=config.batch_size,
            device=device,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_examples),
            "validation": validation_metrics,
        }
        history.append(epoch_row)

        if validation_metrics["loss"] < best_loss:
            best_loss = float(validation_metrics["loss"])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    return model, history


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def run_experiment(
    paths: Sequence[str | Path],
    *,
    config: TrainConfig,
    validation_percent: int,
    split_seed: str,
    device: torch.device | str,
) -> tuple[ARCDecisionDynamics, dict]:
    transitions = load_transition_jsonl(list(paths))
    train, validation = split_by_family(
        transitions,
        validation_percent=validation_percent,
        seed=split_seed,
    )
    _assert_family_disjoint(train, validation)

    model, history = train_model(train, validation, config, device=device)
    final_validation = evaluate_model(
        model,
        validation,
        batch_size=config.batch_size,
        device=device,
    )
    report = {
        "config": asdict(config),
        "split_seed": split_seed,
        "validation_percent": validation_percent,
        "train_transitions": len(train),
        "validation_transitions": len(validation),
        "train_families": sorted({t.game_family for t in train}),
        "validation_families": sorted({t.game_family for t in validation}),
        "persistence_validation": asdict(persistence_report(validation)),
        "best_model_validation": final_validation,
        "history": history,
    }
    return model, report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("transitions", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--validation-percent", type=int, default=20)
    p.add_argument("--split-seed", default="arc3-systemone-v1")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        width=args.width,
        depth=args.depth,
        cond_dim=args.cond_dim,
        seed=args.seed,
    )
    device = resolve_device(args.device)
    model, report = run_experiment(
        args.transitions,
        config=config,
        validation_percent=args.validation_percent,
        split_seed=args.split_seed,
        device=device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "split_seed": args.split_seed,
            "validation_percent": args.validation_percent,
        },
        args.output,
    )
    report_path = args.output.with_suffix(args.output.suffix + ".metrics.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
