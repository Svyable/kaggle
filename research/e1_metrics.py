"""Evaluation metrics and baselines for E1.

Metrics are computed only on valid next-grid cells. This prevents padding or
unchanged background from inflating apparent model quality.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .e1_dataset import PAD_COLOR, TransitionBatch


@dataclass(frozen=True)
class GridMetrics:
    cell_accuracy: float
    exact_grid_accuracy: float
    changed_cell_accuracy: float
    changed_cell_count: int


def grid_metrics_against_current(
    prediction: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> GridMetrics:
    if not (
        prediction.shape == current.shape == target.shape == valid_mask.shape
    ):
        raise ValueError("all grid tensors must have identical shapes")
    valid = valid_mask.bool()
    if not bool(valid.any()):
        raise ValueError("at least one valid cell is required")

    correct = (prediction == target) & valid
    cell_accuracy = correct.sum().item() / valid.sum().item()
    exact = (((prediction == target) | ~valid).flatten(1)).all(dim=1)
    changed = valid & (current != target)
    changed_count = int(changed.sum().item())
    changed_accuracy = (
        ((prediction == target) & changed).sum().item() / changed_count
        if changed_count
        else 1.0
    )
    return GridMetrics(
        cell_accuracy=float(cell_accuracy),
        exact_grid_accuracy=float(exact.float().mean().item()),
        changed_cell_accuracy=float(changed_accuracy),
        changed_cell_count=changed_count,
    )


def persistence_baseline(batch: TransitionBatch) -> tuple[torch.Tensor, GridMetrics]:
    prediction = batch.current.clone()
    # If the next grid is larger than current, padded cells cannot be predicted
    # by persistence; leave the PAD token so the metric counts them as wrong.
    metrics = grid_metrics_against_current(
        prediction, batch.current, batch.next_grid, batch.next_valid_mask
    )
    return prediction, metrics


def binary_brier(logits: torch.Tensor, target: torch.Tensor) -> float:
    if logits.shape != target.shape:
        raise ValueError("logits and target must have identical shapes")
    probs = logits.sigmoid()
    return float(torch.mean((probs - target.float()) ** 2).item())


def expected_calibration_error(
    logits: torch.Tensor, target: torch.Tensor, *, bins: int = 10
) -> float:
    if bins < 2:
        raise ValueError("bins must be >=2")
    if logits.shape != target.shape:
        raise ValueError("logits and target must have identical shapes")
    probs = logits.sigmoid().flatten()
    target = target.float().flatten()
    if probs.numel() == 0:
        raise ValueError("empty tensors are not valid")
    ece = torch.tensor(0.0, device=probs.device)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probs.device)
    for i in range(bins):
        if i == bins - 1:
            mask = (probs >= edges[i]) & (probs <= edges[i + 1])
        else:
            mask = (probs >= edges[i]) & (probs < edges[i + 1])
        if bool(mask.any()):
            weight = mask.float().mean()
            confidence = probs[mask].mean()
            accuracy = target[mask].mean()
            ece = ece + weight * torch.abs(confidence - accuracy)
    return float(ece.item())
