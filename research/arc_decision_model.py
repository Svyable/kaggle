"""Grid-native action-conditioned dynamics model for ARC-AGI-3.

This is the Phase-1 learned replacement for MyAgent._score_candidate().
It does NOT generate language. Given a current/previous grid plus an action,
it predicts:
  - next-cell distribution (16 ARC values),
  - probability of any visible state change,
  - probability of level progress,
  - probability of terminal/game-over transition.

At inference, score every legal action (and selected ACTION6 coordinates) in a
single batched forward pass. Calibrate scalar heads on held-out trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DynamicsOutput:
    next_grid_logits: torch.Tensor  # [B, 16, H, W]
    p_change_logit: torch.Tensor    # [B]
    p_progress_logit: torch.Tensor  # [B]
    p_death_logit: torch.Tensor     # [B]
    valid_mask: torch.Tensor | None = None  # [B,H,W]; False marks padding


class ResidualBlock(nn.Module):
    def __init__(self, width: int, cond_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, width)
        self.norm2 = nn.GroupNorm(8, width)
        self.film = nn.Linear(cond_dim, 2 * width)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.conv1(F.gelu(self.norm1(x)))
        h = h * (1.0 + scale) + shift
        h = self.conv2(F.gelu(self.norm2(h)))
        return x + h


class ARCDecisionDynamics(nn.Module):
    """Small first model; intentionally cheap enough for exhaustive action scoring."""

    def __init__(self, width: int = 96, depth: int = 6, cond_dim: int = 128) -> None:
        super().__init__()
        # ARC colors are 0..15; 16 is an internal padding token for batching.
        self.color_emb = nn.Embedding(17, 24)
        self.prev_color_emb = nn.Embedding(17, 24)
        self.x_emb = nn.Embedding(64, 8)
        self.y_emb = nn.Embedding(64, 8)

        in_channels = 24 + 24 + 8 + 8 + 2  # changed + valid-cell channels
        self.stem = nn.Conv2d(in_channels, width, 3, padding=1)

        self.action_emb = nn.Embedding(8, 48)  # ids 0..7; RESET is normally excluded
        self.click_x_emb = nn.Embedding(65, 24)  # index 64 means N/A
        self.click_y_emb = nn.Embedding(65, 24)
        self.cond = nn.Sequential(
            nn.Linear(48 + 24 + 24, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.blocks = nn.ModuleList([ResidualBlock(width, cond_dim) for _ in range(depth)])
        self.next_grid_head = nn.Conv2d(width, 16, 1)
        self.global_head = nn.Sequential(
            nn.Linear(width + cond_dim, width),
            nn.GELU(),
            nn.Linear(width, 3),
        )

    def forward(
        self,
        current: torch.Tensor,   # [B,H,W], int64 in 0..16 (16=padding)
        previous: torch.Tensor,  # [B,H,W], int64 in 0..16 (16=padding)
        action_id: torch.Tensor, # [B], 1..7
        click_x: torch.Tensor | None = None,
        click_y: torch.Tensor | None = None,
    ) -> DynamicsOutput:
        b, h, w = current.shape
        device = current.device
        if h > 64 or w > 64:
            raise ValueError("ARC grids must be <=64x64")

        if bool(((current < 0) | (current > 16)).any()) or bool(
            ((previous < 0) | (previous > 16)).any()
        ):
            raise ValueError("grid tensors must contain ARC colors 0..15 or padding token 16")

        yy = torch.arange(h, device=device)[None, :, None].expand(b, h, w)
        xx = torch.arange(w, device=device)[None, None, :].expand(b, h, w)
        valid_mask = current != 16
        previous_valid = previous != 16
        changed = (
            (current != previous) & valid_mask & previous_valid
        ).float().unsqueeze(-1)
        valid_feature = valid_mask.float().unsqueeze(-1)

        features = torch.cat(
            [
                self.color_emb(current),
                self.prev_color_emb(previous),
                self.x_emb(xx),
                self.y_emb(yy),
                changed,
                valid_feature,
            ],
            dim=-1,
        ).permute(0, 3, 1, 2)

        na = torch.full((b,), 64, dtype=torch.long, device=device)
        cx = na if click_x is None else torch.where(action_id == 6, click_x, na)
        cy = na if click_y is None else torch.where(action_id == 6, click_y, na)
        cond = self.cond(
            torch.cat(
                [self.action_emb(action_id), self.click_x_emb(cx), self.click_y_emb(cy)],
                dim=-1,
            )
        )

        x = self.stem(features)
        for block in self.blocks:
            x = block(x, cond)

        next_logits = self.next_grid_head(x)
        valid_2d = valid_mask.float().unsqueeze(1)
        denom = valid_2d.sum(dim=(-2, -1)).clamp_min(1.0)
        pooled = (x * valid_2d).sum(dim=(-2, -1)) / denom
        scalar = self.global_head(torch.cat([pooled, cond], dim=-1))
        return DynamicsOutput(
            next_grid_logits=next_logits,
            p_change_logit=scalar[:, 0],
            p_progress_logit=scalar[:, 1],
            p_death_logit=scalar[:, 2],
            valid_mask=valid_mask,
        )


def training_loss(
    out: DynamicsOutput,
    next_grid: torch.Tensor,
    changed: torch.Tensor,
    progressed: torch.Tensor,
    died: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dense world-model loss + proper probabilistic losses for decision heads."""
    # Padding token 16 is deliberately outside the 16-class output space and
    # is ignored rather than rewarded as static background.
    grid_ce = F.cross_entropy(out.next_grid_logits, next_grid.long(), ignore_index=16)
    change_bce = F.binary_cross_entropy_with_logits(out.p_change_logit, changed.float())
    progress_bce = F.binary_cross_entropy_with_logits(out.p_progress_logit, progressed.float())
    death_bce = F.binary_cross_entropy_with_logits(out.p_death_logit, died.float())

    # Sparse outcomes need greater emphasis than the very dense next-grid target.
    total = grid_ce + 0.5 * change_bce + 2.0 * progress_bce + 2.0 * death_bce
    return total, {
        "grid_ce": grid_ce.detach(),
        "change_bce": change_bce.detach(),
        "progress_bce": progress_bce.detach(),
        "death_bce": death_bce.detach(),
    }


@torch.no_grad()
def decision_utility(out: DynamicsOutput) -> torch.Tensor:
    """Initial utility; tune only on held-out games, never on private leaderboard noise."""
    p_change = out.p_change_logit.sigmoid()
    p_progress = out.p_progress_logit.sigmoid()
    p_death = out.p_death_logit.sigmoid()

    # Predictive entropy of next-cell distributions is a useful uncertainty proxy.
    probs = out.next_grid_logits.softmax(dim=1)
    per_cell_entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
    if out.valid_mask is not None:
        mask = out.valid_mask.float()
        entropy = (per_cell_entropy * mask).sum(dim=(-2, -1)) / mask.sum(
            dim=(-2, -1)
        ).clamp_min(1.0)
    else:
        entropy = per_cell_entropy.mean(dim=(-2, -1))
    entropy = entropy / torch.log(torch.tensor(16.0, device=entropy.device))

    # High progress and safe change are good; uncertainty can justify exploration,
    # but never enough to overwhelm substantial death risk.
    return 5.0 * p_progress + 0.8 * p_change + 0.35 * entropy - 4.0 * p_death
