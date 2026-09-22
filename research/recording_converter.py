"""Convert ARC-AGI-3 framework recordings into System-One transitions.

The public ARC agent framework records each post-action FrameData as JSONL:
    {"timestamp": "...", "data": <FrameData.model_dump()>}

A FrameData contains the action_input that produced it. Therefore, for frame
records i-1 -> i, the transition is:
    state = last grid of frame i-1
    action = frame i.action_input
    next_state = last grid of frame i

The first frame record has no recorded predecessor and is intentionally skipped.
RESET transitions are excluded by default because E1 scores candidate actions 1..7.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .trajectory_schema import Transition, write_jsonl


FRAME_REQUIRED_KEYS = {"frame", "state", "levels_completed", "action_input"}


@dataclass(frozen=True)
class ConversionManifest:
    source_recording: str
    source_sha256: str
    source_events: int
    frame_events: int
    transitions: int
    game_ids: list[str]
    include_reset: bool
    arc_agents_commit: str
    arcengine_commit: str
    converter_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_frame_event(data: Any) -> bool:
    return isinstance(data, dict) and FRAME_REQUIRED_KEYS.issubset(data)


def _last_grid(frame_data: dict[str, Any]) -> list[list[int]] | None:
    frames = frame_data.get("frame")
    if not isinstance(frames, list) or not frames:
        return None
    grid = frames[-1]
    if not isinstance(grid, list):
        return None
    return [[int(v) for v in row] for row in grid]


def _action_id(raw: Any) -> int:
    if isinstance(raw, bool):
        raise ValueError("boolean is not a valid action id")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        s = raw.upper()
        if s == "RESET":
            return 0
        if s.startswith("ACTION") and s[6:].isdigit():
            return int(s[6:])
        if s.isdigit():
            return int(s)
    if isinstance(raw, dict):
        for key in ("value", "id"):
            if key in raw:
                return _action_id(raw[key])
    raise ValueError(f"unsupported action id representation: {raw!r}")


def _diff_count(a: list[list[int]], b: list[list[int]]) -> int:
    h = max(len(a), len(b))
    changed = 0
    for y in range(h):
        ra = a[y] if y < len(a) else []
        rb = b[y] if y < len(b) else []
        w = max(len(ra), len(rb))
        for x in range(w):
            va = ra[x] if x < len(ra) else -1
            vb = rb[x] if x < len(rb) else -1
            changed += int(va != vb)
    return changed


def load_frame_events(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    """Return validated frame-like payloads and total non-empty JSONL events."""
    path = Path(path)
    frames: list[dict[str, Any]] = []
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            total += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid JSON at {path}:{lineno}: {e}") from e
            data = event.get("data") if isinstance(event, dict) else None
            if _is_frame_event(data):
                frames.append(data)
    return frames, total


def convert_frame_events(
    frames: Iterable[dict[str, Any]], *, include_reset: bool = False
) -> list[Transition]:
    frames = list(frames)
    out: list[Transition] = []
    if len(frames) < 2:
        return out

    for i in range(1, len(frames)):
        prev = frames[i - 1]
        cur = frames[i]
        current_grid = _last_grid(prev)
        next_grid = _last_grid(cur)
        if current_grid is None or next_grid is None:
            continue

        action_input = cur.get("action_input")
        if not isinstance(action_input, dict):
            continue
        try:
            action_id = _action_id(action_input.get("id", 0))
        except ValueError:
            continue
        if not (0 <= action_id <= 7):
            continue
        if action_id == 0 and not include_reset:
            continue

        data = action_input.get("data")
        data = data if isinstance(data, dict) else {}
        click_x = int(data["x"]) if action_id == 6 and "x" in data else None
        click_y = int(data["y"]) if action_id == 6 and "y" in data else None
        if action_id == 6 and (
            click_x is None or click_y is None or not 0 <= click_x <= 63 or not 0 <= click_y <= 63
        ):
            continue

        game_id = str(cur.get("game_id") or prev.get("game_id") or "unknown")
        family = game_id.split("-")[0]
        prev_levels = int(prev.get("levels_completed", 0))
        cur_levels = int(cur.get("levels_completed", prev_levels))
        state = str(cur.get("state", ""))

        history_grid = current_grid
        if i >= 2:
            maybe_history = _last_grid(frames[i - 2])
            if maybe_history is not None:
                history_grid = maybe_history

        out.append(
            Transition(
                game_family=family,
                level_index=prev_levels,
                current_grid=current_grid,
                previous_grid=history_grid,
                action_id=action_id,
                click_x=click_x,
                click_y=click_y,
                next_grid=next_grid,
                progressed=cur_levels > prev_levels,
                died=state == "GAME_OVER",
                changed_cells=_diff_count(current_grid, next_grid),
            )
        )
    return out


def assign_family_split(
    game_family: str, *, validation_percent: int = 20, seed: str = "arc3-systemone-v1"
) -> str:
    """Deterministically assign an entire family to train or validation."""
    if not 0 <= validation_percent <= 100:
        raise ValueError("validation_percent must be between 0 and 100")
    digest = hashlib.blake2b(
        f"{seed}:{game_family}".encode("utf-8"), digest_size=8
    ).digest()
    bucket = int.from_bytes(digest, "big") % 100
    return "validation" if bucket < validation_percent else "train"


def convert_recording(
    source: str | Path,
    output_jsonl: str | Path,
    manifest_path: str | Path,
    *,
    include_reset: bool = False,
    arc_agents_commit: str = "4743e7d0aaae0ded0d98a89a7e282e63564cd58b",
    arcengine_commit: str = "b495c6acaf253c9681cd7b75c4299d352e9ce6f8",
) -> ConversionManifest:
    source = Path(source)
    frames, total = load_frame_events(source)
    transitions = convert_frame_events(frames, include_reset=include_reset)
    write_jsonl(output_jsonl, transitions)

    manifest = ConversionManifest(
        source_recording=source.name,
        source_sha256=_sha256(source),
        source_events=total,
        frame_events=len(frames),
        transitions=len(transitions),
        game_ids=sorted({str(f.get("game_id", "")) for f in frames if f.get("game_id")}),
        include_reset=include_reset,
        arc_agents_commit=arc_agents_commit,
        arcengine_commit=arcengine_commit,
    )
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return manifest
