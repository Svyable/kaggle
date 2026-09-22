"""Collect action-aligned ARC-AGI-3 trajectories directly.

Why direct collection?
The current public ARC-AGI-3-Agents framework's FrameData conversion omits
raw.action_input before Recorder serializes frames. The documented recording
schema includes action_input, but current recordings can therefore lose the
chosen action. This collector writes Transition rows at action time instead of
reconstructing them later.

The runtime imports are intentionally lazy so repository unit tests do not
need arc-agi installed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

from .trajectory_schema import Transition, write_jsonl


@dataclass(frozen=True)
class GameCollection:
    requested_game_id: str
    observed_game_id: str
    transitions: int
    environment_actions: int
    resets: int
    levels_completed: int
    final_state: str
    action_counts: dict[int, int]


@dataclass(frozen=True)
class CollectionManifest:
    schema_version: int
    collector: str
    starter_sha: str
    framework_sha: str
    arc_agi_version: str
    arcengine_version: str
    max_steps: int
    games: list[GameCollection]
    total_transitions: int


def _last_grid(frame: Any) -> list[list[int]]:
    frames = getattr(frame, "frame", None)
    if not frames:
        return []
    return [[int(v) for v in row] for row in frames[-1]]


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


def _state_name(state: Any) -> str:
    value = getattr(state, "value", state)
    return str(value)


def transition_from_step(
    *,
    game_family: str,
    previous_grid: list[list[int]],
    current_frame: Any,
    action: Any,
    next_frame: Any,
) -> Transition | None:
    """Create one action-aligned Transition; RESET is intentionally excluded."""
    action_id = int(getattr(action, "value"))
    if action_id == 0:
        return None
    if not 1 <= action_id <= 7:
        raise ValueError(f"invalid ARC action id {action_id}")

    current_grid = _last_grid(current_frame)
    next_grid = _last_grid(next_frame)
    if not current_grid or not next_grid:
        raise ValueError("non-RESET transition requires current and next grids")

    click_x = None
    click_y = None
    if action_id == 6:
        action_data = getattr(action, "action_data", None)
        click_x = int(getattr(action_data, "x"))
        click_y = int(getattr(action_data, "y"))
        if not (0 <= click_x <= 63 and 0 <= click_y <= 63):
            raise ValueError(f"ACTION6 coordinates out of bounds: {(click_x, click_y)}")

    before_levels = int(getattr(current_frame, "levels_completed", 0))
    after_levels = int(getattr(next_frame, "levels_completed", before_levels))
    return Transition(
        game_family=game_family,
        level_index=before_levels,
        current_grid=current_grid,
        previous_grid=previous_grid or current_grid,
        action_id=action_id,
        click_x=click_x,
        click_y=click_y,
        next_grid=next_grid,
        progressed=after_levels > before_levels,
        died=_state_name(getattr(next_frame, "state", "")) == "GAME_OVER",
        changed_cells=_diff_count(current_grid, next_grid),
    )


def collect_game(
    *,
    agent: Any,
    env: Any,
    game_family: str,
    max_steps: int,
) -> tuple[list[Transition], GameCollection]:
    """Mirror Agent.main while recording the chosen action before it is lost."""
    transitions: list[Transition] = []
    action_counts: Counter[int] = Counter()
    resets = 0
    previous_grid: list[list[int]] = []

    agent.MAX_ACTIONS = max_steps
    while (
        not agent.is_done(agent.frames, agent.frames[-1])
        and agent.action_counter <= max_steps
    ):
        current = agent._convert_raw_frame_data(env.observation_space)
        action = agent.choose_action(agent.frames, current)
        action_id = int(action.value)
        action_counts[action_id] += 1

        next_frame = agent.take_action(action)
        if next_frame is None:
            raise RuntimeError(
                f"{game_family}: environment returned an invalid frame for action {action_id}"
            )

        if action_id == 0:
            resets += 1
        else:
            transition = transition_from_step(
                game_family=game_family,
                previous_grid=previous_grid or _last_grid(current),
                current_frame=current,
                action=action,
                next_frame=next_frame,
            )
            assert transition is not None
            transitions.append(transition)

        agent.append_frame(next_frame)
        agent.action_counter += 1

        # For a RESET, the returned fresh board is the only meaningful history
        # for the first environment action. Otherwise the current board becomes
        # the previous board on the next decision.
        previous_grid = (
            _last_grid(next_frame) if action_id == 0 else _last_grid(current)
        )

    agent.cleanup()
    final = agent.frames[-1]
    return transitions, GameCollection(
        requested_game_id=game_family,
        observed_game_id=str(getattr(final, "game_id", "") or game_family),
        transitions=len(transitions),
        environment_actions=sum(
            count for action_id, count in action_counts.items() if action_id != 0
        ),
        resets=resets,
        levels_completed=int(getattr(final, "levels_completed", 0)),
        final_state=_state_name(getattr(final, "state", "")),
        action_counts=dict(sorted(action_counts.items())),
    )


def _load_agent_class(agent_file: Path, framework_path: Path) -> type:
    sys.path.insert(0, str(framework_path))
    spec = importlib.util.spec_from_file_location("svyable_arc_agent", agent_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load agent from {agent_file}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec for decorators/introspection that consult sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise RuntimeError(f"{agent_file} must define MyAgent")
    return module.MyAgent


def _requested_game_ids(
    available: Iterable[Any], requested: str | None
) -> list[str]:
    available_ids = [str(e.game_id).split("-")[0] for e in available]
    if requested:
        wanted = [g.strip().split("-")[0] for g in requested.split(",") if g.strip()]
        missing = sorted(set(wanted) - set(available_ids))
        if missing:
            raise ValueError(f"unknown game ids: {missing}")
        return [g for g in available_ids if g in set(wanted)]
    return available_ids


def run_collection(
    *,
    output_dir: Path,
    agent_file: Path,
    framework_path: Path,
    max_steps: int,
    requested_games: str | None,
    starter_sha: str,
    framework_sha: str,
) -> CollectionManifest:
    import arc_agi
    from arc_agi import OperationMode

    MyAgent = _load_agent_class(agent_file, framework_path)
    arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    available = arcade.get_environments()
    game_ids = _requested_game_ids(available, requested_games)
    if not game_ids:
        raise RuntimeError("no public environments available")

    all_transitions: list[Transition] = []
    games: list[GameCollection] = []
    for i, game_id in enumerate(game_ids, 1):
        print(f"=== collect [{i}/{len(game_ids)}] {game_id} ===", flush=True)
        env = arcade.make(game_id)
        if env is None:
            raise RuntimeError(f"could not create environment {game_id}")
        agent = MyAgent(
            card_id="e1-collector",
            game_id=game_id,
            agent_name=f"MyAgent.collect.{game_id}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["e1-collection"],
        )
        transitions, game = collect_game(
            agent=agent,
            env=env,
            game_family=game_id,
            max_steps=max_steps,
        )
        if not transitions:
            raise RuntimeError(f"{game_id}: collected zero non-RESET transitions")
        all_transitions.extend(transitions)
        games.append(game)
        print(
            f"collected {game.transitions} transitions, "
            f"levels={game.levels_completed}, actions={game.action_counts}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "transitions.jsonl", all_transitions)
    manifest = CollectionManifest(
        schema_version=1,
        collector="direct-action-aligned-v1",
        starter_sha=starter_sha,
        framework_sha=framework_sha,
        arc_agi_version=importlib.metadata.version("arc-agi"),
        arcengine_version=importlib.metadata.version("arcengine"),
        max_steps=max_steps,
        games=games,
        total_transitions=len(all_transitions),
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                **asdict(manifest),
                "games": [asdict(g) for g in games],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"collection complete: {len(games)} games, "
        f"{len(all_transitions)} action-aligned transitions",
        flush=True,
    )
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--agent-file", type=Path, default=Path("agent/my_agent.py"))
    p.add_argument("--framework-path", type=Path, required=True)
    p.add_argument("--games", default=None, help="comma-separated short ids; omit for all")
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--starter-sha", default="unknown")
    p.add_argument("--framework-sha", default="unknown")
    args = p.parse_args()

    run_collection(
        output_dir=args.output_dir,
        agent_file=args.agent_file,
        framework_path=args.framework_path,
        max_steps=args.max_steps,
        requested_games=args.games,
        starter_sha=args.starter_sha,
        framework_sha=args.framework_sha,
    )


if __name__ == "__main__":
    main()
