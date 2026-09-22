"""Validate an ARC official-starter E0 gameplay log.

A successful process exit is not enough: a controller can loop on RESET and
still terminate at MAX_ACTIONS. This validator requires every played game to
execute at least one non-RESET environment action.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ACTION_RE = re.compile(r"^\s*([a-zA-Z0-9_-]+)\s+-\s+(RESET|ACTION[1-7]):")
SUMMARY_RE = re.compile(
    r"^\s*([a-zA-Z0-9_-]+)\s+levels=\s*(\d+)\s+actions=\s*(\d+)\s+state=(\S+)"
)
ALL_GAMES_RE = re.compile(r"playing all\s+(\d+)\s+games", re.IGNORECASE)


def validate_log(text: str, *, expected_games: int | None = None) -> dict:
    actions_by_game: dict[str, set[str]] = {}
    summary: dict[str, dict[str, object]] = {}
    announced_all: int | None = None
    in_summary = False

    for raw in text.splitlines():
        line = raw.strip()
        m_all = ALL_GAMES_RE.search(line)
        if m_all:
            announced_all = int(m_all.group(1))

        if "========= SUMMARY =========" in line:
            in_summary = True
            continue

        m_action = ACTION_RE.match(line)
        if m_action:
            game, action = m_action.groups()
            actions_by_game.setdefault(game, set()).add(action)
            continue

        if in_summary:
            m_summary = SUMMARY_RE.match(line)
            if m_summary:
                game, levels, actions, state = m_summary.groups()
                summary[game] = {
                    "levels": int(levels),
                    "actions": int(actions),
                    "state": state,
                }

    target_count = expected_games if expected_games is not None else announced_all
    if target_count is not None and len(summary) != target_count:
        raise ValueError(
            f"expected {target_count} summarized games, found {len(summary)}"
        )
    if not summary:
        raise ValueError("no per-game summary rows found")

    reset_only = sorted(
        game
        for game in summary
        if not any(a.startswith("ACTION") for a in actions_by_game.get(game, set()))
    )
    if reset_only:
        raise ValueError(
            "games executed no non-RESET action: " + ", ".join(reset_only)
        )

    zero_action_rows = sorted(
        game for game, row in summary.items() if int(row["actions"]) <= 1
    )
    if zero_action_rows:
        raise ValueError(
            "games did not advance beyond initialization/reset: "
            + ", ".join(zero_action_rows)
        )

    return {
        "games": len(summary),
        "announced_all": announced_all,
        "reset_only_games": reset_only,
        "levels_completed_total": sum(int(row["levels"]) for row in summary.values()),
        "actions_total": sum(int(row["actions"]) for row in summary.values()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--expected-games", type=int)
    args = p.parse_args()

    report = validate_log(
        args.log.read_text(encoding="utf-8", errors="replace"),
        expected_games=args.expected_games,
    )
    print(
        "E0 log valid: "
        f"{report['games']} games, "
        f"{report['actions_total']} actions, "
        f"{report['levels_completed_total']} levels completed"
    )


if __name__ == "__main__":
    main()
