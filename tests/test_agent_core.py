"""Behavioral smoke tests for the self-contained Kaggle drop-in.

We stub the ARC runtime so this test can run outside Kaggle. The goal is to
exercise our controller logic, not to reimplement ARC.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from enum import Enum


class FakeState(Enum):
    NOT_PLAYED = 0
    NOT_FINISHED = 1
    WIN = 2
    GAME_OVER = 3


class FakeAction:
    _registry = {}

    def __init__(self, action_id: int, name: str):
        self.value = action_id
        self.name = name
        self.data = {}
        self.reasoning = None
        FakeAction._registry[action_id] = self

    @classmethod
    def from_id(cls, action_id: int):
        return cls._registry[action_id]

    def set_data(self, data):
        self.data = dict(data)


for i in range(1, 8):
    setattr(FakeAction, f"ACTION{i}", FakeAction(i, f"ACTION{i}"))
FakeAction.RESET = FakeAction(0, "RESET")


@dataclass
class FakeFrameData:
    frame: list = field(default_factory=list)
    state: FakeState = FakeState.NOT_FINISHED
    levels_completed: int = 0
    available_actions: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    full_reset: bool = False


class FakeAgent:
    def __init__(self, *args, **kwargs):
        self.game_id = kwargs.get("game_id", "fake")

    @property
    def name(self):
        return self.__class__.__name__.lower()


arcengine = types.ModuleType("arcengine")
arcengine.FrameData = FakeFrameData
arcengine.GameAction = FakeAction
arcengine.GameState = FakeState
sys.modules["arcengine"] = arcengine

agents = types.ModuleType("agents")
agent_mod = types.ModuleType("agents.agent")
agent_mod.Agent = FakeAgent
agents.agent = agent_mod
sys.modules["agents"] = agents
sys.modules["agents.agent"] = agent_mod

spec = importlib.util.spec_from_file_location("my_agent_test", "agent/my_agent.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
MyAgent = module.MyAgent
DecisionKey = module.DecisionKey


def frame(grid, *, state=FakeState.NOT_FINISHED, levels=0, actions=None):
    return FakeFrameData(
        frame=[grid],
        state=state,
        levels_completed=levels,
        available_actions=[1, 2, 3, 4, 5, 6, 7] if actions is None else actions,
    )


def test_required_reset():
    a = MyAgent(game_id="x")
    obs = frame([[0]], state=FakeState.NOT_PLAYED)
    assert a.choose_action([], obs).name == "RESET"


def test_proven_noop_is_penalized():
    a = MyAgent(game_id="x")
    obs = frame([[0, 1], [1, 0]], actions=[1, 2])
    first = a.choose_action([], obs)
    assert first.value in (1, 2)

    # Same board after first action => exact state/action no-op learned.
    second = a.choose_action([], obs)
    assert second.value in (1, 2)
    assert second.value != first.value


def test_action6_coordinates_are_bounded():
    a = MyAgent(game_id="x")
    obs = frame([[1, 1, 0], [1, 0, 2]], actions=[6])
    action = a.choose_action([], obs)
    assert action.value == 6
    assert 0 <= action.data["x"] <= 63
    assert 0 <= action.data["y"] <= 63


def test_level_progress_updates_statistics():
    a = MyAgent(game_id="x")
    obs0 = frame([[0]], levels=0, actions=[1])
    action = a.choose_action([], obs0)
    assert action.value == 1

    obs1 = frame([[1]], levels=1, actions=[1])
    a.choose_action([], obs1)
    assert a.action_id_stats[1].progress == 1
    assert a.action_id_stats[1].trials == 1


def test_full_reset_clears_online_memory_and_resets():
    a = MyAgent(game_id="x")
    a.action_id_stats[1].trials = 3
    a.proven_noops.add(("deadbeef", DecisionKey(1)))
    a.recent.append(DecisionKey(1))

    obs = frame([[0]], state=FakeState.NOT_FINISHED, actions=[1])
    obs.full_reset = True
    action = a.choose_action([], obs)

    assert action.name == "RESET"
    assert not a.action_id_stats
    assert not a.proven_noops
    assert not a.recent
    assert a.last_decision is None
