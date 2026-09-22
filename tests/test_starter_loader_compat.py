"""Regression for the official starter's importlib loading pattern.

The starter creates a module from a spec and executes it without inserting the
temporary module into sys.modules first. Some decorators (notably dataclasses on
Python 3.12) fail under that pattern. Keep the Kaggle drop-in import-safe.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


def test_agent_imports_with_official_starter_loader_pattern():
    code = textwrap.dedent(
        r'''
        import importlib.util
        import sys
        import types

        class DummyAgent:
            @property
            def name(self):
                return "dummy"

        class DummyState:
            WIN = "WIN"
            NOT_PLAYED = "NOT_PLAYED"
            GAME_OVER = "GAME_OVER"

        class DummyAction:
            RESET = object()

        arcengine = types.ModuleType("arcengine")
        arcengine.FrameData = object
        arcengine.GameAction = DummyAction
        arcengine.GameState = DummyState
        sys.modules["arcengine"] = arcengine

        agents = types.ModuleType("agents")
        agent_mod = types.ModuleType("agents.agent")
        agent_mod.Agent = DummyAgent
        agents.agent = agent_mod
        sys.modules["agents"] = agents
        sys.modules["agents.agent"] = agent_mod

        spec = importlib.util.spec_from_file_location(
            "user_agent_module", "agent/my_agent.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        assert "user_agent_module" not in sys.modules
        spec.loader.exec_module(module)
        assert module.MyAgent.__name__ == "MyAgent"
        '''
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
