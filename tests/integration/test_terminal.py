"""Only the final action is terminal; Polish never receives Q credit."""

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.rl.controller import Controller


@pytest.mark.parametrize("steps", [1, 3])
def test_trajectory_terminal_boundary(problem, monkeypatch, steps):
    calls = []
    original = Controller.update

    def observe(self, state, action, reward, next_state, terminal=False):
        calls.append((id(self), terminal))
        return original(self, state, action, reward, next_state, terminal)

    monkeypatch.setattr(Controller, "update", observe)
    config = Config(
        population=4,
        neighborhood=2,
        generations=1,
        method="full",
        trigger_mode="always",
        rl_steps=steps,
    )
    result = run(problem, config)
    assert [terminal for _, terminal in calls] == ([False] * (steps - 1) + [True]) * 4
    assert {identity for identity, _ in calls} == {id(result.controller)}
    assert sum(map(sum, result.controller.updates)) == 4 * steps
    assert result.gateway.counts["polish_calls"] == 4
