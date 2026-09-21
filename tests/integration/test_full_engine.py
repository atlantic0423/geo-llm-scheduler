"""End-to-end short RL trajectories with deterministic core traces."""

from dataclasses import replace

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.engine.run import run


def deterministic(trace):
    return [
        {
            **{k: v for k, v in row.items() if k != "elapsed"},
            "steps": [
                {k: v for k, v in step.items() if k not in ("seconds", "construction_seconds")}
                for step in row["steps"]
            ],
        }
        for row in trace
    ]


@pytest.mark.parametrize("controller", ["random", "bandit", "qlearning"])
def test_full_replay(problem, controller):
    config = Config(
        population=4,
        neighborhood=2,
        generations=1,
        rl_steps=2,
        method="full",
        controller=controller,
        trigger_mode="always",
        fixed_budget=3,
    )
    a, b = run(problem, config), run(problem, config)
    assert deterministic(a.trace) == deterministic(b.trace)
    assert a.gateway.archive.attempts == a.gateway.counts["feasible"]
    assert a.gateway.counts["polish_calls"] == 4
    assert sum(a.controller.visits) == 8
    assert all(s["effective"] <= s["budget"] for row in a.trace for s in row["steps"])


@pytest.mark.parametrize("policy", ["fixed", "static", "random", "severity", "coverage"])
def test_budget_policies(problem, policy):
    config = replace(
        Config(),
        population=4,
        neighborhood=2,
        generations=1,
        rl_steps=1,
        method="full",
        trigger_mode="always",
        budget_policy=policy,
    )
    result = run(problem, config)
    assert all(s["budget"] in (3, 6, 10) for row in result.trace for s in row["steps"])
