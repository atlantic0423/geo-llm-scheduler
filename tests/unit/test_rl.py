"""State, reward, seeded action ties and controller lifecycle golden checks."""

import random
from dataclasses import replace

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.rl.controller import Controller, reward
from geo_llm_scheduler.rl.state import encode, extract, preference


def test_42_states():
    assert {encode(p, d, s) for p in range(3) for d in range(7) for s in (False, True)} == set(
        range(42)
    )
    assert preference(0, 100) == 2 and preference(99, 100) == 0
    assert extract(0, (0,) * 6, 0, Config()) == 28
    with pytest.raises(ValueError):
        encode(3, 0, False)


def test_reward_cases():
    assert reward(10, 10, 0) == -1
    assert reward(10, 10, 1) == 0
    assert reward(100, 99, 1) == pytest.approx(1)
    assert reward(100, 50, 1) == 10


def test_controller_reset_ties_and_bandit():
    config = replace(Config(), epsilon_start=0, epsilon_end=0)
    a, b = Controller(config), Controller(config)
    ra, rb = random.Random(4), random.Random(4)
    assert [a.select(0, 0, ra) for _ in range(20)] == [b.select(0, 0, rb) for _ in range(20)]
    a.q[1][0] = 10
    a.update(0, 1, 1, 1)
    assert a.q[0][0] == pytest.approx(0.3 * (1 + 0.7 * 10))
    bandit = Controller(replace(config, controller="bandit"))
    bandit.q[1][0] = 10
    bandit.update(0, 1, 1, 1)
    assert bandit.q[0][0] == 0.3
    assert Controller(config).q == [[0.0] * 8 for _ in range(42)]


def test_exploration_probability_one():
    c = Controller(replace(Config(), epsilon_start=1, epsilon_end=1))
    assert all(c.select(0, 0, random.Random(i))[1] for i in range(10))


def test_terminal_target_ignores_future_value_without_resetting_shared_q():
    controller = Controller(Config())
    controller.q[0][0] = 2.0
    controller.q[1][0] = 1000.0
    controller.update(0, 1, 5.0, 1, terminal=True)
    assert controller.q[0][0] == pytest.approx(2 + 0.3 * (5 - 2))
    assert controller.q[1][0] == 1000.0
    assert controller.updates[0][0] == 1
    controller.update(0, 1, 5.0, 1, terminal=False)
    assert controller.q[0][0] == pytest.approx(2.9 + 0.3 * (5 + 0.7 * 1000 - 2.9))
