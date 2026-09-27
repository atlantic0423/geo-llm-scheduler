"""Counterexamples for state masking, budget cutoffs and independent price instances."""

import random
from dataclasses import replace

import pytest

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.generation20_instances import generate_generation20_instance
from geo_llm_scheduler.io.validation import validate_problem
from geo_llm_scheduler.macrosearch.budget import choose_budget
from geo_llm_scheduler.moead.core import NormalizationContext, weights
from geo_llm_scheduler.rl.controller import Controller


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("none", (1, 2, 3, 4, 5, 6, 7, 8)),
        ("no_a6", (1, 2, 3, 4, 5, 7, 8)),
        ("no_a4a5", (1, 2, 3, 6, 7, 8)),
        ("no_a4a5a6", (1, 2, 3, 7, 8)),
    ],
)
def test_mask_only_applies_at_flow_tou_stagnating(policy, expected):
    controller = Controller(replace(Config(), action_mask_policy=policy))
    assert controller.available_actions(9) == expected
    assert controller.available_actions(8) == tuple(range(1, 9))
    assert controller.available_actions(37) == tuple(range(1, 9))


def test_mask_constrains_exploration_and_bootstrap_without_penalty():
    config = replace(Config(), action_mask_policy="no_a4a5a6", epsilon_start=1, epsilon_end=1)
    controller = Controller(config)
    selected = {controller.select(9, 0, random.Random(seed))[0] for seed in range(100)}
    assert selected <= {1, 2, 3, 7, 8}
    assert selected
    controller.q[9][5] = 1000  # A6 is masked in next state.
    controller.q[9][6] = 2
    controller.update(8, 1, 1, 9)
    assert controller.q[8][0] == pytest.approx(0.3 * (1 + 0.7 * 2))
    assert controller.updates[9][5] == 0
    controller.update(8, 1, 1, 9, terminal=True)
    assert controller.q[8][0] < 1


def test_budget_cutoffs_are_separate_from_state_thresholds():
    base = Config(budget_policy="severity")
    args = (
        1,
        (0.13 * 0.9, 0, 0, 0, 0, 0),
        0,
        0,
        [],
        weights(100),
        NormalizationContext((0, 0), (1, 1)),
    )
    old = choose_budget(*args, base, random.Random(1))
    raised = choose_budget(
        *args, replace(base, severity_budget_cutoffs=(0.75, 1.5)), random.Random(1)
    )
    assert old == 3 and raised == 6
    assert (
        replace(base, severity_budget_cutoffs=(0.75, 1.5)).severity_thresholds
        == base.severity_thresholds
    )
    with pytest.raises(ValueError):
        replace(base, severity_budget_cutoffs=(1.5, 0.75))


def test_heterogeneous_prices_keep_hardware_homogeneous_and_tariffs_valid():
    normal = generate_generation20_instance(116, False)
    varied = generate_generation20_instance(116, True)
    validate_problem(varied)
    assert normal.jobs == varied.jobs and normal.instances == varied.instances
    assert len({r.demand_rate for r in varied.regions}) == 3
    assert len({r.tariffs[0].price for r in varied.regions}) == 3
    assert all(r.tariffs[-1].end == normal.regions[0].tariffs[-1].end for r in varied.regions)
