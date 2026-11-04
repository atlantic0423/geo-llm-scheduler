"""Independent numerical checks for the attributed Python baseline mechanisms."""

from dataclasses import replace

import numpy as np
import pytest
from test_lsv2_fast_rightshift import synthetic

from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.trans_cmoeaaop import de_best, portfolio, run_cmoeaaop, sample
from geo_llm_scheduler.experiments.trans_cmoeacd import (
    diversity_archive,
    run_cmoeacd,
    update_archive,
)
from geo_llm_scheduler.experiments.trans_cmoflcs import (
    fuzzy_probabilities,
    indicators,
    membership,
    run_cmoflcs,
)
from geo_llm_scheduler.experiments.trans_common import (
    Individual,
    SearchContext,
    de,
    decisions,
    dominance_mask,
    ga,
    hv2,
    objectives,
    select_spea,
    spea_fitness,
    tournament,
    truncate,
)
from geo_llm_scheduler.experiments.trans_ddpg import DDPG, Network


def fake_population(values):
    problem, _ = synthetic(4)
    context = SearchContext(problem, 1, 100)
    base = context.initialize(4)[0]
    return [
        Individual(
            np.full(16, i / (len(values) + 1)),
            replace(
                base.candidate,
                evaluation=replace(base.candidate.evaluation, flow=a, tou=b, demand=(0.0,)),
            ),
        )
        for i, (a, b) in enumerate(values)
    ]


def test_known_pareto_strength_truncation_and_hv():
    f = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [3.0, 3.0], [1.0, 4.0]])
    assert dominance_mask(f).tolist() == [0, 1, 2]
    assert dominance_mask(f[::-1]).tolist() == [0, 2, 3]
    assert dominance_mask(np.array([[1.0, 1.0], [1.0 + 1e-12, 1.0]]), round10=True).tolist() == [0]
    assert dominance_mask(np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([1.0, 0.0])).tolist() == [1]
    assert hv2(f, np.array([5.0, 5.0])) == 11
    assert hv2(f + 10, np.array([5.0, 5.0])) == 0
    fitness = spea_fitness(f)
    # Only (2,2) dominates (3,3), so raw strength of the latter is one.
    assert np.floor(fitness).tolist() == [0, 0, 0, 1, 0]
    assert spea_fitness(np.array([[0.0, 0.0], [10.0, 10.0]]), np.array([1.0, 0.0]))[0] > 1
    assert truncate(np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 3) == [1, 2, 3]
    with pytest.raises(ValueError):
        truncate(f, 0)
    pop = fake_population(f)
    selected, _, membership_indices = select_spea(pop, 3)
    assert len(selected) == len(membership_indices) == 3
    assert set(map(tuple, objectives(selected))) == {(1.0, 4.0), (2.0, 2.0), (4.0, 1.0)}
    selected, _, _ = select_spea(pop, 5)
    assert len(selected) == 5
    assert decisions(selected).shape == (5, 16)


def test_cd_archives_diversity_padding_and_truncation():
    pop = fake_population(
        [(1.0, 10.0), (3.0, 8.0), (5.0, 6.0), (7.0, 4.0), (9.0, 2.0), (10.0, 1.0), (11.0, 11.0)]
    )
    ideal = np.array([0.0, 0.0])
    assert len(update_archive(pop, 8, np.random.default_rng(2))) == 8
    assert len(update_archive(pop, 4, np.random.default_rng(2), ideal)) == 4
    associated = diversity_archive(pop, 3, ideal)
    assert associated[0] is pop[0] and associated[-1] is pop[5]
    problem, _ = synthetic(4)
    with pytest.raises(ValueError):
        run_cmoeacd(SearchContext(problem, 1, 1), 4)


def test_fuzzy_rules_known_shoulders_and_ordering():
    np.testing.assert_allclose(
        membership(np.array([0.0, 0.2, 0.35, 0.4, 1.0]), (0.0, 0.0, 0.3, 0.4)), [1, 1, 0.5, 0, 0]
    )
    np.testing.assert_allclose(membership(np.array([0.3, 0.5, 0.7]), (0.3, 0.5, 0.7)), [0, 1, 0])
    np.testing.assert_allclose(fuzzy_probabilities(np.ones(2), 0.5, 0.5), [0.5, 0.5])
    assert fuzzy_probabilities(np.array([9.0, 1.0]), 0.5, 0.0)[0] > 0.75
    assert fuzzy_probabilities(np.array([1.0, 9.0]), 0.0, 0.5)[1] > 0.75
    assert indicators(np.ones((4, 2))) == (0.5, 0.5)
    d, c = indicators(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    assert d == pytest.approx(1 / 3) and c == 0.5


def test_native_variation_and_aop_portfolio():
    pop = fake_population([(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0), (2.0, 4.0), (5.0, 5.0)])
    x = decisions(pop)
    rng = np.random.default_rng(55)
    for offspring in (
        ga(x, rng),
        ga(x, rng, half=False, eta_m=1),
        de(x, x[::-1], x, rng),
        de_best(pop, rng),
        portfolio(pop, np.arange(6), np.array([0.0, 0.0, 1.0]), rng),
    ):
        assert np.isfinite(offspring).all() and np.all((offspring >= 0) & (offspring <= 1))
        assert offspring.shape[1] == x.shape[1]
    assert portfolio(pop, np.arange(6), np.array([0.0, 0.0, 1.0]), rng).shape[0] == 14
    winners = tournament(np.arange(6, dtype=float), 100, rng)
    assert winners.min() >= 0 and winners.max() < 6
    old = np.array([[1.0, 4.0], [4.0, 1.0]])
    new = np.array([[1.0, 2.0], [2.0, 1.0]])
    s, ns, reward = sample(old, new, 0.4)
    np.testing.assert_allclose(s, [4.5, 4.5, 5.0, 5.0, 0.0, 0.4])
    assert reward > 0 and ns[2] == 3
    assert sample(np.zeros((2, 2)), np.zeros((2, 2)), 1)[2] == 0


@pytest.mark.parametrize("tanh", [True, False])
def test_network_gradients_against_finite_difference(tanh):
    rng = np.random.default_rng(23)
    net = Network((3, 4, 3, 2), rng, tanh)
    x = rng.normal(size=(5, 3))
    upstream = rng.normal(size=(5, 2))
    _, activations = net.forward(x)
    grads, input_gradient = net.backward(upstream, activations)
    eps = 1e-6
    for parameter, analytic in zip(net.weights + net.biases, grads, strict=True):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + eps
            plus = float(np.sum(net.forward(x)[0] * upstream))
            parameter[index] = original - eps
            minus = float(np.sum(net.forward(x)[0] * upstream))
            parameter[index] = original
            assert analytic[index] == pytest.approx((plus - minus) / (2 * eps), abs=2e-7)
    for index in np.ndindex(x.shape):
        original = x[index]
        x[index] = original + eps
        plus = float(np.sum(net.forward(x)[0] * upstream))
        x[index] = original - eps
        minus = float(np.sum(net.forward(x)[0] * upstream))
        x[index] = original
        assert input_gradient[index] == pytest.approx((plus - minus) / (2 * eps), abs=2e-7)


def test_ddpg_learning_ring_copy_targets_and_replay():
    rng = np.random.default_rng(9)
    a, b = DDPG(np.random.default_rng(2)), DDPG(np.random.default_rng(2))
    assert a.train() == (0.0, 0.0)
    for _ in range(510):
        state = rng.normal(size=6)
        for model in (a, b):
            action = model.select(state)
            model.store(state, action, 0.2, state + 0.1, True)
    before = a.actor.weights[0].copy()
    before_target = a.actor_target.weights[0].copy()
    np.testing.assert_allclose(a.train(), b.train())
    assert a.pointer == 10 and len(a.buffer) == 500
    assert not np.array_equal(before, a.actor.weights[0])
    np.testing.assert_allclose(
        a.actor_target.weights[0], 0.005 * a.actor.weights[0] + 0.995 * before_target
    )
    np.testing.assert_allclose(
        a.select(np.zeros(6), explore=False), b.select(np.zeros(6), explore=False)
    )
    with pytest.raises(FloatingPointError):
        a.actor.adam([np.full_like(p, np.nan) for p in a.actor.weights + a.actor.biases], 2, 0.1)


@pytest.mark.parametrize("wait", [False, True])
def test_adapter_replay_exact_boundaries_and_validation(wait):
    problem, _ = synthetic(8)
    a, b = [SearchContext(problem, 44, 100, wait_genes=wait) for _ in range(2)]
    pa, pb = a.initialize(12), b.initialize(12)
    assert [x.candidate for x in pa] == [x.candidate for x in pb]
    for x in pa:
        assert x.candidate.evaluation == evaluate(
            problem, x.candidate.genotype, x.candidate.schedule
        )
    assert a.evaluate(np.ones(a.dimensions)).candidate.evaluation.feasible
    assert a.evaluate(np.zeros(a.dimensions)).candidate.evaluation.feasible
    for invalid in (np.array([0.0]), np.full(a.dimensions, np.nan)):
        with pytest.raises(ValueError):
            a.evaluate(invalid)
    with pytest.raises(ValueError):
        a.initialize(3)
    for seconds in (0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SearchContext(problem, 1, seconds)
    with pytest.raises(ValueError):
        SearchContext(problem, 1, 1, max_generations=-1)
    with pytest.raises(ValueError):
        SearchContext(replace(problem, jobs=()), 1, 1)
    a.started -= 200
    assert not a.running() and a.offspring(np.ones((3, a.dimensions))) == []


@pytest.mark.parametrize("runner", [run_cmoeacd, run_cmoflcs, run_cmoeaaop])
def test_native_loops_deterministic_and_exact(runner):
    problem, _ = synthetic(5)
    contexts = [SearchContext(problem, 99, 100, max_generations=3) for _ in range(2)]
    results = [runner(c, 6) for c in contexts]
    assert contexts[0].generations == 3
    assert [p.candidate for p in results[0]] == [p.candidate for p in results[1]]
    for c in contexts:
        for candidate in c.gateway.archive.members:
            assert evaluate(problem, candidate.genotype, candidate.schedule) == candidate.evaluation


def test_aop_transfer_phase_and_actual_training():
    problem, _ = synthetic(4)
    ctx = SearchContext(problem, 9, 100, max_generations=34)
    ctx.progress = lambda: 0.5
    run_cmoeaaop(ctx, 6)
    assert ctx.trace[-1]["replay_size"] == 34
    assert ctx.trace[-1]["transfer"]
    assert np.isfinite(ctx.trace[-1]["critic_loss"])
    assert ctx.trace[-1]["critic_loss"] > 0
