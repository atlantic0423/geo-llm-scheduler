"""PSLGP structural invariants, preference behavior, surrogate and exact P/D adaptation."""

from dataclasses import replace

import numpy as np
import pytest
from test_lsv2_fast_rightshift import synthetic

from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.trans_common import SearchContext
from geo_llm_scheduler.experiments.trans_gp import (
    Node,
    normalize,
    paths,
    random_tree,
    subtree,
    variation,
)
from geo_llm_scheduler.experiments.trans_gp import (
    replace as replace_tree,
)
from geo_llm_scheduler.experiments.trans_pslgp import (
    Dispatcher,
    criteria,
    nearest,
    preselect,
    ranks,
    run_pslgp,
)


def test_gp_primitives_variation_depth_and_pure_inputs():
    features = np.array([[2.0, 0.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.7]])
    expected = {"+": 2.0, "-": 2.0, "*": 0.0, "/": 1.0, "min": 0.0, "max": 2.0}
    for symbol, result in expected.items():
        assert Node(symbol, (Node(0), Node(1))).evaluate(features)[0] == result
    assert Node("/", (Node(0), Node(2))).evaluate(features)[0] == 0.5
    with pytest.raises(ValueError):
        Node("bad", (Node(0), Node(1))).evaluate(features)
    tree = Node("+", (Node(0), Node("*", (Node(1), Node(2)))))
    assert tree.depth() == 3 and len(paths(tree)) == 5
    assert subtree(tree, (1, 1)) == Node(2)
    assert replace_tree(tree, (1, 0), Node(3)).children[1].children[0] == Node(3)
    rng = np.random.default_rng(11)
    policy = (tree, Node(8))
    for _ in range(400):
        other = (random_tree(rng, 6, full=True), random_tree(rng, 4, full=False))
        policy = variation(policy, other, rng)
        assert max(t.depth() for t in policy) <= 8
        assert all(np.isfinite(t.evaluate(features)).all() for t in policy)
    raw = np.vstack((features, features * 2))
    normalized = normalize(raw)
    np.testing.assert_allclose(normalized[:, 8:], raw[:, 8:])
    np.testing.assert_allclose(raw[0], features[0])


def test_pc_detects_preference_changes_even_when_policy_is_reference():
    problem, _ = synthetic(4)
    context = SearchContext(problem, 4, 100)
    dispatcher = Dispatcher(context)
    # One terminal favors first candidate, the other favors last candidate.
    decision = np.zeros((20, 7, 10))
    decision[:, :, 0] = np.arange(7)
    decision[:, :, 3] = np.arange(6, -1, -1)
    dispatcher.pc_sets = [decision.copy(), decision.copy()]
    tree = Node("+", (Node("*", (Node(8), Node(0))), Node("*", (Node(9), Node(3)))))
    policy = (tree, tree)
    pc = dispatcher.characterize(policy, np.array([[1.0, 0.0], [0.0, 1.0]]), policy)
    assert pc.shape == (2, 40) and not np.array_equal(pc[0], pc[1])
    assert context.evaluations == 0


def test_surrogate_known_neighbors_ranks_diversity_and_niches():
    training = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    query = np.tile(np.array([[1.1, 1.1], [0.5, 0.5], [2.0, 2.0]]), (100, 1))
    assert nearest(query, training).tolist() == [1, 0, 2] * 100
    assert ranks(
        np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [1.0, 3.0]])
    ).tolist() == [0, 0, 1, 2, 0]
    pcs = np.array([[[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]])
    values = np.array([[[1.0, 3.0]], [[3.0, 1.0]]])
    keys = criteria(values, pcs, np.array([[1.0, 0.0], [0.0, 1.0]]), np.ones(2))
    assert keys[0][:2] == (0, -0.0) and keys[0][2:] == (2.0, -1.0)
    selected = preselect([(0,), (1,), (2,), (3,)], np.array([[0.0], [0.0], [1.0], [2.0]]), 3)
    assert selected == [0, 2, 3]
    assert preselect([(0,), (1,)], np.zeros((2, 3)), 2) == [0, 1]


def test_dispatch_and_two_generations_use_exact_schedules_only():
    problem, _ = synthetic(8)
    problem = replace(
        problem,
        jobs=tuple(
            replace(j, release=i, decode=replace(j.decode, duration=10 + i))
            for i, j in enumerate(problem.jobs)
        ),
    )
    a, b = [SearchContext(problem, 22, 100, max_generations=2) for _ in range(2)]
    ra, rb = [run_pslgp(c, 8, preferences_count=5) for c in (a, b)]
    assert a.generations == b.generations == 2
    assert a.evaluations == 17  # one manual rule and two populations; surrogate free of exact calls
    assert [x.candidate for x in ra] == [x.candidate for x in rb]
    assert a.trace[0]["brood_multiplier"] == 5
    assert a.trace[0]["main_preference"] != a.trace[1]["main_preference"]
    for candidate in a.gateway.archive.members:
        assert candidate.evaluation == evaluate(problem, candidate.genotype, candidate.schedule)
    with pytest.raises(ValueError):
        run_pslgp(a, 2)


def test_pslgp_stops_on_partial_generation_without_estimating_archive():
    problem, _ = synthetic(4)
    ctx = SearchContext(problem, 33, 100, max_generations=3)
    ctx.progress = lambda: 1.0 if ctx.evaluations >= 3 else 0.0
    result = run_pslgp(ctx, 8, preferences_count=3)
    assert len(result) == 2 and ctx.evaluations == 3
    assert ctx.trace[-1]["partial_generation"]
