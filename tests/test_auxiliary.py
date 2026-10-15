"""Independent geometry, bounded budget, online transition and exact regression checks."""

import itertools
import random
from dataclasses import replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config, load_config
from geo_llm_scheduler.domain.models import (
    Candidate,
    EvaluationResult,
    Genotype,
    ProblemInstance,
    Schedule,
)
from geo_llm_scheduler.engine import auxiliary
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.macrosearch.distance import components, operation_order, trajectory_signal
from geo_llm_scheduler.macrosearch.search import MacroSearchResult, ScoredCandidate, execute
from geo_llm_scheduler.macrosearch.trajectory_budget import ARM_SPECS, allocate
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.rl.controller import Controller
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.numeric import identity
from geo_llm_scheduler.utils.rng import RNGManager


def point(flow, bill, marker=0):
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    return Candidate(
        g,
        Schedule((marker, 100 + marker, marker, 100 + marker)),
        EvaluationResult(True, flow, bill, (), ()),
        "unit",
    )


def test_history_rescores_unselected_points_after_ideal_scale_changes():
    a, b, c = point(1, 10, 1), point(4, 4, 2), point(10, 1, 3)
    history = [a, b, c]
    assert auxiliary.best_history(history, NormalizationContext((0, 0), (10, 10)), (0.5, 0.5)) is b
    assert auxiliary.best_history(history, NormalizationContext((1, 0), (10, 100)), (0.5, 0.5)) is a
    assert history == [a, b, c]
    with pytest.raises(ValueError):
        auxiliary.best_history([], NormalizationContext((0, 0), (10, 10)), (0.5, 0.5))


def test_operation_labels_and_kendall_against_independent_pair_counter(problem):
    rng = random.Random(781)
    for _ in range(50):
        a = [0, 0, 1, 1]
        b = [0, 0, 1, 1]
        rng.shuffle(a)
        rng.shuffle(b)
        ga = Genotype((0, 0, 0, 0), tuple(a))
        gb = replace(ga, os=tuple(b))
        x = Candidate(ga, decode(problem, ga), EvaluationResult(True, 1, 1, (), ()))
        y = Candidate(gb, decode(problem, gb), x.evaluation)
        oa, ob = operation_order(ga.os), operation_order(gb.os)
        ia = {o: i for i, o in enumerate(oa)}
        ib = {o: i for i, o in enumerate(ob)}
        wrong = sum(
            (ia[u] < ia[v]) != (ib[u] < ib[v]) for u, v in itertools.combinations(range(4), 2)
        )
        assert components(problem, x, y).order == wrong / 4
        assert components(problem, x, y) == components(problem, y, x)
    for invalid in ((0,), (0, 0, 0, 0), (0, 2, 0, 2)):
        with pytest.raises(ValueError):
            operation_order(invalid)


def test_timing_only_and_round_trip_efficiency(problem):
    x = point(1, 1, 0)
    y = point(1, 1, 50)
    d = components(problem, x, y)
    assert d.assignment == d.order == 0 and d.timing == pytest.approx(0.375)
    assert d.combined("mean", 7) == pytest.approx(0.125)
    assert d.combined("max", 7) == pytest.approx(0.375)
    assert d.combined("action", 7) == pytest.approx(0.375)
    assert d.combined("action", 1) == 0
    assert trajectory_signal(problem, [x, y, x], "max", 2, "efficiency", 8)[0] == 0
    assert trajectory_signal(problem, [x, y], "max", 2, "efficiency", 8)[0] == pytest.approx(1)
    assert trajectory_signal(problem, [x, x], "max", 2, "efficiency", 8)[0] == 0
    assert components(ProblemInstance((), (), ()), x, y).timing == 0
    with pytest.raises(ValueError):
        components(problem, replace(x, genotype=Genotype((), ())), y)
    for args in (
        ([], "max", 2, "endpoint", 1),
        ([x], "bad", 2, "endpoint", 1),
        ([x], "max", 0, "endpoint", 1),
        ([x], "action", 2, "efficiency", 1),
        ([x], "max", 2, "bad", 1),
    ):
        with pytest.raises(ValueError):
            trajectory_signal(problem, *args)


@pytest.mark.parametrize("name", ["D2", "D6"])
def test_causal_mapping_warmup_and_remaining_credit(name):
    spec = ARM_SPECS[name]
    widths = [
        allocate(spec, s, 1, 30, (0.1, 0.3), (0.4, 0.3, 0.3), random.Random(1))
        for s in (0, 0.2, 0.8)
    ]
    assert widths == ([3, 6, 10] if name == "D2" else [10, 6, 3])
    assert allocate(spec, 0, 0, 30, (0.1, 0.3), (0.4, 0.3, 0.3), random.Random(1)) == 6
    assert allocate(spec, 0.8, 1, 2, (0.1, 0.3), (0.4, 0.3, 0.3), random.Random(1)) == 2
    for signal, step, remaining in ((-1, 0, 10), (2, 0, 10), (0, -1, 10), (0, 0, 0)):
        with pytest.raises(ValueError):
            allocate(spec, signal, step, remaining, (0.1, 0.3), (0.4, 0.3, 0.3), random.Random(1))


def test_random_uses_explicit_stream_and_does_not_inspect_distance():
    a, b = random.Random(17), random.Random(17)
    spec = ARM_SPECS["R0"]
    left = [allocate(spec, 0, s, 30, (0.1, 0.3), (0.4, 0.3, 0.3), a) for s in range(20)]
    right = [allocate(spec, 1, s, 30, (0.1, 0.3), (0.4, 0.3, 0.3), b) for s in range(20)]
    assert left == right and set(left) == {3, 6, 10}


def test_aid_stops_at_first_different_nonworse_under_current_prefix_context():
    x = point(10, 10, 0)
    same = point(10, 10, 0)
    equal = point(10, 10, 1)
    better = point(5, 5, 2)

    class Gateway:
        ideal = (0.0, 0.0)

        def __init__(self):
            self.calls = 0

        def evaluate(self, *args):
            result = (same, equal, better)[self.calls]
            self.calls += 1
            return result

    gateway = Gateway()
    batch = ProposalBatch(
        [Proposal(x.genotype, Schedule((i, 100 + i, i, 100 + i)), (0, 1, 2, 3)) for i in range(3)]
    )
    result = execute(
        batch, x, 3, (1, 0), NormalizationContext((0, 0), (20, 20)), gateway, "unit", aid=True
    )
    assert gateway.calls == 2 and result.exact_evaluations == 2
    assert result.best.evaluation.flow == 10
    assert len(result.evaluated) == 2
    with pytest.raises(ValueError):
        execute(
            batch,
            x,
            3,
            (1, 0),
            NormalizationContext((0, 0), (20, 20)),
            gateway,
            "unit",
            sequential=True,
            aid=True,
        )


def test_walk_online_next_state_terminal_and_protected_output(problem, monkeypatch):
    start = point(10, 10, 0)
    points = [point(12, 10, 1), point(8, 10, 2), point(11, 10, 3)]
    config = Config(method="full", auxiliary_arm="F2", polish=False, enabled_operators=(8,))
    gateway = EvaluationGateway(problem, Archive())
    gateway.ideal = (0.0, 0.0)
    controller = Controller(config)
    terminal = []
    seen = []
    original_update = controller.update

    def update(s, a, r, n, terminal=False):
        seen.append((r, n))
        terminal_flags.append(terminal)
        original_update(s, a, r, n, terminal)

    terminal_flags = terminal
    monkeypatch.setattr(controller, "update", update)
    monkeypatch.setattr(
        auxiliary, "severities", lambda p, c: (float(c.evaluation.flow > 10), 0, 0, 0, 0, 0)
    )

    class Operator:
        action = 8

        def propose(self, *args):
            return ProposalBatch()

    def search(*args, **kwargs):
        candidate = points.pop(0)
        context = NormalizationContext((0, 0), (20, 20))
        return MacroSearchResult(
            10,
            1,
            1,
            1,
            candidate,
            (candidate,),
            (ScoredCandidate(candidate, context.scalar(candidate, (1, 0))),),
            context,
        )

    monkeypatch.setattr(auxiliary, "execute", search)
    output, rows = auxiliary.improve_auxiliary(
        start,
        0,
        0,
        0,
        ((1, 0),),
        (20, 20),
        gateway,
        controller,
        config,
        RNGManager(9),
        operator_overrides={8: Operator()},
    )
    assert [row["scalar_after"] for row in rows] == [0.6, 0.4, 0.55]
    assert output.evaluation.flow == 8 and rows[-1]["position_objectives"][0] == 11
    assert terminal == [False, False, True] and seen[0][0] == 0 and seen[1][0] > 0
    assert seen[0][1] != seen[1][1] and rows[0]["regression"]
    assert len(controller.q) == 42
    with pytest.raises(ValueError):
        auxiliary.improve_auxiliary(
            start,
            0,
            0,
            0,
            ((1, 0),),
            (20, 20),
            gateway,
            controller,
            config,
            RNGManager(9),
            operator_overrides={1: Operator()},
        )


@pytest.mark.parametrize("arm", tuple(ARM_SPECS))
def test_all_arms_keep_exact_feasible_archive_and_joint_caps(problem, arm):
    config = Config(
        population=4,
        neighborhood=2,
        generations=1,
        method="full",
        trigger_mode="always",
        auxiliary_arm=arm,
    )
    result = run(problem, config)
    assert result.offspring_count == 4
    assert len(result.controller.q) == 42
    for row in result.trace:
        steps = row["steps"]
        assert steps and steps[-1]["terminal"] and not any(s["terminal"] for s in steps[:-1])
        assert len(steps) <= ARM_SPECS[arm].steps
        assert sum(s["effective"] for s in steps) <= ARM_SPECS[arm].credit
    assert result.gateway.counts["polish_calls"] == 4
    for c in result.population + result.archive.members:
        assert c.evaluation == evaluate(problem, c.genotype, c.schedule)


def test_strict_reference_genes_timing_q_rng_and_archive_are_unchanged(problem):
    config = Config(
        population=6, neighborhood=3, generations=2, method="full", trigger_mode="always"
    )
    base = run(problem, config)
    strict = run(problem, replace(config, auxiliary_arm="S0"))
    assert [identity(c) for c in base.population] == [identity(c) for c in strict.population]
    assert base.controller.q == strict.controller.q
    assert base.gateway.counts["exact"] == strict.gateway.counts["exact"]
    assert {identity(c) for c in base.archive.members} == {
        identity(c) for c in strict.archive.members
    }


@pytest.mark.parametrize(
    "extra",
    [
        {"auxiliary_arm": "bad"},
        {"auxiliary_arm": "D1"},
        {"method": "full", "auxiliary_arm": "D1", "macro_generation_budget": 10},
        {"auxiliary_cutoffs": (0.3, 0.1)},
        {"auxiliary_cutoffs": (-1, 0.2)},
        {"auxiliary_random_probabilities": (0.2, 0.2, 0.2)},
    ],
)
def test_invalid_controller_config_is_rejected(extra):
    with pytest.raises(ValueError):
        Config(**extra)


def test_yaml_research_tuples_are_explicit(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "method: full\nauxiliary_arm: D2\nauxiliary_cutoffs: [0.0, 0.1]\nauxiliary_random_probabilities: [0.4, 0.3, 0.3]\n"
    )
    c = load_config(p)
    assert c.auxiliary_cutoffs == (0.0, 0.1) and c.auxiliary_random_probabilities == (0.4, 0.3, 0.3)
