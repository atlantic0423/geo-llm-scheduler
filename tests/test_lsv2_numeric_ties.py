"""Regression for the n200 P2_FLOW -> A7 cross-runtime chain divergence."""

from itertools import permutations

import pytest

from geo_llm_scheduler.domain.models import Candidate, EvaluationResult, Genotype, Schedule
from geo_llm_scheduler.lsv2.frontier import FrozenFrame
from geo_llm_scheduler.lsv2.search import _chain_next
from geo_llm_scheduler.utils.numeric import TOL


def candidate(index, flow=0.4, bill=0.2):
    # Synthetic ranking fixture, not an assertion about scheduling feasibility.
    return Candidate(
        Genotype((0, 0), (0, 0)),
        Schedule((float(index), 0.0)),
        EvaluationResult(True, flow, bill, (), ()),
    )


@pytest.mark.parametrize("secondary", [False, True])
def test_observed_roundoff_uses_identity_for_every_input_order(secondary):
    # Recorded gains from snapshot_p1_s1, st_n200_d07_s16901, repeat 2.
    # The two candidate totals differ by about 7e-16 on Python 3.13 only.
    a = candidate(0, 0.41573937525137417, 0.22636661505088587)
    b = candidate(2, 0.41573937525137417, 0.22636661505088518)
    current = candidate(9, 1.0 if secondary else 0.0, 1.0)
    scores = {a: 3.01952970628485e-7, b: 3.019529706287696e-7}
    frame = FrozenFrame((0.0, 0.0), (1.0, 1.0))
    for order in permutations((a, b)):
        chosen, rule = _chain_next(current, order, order, frame, scores.__getitem__)
        assert chosen is a
        assert rule == ("dominate" if secondary else "gap")


def test_ties_are_anchored_not_pairwise_chained():
    a, b, c = (candidate(i) for i in range(3))
    scores = {a: 1e-6, b: 1e-6 + 0.75 * TOL.scalar, c: 1e-6 + 1.5 * TOL.scalar}
    current = candidate(9, 1.0, 1.0)
    for order in permutations((a, b, c)):
        chosen, _ = _chain_next(current, order, [], FrozenFrame((0, 0), (1, 1)), scores.__getitem__)
        assert chosen is b  # a is close to b, but outside the tie set anchored at c.


@pytest.mark.parametrize("gap", [False, True])
def test_meaningful_gain_wins_over_identity_and_secondary_score(gap):
    a, b = candidate(0, 0.1, 0.1), candidate(1, 0.4, 0.4)
    current = candidate(9, 0.0 if gap else 1.0, 1.0)
    scores = {a: 1e-6, b: 1e-6 + 2 * TOL.scalar}
    chosen, _ = _chain_next(
        current, [a, b], [a, b], FrozenFrame((0, 0), (1, 1)), scores.__getitem__
    )
    assert chosen is b


def test_secondary_total_breaks_only_dominating_ties():
    a, b = candidate(0, 0.4, 0.4), candidate(1, 0.2, 0.2)
    frame = FrozenFrame((0, 0), (1, 1))
    assert _chain_next(candidate(9, 1, 1), [a, b], [], frame, lambda c: 1.0)[0] is b
    assert _chain_next(candidate(9, 0, 1), [a, b], [a, b], frame, lambda c: 1.0)[0] is a


def test_secondary_tie_set_is_anchored_at_its_minimum():
    a, b, c = (candidate(i, 0.4, 0.2 - i * 0.75 * TOL.scalar) for i in range(3))
    for order in permutations((a, b, c)):
        assert (
            _chain_next(candidate(9, 1, 1), order, [], FrozenFrame((0, 0), (1, 1)), lambda c: 1.0)[
                0
            ]
            is b
        )


def test_gap_eligibility_stays_strictly_positive_and_requires_archive_entry():
    current, zero, tiny, negative = (candidate(i, 0.0, 1.0) for i in (9, 0, 1, 2))
    scores = {zero: 0.0, tiny: 1e-15, negative: -1e-15}
    frame = FrozenFrame((0, 0), (1, 1))
    assert _chain_next(current, [], [zero, tiny, negative], frame, scores.__getitem__) == (
        tiny,
        "gap",
    )
    assert _chain_next(current, [tiny], [zero, negative], frame, scores.__getitem__) == (
        current,
        "stay",
    )


def test_dominance_priority_does_not_require_positive_gap_gain():
    current, dominating, entrant = candidate(9, 1, 1), candidate(1), candidate(0, 0, 2)
    scores = {dominating: 0.0, entrant: 1.0}
    assert _chain_next(
        current, [dominating, entrant], [entrant], FrozenFrame((0, 0), (1, 1)), scores.__getitem__
    ) == (dominating, "dominate")
