"""Golden and event-jump properties independent of evaluator construction."""

from dataclasses import replace

import pytest

from geo_llm_scheduler.domain.models import Genotype, Profile, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.scheduling.resources import earliest
from geo_llm_scheduler.scheduling.ssgs import decode, refresh_diagnostics
from geo_llm_scheduler.scheduling.timing import critical_times


def test_decode_replay_and_overlap(problem):
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    schedule = decode(problem, g)
    assert schedule.starts == (0, 100, 0, 100)
    assert schedule == decode(problem, g)
    assert evaluate(problem, g, schedule).feasible


@pytest.mark.parametrize("compute,vram", [(1, 2), (0.1, 8)])
def test_blocked_resource_event(problem, compute, vram):
    jobs = tuple(
        replace(j, prefill=Profile(100, compute, vram), decode=Profile(200, compute, vram))
        for j in problem.jobs
    )
    p = replace(problem, jobs=jobs)
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    s = decode(p, g)
    assert s.starts == (0, 200, 100, 400)
    assert s.resource_wait == (0, 100, 100, 200)


def test_kv_release_and_long_jump(problem):
    jobs = (replace(problem.jobs[0], release=1e8),)
    p = replace(problem, jobs=jobs)
    g = Genotype((0, 1), (0, 0))
    assert decode(p, g).starts == (1e8, 1e8 + 130)


def test_future_conflict_is_checked(problem):
    # A future interval can block a long proposed interval despite free resources at EST.
    jobs = tuple(
        replace(j, prefill=Profile(100, 1, 2), decode=Profile(200, 1, 2)) for j in problem.jobs
    )
    p = replace(problem, jobs=jobs)
    g = Genotype((0, 0, 0, 0), (0, 0, 1, 1))
    assert earliest(p, g, Schedule((-1, -1, 50, 150)), 0, 0) == 350


def test_refresh_never_changes_timing(problem):
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    s = Schedule((50, 150, 50, 150))
    updated = refresh_diagnostics(problem, g, s)
    assert updated.starts == s.starts
    assert updated.intentional_wait[0] == 50
    assert updated.resource_wait[0] == 0


def test_no_artificial_a7_horizon(problem):
    g = Genotype((0, 0, 0, 0), (0, 1, 0, 1))
    s = Schedule((0, 100, 200, 300))
    # Existing completion events remain available even beyond old end-duration.
    assert 500 in critical_times(problem, g, s, 1)
