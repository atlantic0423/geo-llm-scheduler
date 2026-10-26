"""Bounded structural decode cache and digest-based repeat counting: exact equivalence."""

import random
from collections import Counter
from dataclasses import replace

import pytest

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.comparison13 import case_problem
from geo_llm_scheduler.experiments.mechanism5 import StructuralObserver, genotype_digest
from geo_llm_scheduler.experiments.slot_assignment import run_arm, study_config
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.scheduling.ssgs import decode


def problem20():
    return case_problem(20, 3, 6, 202610160020, "arrival", 0)


def genotypes(problem, count):
    return initial_genotypes(problem, Config(population=10, neighborhood=5), random.Random(5))[
        :count
    ]


def test_cache_hit_returns_identical_content_and_keeps_logical_counts():
    problem = problem20()
    g = genotypes(problem, 4)
    plain = EvaluationGateway(problem, Archive())
    cached = EvaluationGateway(problem, Archive(), decode_cache_size=2)
    cached.audit_decode_cache = True
    sequence = [g[0], g[1], g[0], g[2], g[0], g[1], g[3], g[1]]
    for genotype in sequence:
        a = plain.evaluate(genotype)
        b = cached.evaluate(genotype)
        assert (a.genotype, a.schedule, a.evaluation, a.origin) == (
            b.genotype,
            b.schedule,
            b.evaluation,
            b.origin,
        )
        assert b.schedule == decode(problem, genotype)
        assert b.evaluation == evaluate(problem, genotype, b.schedule)
    # LRU size 2 hits g0 at positions 3 and 5 and g1 at 8; g1 at 6 was evicted by g2.
    assert cached.counts["ssgs_cache_hits"] == 3
    extra = {"ssgs_cache_hits"}
    assert {k: v for k, v in cached.counts.items() if k not in extra} == dict(plain.counts)
    assert cached.archive.attempts == plain.archive.attempts
    assert [c.evaluation.objectives for c in cached.archive.members] == [
        c.evaluation.objectives for c in plain.archive.members
    ]
    assert len(cached._decoded) == 2


def test_timing_proposals_and_disabled_cache_never_use_the_map():
    problem = problem20()
    g = genotypes(problem, 1)[0]
    gateway = EvaluationGateway(problem, Archive(), decode_cache_size=4)
    first = gateway.evaluate(g)
    gateway.evaluate(g, first.schedule, "timing")
    assert gateway.counts["ssgs"] == 1 and gateway.counts["ssgs_cache_hits"] == 0
    off = EvaluationGateway(problem, Archive())
    off.evaluate(g)
    off.evaluate(g)
    assert off.counts["ssgs"] == 2 and not off._decoded
    with pytest.raises(ValueError):
        EvaluationGateway(problem, Archive(), decode_cache_size=-1)
    with pytest.raises(ValueError):
        Config(decode_cache_size=-1)


def test_audit_detects_a_corrupted_entry():
    problem = problem20()
    g = genotypes(problem, 2)
    gateway = EvaluationGateway(problem, Archive(), decode_cache_size=4)
    gateway.audit_decode_cache = True
    gateway.evaluate(g[0])
    other = gateway.evaluate(g[1])
    gateway._decoded[g[0]] = (other.schedule, other.evaluation)
    with pytest.raises(AssertionError):
        gateway.evaluate(g[0])


class TupleObserver:
    """The original full-genotype implementation, kept as the test oracle."""

    def __init__(self):
        self.initialized, self.last_ssgs = False, 0
        self.seen, self.seen_search, self.counts = set(), set(), Counter()

    def candidate(self, candidate, ssgs):
        rebuilt = ssgs > self.last_ssgs
        self.last_ssgs = ssgs
        if rebuilt:
            stage = "search" if self.initialized else "initialization"
            self.counts[f"{stage}:structural_evaluations"] += 1
            novel = candidate.genotype not in self.seen
            self.counts[f"{stage}:novel_structure_in_run"] += int(novel)
            self.counts[f"{stage}:repeated_structure_in_run"] += int(not novel)
            if self.initialized:
                repeated = candidate.genotype in self.seen_search
                self.counts["search:repeated_structure_in_search"] += int(repeated)
                self.seen_search.add(candidate.genotype)
        elif self.initialized:
            self.counts["search:timing_evaluations"] += 1
        self.seen.add(candidate.genotype)


def test_digest_observer_matches_tuple_oracle_and_separates_ms_from_os():
    a = Genotype((0, 1, 2, 3), (0, 1, 0, 1))
    b = Genotype((0, 1, 2, 4), (0, 1, 0, 1))
    c = Genotype((0, 1, 2, 3), (1, 0, 0, 1))
    assert len({genotype_digest(g) for g in (a, b, c)}) == 3
    assert genotype_digest(a) == genotype_digest(Genotype((0, 1, 2, 3), (0, 1, 0, 1)))
    events = [(a, 1), (a, 1), (b, 2), (a, 3), (c, 4), (c, 4), (b, 5), (a, 6), (a, 6)]
    new, old = StructuralObserver(), TupleObserver()
    for i, (g, ssgs) in enumerate(events):
        if i == 3:
            new.initialized = old.initialized = True
        candidate = Candidate(g, None, None, "x")  # type: ignore[arg-type]
        new.candidate(candidate, ssgs)
        old.candidate(candidate, ssgs)
    assert new.counts == old.counts
    assert len(new.seen) == len(old.seen) and len(new.seen_search) == len(old.seen_search)


@pytest.mark.parametrize("arm", ("BILL_RANDOM", "MAIN_RANDOM"))
def test_fixed_generation_run_is_identical_with_and_without_cache(arm):
    problem = problem20()
    base = replace(study_config(arm, 16101, 300), generations=6, seconds=None)
    streams: dict[int, list] = {}
    results = {}
    for size in (0, 2048):
        stream: list = []

        def observe(event, data, stream=stream):
            if event == "candidate":
                c = data["candidate"]
                stream.append((c.genotype, c.schedule, c.evaluation, c.origin))

        results[size] = run_arm(
            problem, arm, replace(base, decode_cache_size=size), {}, observer=observe
        )
        streams[size] = stream
    off, on = results[0], results[2048]
    assert streams[0] == streams[2048] and len(streams[0]) > 600
    assert on.gateway.counts["ssgs_cache_hits"] > 0
    assert [(c.genotype, c.schedule, c.evaluation) for c in off.population] == [
        (c.genotype, c.schedule, c.evaluation) for c in on.population
    ]
    assert [(c.genotype, c.schedule, c.evaluation) for c in off.archive.members] == [
        (c.genotype, c.schedule, c.evaluation) for c in on.archive.members
    ]
    assert off.controller.q == on.controller.q and off.offspring_count == on.offspring_count
    logical = {k: v for k, v in on.gateway.counts.items() if k != "ssgs_cache_hits"}
    assert logical == dict(off.gateway.counts)
