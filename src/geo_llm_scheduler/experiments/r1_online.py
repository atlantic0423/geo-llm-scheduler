"""Online R1 regional-block mixing with ledger screening (research arms, default off).

After the usual whole-solution replacement of an offspring, with probability
``probability`` the offspring donates single regions to up to ``recipients`` of its
replacement neighbors that hold the same nonempty regional Job set. Regional
ledgers predict each spliced candidate without decoding; only candidates whose
prediction improves the recipient's direction or is nondominated by the Archive
are built, SSGS-decoded and exact-evaluated through the gateway (which attempts
the Archive). In ``population`` mode an exact improvement replaces the recipient;
in ``archive`` mode the population is never changed. At most one accepted
replacement per offspring; candidates are never chained. Ledger and screening
work is charged to the run clock. All draws use the dedicated ``r1`` stream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.regional import (
    RegionLedger,
    combine,
    region_ledgers,
    splice,
    tariff_indexes,
)
from geo_llm_scheduler.moead.core import NormalizationContext, maximum
from geo_llm_scheduler.utils.numeric import TOL, less
from geo_llm_scheduler.utils.rng import RNGManager


@dataclass(frozen=True)
class R1Options:
    """Frozen research settings (provisional values, see open questions)."""

    mode: str = "population"  # or "archive"
    probability: float = 0.25
    recipients: int = 2

    def __post_init__(self) -> None:
        if self.mode not in ("population", "archive"):
            raise ValueError("R1 mode must be population or archive")
        if not 0 < self.probability <= 1 or self.recipients < 1:
            raise ValueError("Invalid R1 probability or recipient count")


def mix_genotype(donor: Genotype, recipient: Genotype, jobs: frozenset[int]) -> Genotype:
    """Copy the donor's MS and relative OS order of ``jobs`` into the recipient."""
    local = iter(j for j in donor.os if j in jobs)
    os = tuple(next(local) if j in jobs else j for j in recipient.os)
    ms = tuple(donor.ms[o] if o // 2 in jobs else m for o, m in enumerate(recipient.ms))
    return Genotype(ms, os)


class RegionalBlockMixer:
    """Engine ``post_replacement`` seam implementing online R1."""

    def __init__(self, problem: ProblemInstance, options: R1Options) -> None:
        """Bind the problem and frozen options; counters start at zero."""
        self.problem = problem
        self.options = options
        self.indexes = tariff_indexes(problem)
        self._cache: dict[int, tuple[Candidate, tuple[RegionLedger, ...]]] = {}
        self.counts = dict(
            activations=0, predicted=0, screened_in=0, exact=0, improved=0, accepted=0, archive=0
        )
        self.seconds = 0.0

    def _ledgers(self, c: Candidate) -> tuple[RegionLedger, ...]:
        hit = self._cache.get(id(c))
        if hit is None or hit[0] is not c:
            hit = (c, region_ledgers(self.problem, c.genotype, c.schedule))
            self._cache[id(c)] = hit
        return hit[1]

    def __call__(
        self,
        population: list[Candidate],
        child: Candidate,
        order: tuple[int, ...],
        lambdas: tuple[tuple[float, float], ...],
        gateway: EvaluationGateway,
        streams: RNGManager,
    ) -> None:
        """Try regional blocks of ``child`` on sampled replacement neighbors."""
        rng = streams.stream("r1")
        if rng.random() >= self.options.probability or not child.evaluation.feasible:
            return
        tick = perf_counter()
        try:
            self._mix(population, child, order, lambdas, gateway, rng)
        finally:
            self.seconds += perf_counter() - tick
            live = {id(c) for c in population}
            self._cache = {k: v for k, v in self._cache.items() if k in live}

    def _mix(self, population, child, order, lambdas, gateway, rng) -> None:
        self.counts["activations"] += 1
        slots = [j for j in order if population[j] is not child]
        if not slots:
            return
        slots = rng.sample(slots, min(self.options.recipients, len(slots)))
        donor = self._ledgers(child)
        for j in slots:
            recipient = population[j]
            mine = self._ledgers(recipient)
            regions = [
                r
                for r in range(len(self.problem.regions))
                if donor[r].jobs
                and donor[r].jobs == mine[r].jobs
                and not donor[r].same_core(mine[r])
            ]
            rng.shuffle(regions)
            for r in regions:
                self.counts["predicted"] += 1
                prediction = combine(self.problem, self.indexes, splice(mine, donor, r))
                context = NormalizationContext(gateway.ideal, maximum(population))
                p = prediction.objectives
                ideal = (min(context.ideal[0], p[0]), min(context.ideal[1], p[1]))
                ctx = NormalizationContext(ideal, context.maximum)
                w = lambdas[j]
                scale = [max(ctx.maximum[k] - ideal[k], 1e-9) for k in range(2)]
                p_scalar = max(w[k] * abs((p[k] - ideal[k]) / scale[k]) for k in range(2))
                improve = less(p_scalar, ctx.scalar(recipient, w), TOL.scalar)
                frontier = not any(
                    m.evaluation.flow <= p[0] and m.evaluation.bill <= p[1]
                    for m in gateway.archive.members
                )
                if not (improve or frontier):
                    continue
                self.counts["screened_in"] += 1
                before = gateway.archive.insertions
                mixed = gateway.evaluate(
                    mix_genotype(child.genotype, recipient.genotype, donor[r].jobs),
                    origin="r1_block",
                )
                self.counts["exact"] += 1
                self.counts["archive"] += gateway.archive.insertions - before
                if not mixed.evaluation.feasible:
                    continue
                exact_ctx = NormalizationContext(gateway.ideal, maximum(population))
                if less(exact_ctx.scalar(mixed, w), exact_ctx.scalar(recipient, w), TOL.scalar):
                    self.counts["improved"] += 1
                    if self.options.mode == "population":
                        population[j] = mixed
                        self.counts["accepted"] += 1
                        return

    def record(self) -> dict:
        """Settings, counters and charged seconds for the run artifacts."""
        return dict(options=asdict(self.options), counts=dict(self.counts), seconds=self.seconds)
