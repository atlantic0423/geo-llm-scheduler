"""Opt-in population retention from already evaluated initialization candidates."""

from collections.abc import Sequence
from math import isfinite

from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.utils.numeric import EPS_NORM, TOL, close, identity, less


def _physical(c: Candidate) -> tuple:
    return c.genotype.ms, c.schedule.starts


def _stable(c: Candidate) -> tuple:
    return identity(c), c.schedule.starts, c.evaluation.objectives, c.origin


def _dominates(a: Candidate, b: Candidate) -> bool:
    # Exact non-worsening makes the relation acyclic; strictness uses the
    # existing unit-aware tolerances. This is retention, not Archive mutation.
    x, y = a.evaluation.objectives, b.evaluation.objectives
    return all(v <= w for v, w in zip(x, y)) and any(
        less(v, w, t) for v, w, t in zip(x, y, (TOL.time, TOL.cost))
    )


def _structure(a: Candidate, b: Candidate) -> int:
    return sum(x != y for x, y in zip(a.genotype.ms, b.genotype.ms)) + sum(
        x != y for x, y in zip(a.genotype.os, b.genotype.os)
    )


def retain_pool(candidates: Sequence[Candidate], size: int) -> list[Candidate]:
    """Keep Pareto layers, then maximize spacing in the cutoff layer.

    Exact MS/starts duplicates first share one canonical representative. If
    there are fewer physical schedules than slots, distinct contract identities
    supply the remainder; no new candidate or evaluation is created. Distinct
    timings of the same genotype are never collapsed by genotype alone.

    The first front fixes objective normalization; a constant axis contributes
    zero distance. Protect both objective extremes if at least two slots exist.
    Distance ties use max-min MS/OS Hamming distance, then stable identity.
    Output follows the input stream's order, not objective or quality order.
    This deterministic selector consumes no RNG and never edits its inputs.
    """
    if type(size) is not int or size < 2:
        raise ValueError("Retention requires an integer size of at least two")
    if not candidates or any(
        not c.evaluation.feasible
        or not all(isfinite(v) for v in (*c.evaluation.objectives, *c.schedule.starts))
        for c in candidates
    ):
        raise ValueError("Retention requires finite, complete feasible exact candidates")
    shape = (len(candidates[0].genotype.ms), len(candidates[0].genotype.os))
    if any(
        (len(c.genotype.ms), len(c.genotype.os)) != shape or len(c.schedule.starts) != shape[0]
        for c in candidates
    ):
        raise ValueError("Candidates must belong to the same operation space")
    unique: dict[tuple, Candidate] = {}
    for c in sorted(candidates, key=_stable):
        unique.setdefault(identity(c), c)
    if len(unique) < size:
        raise ValueError("Evaluated pool cannot supply enough distinct identities")
    physical: dict[tuple, Candidate] = {}
    for c in unique.values():
        physical.setdefault(_physical(c), c)
    pending = list(physical.values())
    chosen: list[Candidate] = []
    first = [c for c in pending if not any(_dominates(a, c) for a in pending)]
    lo = tuple(min(c.evaluation.objectives[k] for c in first) for k in range(2))
    span = tuple(max(c.evaluation.objectives[k] for c in first) - lo[k] for k in range(2))
    positions = {
        id(c): tuple(
            (c.evaluation.objectives[k] - lo[k]) / span[k] if span[k] > EPS_NORM else 0.0
            for k in range(2)
        )
        for c in unique.values()
    }

    def distance(a: Candidate, b: Candidate) -> float:
        return sum((u - v) ** 2 for u, v in zip(positions[id(a)], positions[id(b)]))

    def take_layer(layer: list[Candidate]) -> None:
        if len(chosen) + len(layer) <= size:
            chosen.extend(layer)
            return
        if not chosen:
            for k in range(2):
                c = min(
                    layer,
                    key=lambda c: (
                        *c.evaluation.objectives[k:],
                        *c.evaluation.objectives[:k],
                        _stable(c),
                    ),
                )
                if c not in chosen:
                    chosen.append(c)
            layer = [c for c in layer if c not in chosen]
        nearest = {id(c): min(distance(c, a) for a in chosen) for c in layer}
        while layer and len(chosen) < size:
            best = max(nearest[id(c)] for c in layer)
            tied = [c for c in layer if close(nearest[id(c)], best, TOL.scalar)]
            c = min(
                tied,
                key=lambda c: (-min(_structure(c, a) for a in chosen), _stable(c)),
            )
            chosen.append(c)
            layer.remove(c)
            for other in layer:
                nearest[id(other)] = min(nearest[id(other)], distance(other, c))

    while pending and len(chosen) < size:
        layer = [c for c in pending if not any(_dominates(a, c) for a in pending)]
        take_layer(layer)
        taken = {id(c) for c in layer}
        pending = [c for c in pending if id(c) not in taken]
    if len(chosen) < size:
        used = {identity(c) for c in chosen}
        pending = [c for c in unique.values() if identity(c) not in used]
        while pending and len(chosen) < size:
            layer = [c for c in pending if not any(_dominates(a, c) for a in pending)]
            take_layer(layer)
            taken = {id(c) for c in layer}
            pending = [c for c in pending if id(c) not in taken]
    selected = {id(c) for c in chosen}
    result = []
    for c in candidates:
        if id(c) in selected:
            result.append(c)
            selected.remove(id(c))
    assert len(result) == size
    return result
