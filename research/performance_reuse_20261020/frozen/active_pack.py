"""A7 single-operation active-union packing with 2D Pareto/crowding prescreen."""

import random
from dataclasses import dataclass

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.operators.base import Proposal, ProposalBatch
from geo_llm_scheduler.operators.packing_proxy import PackingBillProxy
from geo_llm_scheduler.scheduling.timing import move, packing_moves

Move = tuple[int, float, float, float]


def rank_crowding(moves: list[Move]) -> tuple[dict[int, int], dict[int, float]]:
    """Nondominated layers for gain maximization and Flow-delta minimization."""
    remaining = set(range(len(moves)))
    ranks: dict[int, int] = {}
    crowd: dict[int, float] = {}
    layer = 0
    while remaining:
        front = [
            i
            for i in sorted(remaining)
            if not any(
                moves[j][2] >= moves[i][2]
                and moves[j][3] <= moves[i][3]
                and (moves[j][2] > moves[i][2] or moves[j][3] < moves[i][3])
                for j in remaining
                if i != j
            )
        ]
        for i in front:
            ranks[i] = layer
            crowd[i] = 0.0
        if len(front) <= 2:
            for i in front:
                crowd[i] = float("inf")
        else:
            for dim in (2, 3):
                order = sorted(front, key=lambda i: (moves[i][dim], i))
                span = moves[order[-1]][dim] - moves[order[0]][dim]
                if span > 0:
                    crowd[order[0]] = crowd[order[-1]] = float("inf")
                    for k in range(1, len(order) - 1):
                        crowd[order[k]] += (
                            moves[order[k + 1]][dim] - moves[order[k - 1]][dim]
                        ) / span
        remaining.difference_update(front)
        layer += 1
    return ranks, crowd


@dataclass(frozen=True)
class PackingAudit:
    """The exact raw, representative, pool and selected moves of one A7 call."""

    raw: tuple[Move, ...]
    representatives: tuple[Move, ...]
    pool: tuple[Move, ...]
    selected: tuple[Move, ...]


def select_packing_moves(
    problem: ProblemInstance,
    incumbent: Candidate,
    budget: int,
    rng: random.Random,
    *,
    representative_policy: str = "compression",
) -> PackingAudit:
    """Expose A7 stages; opt-in bill_proxy changes only the first representative."""
    if representative_policy not in ("compression", "bill_proxy"):
        raise ValueError("Unknown experimental A7 representative policy")
    all_moves = packing_moves(problem, incumbent.genotype, incumbent.schedule)
    proxy = PackingBillProxy(problem, incumbent) if representative_policy == "bill_proxy" else None
    representatives = []
    for o in range(problem.operation_count):
        available = [m for m in all_moves if m[0] == o]
        if not available:
            continue
        old = incumbent.schedule.starts[o]
        if proxy is None:
            first = min(available, key=lambda m: (-m[2], m[3], abs(m[1] - old), m[1]))
        else:
            first = min(
                available,
                key=lambda m: (proxy.score(o, m[1]), -m[2], m[3], abs(m[1] - old), m[1]),
            )
        representatives.append(first)
        available.remove(first)
        if available:
            representatives.append(
                min(available, key=lambda m: (m[3], -m[2], abs(m[1] - old), m[1]))
            )
    ranks, crowd = rank_crowding(representatives)
    tie = {i: rng.random() for i in range(len(representatives))}

    def key(i: int) -> tuple[int, float, float]:
        return ranks[i], -crowd[i], tie[i]

    primary = []
    for o in range(problem.operation_count):
        indices = [i for i, m in enumerate(representatives) if m[0] == o]
        if indices:
            primary.append(min(indices, key=key))
    pool = sorted(primary, key=key)[: 2 * budget]
    secondary = sorted(set(range(len(representatives))) - set(primary), key=key)
    pool += secondary[: max(0, 2 * budget - len(pool))]
    selected = rng.sample(pool, min(budget, len(pool)))
    return PackingAudit(
        tuple(all_moves),
        tuple(representatives),
        tuple(representatives[i] for i in pool),
        tuple(representatives[i] for i in selected),
    )


class ActivePack:
    """A7: select per-operation representatives, then seeded best-front diversity."""

    action = 7

    def propose(
        self,
        problem: ProblemInstance,
        incumbent: Candidate,
        budget: int,
        config: Config,
        rng: random.Random,
    ) -> ProposalBatch:
        """Prescreen at most two moves per operation, then sample from a 2B pool."""
        audit = select_packing_moves(
            problem, incumbent, budget, rng, representative_policy=config.a7_representative_policy
        )
        proposals = []
        for o, t, _, _ in audit.selected:
            schedule = move(problem, incumbent.genotype, incumbent.schedule, o, t)
            assert schedule is not None
            proposals.append(Proposal(incumbent.genotype, schedule, (o,)))
        return ProposalBatch(
            proposals,
            len(audit.raw),
            {"representatives": len(audit.representatives), "pool": len(audit.pool)},
            {
                "positive_compression_moves": len(audit.raw),
                "positive_compression_operations": len({m[0] for m in audit.raw}),
                "g_pack_values": [m[2] for m in audit.selected],
            },
        )
