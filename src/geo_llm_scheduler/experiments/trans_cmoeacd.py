"""CMOEA-CD's default three-archive MATLAB path ported to Python.

Zhe Liu / BIMK PlatEMO, e2d58cb, research-use attribution in trans_python.md.
The scheduling decoder is feasible by construction, so CV is identically zero.
"""

import numpy as np

from geo_llm_scheduler.experiments.trans_common import (
    Array,
    Individual,
    SearchContext,
    de,
    decisions,
    dominance_mask,
    ga,
    objectives,
    truncate,
)


def update_archive(
    pool: list[Individual], size: int, rng: np.random.Generator, ideal: Array | None = None
) -> list[Individual]:
    """Port default FA/FEA nondominance, source rounding, padding and truncation."""
    kept = [pool[int(i)] for i in dominance_mask(objectives(pool), round10=True)]
    if len(kept) <= size:
        return kept + [kept[int(i)] for i in rng.integers(len(kept), size=size - len(kept))]
    values = objectives(kept)
    low = values.min(axis=0) - 1e-6 if ideal is None else ideal
    normalized = (values - low) / np.maximum(values.max(axis=0) - low, 1e-15)
    return [kept[i] for i in truncate(normalized, size)]


def diversity_archive(pool: list[Individual], size: int, ideal: Array) -> list[Individual]:
    """Port DA direction association; all-feasible branch retains minimum angle."""
    kept = [pool[int(i)] for i in dominance_mask(objectives(pool), round10=True)]
    values = objectives(kept)
    normalized = (values - ideal) / np.maximum(values.max(axis=0) - ideal, 1e-15)
    unit = normalized / np.maximum(np.linalg.norm(normalized, axis=1)[:, None], 1e-15)
    weights = np.maximum(np.column_stack((np.linspace(0, 1, size), np.linspace(1, 0, size))), 1e-6)
    weights /= np.linalg.norm(weights, axis=1)[:, None]
    angles = np.sqrt(np.maximum(0, 1 - np.clip(weights @ unit.T, -1, 1) ** 2))
    # With CV=0, the h cone always contains its minimum-angle point, or falls
    # back to that same point. Thus direct argmin is exactly the reachable path.
    return [kept[int(i)] for i in np.argmin(angles, axis=1)]


def run_cmoeacd(context: SearchContext, population_size: int = 100) -> list[Individual]:
    """Run default e1=e2=SPEA2, three archives and a shared GA/DE generation draw."""
    if population_size < 6:
        raise ValueError("CMOEA-CD needs at least six individuals")
    n = population_size
    ns = n // 3
    offspring = context.initialize(n)
    feasible = offspring.copy()
    forward: list[Individual] = []
    diverse: list[Individual] = []
    ideal = objectives(offspring).min(axis=0) - 1e-6
    rng = context.rng
    while context.running():
        ideal = np.minimum(ideal, objectives(offspring).min(axis=0) - 1e-6)
        forward = update_archive(forward + offspring, ns, rng, ideal)
        diverse = diversity_archive(diverse + offspring, ns, ideal)
        feasible = update_archive(feasible + offspring, n, rng)
        pops = [forward, diverse, [feasible[int(i)] for i in rng.integers(n, size=ns)]]
        permutations = [(rng.permutation(len(p)), rng.permutation(len(p))) for p in pops]
        use_de = rng.random() < 0.5
        genes = []
        for pop, (a, b) in zip(pops, permutations, strict=True):
            x = decisions(pop)
            genes.append(
                de(x, x[a], x[b], rng, eta_m=1) if use_de else ga(x[a], rng, half=False, eta_m=1)
            )
        offspring = context.offspring(np.vstack(genes))
        if not offspring:
            break
        offspring = context.polish_population(offspring)
        context.generations += 1
        context.trace.append(
            {
                "generation": context.generations,
                "operator": "DE" if use_de else "GA",
                "offspring": len(offspring),
            }
        )
    return feasible
