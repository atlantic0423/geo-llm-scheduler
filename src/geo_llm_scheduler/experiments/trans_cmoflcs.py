"""CMOFLCS dual-population fuzzy operator control, ported from Yubo Wang's source.

The always-feasible scheduling representation cannot enter the completely
infeasible UPF separation branch. Both populations and the native Archive remain.
"""

import numpy as np

from geo_llm_scheduler.experiments.trans_common import (
    Array,
    Individual,
    SearchContext,
    de,
    decisions,
    ga,
    objectives,
    select_spea,
    spea_fitness,
    tournament,
)


def membership(value: Array, points: tuple[float, ...]) -> Array:
    """Triangular/trapezoidal membership with closed shoulder endpoints."""
    a, b, c, d = points if len(points) == 4 else (*points[:2], *points[1:])
    left = np.ones_like(value) if a == b else np.clip((value - a) / (b - a), 0, 1)
    right = np.ones_like(value) if c == d else np.clip((d - value) / (d - c), 0, 1)
    return np.minimum(left, right) * ((value >= a) & (value <= d))


def fuzzy_probabilities(rewards: Array, diversity: float, convergence: float) -> Array:
    """Mamdani min/max inference over the original 18 rules, 101-point centroid."""
    mf = ((0, 0, 0.3, 0.4), (0.3, 0.5, 0.7), (0.6, 0.7, 1, 1))
    out_mf = ((0, 0, 0.1, 0.3), (0.2, 0.5, 0.7), (0.7, 0.9, 1, 1))
    # Rows are low/mid/high reward, columns low/mid/high diversity/convergence.
    rules = ((1, 0, 0), (2, 1, 0), (2, 1, 1))
    grid: Array = np.linspace(0, 1, 101, dtype=np.float64)
    templates = [membership(grid, p) for p in out_mf]
    result = []
    fractions = rewards / rewards.sum()
    for reward, metric in zip(fractions, (convergence, diversity), strict=True):
        activated = np.zeros_like(grid)
        for i in range(3):
            for j in range(3):
                strength = min(
                    float(membership(np.array(reward), mf[i])),
                    float(membership(np.array(metric), mf[j])),
                )
                activated = np.maximum(activated, np.minimum(strength, templates[rules[i][j]]))
        result.append(float(grid @ activated / activated.sum()) if activated.sum() else 0.5)
    probabilities = np.array(result)
    return probabilities / probabilities.sum()


def indicators(values: Array) -> tuple[float, float]:
    """Original convergence and diversity ratios, neutral 0.5 for a zero span."""
    distance = np.linalg.norm(values - values.min(axis=0), axis=1)
    span = float(np.ptp(distance))
    convergence = float((distance.max() - distance.mean()) / span) if span else 0.5
    pairs = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)[
        np.triu_indices(len(values), 1)
    ]
    span = float(np.ptp(pairs))
    diversity = float((pairs.mean() - pairs.min()) / span) if span else 0.5
    return diversity, convergence


def run_cmoflcs(context: SearchContext, population_size: int = 100) -> list[Individual]:
    """Run native fuzzy-controlled GAhalf/DE with two independently initialized populations."""
    n = population_size
    populations = [context.initialize(n), context.initialize(n)]
    fitness = [spea_fitness(objectives(p)) for p in populations]
    archive, _, _ = select_spea(populations[0] + populations[1], n)
    rewards = np.full(2, 1e-3)
    while context.running():
        if context.evaluations % 20000 == 0:
            rewards[:] = 1e-3
            probabilities: Array = np.full(2, 0.5)
        else:
            probabilities = fuzzy_probabilities(rewards, *indicators(objectives(populations[0])))
        operator = int(context.rng.choice(2, p=probabilities))
        offspring = []
        for pop, fit in zip(populations, fitness, strict=True):
            x = decisions(pop)
            mating = tournament(fit, n if operator == 0 else 2 * n, context.rng)
            genes = (
                ga(x[mating], context.rng)
                if operator == 0
                else de(x, x[mating[:n]], x[mating[n:]], context.rng)
            )
            offspring.append(context.offspring(genes))
        if not any(offspring):
            break
        first, fitness[0], selected = select_spea(populations[0] + offspring[0] + offspring[1], n)
        second, fitness[1], _ = select_spea(populations[1] + offspring[0] + offspring[1], n)
        # Original reward counts selected offspring among the first population.
        rewards[operator] += sum(i >= n for i in selected) / sum(map(len, offspring))
        populations = [context.polish_population(first), context.polish_population(second)]
        if populations[0] is not first or populations[1] is not second:
            fitness = [spea_fitness(objectives(p)) for p in populations]
        archive, _, _ = select_spea(archive + populations[0] + populations[1], n)
        context.generations += 1
        context.trace.append(
            {
                "generation": context.generations,
                "operator": "GAhalf" if operator == 0 else "DE",
                "probabilities": probabilities.tolist(),
                "rewards": rewards.tolist(),
            }
        )
    return archive
