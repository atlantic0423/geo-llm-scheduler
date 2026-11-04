"""CMOEA-AOP Python scheduling adaptation of Shao et al./PlatEMO source.

Preserves source replay of normalized actions and its lagged control state.
These source choices are intentionally not silently replaced by a new DDPG.
"""

import numpy as np

from geo_llm_scheduler.experiments.trans_common import (
    Array,
    Individual,
    SearchContext,
    de,
    decisions,
    ga,
    hv2,
    objectives,
    select_spea,
    spea_fitness,
    tournament,
)
from geo_llm_scheduler.experiments.trans_ddpg import DDPG


def sample(old: Array, new: Array, progress: float) -> tuple[Array, Array, float]:
    """Source GenerateSample: sample variances, raw sums, zero CV, common progress/HV."""
    reference = np.maximum(np.vstack((old, new)).max(axis=0), 1e-15) * 1.1
    previous = hv2(old / reference, np.ones(2))
    current = hv2(new / reference, np.ones(2))
    reward = (current - previous) / previous if previous else 0.0
    states = [np.r_[f.var(axis=0, ddof=1), f.sum(axis=0), 0.0, progress] for f in (old, new)]
    return states[0], states[1], reward


def de_best(population: list[Individual], rng: np.random.Generator) -> Array:
    """Source DEBest, including its non-distinct r2/r3 index sampling behavior."""
    x, f = decisions(population), objectives(population)
    dominated = np.any(
        np.all(f[:, None] <= f[None, :], axis=2) & np.any(f[:, None] < f[None, :], axis=2),
        axis=0,
    )
    best = x[int(rng.choice(np.flatnonzero(~dominated)))]
    out = x.copy()
    for i in range(len(x)):
        scale = float(rng.choice([0.6, 0.8, 1.0]))
        cr = float(rng.choice([0.1, 0.2, 1.0]))
        available = [j for j in range(len(x)) if j != i]
        first = available.pop(int(rng.integers(len(x) - 1)))
        second = available[int(rng.integers(len(x) - 2))]
        third = available[int(rng.integers(len(x) - 3))]
        trial = np.clip(
            x[first] + rng.random() * (best - x[first]) + scale * (x[second] - x[third]), 0, 1
        )
        site = rng.random(x.shape[1]) < cr
        site[int(rng.integers(x.shape[1]))] = True
        out[i, site] = trial[site]
    return out


def portfolio(
    population: list[Individual], mating: np.ndarray, rates: Array, rng: np.random.Generator
) -> Array:
    """Allocate min-four offspring per GAhalf, DE and DEBest as in upstream."""
    n = len(population)
    counts = np.maximum(4, np.floor(rates * n).astype(int))
    x = decisions(population)
    parents = np.tile(mating, 20)
    first = ga(x[parents[: 2 * counts[0]]], rng)
    draws = rng.integers(n, size=(3, counts[1]))
    second = de(x[draws[0]], x[draws[1]], x[draws[2]], rng)
    chosen = [population[int(i)] for i in rng.integers(n, size=counts[2])]
    return np.vstack((first, second, de_best(chosen, rng)))


def run_cmoeaaop(context: SearchContext, population_size: int = 100) -> list[Individual]:
    """Run two populations and source DDPG portfolio; all costs use one wall budget."""
    n = population_size
    populations = [context.initialize(n), context.initialize(n)]
    fitness = [spea_fitness(objectives(p)) for p in populations]
    state, _, _ = sample(objectives(populations[0]), objectives(populations[0]), context.progress())
    agent = DDPG(context.rng)
    transfer = False
    while context.running():
        action = agent.select(state)
        rates = (action + 5) / np.sum(action + 5 + 1e-5)
        previous = populations[0]
        offspring = []
        for pop, fit in zip(populations, fitness, strict=True):
            mating = tournament(fit, n, context.rng) if transfer else context.rng.permutation(n)
            offspring.append(context.offspring(portfolio(pop, mating, rates, context.rng)))
        if not any(offspring):
            break
        success = [0.0, 0.0]
        if transfer:
            for i, other in enumerate((1, 0)):
                _, _, chosen = select_spea(populations[other] + offspring[other], n)
                # Upstream literals 100/50 generalized only for engineering-size tests.
                success[i] = sum(j < n for j in chosen) / n - sum(j >= n for j in chosen) / (n / 2)
        for i in range(2):
            other = 1 - i
            injected = (
                [populations[other][int(j)] for j in context.rng.permutation(n)[: n // 2]]
                if transfer and success[i] > 0
                else offspring[other]
            )
            populations[i], fitness[i], _ = select_spea(populations[i] + offspring[i] + injected, n)
        polished = [context.polish_population(p) for p in populations]
        if any(a is not b for a, b in zip(polished, populations, strict=True)):
            populations = polished
            fitness = [spea_fitness(objectives(p)) for p in populations]
        state, next_state, reward = sample(
            objectives(previous), objectives(populations[0]), context.progress()
        )
        agent.store(state, rates, reward, next_state, False)
        critic, actor = agent.train()
        transfer = transfer or context.progress() >= 0.2
        context.generations += 1
        context.trace.append(
            {
                "generation": context.generations,
                "rates": rates.tolist(),
                "reward": reward,
                "critic_loss": critic,
                "actor_loss": actor,
                "transfer": transfer,
                "replay_size": len(agent.buffer),
            }
        )
    return populations[0]
