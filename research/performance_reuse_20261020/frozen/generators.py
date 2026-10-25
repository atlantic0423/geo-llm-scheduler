"""Ten approximately balanced heuristic modes with bounded duplicate recovery."""

import random

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype, ProblemInstance
from geo_llm_scheduler.moead.variation import paths


def generate(
    problem: ProblemInstance, mode: int, rng: random.Random, perturbation: float
) -> Genotype:
    """Generate FCFS/SPT/LPT/Random x MinLoad/Random, KV-aware or Energy-aware."""
    n = len(problem.jobs)
    order = list(range(n))
    if mode < 8:
        sort = mode // 2
        if sort == 0:
            order.sort(key=lambda j: (problem.jobs[j].release, j))
        elif sort in (1, 2):
            order.sort(
                key=lambda j: sum(problem.profile(2 * j + s).duration for s in (0, 1)),
                reverse=sort == 2,
            )
        else:
            rng.shuffle(order)
    else:
        order.sort(key=lambda j: (problem.jobs[j].release, j))
    ms = [0] * (2 * n)
    loads = [0.0] * len(problem.instances)
    for job in order:
        eligible = paths(problem, job)
        if mode < 8 and mode % 2 == 1 or rng.random() < perturbation:
            pair = rng.choice(eligible)
        else:

            def score(pair: tuple[int, int]) -> tuple[float, ...]:
                p, d = pair
                load = loads[p] + loads[d]
                if mode == 8:
                    return (float(p != d) * problem.jobs[job].kv_delay, load, p, d)
                if mode == 9:
                    region = problem.regions[problem.instances[p].region]
                    price = (
                        sum((t.end - t.start) * t.price for t in region.tariffs)
                        / region.tariffs[-1].end
                    )
                    return (price, load, p, d)
                return (load, p, d)

            pair = min(eligible, key=score)
        ms[2 * job : 2 * job + 2] = pair
        for phase, m in enumerate(pair):
            profile = problem.profile(2 * job + phase)
            loads[m] += profile.duration * max(
                profile.compute, profile.vram / problem.instances[m].vram
            )
    os = [j for j in order for _ in (0, 1)]
    if mode in (6, 7):
        rng.shuffle(os)
    elif len(os) > 1 and rng.random() < perturbation:
        i, j = rng.sample(range(len(os)), 2)
        os[i], os[j] = os[j], os[i]
    return Genotype(tuple(ms), tuple(os))


def initial_genotypes(
    problem: ProblemInstance, config: Config, rng: random.Random
) -> list[Genotype]:
    """Balance ten quotas, then transfer repeated-mode shortages to random generation."""
    result: list[Genotype] = []
    seen: set[Genotype] = set()
    quotas = [config.population // 10 + (i < config.population % 10) for i in range(10)]
    for attempt in range(config.initialization_attempts):
        pending = [i for i, q in enumerate(quotas) if q > 0]
        if not pending:
            break
        mode = pending[attempt % len(pending)] if attempt < config.population * 20 else 7
        g = generate(problem, mode, rng, config.initialization_perturbation)
        if g not in seen:
            seen.add(g)
            result.append(g)
            target = mode if quotas[mode] > 0 else pending[0]
            quotas[target] -= 1
        if len(result) == config.population:
            rng.shuffle(result)
            return result
    raise ValueError("Unique initialization population unavailable within attempt budget")
