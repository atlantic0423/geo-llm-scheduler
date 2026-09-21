"""Experimental normalized workloads shared by severity and structural proxies."""

from geo_llm_scheduler.domain.models import Genotype, ProblemInstance


def machine_loads(problem: ProblemInstance, genotype: Genotype) -> tuple[float, ...]:
    """Compute max of normalized compute-work and memory-work per instance."""
    c, v = [0.0] * len(problem.instances), [0.0] * len(problem.instances)
    for o, m in enumerate(genotype.ms):
        p = problem.profile(o)
        c[m] += p.duration * p.compute
        v[m] += p.duration * p.vram / problem.instances[m].vram
    return tuple(max(a, b) for a, b in zip(c, v))


def region_loads(problem: ProblemInstance, genotype: Genotype) -> tuple[float, ...]:
    """Aggregate resource work before normalization by total regional capacity."""
    result = []
    for r in range(len(problem.regions)):
        machines = [m for m, s in enumerate(problem.instances) if s.region == r]
        c = v = 0.0
        for o, m in enumerate(genotype.ms):
            if m in machines:
                p = problem.profile(o)
                c += p.duration * p.compute
                v += p.duration * p.vram
        result.append(
            max(c / len(machines), v / sum(problem.instances[m].vram for m in machines))
            if machines
            else 0.0
        )
    return tuple(result)


def imbalance(values: tuple[float, ...]) -> float:
    """Return normalized max-min imbalance, zero for no workload."""
    high = max(values, default=0.0)
    return (high - min(values)) / high if high > 0 else 0.0
