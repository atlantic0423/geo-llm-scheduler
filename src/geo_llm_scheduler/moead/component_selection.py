"""Bounded opt-in v4 selection and routing rules; exact acceptance stays in the engine."""

from random import Random

from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.moead.variation import paths
from geo_llm_scheduler.rl.state import associate

PARENTS = ("random_pair", "incumbent_partner", "tournament2", "diverse_partner")
CROSS_REGION = ("random_legal_path", "load_proxy", "tou_proxy")
REPLACEMENT = ("random", "fixed", "improvement_first", "direction_match")
TERMINAL = ("flow_ascending", "bill_ascending", "alternate_extremes", "crowding_first")


def genotype_distance(a: Genotype, b: Genotype) -> float:
    """Mean of MS mismatch fraction and normalized occurrence-position distance."""
    if not a.ms:
        return 0.0
    assignment = sum(x != y for x, y in zip(a.ms, b.ms)) / len(a.ms)
    positions = []
    for g in (a, b):
        seen: dict[int, int] = {}
        location = {}
        for index, job in enumerate(g.os):
            occurrence = seen.get(job, 0)
            location[job, occurrence] = index
            seen[job] = occurrence + 1
        positions.append(location)
    order = sum(abs(i - positions[1][key]) for key, i in positions[0].items())
    return (assignment + order / max(1, len(a.os) * (len(a.os) - 1))) / 2


def select_parents(
    population: list[Candidate],
    birth: int,
    pool: tuple[int, ...],
    rule: str,
    context: NormalizationContext,
    weight: tuple[float, float],
    rng: Random,
) -> tuple[int, int]:
    """Select distinct parents using one frozen birth-subproblem comparison frame."""
    if rule not in PARENTS or len(set(pool)) < 2:
        raise ValueError("Unknown parent rule or insufficient distinct parents")
    if rule == "random_pair":
        a, b = rng.sample(pool, 2)
        return a, b
    if rule == "incumbent_partner":
        return birth, rng.choice([j for j in pool if j != birth])
    if rule == "tournament2":

        def tournament(eligible: list[int]) -> int:
            contestants = rng.sample(eligible, min(2, len(eligible)))
            return min(contestants, key=lambda j: (context.scalar(population[j], weight), j))

        a = tournament(list(pool))
        return a, tournament([j for j in pool if j != a])
    a = rng.choice(pool)
    others = [j for j in pool if j != a]
    sampled = rng.sample(others, min(4, len(others)))
    b = min(
        sampled,
        key=lambda j: (-genotype_distance(population[a].genotype, population[j].genotype), j),
    )
    return a, b


def tariff_proxy(problem: ProblemInstance, job: int, path: tuple[int, int]) -> float:
    """No-queue release-based active energy proxy in CNY; excludes demand and idle tail."""
    start = problem.jobs[job].release
    cost = 0.0
    for phase, machine in enumerate(path):
        if phase and path[0] != path[1]:
            start += problem.jobs[job].kv_delay
        op = 2 * job + phase
        end = start + problem.profile(op).duration
        server = problem.instances[machine]
        power = server.idle_kw + problem.floor_kw(machine) + problem.load_kw(op, machine)
        tariffs = problem.regions[server.region].tariffs
        if not tariffs or tariffs[-1].end < end:
            raise ValueError("TOU proxy requires tariff coverage of the complete path")
        cost += (
            power
            * sum(max(0.0, min(end, t.end) - max(start, t.start)) * t.price for t in tariffs)
            / 3600
        )
        start = end
    return cost


def mutate_cross_region(
    problem: ProblemInstance,
    g: Genotype,
    rule: str,
    rng: Random,
) -> Genotype:
    """Choose one other-region legal path using bounded load or TOU shortlists."""
    if rule not in CROSS_REGION:
        raise ValueError("Unknown cross-region mutation")
    if not problem.jobs:
        return g
    job = rng.randrange(len(problem.jobs))
    region = problem.instances[g.ms[2 * job]].region
    legal = [pair for pair in paths(problem, job) if problem.instances[pair[0]].region != region]
    if not legal:
        return g
    if rule == "load_proxy":
        loads = [0.0] * len(problem.regions)
        for op, machine in enumerate(g.ms):
            loads[problem.instances[machine].region] += (
                problem.profile(op).duration * problem.profile(op).compute
            )
        capacities = [
            sum(m.region == r for m in problem.instances) for r in range(len(problem.regions))
        ]
        regions = sorted(
            {problem.instances[p].region for p, _ in legal},
            key=lambda r: (loads[r] / capacities[r], r),
        )[:3]
        chosen_region = rng.choice(regions)
        legal = [pair for pair in legal if problem.instances[pair[0]].region == chosen_region]
    elif rule == "tou_proxy":
        legal = sorted(legal, key=lambda pair: (tariff_proxy(problem, job, pair), pair))[:3]
    chosen = rng.choice(legal)
    ms = list(g.ms)
    ms[2 * job : 2 * job + 2] = chosen
    return Genotype(tuple(ms), g.os)


def order_replacements(
    population: list[Candidate],
    child: Candidate,
    neighbors: tuple[int, ...],
    lambdas: tuple[tuple[float, float], ...],
    context: NormalizationContext,
    rule: str,
    rng: Random | None = None,
) -> tuple[int, ...]:
    """Permute only the birth neighborhood; never change own-weight acceptance."""
    if rule not in REPLACEMENT:
        raise ValueError("Unknown replacement rule")
    if rule == "random":
        if rng is None:
            raise ValueError("Random replacement requires an explicit RNG")
        indices = list(neighbors)
        rng.shuffle(indices)
        return tuple(indices)
    if rule == "fixed":
        return neighbors
    if rule == "improvement_first":
        return tuple(
            sorted(
                neighbors,
                key=lambda j: (
                    -(context.scalar(population[j], lambdas[j]) - context.scalar(child, lambdas[j]))
                ),
            )
        )
    direction = lambdas[associate(context.normalize(child), lambdas)]
    return tuple(
        sorted(neighbors, key=lambda j: sum((a - b) ** 2 for a, b in zip(direction, lambdas[j])))
    )


def crowding(points: list[tuple[float, float]]) -> list[float]:
    """Normalized two-objective crowding distances; stable indices break ties."""
    scores = [0.0] * len(points)
    if not points:
        return scores
    for axis in range(2):
        order = sorted(range(len(points)), key=lambda i: (points[i][axis], i))
        span = points[order[-1]][axis] - points[order[0]][axis]
        if span <= 0:
            continue
        scores[order[0]] = scores[order[-1]] = float("inf")
        for before, index, after in zip(order, order[1:], order[2:]):
            scores[index] += (points[after][axis] - points[before][axis]) / span
    return scores


def select_pool(candidates: list[Candidate], count: int) -> list[Candidate]:
    """Unique-genotype nondominated-layer/crowding truncation with stable ties."""
    unique = {c.genotype: c for c in reversed(candidates)}
    remaining = [
        c
        for i, c in enumerate(candidates)
        if unique.get(c.genotype) is c
        and all(candidates[j].genotype != c.genotype for j in range(i))
    ]
    if len(remaining) < count:
        raise ValueError("Candidate pool cannot supply unique population")
    selected: list[Candidate] = []
    while len(selected) < count:
        front = [
            c
            for c in remaining
            if not any(
                all(a <= b for a, b in zip(d.evaluation.objectives, c.evaluation.objectives))
                and any(a < b for a, b in zip(d.evaluation.objectives, c.evaluation.objectives))
                for d in remaining
            )
        ]
        scores = crowding([c.evaluation.objectives for c in front])
        ordered = sorted(range(len(front)), key=lambda i: (-scores[i], i))
        selected.extend(front[i] for i in ordered[: count - len(selected)])
        chosen = {id(c) for c in front}
        remaining = [c for c in remaining if id(c) not in chosen]
    return selected


class TerminalSelector:
    """Per-run target ordering with persistent endpoint alternation and no RNG."""

    def __init__(self, rule: str) -> None:
        if rule not in TERMINAL:
            raise ValueError("Unknown terminal target rule")
        self.rule = rule
        self.high_next = False

    def __call__(self, todo: list[Candidate], front: list[tuple[float, float]]) -> Candidate:
        """Choose among unvisited physical representatives using the frozen front."""
        if not todo:
            raise ValueError("No terminal targets")
        ordered = sorted(todo, key=lambda c: c.evaluation.objectives)
        if self.rule == "bill_ascending":
            return min(todo, key=lambda c: c.evaluation.objectives[::-1])
        if self.rule == "alternate_extremes":
            target = ordered[-1 if self.high_next else 0]
            self.high_next = not self.high_next
            return target
        if self.rule == "crowding_first":
            scores = dict(zip(front, crowding(front)))
            return min(
                ordered, key=lambda c: (-scores[c.evaluation.objectives], c.evaluation.objectives)
            )
        return ordered[0]
