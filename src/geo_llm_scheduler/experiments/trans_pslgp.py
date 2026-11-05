"""Paper-guided PSLGP-H-PD: preference-conditioned GP with nearest-PC surrogate.

Algorithms 1--4 of Xu et al. TEVC 2025, with source GP/brood defaults.
The online static P/D dispatcher replaces their offline dynamic FJSP simulator.
This is an attributed problem adaptation, not a reproduction of published scores.
"""

from dataclasses import asdict

import numpy as np

from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.experiments.trans_common import (
    Array,
    Individual,
    SearchContext,
    hv2,
    objectives,
)
from geo_llm_scheduler.experiments.trans_gp import Node, Policy, normalize, random_tree, variation


class Dispatcher:
    """Map two GP rules to same-region P/D routes and precedence-safe dispatch priorities."""

    def __init__(self, context: SearchContext) -> None:
        self.context = context
        self.situations: list[list[Array]] = [[], []]
        self.pc_sets: list[Array] = []

    def dispatch(self, policy: Policy, preference: Array, *, collect: bool = False) -> Individual:
        """Build assignment/priorities from online prefix workload proxies, then exact SSGS.

        Workload is a feature, not a serial-machine constraint: exact decoding
        still allows simultaneous operations subject to compute and VRAM.
        """
        ctx, problem = self.context, self.context.problem
        n = len(problem.jobs)
        workload = np.zeros(len(problem.instances))
        assignment = [-1] * (2 * n)
        ready = np.asarray([j.release for j in problem.jobs])
        seen = np.zeros(n, dtype=int)
        order = []
        for _ in range(2 * n):
            active = np.flatnonzero(seen < 2)
            sequence = []
            for j in active:
                job = problem.jobs[int(j)]
                profile = (job.prefill, job.decode)[int(seen[j])]
                machine = assignment[2 * j + int(seen[j])]
                load = float(workload[machine]) if machine >= 0 else float(workload.min())
                sequence.append(
                    [
                        profile.duration,
                        ready[j],
                        profile.duration + (job.decode.duration if seen[j] == 0 else 0),
                        load,
                        profile.compute,
                        profile.vram,
                        profile.compute * profile.duration,
                        0.0,
                        *preference,
                    ]
                )
            sf = normalize(np.asarray(sequence))
            if collect:
                self.situations[0].append(sf.copy())
            j = int(active[int(np.argmin(policy[0].evaluate(sf)))])
            job = problem.jobs[j]
            if seen[j] == 0:
                choices = ctx.legal[j]
                routes = []
                for p, d in choices:
                    server = problem.instances[p]
                    region = problem.regions[server.region]
                    start = max(job.release, float(workload[p]))
                    price = next(
                        (t.price for t in region.tariffs if t.start <= start < t.end),
                        region.tariffs[-1].price,
                    )
                    energy = (
                        server.active_kw * job.prefill.duration
                        + problem.instances[d].active_kw * job.decode.duration
                    )
                    routes.append(
                        [
                            job.prefill.duration,
                            start,
                            job.prefill.duration + job.decode.duration,
                            workload[p] + workload[d],
                            job.prefill.compute,
                            job.prefill.vram,
                            energy / 3600,
                            price,
                            *preference,
                        ]
                    )
                rf = normalize(np.asarray(routes))
                if collect:
                    self.situations[1].append(rf.copy())
                assignment[2 * j : 2 * j + 2] = choices[int(np.argmin(policy[1].evaluate(rf)))]
            phase = int(seen[j])
            profile = (job.prefill, job.decode)[phase]
            machine = assignment[2 * j + phase]
            workload[machine] += profile.duration * profile.compute
            ready[j] += profile.duration + (
                job.kv_delay if phase == 0 and assignment[2 * j] != assignment[2 * j + 1] else 0.0
            )
            seen[j] += 1
            order.append(j)
        genotype = Genotype(tuple(assignment), tuple(order))
        candidate = ctx.gateway.evaluate(genotype, origin="PSLGP_H_PD")
        ctx.evaluations += 1
        return Individual(np.empty(0), candidate)

    def prepare(self) -> Array:
        """Charge manual-rule reference schedules and freeze 20+20 decision situations."""
        # Adapt paper manual-rule normalization to P/D SPT / minimum load+energy.
        rule = (Node(0), Node("+", (Node(3), Node(6))))
        reference = self.dispatch(rule, np.array([0.5, 0.5]), collect=True)
        for raw in self.situations:
            eligible = [s for s in raw if len(s) >= 7] or raw
            chosen = self.context.rng.choice(len(eligible), 20, replace=len(eligible) < 20)
            batch = []
            for i in chosen:
                s = eligible[int(i)]
                # Small toy problems may have <7 distinct candidates: repeat padding.
                indices = self.context.rng.choice(len(s), 7, replace=len(s) < 7)
                batch.append(s[indices])
            self.pc_sets.append(np.asarray(batch))
        self.situations = [[], []]
        return np.maximum(reference.candidate.evaluation.objectives, 1e-15)

    def characterize(
        self,
        policy: Policy,
        preferences: Array,
        reference: Policy,
        reference_preference: Array | None = None,
    ) -> Array:
        """Return 40 reference-rule ranks per preference, without exact evaluations."""
        parts = []
        for tree, ref, decisions in zip(policy, reference, self.pc_sets, strict=True):
            features = np.broadcast_to(decisions, (len(preferences), *decisions.shape)).copy()
            features[..., 8:] = preferences[:, None, None, :]
            chosen = np.argmin(tree.evaluate(features), axis=2)
            # Fix the reference ranking across target preferences. Otherwise a
            # rule compared with itself would always have PC=1, hiding changes.
            fixed = decisions.copy()
            fixed[..., 8:] = (
                np.array([0.5, 0.5]) if reference_preference is None else reference_preference
            )
            ranks = (
                np.argsort(
                    np.argsort(ref.evaluate(fixed), axis=1, kind="stable"), axis=1, kind="stable"
                )
                + 1
            )
            ranks = np.broadcast_to(ranks, chosen.shape + (7,))
            parts.append(np.take_along_axis(ranks, chosen[..., None], axis=2)[..., 0])
        return np.column_stack(parts).astype(float)


def nearest(query: Array, training: Array) -> np.ndarray:
    """Exact 1-nearest-PC lookup in bounded chunks; paper Algorithm 3, first ties."""
    result = []
    for offset in range(0, len(query), 128):
        q = query[offset : offset + 128]
        distance = (
            (q * q).sum(axis=1)[:, None]
            + (training * training).sum(axis=1)[None, :]
            - 2 * q @ training.T
        )
        result.extend(np.argmin(distance, axis=1).tolist())
    return np.asarray(result, dtype=int)


def ranks(values: Array) -> np.ndarray:
    """Nondomination layer ranks with equal objective vectors in the same layer."""
    domination = np.all(values[:, None] <= values[None, :], axis=2) & np.any(
        values[:, None] < values[None, :], axis=2
    )
    counts = domination.sum(axis=0)
    result = np.full(len(values), -1, dtype=int)
    layer = 0
    while np.any(result < 0):
        front = np.flatnonzero((counts == 0) & (result < 0))
        result[front] = layer
        counts -= domination[front].sum(axis=0)
        layer += 1
    return result


def criteria(values: Array, pcs: Array, preferences: Array, scale: Array) -> list[tuple]:
    """Paper rank, descending HV, manual-normalized weighted sum, PC diversity."""
    true = values[:, 0]
    training = pcs[:, 0]
    inferred = values[nearest(pcs.reshape(-1, pcs.shape[-1]), training), 0].reshape(
        len(pcs), len(preferences), 2
    )
    inferred[:, 0] = true
    reference = true.max(axis=0)
    front = ranks(true)
    result = []
    for i, estimates in enumerate(inferred):
        hv = hv2(estimates / scale, reference / scale)
        weighted = float(np.sum(estimates / scale * preferences))
        # Paper Eq. 6 counts unique behaviors; text/figure also describe distances.
        diversity = len(np.unique(pcs[i], axis=0)) / len(preferences)
        result.append((int(front[i]), -hv, weighted, -diversity))
    return result


def preselect(keys: list[tuple], pcs: Array, size: int) -> list[int]:
    """Sort surrogate criteria and clear identical main-PC niches (radius 0, cap 1)."""
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    selected, rejected = [], []
    seen = set()
    for i in order:
        signature = tuple(pcs[i])
        if signature in seen:
            rejected.append(i)
        else:
            selected.append(i)
            seen.add(signature)
    # Source clearing penalizes duplicates; they fill if fewer than N niches exist.
    return (selected + rejected)[:size]


def run_pslgp(
    context: SearchContext, population_size: int = 1000, *, preferences_count: int = 101
) -> list[Individual]:
    """Online PSLGP-H-PD with charged learning, 5N brood, rotating main preference.

    No cross-instance model is trained or reused. Native GP population is 1000;
    engineering tests can explicitly override this and preference count.
    """
    if population_size < 4 or preferences_count < 3:
        raise ValueError("PSLGP needs >=4 policies and >=3 preferences")
    rng = context.rng
    dispatcher = Dispatcher(context)
    scale = dispatcher.prepare()
    grid = np.linspace(0, 1, preferences_count - 1)
    weights = np.column_stack((grid, 1 - grid))
    rotation = rng.permutation(len(weights))
    population: list[Policy] = [
        (
            random_tree(rng, int(rng.integers(2, 7)), full=bool(rng.integers(2))),
            random_tree(rng, int(rng.integers(2, 7)), full=bool(rng.integers(2))),
        )
        for _ in range(population_size)
    ]
    reference: Policy = (Node(0), Node(3))
    evaluated: list[Individual] = []
    while context.running():
        main = weights[int(rotation[context.generations % len(rotation)])]
        preferences = np.vstack((main, weights))
        evaluated = []
        for policy in population:
            if context.progress() >= 1:
                break
            evaluated.append(dispatcher.dispatch(policy, main))
        if len(evaluated) != len(population):
            context.trace.append({"partial_generation": True, "evaluated": len(evaluated)})
            break
        pcs = np.asarray(
            [dispatcher.characterize(policy, preferences, reference) for policy in population]
        )
        values = objectives(evaluated)[:, None, :]
        keys = criteria(values, pcs, preferences, scale)
        best = min(range(len(keys)), key=lambda i: keys[i])
        context.generations += 1
        context.trace.append(
            {
                "generation": context.generations,
                "main_preference": main.tolist(),
                "best_criteria": keys[best],
                "exact_policies": len(evaluated),
                "brood_multiplier": 5,
            }
        )
        # Save a portable best policy; archive contains only actual exact schedules.
        context.trace[-1]["best_policy"] = [asdict(t) for t in population[best]]
        if not context.running():
            break

        def parent() -> Policy:
            competitors = rng.integers(len(population), size=7)
            return population[min(competitors, key=lambda i: keys[int(i)])]

        elites = sorted(range(len(keys)), key=lambda i: keys[i])[
            : min(10, max(1, population_size // 100))
        ]
        brood = [population[i] for i in elites]
        while len(brood) < 5 * population_size and context.progress() < 1:
            brood.append(variation(parent(), parent(), rng))
        if context.progress() >= 1:
            break
        next_main = weights[int(rotation[context.generations % len(rotation)])]
        # Train and query use the same reference rule and frozen situations.
        reference = population[best]
        training = np.asarray(
            [dispatcher.characterize(p, main[None, :], reference)[0] for p in population]
        )
        brood_pc = np.asarray(
            [dispatcher.characterize(p, next_main[None, :], reference)[0] for p in brood]
        )
        estimates = nearest(brood_pc, training)
        picked = preselect([keys[int(i)] for i in estimates], brood_pc, population_size)
        population = [brood[i] for i in picked]
    return evaluated
