"""Invocation-local assignment workloads with reference operation-order sums."""

from geo_llm_scheduler.diagnostics.workload import imbalance, region_loads
from geo_llm_scheduler.domain.models import Genotype, ProblemInstance


class AssignmentWorkloads:
    """Recompute affected groups, never subtract rounded resource totals.

    One invocation owns one immutable input/incumbent. Machine sums keep the
    reference operation order and per-operation division; regional sums keep
    their separate normalize-after-summing convention.
    """

    def __init__(self, problem: ProblemInstance, genotype: Genotype) -> None:
        self.problem = problem
        self.genotype = genotype
        self.profiles = tuple(problem.profile(o) for o in range(problem.operation_count))
        self.machine_ops: list[list[int]] = [[] for _ in problem.instances]
        self.region_ops: list[list[int]] = [[] for _ in problem.regions]
        for operation, machine in enumerate(genotype.ms):
            self.machine_ops[machine].append(operation)
            self.region_ops[problem.instances[machine].region].append(operation)
        machines = [
            [m for m, server in enumerate(problem.instances) if server.region == region]
            for region in range(len(problem.regions))
        ]
        self.capacities = tuple(
            (len(group), sum(problem.instances[m].vram for m in group)) for group in machines
        )
        self.original_regions = region_loads(problem, genotype)
        self.machine_cache: dict[tuple[int, int, bool, bool], float] = {}
        self.region_cache: dict[tuple[int, int], float] = {}

    def _machine(self, job: int, machine: int, prefill: int, decode: int) -> float:
        key = (job, machine, prefill == machine, decode == machine)
        if key not in self.machine_cache:
            operations = [o for o in self.machine_ops[machine] if o // 2 != job]
            if prefill == machine:
                operations.append(2 * job)
            if decode == machine:
                operations.append(2 * job + 1)
            compute = memory = 0.0
            for operation in sorted(operations):
                profile = self.profiles[operation]
                compute += profile.duration * profile.compute
                memory += profile.duration * profile.vram / self.problem.instances[machine].vram
            self.machine_cache[key] = max(compute, memory)
        return self.machine_cache[key]

    def path_load(self, job: int, prefill: int, decode: int) -> float:
        """Return exactly the reference loads of the two destination machines."""
        return max(
            self._machine(job, prefill, prefill, decode),
            self._machine(job, decode, prefill, decode),
        )

    def region_imbalance(self, job: int, destination: int) -> float:
        """Reuse the regional score for every same-region P/D destination path."""
        key = (job, destination)
        if key not in self.region_cache:
            original = self.problem.instances[self.genotype.ms[2 * job]].region
            loads = list(self.original_regions)
            for region in sorted({original, destination}):
                operations = [o for o in self.region_ops[region] if o // 2 != job]
                if region == destination:
                    operations.extend((2 * job, 2 * job + 1))
                compute = memory = 0.0
                for operation in sorted(operations):
                    profile = self.profiles[operation]
                    compute += profile.duration * profile.compute
                    memory += profile.duration * profile.vram
                count, vram = self.capacities[region]
                loads[region] = max(compute / count, memory / vram) if count else 0.0
            self.region_cache[key] = imbalance(tuple(loads))
        return self.region_cache[key]
