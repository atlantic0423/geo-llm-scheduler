"""Frozen trigger/Polish factorial diagnostic; default algorithms are unchanged."""

from __future__ import annotations

import hashlib
from array import array
from collections import Counter
from dataclasses import replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype
from geo_llm_scheduler.experiments.comparison13 import BUDGETS, PRESSURES, SIZES, arm_config

ARMS = ("A_CURRENT", "B_STRICT", "C_NO_POLISH", "D_STRICT_NO_POLISH", "E_MOEAD")
SEEDS = (19101, 19201)
TOPOLOGIES = ((3, 12), (2, 12), (3, 24), (5, 12))


def mechanism_config(arm: str, seed: int, seconds: float) -> Config:
    """Resolve exactly the five user-authorized gates and trajectory Polish flags."""
    if arm not in ARMS:
        raise ValueError("Unknown mechanism arm")
    original = arm_config("MOEAD" if arm == "E_MOEAD" else "M11_I4", seed, seconds)
    return replace(
        original,
        trigger_mode="strict" if arm in ("B_STRICT", "D_STRICT_NO_POLISH") else "preference",
        trigger_delta=0.0 if arm in ("B_STRICT", "D_STRICT_NO_POLISH") else 0.10,
        polish=arm in ("A_CURRENT", "B_STRICT"),
    )


def balanced_order(index: int) -> list[str]:
    """Use ten Williams orders, balancing all five positions and predecessors."""
    row = [0, 1, 4, 2, 3]
    order = [ARMS[(j + index % 5) % 5] for j in row]
    return order if index % 10 < 5 else order[::-1]


def study_specs(pilot: bool = False) -> list[dict]:
    """Freeze sixteen fresh parents, two paired variants and two algorithm seeds."""
    specs = []
    for variant in (0, 1):
        for seed_index, seed in enumerate(SEEDS):
            for pressure_index, pressure in enumerate(PRESSURES):
                for size_index, n in enumerate(SIZES):
                    # Pilot is a separate set of parents, never a resampled formal input.
                    base = (
                        (202610199000 if pilot else 202610190000) + pressure_index * 10 + size_index
                    )
                    r, m = TOPOLOGIES[(pressure_index + size_index) % 4]
                    key = f"m5_n{n}_b{base}_v{variant}_s{seed}"
                    specs.append(
                        dict(
                            key=key,
                            n=n,
                            regions=r,
                            machines=m,
                            pressure=pressure,
                            variant=variant,
                            base_seed=base,
                            seed=seed,
                            bank_seed=seed + 100000,
                            seconds=({20: 45, 50: 90, 100: 180, 200: 480} if pilot else BUDGETS)[n],
                            # Four available versions per parent rotate order independently.
                            order=balanced_order(pressure_index * 4 + variant * 2 + seed_index),
                        )
                    )
    return specs


def genotype_digest(genotype: Genotype) -> bytes:
    """Return a 128-bit digest of MS/OS so repeat counting needs O(1) memory per genotype."""
    data = array("q", genotype.ms)
    data.append(-1)
    data.extend(genotype.os)
    return hashlib.blake2b(data.tobytes(), digest_size=16).digest()


class StructuralObserver:
    """Count exact structural repeats without suppressing any evaluation or RNG draw.

    Seen genotypes are stored as 16-byte digests rather than full tuples, so memory
    no longer grows with problem size times offspring count.
    """

    def __init__(self) -> None:
        self.initialized = False
        self.last_ssgs = 0
        self.seen: set[bytes] = set()
        self.seen_search: set[bytes] = set()
        self.counts: Counter[str] = Counter()

    def candidate(self, candidate: Candidate, ssgs: int) -> None:
        """Use the gateway rebuild counter to distinguish structural and timing work."""
        rebuilt = ssgs > self.last_ssgs
        self.last_ssgs = ssgs
        digest = genotype_digest(candidate.genotype)
        if rebuilt:
            stage = "search" if self.initialized else "initialization"
            self.counts[f"{stage}:structural_evaluations"] += 1
            novel = digest not in self.seen
            self.counts[f"{stage}:novel_structure_in_run"] += int(novel)
            self.counts[f"{stage}:repeated_structure_in_run"] += int(not novel)
            if self.initialized:
                repeated = digest in self.seen_search
                self.counts["search:repeated_structure_in_search"] += int(repeated)
                self.seen_search.add(digest)
        elif self.initialized:
            self.counts["search:timing_evaluations"] += 1
        self.seen.add(digest)

    def summary(self, population: list[Candidate], archive: list[Candidate]) -> dict:
        """Return final genotype multiplicities; timing-distinct members remain intact."""
        return dict(
            counts=dict(self.counts),
            all_unique_genotypes=len(self.seen),
            search_unique_structural_genotypes=len(self.seen_search),
            population_unique_genotypes=len({c.genotype for c in population}),
            archive_unique_genotypes=len({c.genotype for c in archive}),
            archive_members=len(archive),
        )
