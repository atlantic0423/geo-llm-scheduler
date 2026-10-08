"""Bounded, objective-blind reservoirs of actual F6 structural evaluation calls."""

from __future__ import annotations

import random
from collections import OrderedDict, defaultdict
from dataclasses import asdict
from time import perf_counter

from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.experiments.d1 import extract_intent
from geo_llm_scheduler.experiments.d1_routing import gate_transfer
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.rl.state import preference
from geo_llm_scheduler.utils.numeric import EPS_NORM, TOL
from geo_llm_scheduler.utils.rng import RNGManager

WINDOWS = ((0.0, 0.2), (0.4, 0.6), (0.8, 1.0))


class RealPathObserver:
    """Sample sixteen events per phase/preference without consuming algorithm RNG.

    Source candidates are immutable references. Storage is bounded by nine
    reservoirs plus a 32-entry positive-wait cache, independent of run length.
    All structural calls, including unchanged genotypes, enter the denominators.
    """

    def __init__(self, problem: ProblemInstance, seconds: float, seed: int, cap: int = 16) -> None:
        """Create explicit observation streams and fixed-capacity reservoirs."""
        if seconds <= 0 or cap < 1:
            raise ValueError("Positive duration and reservoir capacity required")
        self.problem = problem
        self.seconds = seconds
        self.cap = cap
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.seen = {(s, p): 0 for s in range(3) for p in range(3)}
        self.samples: dict[tuple[int, int], list[dict]] = {k: [] for k in self.seen}
        self.streams = {k: RNGManager(seed).stream(f"D1R:reservoir:{k}") for k in self.seen}
        self.cache: OrderedDict[int, tuple[Candidate, bool]] = OrderedDict()
        self.events = 0
        self.observer_seconds = 0.0

    def __call__(self, event: str, values: dict) -> None:
        """Count and reservoir-sample each actual structural evaluation attempt."""
        if event != "structure":
            return
        tick = perf_counter()
        self.events += 1
        source, target = values["source"], values["target"]
        unchanged = source.genotype == target
        os_only = source.genotype.ms == target.ms and source.genotype.os != target.os
        gated = False
        if os_only:
            key = id(source)
            if key not in self.cache:
                positive = any(w > TOL.time for w in extract_intent(self.problem, source))
                self.cache[key] = (source, positive)
                if len(self.cache) > 32:
                    self.cache.popitem(last=False)
            gated = self.cache[key][1]
        fraction = values["elapsed"] / self.seconds
        stage = next((s for s, (lo, hi) in enumerate(WINDOWS) if lo <= fraction < hi), None)
        path = values["path"]
        for name in ("all", f"path:{path}", f"stage:{stage}"):
            counts = self.counts[name]
            counts["events"] += 1
            counts["unchanged"] += int(unchanged)
            counts["os_only"] += int(os_only)
            counts["ms_changed"] += int(source.genotype.ms != target.ms)
            counts["gate_true"] += int(gated)
        if stage is not None:
            p = preference(values["subproblem"], len(values["population"]))
            bucket = (stage, p)
            self.seen[bucket] += 1
            n = self.seen[bucket]
            sample = self.samples[bucket]
            slot = len(sample) if len(sample) < self.cap else self.streams[bucket].randrange(n)
            if slot < self.cap:
                # Decide inclusion before serializing or evaluating a target.
                item = {
                    "event": self.events,
                    "stage": stage,
                    "preference": p,
                    "elapsed": values["elapsed"],
                    "generation": values["generation"],
                    "subproblem": values["subproblem"],
                    "path": path,
                    "step": values.get("step"),
                    "parents": values.get("parents"),
                    "weight": values["weight"],
                    "context": values["context"],
                    "source": source,
                    "target": target,
                    "natural_gate": gated,
                }
                if slot == len(sample):
                    sample.append(item)
                else:
                    sample[slot] = item
        self.observer_seconds += perf_counter() - tick

    def export(self) -> dict:
        """Serialize only bounded samples and complete event denominators."""
        rows = []
        for bucket, samples in self.samples.items():
            for item in sorted(samples, key=lambda x: x["event"]):
                rows.append(
                    {
                        **item,
                        "source": solution(item["source"]),
                        "target": asdict(item["target"]),
                        "context": asdict(item["context"]),
                        "bucket_events": self.seen[bucket],
                        "bucket_sampled": len(samples),
                        "probability": len(samples) / self.seen[bucket],
                    }
                )
        return {
            "rows": rows,
            "counts": dict(self.counts),
            "buckets": {f"{s}:{p}": self.seen[s, p] for s, p in self.seen},
            "observer_seconds": self.observer_seconds,
            "cap": self.cap,
            "windows": WINDOWS,
        }


def allocate_routes(rows: list[dict], intents: list[tuple[float, ...]], rng: random.Random) -> dict:
    """Match TRUE counts within phase/preference, keeping the positive-wait guard.

    Sources can differ. All changed targets with positive source wait are
    eligible, including assignment changes. Objectives never enter allocation.
    Degenerate blocks are retained and explicitly reported.
    """
    if len(rows) != len(intents):
        raise ValueError("Missing source intent")
    blocks: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        blocks[row["stage"], row["preference"]].append(i)
    selected: set[int] = set()
    details = []
    for key, indices in sorted(blocks.items()):
        gates = [
            i
            for i in indices
            if gate_transfer(rows[i]["source"].genotype, rows[i]["target"], intents[i])
        ]
        pool = [
            i
            for i in indices
            if any(w > TOL.time for w in intents[i])
            and rows[i]["source"].genotype != rows[i]["target"]
        ]
        if not set(gates).issubset(pool):
            raise ValueError("Conditional allocation exceeds eligible pool")
        chosen = set(rng.sample(pool, len(gates)))
        selected.update(chosen)
        details.append(
            {
                "stage": key[0],
                "preference": key[1],
                "sampled": len(indices),
                "eligible": len(pool),
                "true": len(gates),
                "symmetric_difference": len(set(gates) ^ chosen),
                "degenerate": len(gates) in (0, len(pool)),
            }
        )
    return {"routes": tuple(i in selected for i in range(len(rows))), "blocks": details}


def retained_gains(row: dict, updated: bool = True) -> dict[str, float]:
    """Compare actually retained outputs under one common candidate-updated ideal.

    Each policy selects with its own evaluated-candidate ideal. This evaluation
    context is common across policies; it cannot choose or generate candidates.
    """
    objectives = [row["base_objectives"]] + [
        p["objectives"]
        for arm in row["groups"].values()
        for p in arm["proposals"]
        if p["on_time"] and p["feasible"]
    ]
    ideal = (
        tuple(min(row["context"]["ideal"][m], *(v[m] for v in objectives)) for m in range(2))
        if updated
        else row["context"]["ideal"]
    )
    maximum, weight = row["context"]["maximum"], row["weight"]

    def scalar(values: list) -> float:
        return max(
            weight[m] * abs((values[m] - ideal[m]) / max(maximum[m] - ideal[m], EPS_NORM))
            for m in range(2)
        )

    base = scalar(row["base_objectives"])
    return {
        g: (base - scalar(a["objectives"])) / max(base, 1e-12) for g, a in row["groups"].items()
    }
