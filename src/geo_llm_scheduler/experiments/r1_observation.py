"""Bounded, passive sampling of live structural offspring before any R1 search."""

from collections import Counter
from dataclasses import asdict
from random import Random
from time import perf_counter

from geo_llm_scheduler.moead.core import NormalizationContext, maximum, weights


class RegionalObserver:
    """Uniform per-stage reservoirs, independent of every algorithm random stream.

    Sampling precedes eligibility inspection. Immutable candidates are retained by
    reference; objective fronts and slot membership are copied at observation time.
    No decoding, billing, archive insertions or eligibility scans happen online.
    """

    def __init__(self, seconds: float, seed: int, capacity: int = 16, recipients: int = 2) -> None:
        """Set the declared total budget and independent diagnostic master seed."""
        if seconds <= 0 or capacity < 1 or recipients < 1:
            raise ValueError("Positive budget and sampling limits required")
        self.seconds = seconds
        self.capacity = capacity
        self.recipients = recipients
        self.rngs = [Random(f"r1-reservoir:{seed}:{stage}") for stage in range(3)]
        self.pair_rng = Random(f"r1-recipients:{seed}")
        self.initialized: float | None = None
        self.seen = [0, 0, 0]
        self.routes: list[Counter] = [Counter(), Counter(), Counter()]
        self.samples: list[list[dict]] = [[], [], []]
        self.pending: dict | None = None
        self.seconds_used = 0.0

    def __call__(self, event: str, data: dict) -> None:
        """Observe existing engine events without mutating any supplied object."""
        tick = perf_counter()
        try:
            if event in ("initialization", "before_replacement", "offspring"):
                self._observe(event, data)
        finally:
            self.seconds_used += perf_counter() - tick

    def _observe(self, event: str, data: dict) -> None:
        if event == "initialization":
            self.initialized = data["elapsed"]
            return
        if event == "before_replacement":
            self.pending = None
            assert self.initialized is not None
            span = 0.95 * self.seconds - self.initialized
            elapsed = data["elapsed"]
            if span <= 0 or elapsed >= 0.95 * self.seconds:
                return
            stage = min(2, max(0, int(3 * (elapsed - self.initialized) / span)))
            self.seen[stage] += 1
            self.routes[stage][data["candidate"].origin] += 1
            index = self.seen[stage] - 1
            if index >= self.capacity:
                index = self.rngs[stage].randrange(self.seen[stage])
            if index >= self.capacity:
                return
            neighbors = tuple(data["neighbors"])
            selected = self.pair_rng.sample(list(neighbors), min(self.recipients, len(neighbors)))
            self.pending = dict(
                stage=stage,
                index=index,
                elapsed=elapsed,
                generation=data["generation"],
                subproblem=data["subproblem"],
                donor=data["candidate"],
                replacement_context=data["context"],
                replacement_neighbors=neighbors,
                selected=selected,
            )
        elif self.pending is not None:
            pending, self.pending = self.pending, None
            stage, index = pending.pop("stage"), pending.pop("index")
            population, gateway = data["population"], data["gateway"]
            lambdas = weights(len(population))
            pending.update(
                stage=stage,
                processed=data["processed"],
                replaced=data["row"]["replaced"],
                context=NormalizationContext(gateway.ideal, maximum(population)),
                front=tuple(sorted({c.evaluation.objectives for c in gateway.archive.members})),
                archive=tuple(gateway.archive.members),
                receivers=tuple(
                    dict(slot=j, weight=lambdas[j], candidate=population[j])
                    for j in pending.pop("selected")
                ),
            )
            if index == len(self.samples[stage]):
                self.samples[stage].append(pending)
            else:
                self.samples[stage][index] = pending

    def record(self) -> dict:
        """Serialize after search; expensive JSON conversion is outside the solver."""
        snapshots = []
        for rows in self.samples:
            for row in sorted(rows, key=lambda x: x["processed"]):
                snapshots.append(
                    {
                        **row,
                        "donor": asdict(row["donor"]),
                        "archive": [asdict(c) for c in row["archive"]],
                        "context": asdict(row["context"]),
                        "replacement_context": asdict(row["replacement_context"]),
                        "receivers": [
                            {**r, "candidate": asdict(r["candidate"])} for r in row["receivers"]
                        ],
                    }
                )
        return dict(
            protocol="r1-live-observation/v1",
            capacity_per_stage=self.capacity,
            recipients_per_snapshot=self.recipients,
            initialization_end=self.initialized,
            search_end=0.95 * self.seconds,
            seen=self.seen,
            routes=[dict(x) for x in self.routes],
            snapshot_inclusion=[min(1.0, self.capacity / n) if n else 0.0 for n in self.seen],
            observer_seconds=self.seconds_used,
            snapshots=snapshots,
        )
