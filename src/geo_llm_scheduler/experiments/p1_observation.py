"""Bounded P1 observations and recoverable, stratified phenotype checkpoints."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TextIO

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, EvaluationResult, Genotype, Schedule
from geo_llm_scheduler.experiments.campaign.support import atomic_json, process_peak_rss_gb
from geo_llm_scheduler.experiments.runner import digest, solution
from geo_llm_scheduler.moead.core import NormalizationContext, maximum, neighborhoods, weights
from geo_llm_scheduler.rl.state import associate, preference
from geo_llm_scheduler.utils.numeric import TOL, identity, less
from geo_llm_scheduler.utils.rng import RNGManager


def restore_candidate(data: dict) -> Candidate:
    """Restore a checkpoint value, retaining every timing and diagnostic field."""
    g, s, e = data["genotype"], data["schedule"], data["evaluation"]
    return Candidate(
        Genotype(tuple(g["ms"]), tuple(g["os"])),
        Schedule(tuple(s["starts"]), tuple(s["resource_wait"]), tuple(s["intentional_wait"])),
        EvaluationResult(
            e["feasible"],
            e["flow"],
            e["tou"],
            tuple(e["demand"]),
            tuple(tuple(w) for w in e["windows"]),
            tuple(e["violations"]),
        ),
        data["origin"],
    )


def sample_population(population: list[Candidate], rng: random.Random) -> list[dict]:
    """Draw one unique-phenotype representative per preference, then one remainder.

    A phenotype selected twice is recorded as missing, without replacement from
    the same stratum. Probabilities refer to distinct phenotypes, not copies.
    """
    rows: list[dict] = []
    seen: set[tuple] = set()
    for group in range(3):
        candidates: dict[tuple, list[int]] = {}
        for i, c in enumerate(population):
            if preference(i, len(population)) == group:
                candidates.setdefault(identity(c), []).append(i)
        keys = sorted(candidates)
        row: dict = {"stratum": group, "size": len(keys), "probability": 0.0, "index": None}
        if keys:
            key = rng.choice(keys)
            indices = candidates[key]
            index = rng.choice(indices)
            row.update(
                probability=1 / len(keys),
                index_probability=1 / len(indices),
                missing="duplicate" if key in seen else None,
            )
            if key not in seen:
                row["index"] = index
                seen.add(key)
        else:
            row["missing"] = "empty"
        rows.append(row)
    remaining: dict[tuple, list[int]] = {}
    for i, c in enumerate(population):
        if identity(c) not in seen:
            remaining.setdefault(identity(c), []).append(i)
    keys = sorted(remaining)
    extra: dict = {
        "stratum": "remainder",
        "size": len(keys),
        "probability": 0.0,
        "index": None,
        "missing": "empty",
    }
    if keys:
        indices = remaining[rng.choice(keys)]
        extra.update(
            index=rng.choice(indices),
            probability=1 / len(keys),
            index_probability=1 / len(indices),
            missing=None,
        )
    rows.append(extra)
    return rows


def replacement_audit(
    population: list[Candidate],
    child: Candidate,
    context: NormalizationContext,
    birth: int,
    config: Config,
    rng: random.Random,
) -> dict:
    """Audit own-weight improvements, independently separating neighborhood and cap."""
    lambdas = weights(len(population))
    neighbors = neighborhoods(lambdas, config.neighborhood)
    gains = [context.scalar(c, w) - context.scalar(child, w) for c, w in zip(population, lambdas)]
    improving = [i for i, gain in enumerate(gains) if less(0.0, gain, TOL.scalar)]
    original = [j for j in neighbors[birth] if j in improving]
    random_order = list(neighbors[birth])
    rng.shuffle(random_order)
    direction = associate(context.normalize(child), lambdas)
    by_direction = [j for j in neighbors[direction] if j in improving]
    cap = config.replacement_cap
    return {
        "ideal": context.ideal,
        "maximum": context.maximum,
        "birth": birth,
        "direction": direction,
        "improving": improving,
        "gains": gains,
        "outside_birth": [j for j in improving if j not in neighbors[birth]],
        "original": original[:cap],
        "random": [j for j in random_order if j in improving][:cap],
        "gain_order": sorted(original, key=lambda j: (-gains[j], j))[:cap],
        "direction_neighbors": by_direction[:cap],
        "cap_truncated": max(0, len(original) - cap),
    }


def _semantic(value: object) -> object:
    if isinstance(value, dict):
        return {
            k: _semantic(v)
            for k, v in value.items()
            if k != "elapsed" and k != "archive_objectives" and not k.endswith("seconds")
        }
    if isinstance(value, (list, tuple)):
        return [_semantic(v) for v in value]
    return value


class SemanticRecorder:
    """Hash complete candidate and decision order in constant historical memory."""

    def __init__(self) -> None:
        self.chain = hashlib.sha256()
        self.decision_chain = hashlib.sha256()
        self.rows = 0

    def __call__(self, event: str, data: dict) -> None:
        """Consume immutable candidates and semantic offspring rows synchronously."""
        if event == "candidate":
            self.chain.update(digest(solution(data["candidate"])).encode())
        elif event == "offspring":
            self.decision_chain.update(digest(_semantic(data["row"])).encode())
            self.rows += 1

    def signature(self, result: object) -> dict:
        """Combine ordered history with final population/archive/Q and exact accounting."""
        # RunResult is imported lazily to avoid an observation/engine cycle.
        from geo_llm_scheduler.engine.run import RunResult

        assert isinstance(result, RunResult)
        return {
            "chain": self.chain.hexdigest(),
            "decision_chain": self.decision_chain.hexdigest(),
            "offspring": self.rows,
            "population": [digest(solution(c)) for c in result.population],
            "archive": sorted(digest(solution(c)) for c in result.archive.members),
            "q": result.controller.q,
            "visits": result.controller.visits,
            "updates": result.controller.updates,
            "selections": result.controller.selections,
            "counts": dict(result.gateway.counts),
            "reason": result.termination_reason,
        }


class P1Observer:
    """Stream compact events and save only the three prespecified full checkpoints."""

    def __init__(
        self, output: Path, config: Config, fractions: tuple[float, ...] = (0.1, 0.5, 0.9)
    ):
        if config.seconds is None:
            raise ValueError("P1 sampling requires a time budget")
        self.output, self.config, self.fractions = output, config, fractions
        self.budget_seconds = config.seconds
        output.mkdir(parents=True, exist_ok=False)
        self.handle: TextIO = gzip.open(
            output / "events.jsonl.gz", "wt", encoding="utf-8", compresslevel=1
        )
        self.rng = RNGManager(config.seed).stream("P1:observation")
        self.checkpoints: list[str] = []
        self.generation = self.subproblem = -1
        self.last_heartbeat = -float("inf")
        self.start = perf_counter()
        self.seconds = 0.0
        self.recorder = SemanticRecorder()

    def _write(self, event: str, data: dict) -> None:
        self.handle.write(json.dumps({"event": event, **data}, separators=(",", ":")) + "\n")

    def __call__(self, event: str, data: dict) -> None:
        """Observe without consuming any RNG stream belonging to the search engine."""
        begin = perf_counter()
        self.recorder(event, data)
        if event == "before_offspring":
            self.generation, self.subproblem = data["generation"], data["subproblem"]
        elif event == "candidate":
            c, gateway = data["candidate"], data["gateway"]
            self._write(
                event,
                {
                    "generation": self.generation,
                    "subproblem": self.subproblem,
                    "elapsed": data["elapsed"],
                    "exact": gateway.counts["exact"],
                    "identity": digest(identity(c)),
                    "origin": c.origin,
                    "flow": c.evaluation.flow,
                    "bill": c.evaluation.bill,
                    "tou": c.evaluation.tou,
                    "demand": c.evaluation.demand,
                    "feasible": c.evaluation.feasible,
                    "archive_insertions": gateway.archive.insertions,
                },
            )
        elif event == "before_replacement":
            self._write(
                "replacement",
                {
                    "elapsed": data["elapsed"],
                    "generation": self.generation,
                    "subproblem": self.subproblem,
                    **replacement_audit(
                        data["population"],
                        data["candidate"],
                        data["context"],
                        self.subproblem,
                        self.config,
                        self.rng,
                    ),
                },
            )
        elif event == "initialization":
            with gzip.open(self.output / "initial_population.json.gz", "wt", encoding="utf-8") as h:
                json.dump([solution(c) for c in data["population"]], h)
        elif event == "offspring":
            self._write(event, data["row"])
            for fraction in self.fractions:
                name = f"checkpoint_{fraction:.1f}.json.gz"
                if (
                    data["elapsed"] >= fraction * self.budget_seconds
                    and name not in self.checkpoints
                ):
                    self._checkpoint(name, fraction, data)
            if perf_counter() - self.last_heartbeat >= 30:
                self.handle.flush()
                atomic_json(
                    self.output / "heartbeat.json",
                    {
                        "phase": "sampling",
                        "engine_elapsed": data["elapsed"],
                        "processed": data["processed"],
                        "exact": data["gateway"].counts["exact"],
                        "peak_rss_gb": process_peak_rss_gb(),
                        "checkpoints": self.checkpoints,
                    },
                )
                self.last_heartbeat = perf_counter()
        self.seconds += perf_counter() - begin

    def _checkpoint(self, name: str, fraction: float, data: dict) -> None:
        population, gateway = data["population"], data["gateway"]
        context = NormalizationContext(gateway.ideal, maximum(population))
        payload = {
            "fraction": fraction,
            "elapsed": data["elapsed"],
            "processed": data["processed"],
            "generation": self.generation,
            "subproblem": self.subproblem,
            "config": asdict(self.config),
            "population": [solution(c) for c in population],
            "archive": [solution(c) for c in gateway.archive.members],
            "ideal": context.ideal,
            "maximum": context.maximum,
            "stagnation": list(data["stagnation"]),
            "q": data["controller"].q,
            "rng_states": data["streams"].snapshot(),
            "sample": sample_population(population, self.rng),
        }
        with gzip.open(self.output / name, "wt", encoding="utf-8", compresslevel=1) as h:
            json.dump(payload, h)
        self.checkpoints.append(name)

    def close(self) -> None:
        """Flush the gzip footer after the run, including abnormal caller cleanup."""
        self.handle.close()
