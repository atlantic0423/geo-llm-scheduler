"""One run, including native initialization and bounded diagnostic observers."""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json, process_peak_rss_gb
from geo_llm_scheduler.experiments.runner import digest, solution
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.macrosearch.distance import trajectory_signal
from geo_llm_scheduler.macrosearch.trajectory_budget import ARM_SPECS
from geo_llm_scheduler.utils.numeric import EPS_NORM, TOL, close, less, schedule_identity

from .common import config_from, sha, source, stream_json, validate_complete


class Observer:
    """Keep fixed-size trajectory reservoirs and complete aggregate accounting."""

    def __init__(
        self,
        problem: object,
        seconds: float,
        seed: int,
        calibration: bool = False,
        heartbeat: Path | None = None,
    ):
        self.problem = problem
        self.seconds = seconds
        self.rng = random.Random(seed ^ 0xA017CA1)
        self.calibration = calibration
        self.heartbeat = heartbeat
        self.last_heartbeat = 0.0
        self.counts: Counter = Counter()
        self.signals: dict[str, list[float]] = {a: [] for a in ("D1", "D2", "D3", "D4", "D5", "D6")}
        self.signal_seen: Counter = Counter()
        self.paths: list[dict] = []
        self.anytime: list[dict] = []
        self.initial_hash = ""
        self.initial_seconds = 0.0
        self.thresholds = iter((0.0, 0.1, 0.25, 0.5, 0.75, 1.0))
        self.next_threshold = next(self.thresholds)

    def __call__(self, event: str, data: dict) -> None:
        """Observe completed events without consuming any algorithm random stream."""
        now = time.monotonic()
        if self.heartbeat is not None and now - self.last_heartbeat >= 15:
            atomic_json(
                self.heartbeat,
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "utc_epoch": time.time(),
                    "elapsed": data["elapsed"],
                    "exact": data["gateway"].counts["exact"],
                },
            )
            self.last_heartbeat = now
        if event == "initialization":
            self.initial_hash = digest([asdict(c.genotype) for c in data["population"]])
            self.initial_seconds = data["elapsed"]
        if event in ("initialization", "offspring"):
            elapsed = data["elapsed"]
            if elapsed / self.seconds >= self.next_threshold:
                self.anytime.append(
                    {
                        "elapsed": elapsed,
                        "points": [
                            c.evaluation.objectives for c in data["gateway"].archive.members
                        ],
                    }
                )
                while self.next_threshold <= elapsed / self.seconds:
                    self.next_threshold = next(self.thresholds, float("inf"))
        if event != "auxiliary_episode":
            return
        rows = data["records"]
        context, weight = data["context"], data["weight"]
        positions = data["positions"]
        self.counts["episodes"] += 1
        self.counts[f"depth:{len(rows)}"] += 1
        best = context.scalar(positions[0], weight)
        pending: list[tuple[int, float]] = []
        recovery: list[tuple[int, float]] = []
        for j, row in enumerate(rows):
            for name in ("accepted", "regression", "revisit", "effective", "attempts"):
                self.counts[name] += row[name]
            self.counts[f"action:{row['action']}"] += 1
            self.counts[f"budget:{row['budget']}"] += 1
            if j > 0:
                # Exclude warm-up and credit-truncated tails from the matched marginal.
                remaining = (
                    ARM_SPECS[data["arm"]].credit - rows[j - 1]["budget_details"]["credit_used"]
                )
                if remaining >= 10:
                    self.counts[f"full_request:{row['budget']}"] += 1
            score = context.scalar(positions[j + 1], weight)
            before = context.scalar(positions[j], weight)
            if less(before, score, TOL.scalar):
                pending.append((j, best))
                recovery.append((j, before))
                self.counts["fixed_context_regressions"] += 1
            current_best = min(
                max(
                    weight[k]
                    * abs(
                        (p[k] - context.ideal[k])
                        / max(context.maximum[k] - context.ideal[k], EPS_NORM)
                    )
                    for k in range(2)
                )
                for p in row["historical_objectives"]
            )
            retained = []
            for start, old_best in pending:
                if less(current_best, old_best, TOL.scalar):
                    self.counts[f"breakthrough_after:{j - start}"] += 1
                else:
                    retained.append((start, old_best))
            pending = retained
            remaining_recovery = []
            for start, baseline in recovery:
                if not less(baseline, score, TOL.scalar):
                    self.counts[f"position_recovery_after:{j - start}"] += 1
                else:
                    remaining_recovery.append((start, baseline))
            recovery = remaining_recovery
            best = current_best
            if self.calibration and j:
                for arm, reservoir in self.signals.items():
                    spec = ARM_SPECS[arm]
                    signal, _ = trajectory_signal(
                        self.problem,
                        positions[: j + 1],
                        spec.metric,
                        spec.window,
                        spec.signal,
                        rows[j - 1]["action"],
                    )
                    self.signal_seen[arm] += 1
                    if len(reservoir) < 1024:
                        reservoir.append(signal)
                    else:
                        k = self.rng.randrange(self.signal_seen[arm])
                        if k < len(reservoir):
                            reservoir[k] = signal
        self.counts["unrecovered_regressions"] += len(pending)
        self.counts["unrecovered_positions"] += len(recovery)
        self.counts["last_best_gap_sum"] += max(
            0, context.scalar(positions[-1], weight) - context.scalar(data["selected"], weight)
        )
        self.counts["physical_zero_steps"] += sum(
            schedule_identity(a) == schedule_identity(b) for a, b in zip(positions, positions[1:])
        )
        sample = {
            "context": asdict(context),
            "weight": weight,
            "rows": rows,
            "positions": [
                {
                    "genotype": asdict(c.genotype),
                    "starts": c.schedule.starts,
                    "objectives": c.evaluation.objectives,
                }
                for c in positions
            ],
        }
        if len(self.paths) < 3:
            self.paths.append(sample)
        else:
            k = self.rng.randrange(self.counts["episodes"])
            if k < len(self.paths):
                self.paths[k] = sample


def execute_run(spec: dict, control: Path, output: Path, provenance: dict) -> dict:
    """Run one immutable input/config and atomically commit fully audited artifacts."""
    if socket.gethostname() != spec["host"]:
        raise ValueError("Wrong actual host for paired parent")
    if source() != provenance:
        raise ValueError("Source changed or dirty before run")
    if output.exists():
        return validate_complete(output, spec, provenance)
    output.mkdir(parents=True)
    config = config_from(spec["config"])
    input_path = control / spec["input"]
    if sha(input_path) != spec["input_sha256"]:
        raise ValueError("Input bytes changed")
    problem = load_instance(input_path)
    observer = Observer(
        problem,
        config.seconds,
        config.seed,
        spec.get("calibration", False),
        control / "heartbeats" / f"{spec['id']}.json",
    )
    tick = time.perf_counter()
    result = run(problem, config, retain_trace=False, observer=observer)
    search_wall = time.perf_counter() - tick
    if observer.initial_hash != spec["initial_hash"] or not result.offspring_count:
        raise ValueError("Wrong initial genes or no evolution before time limit")
    audit_tick = time.perf_counter()
    checked = 0
    for candidate in (*result.population, *result.archive.members):
        fresh = evaluate(problem, candidate.genotype, candidate.schedule)
        if (
            fresh != candidate.evaluation
            or not fresh.feasible
            or not candidate.evaluation.feasible
            or any(
                not close(a, b, TOL.cost if k else TOL.time)
                for k, (a, b) in enumerate(zip(fresh.objectives, candidate.evaluation.objectives))
            )
        ):
            raise ValueError("Final independent exact mismatch/infeasibility")
        checked += 1
    audit_wall = time.perf_counter() - audit_tick
    q = result.controller
    assert len(q.q) == 42 and all(len(row) == 8 for row in q.q)
    atomic_json(
        output / "q.json",
        {"q": q.q, "updates": q.updates, "selections": q.selections, "visits": q.visits},
    )
    atomic_json(
        output / "diagnostics.json",
        {
            "counts": dict(observer.counts),
            "paths": observer.paths,
            "anytime": observer.anytime,
            "signals": observer.signals,
            "signal_seen": dict(observer.signal_seen),
        },
    )
    export_tick = time.perf_counter()
    stream_meta = stream_json(
        output / "solutions.json.gz",
        (
            {"set": label, **solution(c)}
            for label, candidates in (
                ("population", result.population),
                ("archive", result.archive.members),
            )
            for c in candidates
        ),
    )
    summary = {
        "spec": spec,
        "source": provenance,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "initial_hash": observer.initial_hash,
        "initialization_seconds": observer.initial_seconds,
        "population": len(result.population),
        "archive_size": len(result.archive.members),
        "archive_points": [c.evaluation.objectives for c in result.archive.members],
        "independent_exact": checked,
        "offspring": result.offspring_count,
        "production_seconds": result.elapsed,
        "engine_call_seconds": search_wall,
        "overshoot": max(0, result.elapsed - config.seconds),
        "termination": result.termination_reason,
        "audit_seconds": audit_wall,
        "export_seconds": time.perf_counter() - export_tick,
        "peak_rss_gib": process_peak_rss_gb(),
        "counts": dict(result.gateway.counts),
        "timings": dict(result.gateway.seconds),
        "observer_seconds": result.observer_seconds,
        "streams": stream_meta,
        "diagnostic_counts": dict(observer.counts),
    }
    atomic_json(output / "summary.json", summary)
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in output.iterdir()}
    marker = {
        "spec_hash": digest(spec),
        "source": provenance,
        "files": files,
        "completed_at": time.time(),
    }
    atomic_json(output / "complete.json", marker)
    validate_complete(output, spec, provenance)
    # Small shared index supports stage selection even if the collector is disconnected.
    atomic_json(
        control / "indexes" / f"{spec['id']}.json", {"summary": summary, "complete": marker}
    )
    return marker


def main() -> None:
    """Run the immutable spec selected by a supervisor, with a Linux AS hard cap."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-mib", type=int, default=256)
    args = parser.parse_args()
    if os.name == "posix":
        import resource

        limit = args.memory_mib * 1024**2
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        os.nice(10)
    spec = json.loads(args.spec.read_text())
    provenance = json.loads((args.control / "source.json").read_text())
    execute_run(spec, args.control, args.output, provenance)


if __name__ == "__main__":
    main()
