"""Freeze and execute paired initialization-only blocks without touching D8."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

from run_d8_campaign import lease, resources

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.campaign.support import atomic_json, process_peak_rss_gb
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.d5_worker import canonical_source_hash
from geo_llm_scheduler.experiments.initial_frontier import (
    METHODS,
    SCENARIOS,
    diagnostic_problem,
    evaluate_population,
    sample_population,
)
from geo_llm_scheduler.experiments.runner import git_metadata, solution
from geo_llm_scheduler.io.loaders import load_instance, save_instance


def prepare(root: Path, node: int, pilot: bool = False, seed_count: int = 10) -> dict:
    """Freeze disjoint cases, paired method blocks and ten explicit initialization seeds."""
    if root.exists():
        raise FileExistsError("Do not overwrite an existing initialization campaign")
    if not 1 <= seed_count <= 10:
        raise ValueError("Initialization repeats must be between one and ten")
    provenance = git_metadata()
    if provenance["git_dirty"] or provenance["git_commit"] is None:
        raise ValueError("Preparation requires a clean versioned diagnostic checkout")
    root.mkdir(parents=True)
    (root / "instances").mkdir()
    (root / "specs").mkdir()
    blocks = []
    case_index = 0
    sizes = (50, 100, 200)
    repeats = 1 if pilot else 2
    seeds = (11901,) if pilot else tuple(range(11001, 11001 + seed_count))
    for jobs in sizes:
        for scenario_index, scenario in enumerate(SCENARIOS):
            for repeat in range(repeats):
                base_seed = (
                    (1490000 if pilot else 1400000) + jobs * 100 + scenario_index * 10 + repeat + 1
                )
                assigned = case_index % 2
                case_index += 1
                if assigned != node:
                    continue
                for tariff in ("H", "T"):
                    case = f"{jobs}_{scenario}_{base_seed}_{tariff}"
                    path = root / "instances" / (case + ".json")
                    save_instance(diagnostic_problem(jobs, base_seed, tariff, scenario), path)
                    for seed_index, seed in enumerate(seeds):
                        key = f"{case}_{seed}"
                        # Balanced cyclic order avoids binding methods to an execution position.
                        shift = (seed_index + case_index) % len(METHODS)
                        order = METHODS[shift:] + METHODS[:shift]
                        spec = {
                            "key": key,
                            "jobs": jobs,
                            "scenario": scenario,
                            "base_seed": base_seed,
                            "tariff": tariff,
                            "seed": seed,
                            "instance": path.relative_to(root).as_posix(),
                            "instance_sha256": file_hash(path),
                            "methods": order,
                            "population": 100,
                            "perturbation": 0.1,
                            "attempt_limit": 10000,
                            "source_commit": provenance["git_commit"],
                            "source_hash": canonical_source_hash(),
                        }
                        atomic_json(root / "specs" / (key + ".json"), spec)
                        blocks.append(key)
    # Complete scene coverage in the first seed round before adding repeats.
    blocks.sort(key=lambda key: (int(key.rsplit("_", 1)[1]), key))
    manifest = {
        "source_commit": provenance["git_commit"],
        "source_hash": canonical_source_hash(),
        "node": node,
        "nodes": 2,
        "pilot": pilot,
        "blocks": blocks,
        "block_count": len(blocks),
        "population_count": len(blocks) * len(METHODS),
        "input_files": {
            p.relative_to(root).as_posix(): file_hash(p)
            for folder in ("instances", "specs")
            for p in (root / folder).glob("*.json")
        },
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", {"sha256": file_hash(root / "manifest.json")})
    return manifest


def complete(root: Path, key: str, method: str, spec: dict) -> bool:
    """Check input identity and every saved solution/summary byte before reuse."""
    folder = root / "runs" / key / method
    marker = folder / "complete.json"
    if not marker.exists():
        return False
    data = json.loads(marker.read_text())
    if data["host"] != platform.node() or data["spec_sha256"] != file_hash(
        root / "specs" / (key + ".json")
    ):
        raise ValueError("Do not reuse a partial block across hosts or specifications")
    for name, digest in data["files"].items():
        if file_hash(folder / name) != digest:
            raise ValueError("Initialization artifact hash mismatch")
    summary = json.loads((folder / "summary.json").read_text())
    if (
        summary["source_commit"] != spec["source_commit"]
        or summary["source_hash"] != spec["source_hash"]
    ):
        raise ValueError("Initialization source mismatch")
    return True


def worker(root: Path, key: str) -> None:
    """Finish all methods for one case/seed on one host, publishing checked artifacts."""
    spec = json.loads((root / "specs" / (key + ".json")).read_text())
    if (
        canonical_source_hash() != spec["source_hash"]
        or git_metadata()["git_commit"] != spec["source_commit"]
        or git_metadata()["git_dirty"]
    ):
        raise ValueError("Worker requires the exact frozen diagnostic source")
    if file_hash(root / spec["instance"]) != spec["instance_sha256"]:
        raise ValueError("Initialization input hash mismatch")
    problem = load_instance(root / spec["instance"])
    cfg = replace(
        Config(),
        population=spec["population"],
        neighborhood=min(20, spec["population"]),
        seed=spec["seed"],
        initialization_attempts=spec["attempt_limit"],
        initialization_perturbation=spec["perturbation"],
    )
    for method in spec["methods"]:
        if complete(root, key, method, spec):
            continue
        folder = root / "runs" / key / method
        if folder.exists():
            raise FileExistsError("Keep unfinished evidence; inspect before explicit task recovery")
        folder.mkdir(parents=True)
        start = time.perf_counter()
        trace = sample_population(problem, cfg, method)
        population, archive, stats = evaluate_population(problem, trace)
        summary = {
            **spec,
            "method": method,
            "host": platform.node(),
            "pid": os.getpid(),
            "timestamp": time.time(),
            "elapsed": time.perf_counter() - start,
            "rss_peak_gib": process_peak_rss_gb(),
            "actual_population": len(population),
            "population_complete": len(population) == cfg.population,
            "attempts": trace.attempts,
            "duplicates": trace.duplicates,
            "generation_seconds": trace.generation_seconds,
            "original_replay_seconds": trace.replay_seconds,
            "original_replay_verified": trace.original_replay_verified,
            **stats,
        }
        atomic_json(folder / "population.json", [solution(c) for c in population])
        atomic_json(folder / "archive.json", [solution(c) for c in archive.members])
        atomic_json(folder / "summary.json", summary)
        atomic_json(folder / "config.json", asdict(cfg))
        atomic_json(
            folder / "complete.json",
            {
                "host": platform.node(),
                "timestamp": time.time(),
                "spec_sha256": file_hash(root / "specs" / (key + ".json")),
                "files": {
                    name: file_hash(folder / name)
                    for name in ("population.json", "archive.json", "summary.json", "config.json")
                },
            },
        )
        atomic_json(
            root / "ops" / "heartbeats" / (key + ".json"),
            {
                "pid": os.getpid(),
                "host": platform.node(),
                "timestamp": time.time(),
                "method": method,
                "exact_verified": len(population),
                "elapsed_last_method": summary["elapsed"],
            },
        )


def run(root: Path, workers: int, deadline: float) -> None:
    """Dispatch full paired blocks under an exclusive lock and safe deadline drain."""
    with lease(root / "ops" / "supervisor.lock"):
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            file_hash(root / "manifest.json")
            != json.loads((root / "manifest.sha256.json").read_text())["sha256"]
        ):
            raise ValueError("Frozen initialization matrix hash mismatch")
        if (
            canonical_source_hash() != manifest["source_hash"]
            or git_metadata()["git_commit"] != manifest["source_commit"]
        ):
            raise ValueError("Frozen initialization source mismatch")
        for name, digest in manifest["input_files"].items():
            if file_hash(root / name) != digest:
                raise ValueError("Frozen initialization input mismatch")
        if list((root / "ops" / "failures").glob("*.json")):
            raise RuntimeError("Failure evidence requires user review; no automatic retry")
        pending = []
        done = []
        for key in manifest["blocks"]:
            spec = json.loads((root / "specs" / (key + ".json")).read_text())
            if all(complete(root, key, m, spec) for m in spec["methods"]):
                done.append(key)
            else:
                pending.append(key)
        active: dict[str, subprocess.Popen] = {}
        failure = False
        started = time.time()
        while pending or active:
            stopped = (
                (root / "stop.request").exists()
                or (root.parent / "stop.request").exists()
                or time.time() >= deadline
                or failure
            )
            for key, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[key]
                if code:
                    atomic_json(
                        root / "ops" / "failures" / (key + ".json"),
                        {"returncode": code, "timestamp": time.time()},
                    )
                    failure = True
                    stopped = True
                    # Isolated diagnostic parent shared by both shards; never touches D8.
                    (root.parent / "stop.request").write_text("Worker failure; preserve evidence")
                else:
                    done.append(key)
            budget = resources(root)
            while (
                pending
                and len(active) < workers
                and not stopped
                and budget["available_memory_gib"] >= 6
                and budget["free_disk_gib"] >= 20
            ):
                key = pending.pop(0)
                log = root / "ops" / "logs" / (key + ".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("ab") as stream:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "worker",
                            "--root",
                            str(root),
                            "--key",
                            key,
                        ],
                        stdout=stream,
                        stderr=stream,
                        stdin=subprocess.DEVNULL,
                    )
                active[key] = process
                budget = resources(root)
            state = "running" if not stopped else "draining"
            atomic_json(
                root / "ops" / "status.json",
                {
                    "state": state,
                    "pid": os.getpid(),
                    "host": platform.node(),
                    "timestamp": time.time(),
                    "started": started,
                    "active": {k: p.pid for k, p in active.items()},
                    "completed_blocks": len(done),
                    "pending_blocks": len(pending),
                    "failed": failure,
                    "deadline": deadline,
                    "resources": budget,
                },
            )
            if stopped and not active:
                break
            time.sleep(1)
        state = "failed" if failure else "complete" if not pending else "deadline_or_stop"
        result = {
            "state": state,
            "pid": os.getpid(),
            "host": platform.node(),
            "timestamp": time.time(),
            "started": started,
            "active": {},
            "completed_blocks": len(done),
            "pending_blocks": len(pending),
            "failed": failure,
            "deadline": deadline,
        }
        atomic_json(root / "ops" / "status.json", result)
        atomic_json(root / "ops" / "exit.json", result)


def main() -> None:
    """Provide explicit prepare/worker/run commands with no implicit deployment."""
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "worker", "run"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--node", type=int, choices=(0, 1), default=0)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--seed-count", type=int, default=10)
    p.add_argument("--key")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--deadline", type=float, default=0)
    args = p.parse_args()
    try:
        if args.mode == "prepare":
            print(json.dumps(prepare(args.root, args.node, args.pilot, args.seed_count)))
        elif args.mode == "worker":
            if args.key is None:
                p.error("worker requires --key")
            worker(args.root, args.key)
        else:
            if args.deadline <= time.time() or args.workers < 1:
                p.error("run requires a future deadline and positive workers")
            run(args.root, args.workers, args.deadline)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
