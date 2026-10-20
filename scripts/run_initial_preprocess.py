"""Freeze paired Step-1 parents and run bounded Step-2 preprocessing blocks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

from run_d8_campaign import lease, resources
from run_initial_frontier import complete as original_complete

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json, process_peak_rss_gb
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.d5_worker import canonical_source_hash
from geo_llm_scheduler.experiments.initial_preprocess import ARMS, preprocess
from geo_llm_scheduler.experiments.p1_observation import restore_candidate
from geo_llm_scheduler.experiments.runner import digest, git_metadata, solution
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.scheduling.ssgs import decode

FILES = ("population.json", "archive.json", "summary.json", "config.json", "observations.json")


def prepare(root: Path, baseline: Path, pilot: bool, seed_count: int = 2) -> dict:
    """Copy immutable paired parents after validating their original completion markers.

    Existing campaigns are never overwritten. Formal uses the first two predeclared
    initialization seeds across all original 36 bases; pilot uses the separate 18
    technical bases. Baseline data are copied byte-for-byte and bound to the new
    manifest, while old source and completion provenance remain explicit.
    """
    if root.exists():
        raise FileExistsError("Do not overwrite a preprocessing campaign")
    provenance = git_metadata()
    if provenance["git_dirty"] or provenance["git_commit"] is None:
        raise ValueError("A clean versioned checkout is required")
    if not 1 <= seed_count <= 10:
        raise ValueError("Seed count must be within original ten frozen seeds")
    old = json.loads((baseline / "manifest.json").read_text())
    if (
        old["pilot"] != pilot
        or file_hash(baseline / "manifest.json")
        != json.loads((baseline / "manifest.sha256.json").read_text())["sha256"]
    ):
        raise ValueError("Baseline matrix identity mismatch")
    if old["source_commit"] != "e57d385b2a4dcc7546e2f54e920cccbe2b6eb7a1":
        raise ValueError("Step-1 source was not the frozen original initializer")
    root.mkdir(parents=True)
    for name in ("instances", "parents", "specs"):
        (root / name).mkdir()
    blocks = []
    for key in old["blocks"]:
        if not pilot and int(key.rsplit("_", 1)[1]) not in range(11001, 11001 + seed_count):
            continue
        original = json.loads((baseline / "specs" / (key + ".json")).read_text())
        if not original_complete(baseline, key, "MIXED", original):
            raise ValueError("Baseline MIXED population has not passed original validator")
        folder = baseline / "runs" / key / "MIXED"
        summary = json.loads((folder / "summary.json").read_text())
        if not summary["original_replay_verified"] or summary["actual_population"] != 100:
            raise ValueError("Require 100 replay-matched original MIXED parents")
        for old_path, new_path in (
            (baseline / original["instance"], root / original["instance"]),
            (folder / "population.json", root / "parents" / (key + ".json")),
            (folder / "complete.json", root / "parents" / (key + "_complete.json")),
        ):
            new_path.write_bytes(old_path.read_bytes())
        shift = (original["seed"] + original["base_seed"]) % len(ARMS)
        spec = {
            **{
                k: original[k]
                for k in (
                    "key",
                    "jobs",
                    "scenario",
                    "base_seed",
                    "tariff",
                    "seed",
                    "instance",
                    "instance_sha256",
                    "population",
                )
            },
            "parent_population": "parents/" + key + ".json",
            "parent_population_sha256": file_hash(root / "parents" / (key + ".json")),
            "baseline_source_commit": original["source_commit"],
            "baseline_source_hash": original["source_hash"],
            "baseline_marker_sha256": file_hash(folder / "complete.json"),
            "methods": ARMS[shift:] + ARMS[:shift],
            "source_commit": provenance["git_commit"],
            "source_hash": canonical_source_hash(),
            "runner_sha256": file_hash(Path(__file__)),
            "rng_version": "step2_per_parent_20261011",
        }
        atomic_json(root / "specs" / (key + ".json"), spec)
        blocks.append(key)
    blocks.sort(key=lambda key: (int(key.rsplit("_", 1)[1]), key))
    manifest = {
        "source_commit": provenance["git_commit"],
        "source_hash": canonical_source_hash(),
        "runner_sha256": file_hash(Path(__file__)),
        "baseline_manifest_sha256": file_hash(baseline / "manifest.json"),
        "node": old["node"],
        "pilot": pilot,
        "blocks": blocks,
        "block_count": len(blocks),
        "population_count": len(blocks) * len(ARMS),
        "input_files": {
            p.relative_to(root).as_posix(): file_hash(p)
            for folder_name in ("instances", "parents", "specs")
            for p in (root / folder_name).glob("*.json")
        },
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", {"sha256": file_hash(root / "manifest.json")})
    return manifest


def complete(root: Path, key: str, method: str, spec: dict) -> bool:
    """Validate host, input, parent, runner, source and every published result byte."""
    folder = root / "runs" / key / method
    if not (folder / "complete.json").exists():
        return False
    marker = json.loads((folder / "complete.json").read_text())
    if marker["host"] != platform.node() or marker["spec_sha256"] != file_hash(
        root / "specs" / (key + ".json")
    ):
        raise ValueError("Do not reuse blocks across hosts or changed specifications")
    if set(marker["files"]) != set(FILES):
        raise ValueError("Incomplete preprocessing artifact set")
    for name, sha in marker["files"].items():
        if file_hash(folder / name) != sha:
            raise ValueError("Published result hash mismatch")
    if (
        file_hash(root / spec["instance"]) != spec["instance_sha256"]
        or file_hash(root / spec["parent_population"]) != spec["parent_population_sha256"]
    ):
        raise ValueError("Original frozen parent/input changed")
    summary = json.loads((folder / "summary.json").read_text())
    for name in ("source_commit", "source_hash", "runner_sha256", "parent_population_sha256"):
        if summary[name] != spec[name]:
            raise ValueError("Preprocessing provenance mismatch")
    if summary["selected_exact_verified"] != spec["population"]:
        raise ValueError("Final exact evidence is incomplete")
    return True


def worker(root: Path, key: str) -> None:
    """Process all five matched arms on one host with per-individual live heartbeats."""
    spec = json.loads((root / "specs" / (key + ".json")).read_text())
    if (
        canonical_source_hash() != spec["source_hash"]
        or git_metadata()["git_commit"] != spec["source_commit"]
        or git_metadata()["git_dirty"]
        or file_hash(Path(__file__)) != spec["runner_sha256"]
    ):
        raise ValueError("Worker is not running the frozen source")
    if (
        file_hash(root / spec["parent_population"]) != spec["parent_population_sha256"]
        or file_hash(root / spec["instance"]) != spec["instance_sha256"]
    ):
        raise ValueError("Parent or input hash mismatch")
    problem = load_instance(root / spec["instance"])
    parents = [
        restore_candidate(row) for row in json.loads((root / spec["parent_population"]).read_text())
    ]
    if len(parents) != spec["population"] or len({p.genotype for p in parents}) != len(parents):
        raise ValueError("Require the complete unique original population")
    tick = time.perf_counter()
    for parent in parents:
        if (
            decode(problem, parent.genotype) != parent.schedule
            or evaluate(problem, parent.genotype, parent.schedule) != parent.evaluation
        ):
            raise AssertionError("Original decode/exact replay mismatch")
    baseline_replay_seconds = time.perf_counter() - tick
    for arm in spec["methods"]:
        if complete(root, key, arm, spec):
            continue
        folder = root / "runs" / key / arm
        if folder.exists():
            raise FileExistsError("Preserve unfinished evidence; no automatic retry")
        folder.mkdir(parents=True)
        population = []
        observations = []
        archive = Archive()
        start = time.perf_counter()
        for index, parent in enumerate(parents):
            # Both structural budgets use identical parent streams and maximum pool.
            seed = int(
                digest(
                    (spec["seed"], spec["base_seed"], spec["tariff"], index, spec["rng_version"])
                )[:32],
                16,
            )
            selected, explored, stats = preprocess(problem, parent, arm, random.Random(seed))
            population.append(selected)
            for candidate in explored.members:
                archive.consider(candidate)
            observations.append(
                {
                    "index": index,
                    "parent_flow": parent.evaluation.flow,
                    "parent_bill": parent.evaluation.bill,
                    **stats,
                }
            )
            atomic_json(
                root / "ops" / "heartbeats" / (key + ".json"),
                {
                    "pid": os.getpid(),
                    "host": platform.node(),
                    "timestamp": time.time(),
                    "method": arm,
                    "processed": index + 1,
                    "exact_verified": sum(s["independent_exact_verified"] for s in observations),
                    "elapsed": time.perf_counter() - start,
                },
            )
        selected_archive = Archive()
        for candidate in population:
            selected_archive.consider(candidate)
        # Every retained explored front point is independently checked as well.
        for candidate in archive.members:
            if evaluate(problem, candidate.genotype, candidate.schedule) != candidate.evaluation:
                raise AssertionError("Explored archive failed independent exact")
        summary = {
            **spec,
            "method": arm,
            "host": platform.node(),
            "pid": os.getpid(),
            "timestamp": time.time(),
            "elapsed": time.perf_counter() - start,
            "rss_peak_gib": process_peak_rss_gb(),
            "actual_population": len(population),
            "selected_exact_verified": len(population),
            "baseline_replay_seconds": baseline_replay_seconds,
            "baseline_replay_verified": len(parents),
            "exact": sum(s["exact"] for s in observations),
            "independent_exact_verified": sum(
                s["independent_exact_verified"] for s in observations
            ),
            "production_seconds": sum(s["production_seconds"] for s in observations),
            "verification_seconds": sum(s["independent_exact_seconds"] for s in observations),
            "accepted": sum(s["accepted"] for s in observations),
            "ssgs": sum(s["ssgs"] for s in observations),
            "selected_nd": len(selected_archive.members),
            "explored_nd": len(archive.members),
            "explored_archive_exact_verified": len(archive.members),
        }
        atomic_json(folder / "population.json", [solution(c) for c in population])
        atomic_json(folder / "archive.json", [solution(c) for c in archive.members])
        atomic_json(folder / "summary.json", summary)
        atomic_json(
            folder / "config.json",
            {
                "methods": ARMS,
                "per_parent_budget": 0 if arm == "RAW" else int(arm[-2:]),
                "selection": "strict Bill improvement, no Flow increase; one selected move",
            },
        )
        atomic_json(folder / "observations.json", observations)
        atomic_json(
            folder / "complete.json",
            {
                "host": platform.node(),
                "timestamp": time.time(),
                "spec_sha256": file_hash(root / "specs" / (key + ".json")),
                "files": {name: file_hash(folder / name) for name in FILES},
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
            or git_metadata()["git_dirty"]
            or file_hash(Path(__file__)) != manifest["runner_sha256"]
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
    """Require explicit paths, a future safe-dispatch deadline and positive concurrency."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "worker", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--key")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--deadline", type=float, default=0)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.baseline is None:
            parser.error("prepare requires --baseline")
        print(json.dumps(prepare(args.root, args.baseline, args.pilot, args.seed_count)))
    elif args.mode == "worker":
        if args.key is None:
            parser.error("worker requires --key")
        worker(args.root, args.key)
    else:
        if args.deadline <= time.time() or args.workers < 1:
            parser.error("run requires future deadline and positive workers")
        run(args.root, args.workers, args.deadline)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
