"""Isolated bill-mixture diagnostic, reusing immutable Step-2 original parents."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

from run_d8_campaign import lease, resources

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json, process_peak_rss_gb
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.d5_worker import canonical_source_hash
from geo_llm_scheduler.experiments.initial_bill_mixture import ARMS, assemble, seed_bank
from geo_llm_scheduler.experiments.p1_observation import restore_candidate
from geo_llm_scheduler.experiments.runner import digest, git_metadata, solution
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.scheduling.ssgs import decode

OLD_SOURCE = "5bb666874a86ae59e38aa2513a1c4f7ed1fa8fcb"
OLD_CANONICAL = "34dd838db2cf62124b967b45ba5049419f6ec96ddab82e6375104d106a1a7f02"
HOSTS = ("b518ece5cd01", "878be0fed7b0")


def prepare(root: Path, baseline: Path, seeds: int, trials: int, pilot: bool) -> dict:
    """Freeze byte-identical paired inputs after checking Step-2 RAW evidence.

    Technical pilot uses only the first H/T base at each scale, with its distinct
    old pilot seed. Formal seed count is decided solely from technical cost.
    """
    if root.exists() or seeds not in (1, 2) or trials not in (1, 4):
        raise ValueError("New campaign required; seeds 1/2 and trials 1/4 only")
    source = git_metadata()
    old = json.loads((baseline / "manifest.json").read_text())
    if source["git_dirty"] or not source["git_commit"]:
        raise ValueError("Clean source required")
    if old["source_commit"] != OLD_SOURCE or old["source_hash"] != OLD_CANONICAL:
        raise ValueError("Not the frozen Step-2 input")
    if (
        old["pilot"] != pilot
        or file_hash(baseline / "manifest.json")
        != json.loads((baseline / "manifest.sha256.json").read_text())["sha256"]
    ):
        raise ValueError("Baseline manifest mismatch")
    for name, expected in old["input_files"].items():
        if file_hash(baseline / name) != expected:
            raise ValueError("Baseline frozen input mismatch")
    node = old["node"]
    chosen: dict[tuple[int, str], int] = {}
    blocks = []
    root.mkdir(parents=True)
    for key in old["blocks"]:
        spec = json.loads((baseline / "specs" / (key + ".json")).read_text())
        if not pilot and spec["seed"] not in range(11001, 11001 + seeds):
            continue
        if pilot:
            pair = (spec["jobs"], spec["tariff"])
            chosen.setdefault(pair, spec["base_seed"])
            if chosen[pair] != spec["base_seed"]:
                continue
        folder = baseline / "runs" / key / "RAW"
        marker = json.loads((folder / "complete.json").read_text())
        if marker["host"] != HOSTS[node] or marker["spec_sha256"] != file_hash(
            baseline / "specs" / (key + ".json")
        ):
            raise ValueError("Old block identity mismatch")
        for name, sha in marker["files"].items():
            if file_hash(folder / name) != sha:
                raise ValueError("Old RAW result corrupted")
        if file_hash(baseline / spec["parent_population"]) != spec["parent_population_sha256"]:
            raise ValueError("Old parent changed")
        if (folder / "population.json").read_bytes() != (
            baseline / spec["parent_population"]
        ).read_bytes():
            raise ValueError("RAW is not the original parent")
        for path in (spec["instance"], spec["parent_population"]):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((baseline / path).read_bytes())
        old_evidence = root / "baseline_evidence" / key
        old_evidence.mkdir(parents=True)
        for name in ("complete.json", "summary.json"):
            (old_evidence / name).write_bytes((folder / name).read_bytes())
        new = {
            **spec,
            "methods": ARMS,
            "source_commit": source["git_commit"],
            "source_hash": canonical_source_hash(),
            "runner_sha256": file_hash(Path(__file__)),
            "expected_host": HOSTS[node],
            "baseline_step2_commit": OLD_SOURCE,
            "baseline_step2_hash": OLD_CANONICAL,
            "baseline_raw_marker_sha256": file_hash(folder / "complete.json"),
            "rng_version": "bill_seed_bank_20261012_v1",
            "trials_per_slot": trials,
        }
        atomic_json(root / "specs" / (key + ".json"), new)
        blocks.append(key)
    blocks.sort(key=lambda k: (-int(k.split("_", 1)[0]), int(k.rsplit("_", 1)[1]), k))
    manifest = {
        "node": node,
        "pilot": pilot,
        "source_commit": source["git_commit"],
        "source_hash": canonical_source_hash(),
        "runner_sha256": file_hash(Path(__file__)),
        "expected_host": HOSTS[node],
        "blocks": blocks,
        "block_count": len(blocks),
        "population_count": len(blocks) * len(ARMS),
        "baseline_manifest_sha256": file_hash(baseline / "manifest.json"),
        "input_files": {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob("*.json")},
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", {"sha256": file_hash(root / "manifest.json")})
    return manifest


def complete(root: Path, key: str, spec: dict) -> bool:
    """Validate the whole ten-arm block, provenance, exact evidence and every file."""
    folder = root / "runs" / key
    if not (folder / "complete.json").exists():
        return False
    marker = json.loads((folder / "complete.json").read_text())
    if marker["host"] != spec["expected_host"] or marker["spec_sha256"] != file_hash(
        root / "specs" / (key + ".json")
    ):
        raise ValueError("Block host or spec mismatch")
    expected = {"bank.json", "observations.json", "summary.json", "config.json"}
    expected |= {f"{arm}/{name}.json" for arm in ARMS for name in ("population", "front")}
    if set(marker["files"]) != expected:
        raise ValueError("Missing bill-mixture artifacts")
    for name, sha in marker["files"].items():
        if file_hash(folder / name) != sha:
            raise ValueError("Result hash mismatch")
    for path, expected_sha in (
        (spec["instance"], spec["instance_sha256"]),
        (spec["parent_population"], spec["parent_population_sha256"]),
    ):
        if file_hash(root / path) != expected_sha:
            raise ValueError("Frozen paired input mismatch")
    summary = json.loads((folder / "summary.json").read_text())
    for name in ("source_commit", "source_hash", "runner_sha256", "parent_population_sha256"):
        if summary[name] != spec[name]:
            raise ValueError("Result source provenance mismatch")
    if summary["selected_exact_verified"] != 1000 or summary["populations"] != 10:
        raise ValueError("Final exact evidence incomplete")
    return True


def worker(root: Path, key: str, finish_deadline: float = float("inf")) -> None:
    """Build a shared seed bank; replay old parents and all P100 mixtures exactly."""
    spec = json.loads((root / "specs" / (key + ".json")).read_text())
    if platform.node() != spec["expected_host"]:
        raise ValueError("Run all arms of the block on the expected host")
    if (
        canonical_source_hash() != spec["source_hash"]
        or git_metadata()["git_commit"] != spec["source_commit"]
        or git_metadata()["git_dirty"]
        or file_hash(Path(__file__)) != spec["runner_sha256"]
    ):
        raise ValueError("Worker source changed")
    if complete(root, key, spec):
        return
    folder = root / "runs" / key
    if folder.exists():
        raise FileExistsError("Preserve unfinished evidence; no automatic retry")
    folder.mkdir(parents=True)
    for path, expected in (
        (spec["instance"], spec["instance_sha256"]),
        (spec["parent_population"], spec["parent_population_sha256"]),
    ):
        if file_hash(root / path) != expected:
            raise ValueError("Paired input hash mismatch")
    problem = load_instance(root / spec["instance"])
    parents = [
        restore_candidate(r) for r in json.loads((root / spec["parent_population"]).read_text())
    ]
    started = time.perf_counter()
    for p in parents:
        if (
            decode(problem, p.genotype) != p.schedule
            or evaluate(problem, p.genotype, p.schedule) != p.evaluation
        ):
            raise AssertionError("Original parent replay mismatch")
    replay_seconds = time.perf_counter() - started

    def progress(row: dict) -> None:
        if time.time() >= finish_deadline:
            raise TimeoutError("Morning deadline; preserve partial evidence")
        atomic_json(
            root / "ops" / "heartbeats" / (key + ".json"),
            {
                **row,
                "pid": os.getpid(),
                "host": platform.node(),
                "timestamp": time.time(),
            },
        )

    seed = int(
        digest((spec["seed"], spec["base_seed"], spec["tariff"], spec["rng_version"]))[:32], 16
    )
    bank, stats = seed_bank(
        problem, parents, random.Random(seed), spec["trials_per_slot"], progress
    )
    arm_seconds = {}
    for arm in ARMS:
        tick = time.perf_counter()
        members = assemble(parents, bank, arm)
        archive = Archive()
        for c in members:
            if (
                decode(problem, c.genotype) != c.schedule
                or evaluate(problem, c.genotype, c.schedule) != c.evaluation
            ):
                raise AssertionError("Final mixture exact/SSGS mismatch")
            archive.consider(c)
        atomic_json(folder / arm / "population.json", [solution(c) for c in members])
        atomic_json(folder / arm / "front.json", [solution(c) for c in archive.members])
        arm_seconds[arm] = time.perf_counter() - tick
    atomic_json(
        folder / "bank.json", {f: [solution(c) for c in members] for f, members in bank.items()}
    )
    atomic_json(folder / "observations.json", stats.pop("observations"))
    atomic_json(
        folder / "config.json",
        {
            "methods": ARMS,
            "trials_per_slot": spec["trials_per_slot"],
            "population": 100,
            "original_quotas": {"10": [9] * 10, "20": [8] * 10},
            "flow_weights": [0, 0.1, 0.25, 0.5],
            "restricted_list_fraction": 0.05,
            "bank_shared": True,
            "no_preprocessing_or_evolution": True,
        },
    )
    atomic_json(
        folder / "summary.json",
        {
            **spec,
            **stats,
            "host": platform.node(),
            "pid": os.getpid(),
            "timestamp": time.time(),
            "total_elapsed": time.perf_counter() - started,
            "rss_peak_gib": process_peak_rss_gb(),
            "baseline_replay_seconds": replay_seconds,
            "baseline_replay_verified": 100,
            "selected_exact_verified": 1000,
            "populations": 10,
            "arm_verification_export_seconds": arm_seconds,
        },
    )
    atomic_json(
        folder / "complete.json",
        {
            "host": platform.node(),
            "timestamp": time.time(),
            "spec_sha256": file_hash(root / "specs" / (key + ".json")),
            "files": {
                p.relative_to(folder).as_posix(): file_hash(p) for p in folder.rglob("*.json")
            },
        },
    )
    if not complete(root, key, spec):
        raise AssertionError("Final block validation failed")


def dispatch_allowed(budget: dict, now: float, deadline: float, bound: float) -> bool:
    """Reserve 2 GiB plus a capped worker and admit only full blocks before 08:00."""
    return (
        budget["available_memory_gib"] >= 2.25
        and budget["free_disk_gib"] >= 10
        and now + bound + 90 < deadline
    )


def child_limits() -> None:
    """Lower only this experiment's priority and cap its worker virtual memory."""
    import resource

    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))


def run(root: Path, workers: int, deadline: float, bound: float) -> None:
    """Resume completed blocks only, preserving failures, other processes and files."""
    with lease(root / "ops" / "supervisor.lock"):
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            file_hash(root / "manifest.json")
            != json.loads((root / "manifest.sha256.json").read_text())["sha256"]
        ):
            raise ValueError("Manifest changed")
        if (
            platform.node() != manifest["expected_host"]
            or canonical_source_hash() != manifest["source_hash"]
            or git_metadata()["git_commit"] != manifest["source_commit"]
            or git_metadata()["git_dirty"]
            or file_hash(Path(__file__)) != manifest["runner_sha256"]
        ):
            raise ValueError("Supervisor source/host changed")
        for path, sha in manifest["input_files"].items():
            if file_hash(root / path) != sha:
                raise ValueError("Input changed")
        if list((root / "ops" / "failures").glob("*.json")):
            raise RuntimeError("Failure history; do not auto-retry")
        pending, done = [], []
        for key in manifest["blocks"]:
            spec = json.loads((root / "specs" / (key + ".json")).read_text())
            (done if complete(root, key, spec) else pending).append(key)
        active: dict[str, subprocess.Popen] = {}
        started = time.time()
        failure = False
        while pending or active:
            for key, proc in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                del active[key]
                if code:
                    atomic_json(
                        root / "ops" / "failures" / (key + ".json"),
                        {"returncode": code, "timestamp": time.time()},
                    )
                    (root.parent / "stop.request").write_text(
                        "Own worker failed; drain own two shards"
                    )
                    failure = True
                else:
                    spec = json.loads((root / "specs" / (key + ".json")).read_text())
                    if not complete(root, key, spec):
                        raise AssertionError("Exited without a complete block")
                    done.append(key)
            stopped = (
                failure
                or (root / "stop.request").exists()
                or (root.parent / "stop.request").exists()
                or time.time() + bound + 90 >= deadline
            )
            budget = resources(root)
            while (
                pending
                and len(active) < workers
                and not stopped
                and dispatch_allowed(budget, time.time(), deadline, bound)
            ):
                key = pending.pop(0)
                log = root / "ops" / "logs" / (key + ".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("ab") as stream:
                    active[key] = subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "worker",
                            "--root",
                            str(root),
                            "--key",
                            key,
                            "--deadline",
                            str(deadline),
                        ],
                        stdout=stream,
                        stderr=stream,
                        stdin=subprocess.DEVNULL,
                        preexec_fn=child_limits,
                    )
                budget = resources(root)
            result = {
                "state": "draining" if stopped else "running",
                "pid": os.getpid(),
                "host": platform.node(),
                "timestamp": time.time(),
                "started": started,
                "active": {k: p.pid for k, p in active.items()},
                "completed_blocks": len(done),
                "pending_blocks": len(pending),
                "failed": failure,
                "deadline": deadline,
                "admission_bound_seconds": bound,
                "resources": budget,
            }
            atomic_json(root / "ops" / "status.json", result)
            if stopped and not active:
                break
            time.sleep(2)
        result.update(
            state="failed" if failure else "complete" if not pending else "deadline_or_stop",
            active={},
            timestamp=time.time(),
        )
        atomic_json(root / "ops" / "status.json", result)
        atomic_json(root / "ops" / "exit.json", result)


def main() -> None:
    """Run explicit new matrices with capped concurrency and a morning deadline."""
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "worker", "run", "validate"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--key")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--deadline", type=float, default=0)
    p.add_argument("--bound", type=float, default=3600)
    args = p.parse_args()
    if args.mode == "prepare":
        if args.baseline is None:
            p.error("Baseline required")
        print(json.dumps(prepare(args.root, args.baseline, args.seeds, args.trials, args.pilot)))
    elif args.mode == "worker":
        if args.key is None:
            p.error("Key required")
        worker(args.root, args.key, args.deadline or float("inf"))
    elif args.mode == "run":
        if not 1 <= args.workers <= 8 or args.deadline <= time.time() or args.bound <= 0:
            p.error("Require <=8 workers, future deadline and positive block bound")
        run(args.root, args.workers, args.deadline, args.bound)
    else:
        m = json.loads((args.root / "manifest.json").read_text())
        result = [
            complete(args.root, k, json.loads((args.root / "specs" / (k + ".json")).read_text()))
            for k in m["blocks"]
        ]
        if not all(result):
            raise ValueError("Not all blocks complete")
        print(json.dumps({"validated_blocks": len(result), "populations": len(result) * 10}))


if __name__ == "__main__":
    main()
