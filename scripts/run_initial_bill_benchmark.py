"""Independent four-arm initialization clocks; no reuse of candidate construction."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from run_d8_campaign import lease, resources
from run_initial_bill_mixture import child_limits

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.campaign.support import atomic_json, process_peak_rss_gb
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.d5_worker import canonical_source_hash
from geo_llm_scheduler.experiments.initial_bill_benchmark import (
    METHODS,
    ORDERS,
    benchmark_population,
)
from geo_llm_scheduler.experiments.runner import digest, git_metadata, solution
from geo_llm_scheduler.io.loaders import load_instance

HOSTS = ("b518ece5cd01", "878be0fed7b0")


def prepare(root: Path, baseline: Path, pilot: bool = False) -> dict:
    """Copy immutable inputs only and freeze 144 paired blocks, without old outputs."""
    if root.exists():
        raise FileExistsError("New campaign required")
    old = json.loads((baseline / "manifest.json").read_text())
    if old["source_commit"] != "5bb666874a86ae59e38aa2513a1c4f7ed1fa8fcb":
        raise ValueError("Not original Step-2 input")
    if (
        file_hash(baseline / "manifest.json")
        != json.loads((baseline / "manifest.sha256.json").read_text())["sha256"]
    ):
        raise ValueError("Original input manifest corrupted")
    source = git_metadata()
    if source["git_dirty"]:
        raise ValueError("Clean source required")
    root.mkdir(parents=True)
    node, blocks, chosen = old["node"], [], {}
    base_seeds = sorted(
        {
            json.loads((baseline / "specs" / (key + ".json")).read_text())["base_seed"]
            for key in old["blocks"]
        }
    )
    for key in old["blocks"]:
        specpath = baseline / "specs" / (key + ".json")
        if file_hash(specpath) != old["input_files"]["specs/" + key + ".json"]:
            raise ValueError("Old spec corrupted")
        spec = json.loads(specpath.read_text())
        if spec["seed"] not in (11001, 11002):
            continue
        if pilot:
            pair = (spec["jobs"], spec["tariff"])
            chosen.setdefault(pair, spec["base_seed"])
            if chosen[pair] != spec["base_seed"] or spec["seed"] != 11001:
                continue
        path = baseline / spec["instance"]
        if file_hash(path) != spec["instance_sha256"]:
            raise ValueError("Old instance corrupted")
        target = root / spec["instance"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        order_row = (
            base_seeds.index(spec["base_seed"]) + 2 * (spec["tariff"] == "T") + spec["seed"] - 11001
        ) % 4
        new = {
            k: spec[k]
            for k in (
                "key",
                "jobs",
                "scenario",
                "base_seed",
                "tariff",
                "seed",
                "instance",
                "instance_sha256",
            )
        }
        new.update(
            methods=[METHODS[i] for i in ORDERS[order_row]],
            source_commit=source["git_commit"],
            source_hash=canonical_source_hash(),
            runner_sha256=file_hash(Path(__file__)),
            expected_host=HOSTS[node],
            population=100,
            trials=4,
            rng_version="isolated_family_benchmark_v1",
        )
        atomic_json(root / "specs" / (key + ".json"), new)
        blocks.append(key)
    blocks.sort(key=lambda k: (-int(k.split("_", 1)[0]), k))
    manifest = {
        "node": node,
        "pilot": pilot,
        "source_commit": source["git_commit"],
        "source_hash": canonical_source_hash(),
        "runner_sha256": file_hash(Path(__file__)),
        "expected_host": HOSTS[node],
        "blocks": blocks,
        "block_count": len(blocks),
        "population_count": len(blocks) * 4,
        "input_files": {f.relative_to(root).as_posix(): file_hash(f) for f in root.rglob("*.json")},
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", {"sha256": file_hash(root / "manifest.json")})
    return manifest


def complete(root: Path, key: str, spec: dict) -> bool:
    """Verify all four artifacts per arm, clocks, input and exact provenance."""
    folder = root / "runs" / key
    marker = folder / "complete.json"
    if not marker.exists():
        return False
    data = json.loads(marker.read_text())
    expected = {
        f"{m}/{n}.json" for m in METHODS for n in ("population", "front", "summary", "observations")
    } | {"summary.json", "config.json"}
    if (
        data["host"] != spec["expected_host"]
        or data["spec_sha256"] != file_hash(root / "specs" / (key + ".json"))
        or set(data["files"]) != expected
    ):
        raise ValueError("Block host/spec/artifact mismatch")
    for name, sha in data["files"].items():
        if file_hash(folder / name) != sha:
            raise ValueError("Result hash mismatch")
    if file_hash(root / spec["instance"]) != spec["instance_sha256"]:
        raise ValueError("Input mismatch")
    for method in METHODS:
        s = json.loads((folder / method / "summary.json").read_text())
        for name in ("source_commit", "source_hash", "runner_sha256"):
            if s[name] != spec[name]:
                raise ValueError("Source provenance mismatch")
        expected_exact = 0 if method == "BASE" else int(method[-2:]) * 4
        if (
            s["final_independent_exact_verified"] != 100
            or s["new_candidate_exact"] != expected_exact
            or s["production_wall_seconds"] <= 0
            or s["production_cpu_seconds"] <= 0
        ):
            raise ValueError("Clock/exact evidence mismatch")
        pop = json.loads((folder / method / "population.json").read_text())
        if len(pop) != 100:
            raise ValueError("P100 mismatch")
    return True


def worker(root: Path, key: str, finish_deadline: float = float("inf")) -> None:
    """Run independently constructed methods sequentially in balanced order."""
    spec = json.loads((root / "specs" / (key + ".json")).read_text())
    if (
        platform.node() != spec["expected_host"]
        or canonical_source_hash() != spec["source_hash"]
        or git_metadata()["git_commit"] != spec["source_commit"]
        or git_metadata()["git_dirty"]
        or file_hash(Path(__file__)) != spec["runner_sha256"]
    ):
        raise ValueError("Source/host mismatch")
    if complete(root, key, spec):
        return
    folder = root / "runs" / key
    if folder.exists():
        raise FileExistsError("Keep partial evidence; no automatic retry")
    folder.mkdir(parents=True)
    if file_hash(root / spec["instance"]) != spec["instance_sha256"]:
        raise ValueError("Input mismatch")
    problem = load_instance(root / spec["instance"])
    cfg = Config(seed=spec["seed"], initialization_perturbation=0.1, initialization_attempts=10000)
    started, exact = time.perf_counter(), 0
    parent_hash = None
    for method in spec["methods"]:

        def progress(row: dict) -> None:
            if time.time() >= finish_deadline:
                raise TimeoutError("Hard deadline; preserve partial evidence")
            atomic_json(
                root / "ops/heartbeats" / (key + ".json"),
                {
                    **row,
                    "method": method,
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "host": platform.node(),
                    "block_exact": exact + row.get("exact", 0),
                },
            )

        progress({"exact": 0, "stage": "parent_generation"})
        family = "BILL" if method.startswith("BILL") else method
        bank_seed = int(
            digest((spec["base_seed"], spec["tariff"], spec["seed"], family, spec["rng_version"]))[
                :32
            ],
            16,
        )
        members, front, stats = benchmark_population(problem, cfg, method, bank_seed, progress)
        hashes = digest(stats.pop("parents"))
        if parent_hash is not None and hashes != parent_hash:
            raise AssertionError("Fresh parent generation differs across arms")
        parent_hash = hashes
        bank = stats.pop("bank_stats")
        observations = bank.pop("observations")
        stats.update(
            source_commit=spec["source_commit"],
            source_hash=spec["source_hash"],
            runner_sha256=spec["runner_sha256"],
            parent_genotypes_sha256=hashes,
            bank_seed=bank_seed,
            bank_timing=bank,
        )
        tick = time.perf_counter()
        atomic_json(folder / method / "population.json", [solution(c) for c in members])
        atomic_json(folder / method / "front.json", [solution(c) for c in front])
        atomic_json(folder / method / "observations.json", observations)
        stats["export_wall_seconds"] = time.perf_counter() - tick
        atomic_json(folder / method / "summary.json", stats)
        exact += stats["parent_exact"] + stats["new_candidate_exact"]
        progress({"exact": 0, "stage": "arm_complete"})
    atomic_json(folder / "config.json", spec)
    atomic_json(
        folder / "summary.json",
        {
            "key": key,
            "jobs": spec["jobs"],
            "total_elapsed": time.perf_counter() - started,
            "rss_peak_gib": process_peak_rss_gb(),
            "exact": exact,
            "selected_exact_verified": 400,
            "populations": 4,
            "host": platform.node(),
            "source_commit": spec["source_commit"],
        },
    )
    files = {p.relative_to(folder).as_posix(): file_hash(p) for p in folder.rglob("*.json")}
    atomic_json(
        folder / "complete.json",
        {
            "host": platform.node(),
            "timestamp": time.time(),
            "spec_sha256": file_hash(root / "specs" / (key + ".json")),
            "files": files,
        },
    )
    if not complete(root, key, spec):
        raise AssertionError("Final block validation failed")


def dispatch_allowed(budget: dict, now: float, deadline: float, bound: float) -> bool:
    """Reserve 1.5GiB including a capped 256MiB child; no external process changes."""
    return (
        budget["available_memory_gib"] >= 1.5
        and budget["free_disk_gib"] >= 10
        and now + bound + 90 < deadline
    )


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
    """Prepare, run, validate or execute one frozen four-arm block."""
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "worker", "run", "validate"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--key")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--deadline", type=float, default=0)
    p.add_argument("--bound", type=float, default=5400)
    a = p.parse_args()
    if a.mode == "prepare":
        if a.baseline is None:
            p.error("Require immutable baseline input")
        print(json.dumps(prepare(a.root, a.baseline, a.pilot)))
    elif a.mode == "worker":
        if not a.key:
            p.error("Require key")
        worker(a.root, a.key, a.deadline or float("inf"))
    elif a.mode == "run":
        if not 1 <= a.workers <= 8 or a.deadline <= time.time() or a.bound <= 0:
            p.error("Require <=8 workers, future deadline and positive bound")
        run(a.root, a.workers, a.deadline, a.bound)
    else:
        m = json.loads((a.root / "manifest.json").read_text())
        if not all(
            complete(a.root, k, json.loads((a.root / "specs" / (k + ".json")).read_text()))
            for k in m["blocks"]
        ):
            raise ValueError("Incomplete campaign")
        print(
            json.dumps({"validated_blocks": m["block_count"], "populations": m["population_count"]})
        )


if __name__ == "__main__":
    main()
