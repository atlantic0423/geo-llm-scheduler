"""Freeze and resume the eight-arm, two-host D7 single-factor screening with paired blocks."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterator

from geo_llm_scheduler.config import Config, load_config
from geo_llm_scheduler.experiments.campaign.support import (
    alive,
    atomic_json,
    available_memory_gb,
    free_disk_gb,
    keep_awake,
)
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.d5_worker import canonical_source_hash, validate_complete
from geo_llm_scheduler.experiments.e15_instances import generate_e15_pair
from geo_llm_scheduler.experiments.runner import git_metadata
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.utils.rng import RNGManager

ARMS = ("BASE", "STATE6", "STATE84", "B3", "B10", "SEQ", "A3_TOU", "RPERM")


@contextmanager
def lease(path: Path) -> Iterator[None]:
    """Hold an OS-released exclusive campaign lock, including after process crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if sys.platform == "win32":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def resources(root: Path) -> dict:
    """Bound available memory and CPU by container limits as well as host resources."""
    memory = available_memory_gb()
    cpus = os.cpu_count() or 1
    group = Path("/sys/fs/cgroup")
    limit, current = group / "memory.max", group / "memory.current"
    if limit.exists() and current.exists() and limit.read_text().strip() != "max":
        memory = min(memory, (int(limit.read_text()) - int(current.read_text())) / 1024**3)
    if (group / "cpu.max").exists():
        quota, period = (group / "cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, math.ceil(int(quota) / int(period))))
    if hasattr(os, "sched_getaffinity"):
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    return {
        "available_memory_gib": memory,
        "logical_cpus": cpus,
        "free_disk_gib": free_disk_gb(root),
        "host": platform.node(),
        "system": platform.platform(),
        "python": platform.python_version(),
    }


def prepare(
    root: Path,
    repository: Path,
    pilot: bool = False,
    seconds: float = 60,
    node: int = 0,
) -> dict:
    """Freeze paired eight-arm blocks on a disjoint base-assigned host shard."""
    if (root / "manifest.json").exists():
        raise FileExistsError("Campaign is already frozen")
    if node not in (0, 1):
        raise ValueError("Node must be 0 or 1")
    provenance = git_metadata(repository)
    if not provenance["git_commit"] or provenance["git_dirty"]:
        raise ValueError("Freeze requires a clean Git commit")
    for directory in ("inputs", "specs", "runs", "ops", "logs", "quarantine"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    baseline = load_config(repository / "configs" / "p2_local" / "F6.yaml")
    configs = {
        "BASE": baseline,
        "STATE6": replace(baseline, rl_state_policy="pooled"),
        "STATE84": replace(baseline, rl_state_policy="compound"),
        "B3": replace(baseline, fixed_budget=3),
        "B10": replace(baseline, fixed_budget=10),
        "SEQ": replace(baseline, budget_policy="sequential"),
        "A3_TOU": replace(baseline, a3_region_policy="tariff"),
        "RPERM": replace(baseline, replacement_policy="permuted"),
    }
    source_hash = canonical_source_hash()
    shutil.copyfile(
        repository / "docs/experiments/d7_state_operator_budget.md", root / "inputs/protocol.md"
    )
    blocks = []
    # Even-order Williams design: each arm occupies every position and follows
    # each other arm once in the eight sequences (first-order carryover balance).
    row = (0, 1, 7, 2, 6, 3, 5, 4)
    orders = [tuple(ARMS[(offset + j) % 8] for j in row) for offset in range(8)]
    sizes = (50, 100)
    for n in sizes:
        bases = (
            list(range(1225001 if n == 50 else 1235001, 1225007 if n == 50 else 1235007))
            if pilot
            else list(range(1205001 if n == 50 else 1215001, 1205017 if n == 50 else 1215017))
        )
        bases = [base for position, base in enumerate(bases) if position % 2 == node]
        seeds = (6606, 7707) if pilot else (1101, 2202)
        for base in bases:
            pair = generate_e15_pair(n, base)
            paths = []
            for tariff, problem in zip(("H", "T"), pair):
                path = f"inputs/{n}_{base}_{tariff}.json"
                save_instance(problem, root / path)
                paths.append(path)
            for seed in seeds:
                initial_path = f"inputs/initial_{n}_{base}_{seed}.json"
                initial = initial_genotypes(
                    pair[0], Config(seed=seed), RNGManager(seed).stream("initialization")
                )
                atomic_json(root / initial_path, [asdict(g) for g in initial])
                for tariff, path in zip(("H", "T"), paths):
                    block = f"n{n}_i{base}_{tariff}_a{seed}"
                    keys = []
                    for arm in orders[len(blocks) % len(orders)]:
                        key = f"{block}_{arm}"
                        config = replace(
                            configs[arm],
                            seed=seed,
                            instance=path,
                            output=f"runs/{key}",
                            seconds=(seconds if n == 50 else 2 * seconds)
                            if pilot
                            else (900.0 if n == 50 else 1800.0),
                            generations=1_000_000,
                        )
                        spec = {
                            "key": key,
                            "block": block,
                            "arm": arm,
                            "jobs": n,
                            "base_seed": base,
                            "tariff": tariff,
                            "algorithm_seed": seed,
                            "config": asdict(config),
                            "instance": path,
                            "instance_hash": file_hash(root / path),
                            "initial": initial_path,
                            "initial_hash": file_hash(root / initial_path),
                            "source_commit": provenance["git_commit"],
                            "source_hash": source_hash,
                        }
                        atomic_json(root / "specs" / f"{key}.json", spec)
                        keys.append(key)
                    blocks.append({"key": block, "jobs": n, "runs": keys})
    RNGManager(20261009).stream("d7_block_order").shuffle(blocks)
    blocks.sort(key=lambda b: -b["jobs"])
    manifest = {
        "schema": 1,
        "kind": "pilot" if pilot else "D7_SINGLE_FACTOR_1024",
        "node": node,
        "nodes": 2,
        "assignment": "base-position modulo 2; both tariffs and all seeds stay together",
        "source_commit": provenance["git_commit"],
        "source_hash": source_hash,
        "source_hash_format": "UTF8-universal-newline-JSON-sha256-v1",
        "arms": ARMS,
        "blocks": blocks,
        "run_count": len(blocks) * len(ARMS),
        "input_files": {
            p.relative_to(root).as_posix(): file_hash(p)
            for directory in ("inputs", "specs")
            for p in sorted((root / directory).glob("*"))
        },
        "nominal_worker_hours": sum(
            json.loads((root / "specs" / f"{key}.json").read_text())["config"]["seconds"]
            for block in blocks
            for key in block["runs"]
        )
        / 3600,
        "resume": "completed eight-arm blocks portable; partial blocks rerun together on migration",
        "intervention": "one factor per arm; original A8; all construction and evaluation cost counted; bounded state/action observations",
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(
        root / "manifest.sha256.json", {"manifest_sha256": file_hash(root / "manifest.json")}
    )
    return manifest


def recover_block(root: Path, block: dict, host: str) -> bool:
    """Reuse complete blocks; preserve and rerun foreign partial blocks as a unit."""
    valid = []
    hosts = set()
    for key in block["runs"]:
        result = root / "runs" / key
        spec = json.loads((root / "specs" / f"{key}.json").read_text())
        if (root / "ops" / "failures" / f"{key}.json").exists() or (
            root / "ops" / "failures" / f"supervisor_{key}.json"
        ).exists():
            raise RuntimeError(f"Recorded crash requires review: {key}")
        if result.exists():
            checked, reason = validate_complete(result, spec)
            if not checked:
                raise RuntimeError(f"Invalid completed result {key}: {reason}")
            valid.append(key)
            hosts.add(json.loads((result / "complete.json").read_text())["host"])
    if len(hosts) > 1:
        raise RuntimeError("Triad mixes hosts")
    if len(valid) == len(block["runs"]):
        return True
    attempts = [p for key in block["runs"] for p in (root / "runs").glob(f".{key}.attempt-*")]
    foreign = bool(valid and host not in hosts)
    if attempts or foreign:
        destination = root / "quarantine" / f"{block['key']}_{time.time_ns()}"
        destination.mkdir(parents=True)
        for path in attempts + ([root / "runs" / key for key in valid] if foreign else []):
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Recovery path escaped campaign")
            path.rename(destination / path.name)
        atomic_json(
            destination / "recovery.json",
            {
                "reason": "interrupted foreign block" if foreign else "interrupted attempt",
                "target_host": host,
                "prior_hosts": sorted(hosts),
            },
        )
    return False


def supervise(root: Path, workers: int, reserve_gib: float = 6.0) -> None:
    """Dispatch blocks under resource guards, drain on stop, stop on the first crash."""
    with lease(root / "ops" / "supervisor.lock"):
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            file_hash(root / "manifest.json")
            != json.loads((root / "manifest.sha256.json").read_text())["manifest_sha256"]
        ):
            raise ValueError("Manifest changed after freeze")
        if canonical_source_hash() != manifest["source_hash"]:
            raise ValueError("Frozen source mismatch")
        if any(file_hash(root / path) != sha for path, sha in manifest["input_files"].items()):
            raise ValueError("Frozen matrix inputs changed")
        old_path = root / "ops" / "status.json"
        if old_path.exists():
            old = json.loads(old_path.read_text())
            if old["host"] == platform.node() and any(alive(r["pid"]) for r in old["active"]):
                raise RuntimeError(
                    "Prior workers are still alive; do not start a second supervisor"
                )
        pending = [b for b in manifest["blocks"] if not recover_block(root, b, platform.node())]
        completed = len(manifest["blocks"]) - len(pending)
        worker_cap = min(workers, resources(root)["logical_cpus"])
        if worker_cap < 1:
            raise ValueError("Positive worker count required")
        active: list[dict] = []
        failed = False
        keep_awake(True)
        epoch = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "workers": worker_cap,
            "resources": resources(root),
            "manifest_sha256": file_hash(root / "manifest.json"),
        }
        atomic_json(root / "ops" / f"epoch_{time.time_ns()}.json", epoch)

        def launch(block: dict) -> dict | None:
            for key in block["runs"]:
                if (root / "runs" / key).exists():
                    continue
                flags = (
                    subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
                    if sys.platform == "win32"
                    else 0
                )
                environment = dict(
                    os.environ,
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    PYTHONUTF8="1",
                )
                with (root / "logs" / f"{key}.log").open("ab") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "geo_llm_scheduler.experiments.d7_worker",
                            str(root),
                            key,
                        ],
                        cwd=root,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        creationflags=flags,
                    )
                return {
                    "block": block,
                    "key": key,
                    "pid": process.pid,
                    "process": process,
                    "started": time.time(),
                }
            return None

        try:
            while pending or active:
                for item in list(active):
                    code = item["process"].poll()
                    spec = json.loads((root / "specs" / f"{item['key']}.json").read_text())
                    if (
                        code is None
                        and time.time() - item["started"] > spec["config"]["seconds"] * 4 + 600
                    ):
                        item["process"].terminate()
                        atomic_json(
                            root / "ops" / "failures" / f"{item['key']}.json",
                            {
                                "reason": "owned worker exceeded hard watchdog deadline",
                                "pid": item["pid"],
                            },
                        )
                        failed = True
                        continue
                    if code is None:
                        continue
                    active.remove(item)
                    checked, reason = validate_complete(root / "runs" / item["key"], spec)
                    if code != 0 or not checked:
                        atomic_json(
                            root / "ops" / "failures" / f"supervisor_{item['key']}.json",
                            {"exit_code": code, "reason": reason, "timestamp": time.time()},
                        )
                        failed = True
                    elif not failed and not (root / "stop.request").exists():
                        successor = launch(item["block"])
                        if successor:
                            active.append(successor)
                        else:
                            completed += 1
                    elif recover_block(root, item["block"], platform.node()):
                        completed += 1
                limits = resources(root)
                stopping = (root / "stop.request").exists() or failed
                guarded = (
                    limits["available_memory_gib"] < reserve_gib or limits["free_disk_gib"] < 20
                )
                if not stopping and not guarded:
                    while pending and len(active) < worker_cap:
                        successor = launch(pending.pop(0))
                        if successor:
                            active.append(successor)
                runs_done = sum(1 for p in (root / "runs").glob("*/complete.json"))
                atomic_json(
                    old_path,
                    {
                        "pid": os.getpid(),
                        "host": platform.node(),
                        "timestamp": time.time(),
                        "workers": worker_cap,
                        "state": "failed_draining"
                        if failed
                        else "draining"
                        if stopping
                        else "resource_wait"
                        if guarded
                        else "running",
                        "completed_blocks": completed,
                        "total_blocks": len(manifest["blocks"]),
                        "completed_runs": runs_done,
                        "total_runs": manifest["run_count"],
                        "pending_blocks": len(pending),
                        "resources": limits,
                        "active": [
                            {k: v for k, v in item.items() if k in ("key", "pid", "started")}
                            for item in active
                        ],
                    },
                )
                if stopping and not active:
                    break
                time.sleep(2)
            atomic_json(
                root / "ops" / "exit.json",
                {
                    "timestamp": time.time(),
                    "failed": failed,
                    "completed_blocks": completed,
                    "pending_blocks": len(manifest["blocks"]) - completed,
                    "active": [],
                    "state": "failed"
                    if failed
                    else "paused"
                    if completed < len(manifest["blocks"])
                    else "complete",
                },
            )
            status = (
                json.loads(old_path.read_text())
                if old_path.exists()
                else {
                    "pid": os.getpid(),
                    "host": platform.node(),
                    "active": [],
                    "completed_blocks": completed,
                    "completed_runs": completed * len(ARMS),
                    "total_runs": manifest["run_count"],
                    "workers": worker_cap,
                }
            )
            status["state"] = (
                "failed"
                if failed
                else "paused"
                if completed < len(manifest["blocks"])
                else "complete"
            )
            status["timestamp"] = time.time()
            atomic_json(old_path, status)
        finally:
            keep_awake(False)


def main() -> None:
    """Freeze or supervise a campaign with an explicitly supplied root and worker cap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--node", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.operation == "prepare":
        print(
            json.dumps(
                prepare(
                    args.root.resolve(),
                    Path(__file__).resolve().parents[1],
                    args.pilot,
                    args.seconds,
                    args.node,
                ),
                ensure_ascii=False,
            )
        )
    else:
        supervise(args.root.resolve(), args.workers)


if __name__ == "__main__":
    main()
