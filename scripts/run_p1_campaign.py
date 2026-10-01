"""Freeze P0/P1 inputs and run an isolated, memory-guarded, zero-retry campaign."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.campaign.model import JobSpec
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.e15_instances import generate_e15_pair
from geo_llm_scheduler.experiments.p1_worker import validate_spec
from geo_llm_scheduler.experiments.runner import source_digest
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.utils.rng import RNGManager


def resources() -> dict:
    """Record current cgroup working set, limits, OOM counters, CPU and disk evidence."""
    root = Path("/sys/fs/cgroup")

    def read(name: str, default: str = "0") -> str:
        p = root / name
        return p.read_text().strip() if p.exists() else default

    limit_text = read("memory.max", "max")
    limit = int(limit_text) if limit_text != "max" else 32 * 1024**3
    current = int(read("memory.current"))
    stats = dict(line.split() for line in read("memory.stat", "inactive_file 0").splitlines())
    events = dict(line.split() for line in read("memory.events", "oom_kill 0").splitlines())
    working = max(0, current - int(stats.get("inactive_file", 0)))
    return {
        "limit": limit,
        "current": current,
        "working": working,
        "fraction": working / limit,
        "oom_kill": int(events.get("oom_kill", 0)),
        "cpu_max": read("cpu.max", "unknown"),
        "cpuset": read("cpuset.cpus.effective", "unknown"),
        "host": platform.node(),
        "timestamp": time.time(),
    }


def prepare(root: Path, plan_path: Path, node: int, commit: str, source_hash: str) -> dict:
    """Generate all immutable H/T instances and shared initialization before dispatch."""
    if node not in (0, 1):
        raise ValueError("Expected node 0 or 1")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (root / "manifest.json").exists():
        raise FileExistsError("Campaign already frozen")
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("inputs", "specs", "runs", "ops", "logs"):
        (root / directory).mkdir(exist_ok=True)
    if source_digest() != source_hash:
        raise ValueError("Freeze source mismatch")
    specs: list[dict] = []
    instances: list[dict] = []
    for dataset in ("pilot", "development"):
        for n in (50, 100):
            for base_index, base_seed in enumerate(plan["datasets"][dataset][str(n)]):
                pair = generate_e15_pair(n, base_seed)
                paths = []
                for tariff, problem in zip(("H", "T"), pair):
                    path = root / "inputs" / f"{n}_{base_seed}_{tariff}.json"
                    save_instance(problem, path)
                    paths.append(path)
                    instances.append(
                        {
                            "dataset": dataset,
                            "jobs": n,
                            "seed": base_seed,
                            "tariff": tariff,
                            "sha256": file_hash(path),
                        }
                    )
                seeds = (6606, 7707) if dataset == "pilot" else tuple(plan["algorithm_seeds"])
                for algorithm_index, seed in enumerate(seeds):
                    initial_path = root / "inputs" / f"initial_{n}_{base_seed}_{seed}.json"
                    initial = initial_genotypes(
                        pair[0], Config(seed=seed), RNGManager(seed).stream("initialization")
                    )
                    atomic_json(initial_path, [asdict(g) for g in initial])

                    def spec(stage: str, arm: str, tariff: int, config: Config) -> dict:
                        value = JobSpec(
                            stage,
                            arm,
                            base_seed * 10 + tariff,
                            seed,
                            asdict(config),
                            str(paths[tariff]),
                            file_hash(paths[tariff]),
                            str(initial_path),
                            file_hash(initial_path),
                            source_hash,
                            commit,
                        )
                        path = root / "specs" / f"{value.key}.json"
                        atomic_json(path, asdict(value))
                        return {
                            "key": value.key,
                            "path": str(path),
                            "stage": stage,
                            "arm": arm,
                            "jobs": n,
                            "base_seed": base_seed,
                            "algorithm_seed": seed,
                            "tariff": tariff,
                        }

                    if dataset == "pilot" and seed == 6606:
                        if base_index % 2 == node:
                            for arm, method in (("F6", "full"), ("M0", "plain"), ("N0", "nsga2")):
                                for repeat in (1, 2):
                                    config = Config(
                                        method=method,
                                        generations=200,
                                        seed=seed,
                                        instance=str(paths[0]),
                                    )
                                    specs.append(spec(f"P0R{repeat}", arm, 0, config))
                    elif dataset == "pilot" and seed == 7707 and base_index == 0 and n == 100:
                        for arm, method in (("F6", "full"), ("M0", "plain")):
                            config = Config(
                                method=method,
                                generations=1_000_000,
                                seconds=1200,
                                seed=seed,
                                instance=str(paths[0]),
                            )
                            specs.append(spec("RSS", arm, 0, config))
                        for level in (1, 2, 4, 8):
                            for replicate in range(8):
                                config = Config(
                                    method="full", generations=2, seed=seed, instance=str(paths[0])
                                )
                                specs.append(spec(f"THR{level}_{replicate}", "F6", 0, config))
                    elif dataset == "development":
                        # Keep each base/tariff/seed F6-M0 block together; both nodes receive all arms.
                        if (base_index + algorithm_index) % 2 == node:
                            for tariff in (0, 1):
                                for arm, method in (("F6", "full"), ("M0", "plain")):
                                    config = Config(
                                        method=method,
                                        generations=1_000_000,
                                        seconds=600 if n == 50 else 1200,
                                        seed=seed,
                                        instance=str(paths[tariff]),
                                    )
                                    specs.append(spec("P1", arm, tariff, config))
    hashes = [item["sha256"] for item in instances]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate input hashes across pilot/development/tariffs")
    p1 = [item for item in specs if item["stage"] == "P1"]
    if len(p1) != 240:
        raise ValueError("Incomplete per-node matrix")
    rng = RNGManager(20261001).stream(f"queue:{node}")
    rng.shuffle(specs)
    manifest = {
        "node": node,
        "source_commit": commit,
        "source_hash": source_hash,
        "plan_sha256": file_hash(plan_path),
        "instances": instances,
        "specs": specs,
        "resources_at_freeze": resources(),
        "P1_runs": len(p1),
        "expiry_verified": False,
        "migration_target": "2026-10-04",
        "retries": 0,
        "generation_cap": 1_000_000,
        "time_protocol": {"50": 600, "100": 1200},
        "prepared_at": time.time(),
    }
    atomic_json(root / "manifest.json", manifest)
    return manifest


def run_batch(root: Path, items: list[dict], workers: int, label: str) -> dict:
    """Drain one frozen wave; stop new dispatch after failure/OOM without deleting evidence."""
    if workers < 1 or workers > 8:
        raise ValueError("Workers outside accepted pilot ladder")
    begin = time.monotonic()
    baseline = resources()
    pending = list(items)
    active: dict[str, tuple[subprocess.Popen, float, dict]] = {}
    done, failed, skipped = [], [], []
    peak_fraction = baseline["fraction"]
    poll_rows = root / "ops" / f"resources_{label}.jsonl"
    stop = False
    while pending or active:
        snapshot = resources()
        peak_fraction = max(peak_fraction, snapshot["fraction"])
        if snapshot["oom_kill"] > baseline["oom_kill"]:
            stop = True
            atomic_json(root / "ops" / f"oom_{label}.json", snapshot)
        if (root / "stop.request").exists() or shutil.disk_usage(root).free < 20 * 1024**3:
            stop = True
        for key, (process, started, item) in list(active.items()):
            code = process.poll()
            timeout = 14400 if item["stage"].startswith("P0") else 10800
            if code is None and time.monotonic() - started > timeout:
                process.terminate()
                stop = True
                atomic_json(
                    root / "ops" / f"timeout_{key}.json", {"pid": process.pid, "timeout": timeout}
                )
            elif code is not None:
                spec = JobSpec(**json.loads(Path(item["path"]).read_text(encoding="utf-8")))
                if code == 0 and validate_spec(root / "runs" / key, spec):
                    done.append(key)
                else:
                    failed.append({"key": key, "exit": code})
                    stop = True
                del active[key]
        while pending and len(active) < workers and snapshot["fraction"] < 0.70 and not stop:
            item = pending.pop(0)
            spec = JobSpec(**json.loads(Path(item["path"]).read_text(encoding="utf-8")))
            output = root / "runs" / item["key"]
            if validate_spec(output, spec):
                skipped.append(item["key"])
                continue
            if output.exists() or output.with_suffix(".failure.json").exists():
                failed.append({"key": item["key"], "exit": "prior_incomplete_no_retry"})
                stop = True
                break
            env = {
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            with (root / "logs" / f"{item['key']}.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "geo_llm_scheduler.experiments.p1_worker",
                        item["path"],
                        str(output),
                    ],
                    stdout=log,
                    stderr=log,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            active[item["key"]] = (process, time.monotonic(), item)
        status = {
            "phase": label,
            "pid": os.getpid(),
            "workers": workers,
            "done": len(done),
            "skipped": len(skipped),
            "failed": failed,
            "pending": len(pending),
            "active": {k: p.pid for k, (p, _, _) in active.items()},
            "resource": snapshot,
            "elapsed": time.monotonic() - begin,
            "stopped": stop,
            "sampling_completed": sum(
                (root / "runs" / i["key"] / "sampling_complete.json").exists() for i in items
            ),
        }
        atomic_json(root / "status.json", status)
        with poll_rows.open("a", encoding="utf-8") as h:
            h.write(json.dumps({**snapshot, "active": len(active)}) + "\n")
        if stop and not active:
            break
        if not pending and not active:
            break
        time.sleep(5)
    result = {
        **status,
        "completed_keys": done,
        "skipped_keys": skipped,
        "peak_fraction": peak_fraction,
        "success": not stop and not failed and not pending,
    }
    atomic_json(root / "ops" / f"batch_{label}.json", result)
    return result


def main() -> int:
    """Prepare or execute an explicit phase; formal P1 requires a signed local gate file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "batch"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--commit")
    parser.add_argument("--source-hash")
    parser.add_argument("--phase", default="P0")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.operation == "prepare":
        if args.plan is None or args.commit is None or args.source_hash is None:
            parser.error("prepare requires plan, commit and source hash")
        prepare(args.root, args.plan, args.node, args.commit, args.source_hash)
        return 0
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    if args.phase == "P1":
        gate = json.loads((args.root / "ops" / "launch_gate.json").read_text(encoding="utf-8"))
        if not gate.get("accepted") or gate["manifest_sha256"] != file_hash(
            args.root / "manifest.json"
        ):
            raise ValueError("Formal P1 gate missing or stale")
    items = [s for s in manifest["specs"] if s["stage"].startswith(args.phase)]
    if not items:
        raise ValueError("Phase has no jobs")
    result = run_batch(args.root, items, args.workers, args.phase)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
