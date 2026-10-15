"""Frozen inputs, paired-host scheduling, bounded workers and automatic phase changes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.auxiliary_instances import PRESSURES, TOPOLOGIES, paired_instance
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.runner import digest
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import load_instance, save_instance
from geo_llm_scheduler.macrosearch.trajectory_budget import ARM_SPECS
from geo_llm_scheduler.utils.rng import RNGManager

from .common import config_from, sha, source, validate_complete
from .selection import choose

SIZES = (20, 50, 100, 200)
BASE_SECONDS = (150, 300, 600, 1200)
STAGES = ("screen", "confirm", "long")


def williams(count: int, row: int) -> list[int]:
    """Balanced even-treatment Williams rows, including all 18 screening arms."""
    if count < 2 or count % 2:
        raise ValueError("Even treatment count required")
    base = [0]
    for k in range(1, count):
        base.append((k + 1) // 2 if k % 2 else count - k // 2)
    return [(v + row) % count for v in base]


def make_config(
    arm: str,
    seed: int,
    seconds: float,
    calibration: dict,
    selected: dict | None = None,
    population: int = 100,
) -> Config:
    """Create the common online-Q42/native-generation research background."""
    return Config(
        method="full",
        a3_region_policy="tariff",
        generations=10**9,
        seconds=seconds,
        seed=seed,
        auxiliary_arm=arm,
        population=population,
        neighborhood=min(20, population),
        auxiliary_cutoffs=tuple(calibration.get(arm, calibration.get("D2", (0.05, 0.2)))),
        auxiliary_random_probabilities=tuple(selected["probabilities"])
        if arm == "R0" and selected
        else (0.4, 0.3, 0.3),
        auxiliary_random_aid=bool(arm == "R0" and selected and selected["random_aid"]),
    )


def _initial(record: dict, control: Path) -> dict:
    problem = load_instance(control / record["input"])
    config = config_from(record["initial_config"])
    values = initial_genotypes(problem, config, RNGManager(config.seed).stream("initialization"))
    record["initial_hash"] = digest([asdict(g) for g in values])
    return record


def prepare(control: Path, workers: int, calibration: dict, initial_jobs: int = 16) -> dict:
    """Freeze all 320 independent parents and native P100 gene fingerprints before T0."""
    if workers not in (16, 24):
        raise ValueError("Use the preregistered 32/48 total-worker tiers")
    if (control / "plan.json").exists():
        raise FileExistsError("An existing frozen matrix must not be overwritten")
    (control / "inputs").mkdir(parents=True, exist_ok=True)
    records = []
    hosts = json.loads((control / "hosts.json").read_text())
    for stage_index, stage in enumerate(STAGES):
        node_orders = [0, 0]
        repeats = 2 if stage == "confirm" else 1
        seeds = 2 if workers == 24 or stage == "confirm" else 1
        for repeat in range(repeats):
            for size_index, jobs in enumerate(SIZES):
                for topology_index, (regions, servers) in enumerate(TOPOLOGIES):
                    for pressure_index, pressure in enumerate(PRESSURES):
                        parent = f"{stage}-r{repeat}-n{jobs}-t{topology_index}-p{pressure_index}"
                        base_seed = (
                            202610160000
                            + stage_index * 100000
                            + repeat * 10000
                            + size_index * 1000
                            + topology_index * 100
                            + pressure_index
                        )
                        node = (repeat + size_index + topology_index + pressure_index) % 2
                        for version in (0, 1):
                            path = control / "inputs" / f"{parent}-v{version}.json"
                            problem = paired_instance(
                                jobs, regions, servers, pressure, version, random.Random(base_seed)
                            )
                            save_instance(problem, path)
                            for seed_index in range(seeds):
                                seed = 161001 + seed_index * 10000
                                case = f"{parent}-v{version}-s{seed}"
                                config = make_config(
                                    "W1",
                                    seed,
                                    BASE_SECONDS[size_index] * 2**stage_index,
                                    calibration,
                                )
                                records.append(
                                    {
                                        "parent": parent,
                                        "case": case,
                                        "stage": stage,
                                        "jobs": jobs,
                                        "topology": [regions, servers],
                                        "pressure": pressure,
                                        "version": version,
                                        "seed": seed,
                                        "node": node,
                                        "host": hosts[node],
                                        "base_seed": base_seed,
                                        "input": path.relative_to(control).as_posix(),
                                        "input_sha256": sha(path),
                                        "initial_config": asdict(config),
                                        "order_row": node_orders[node]
                                        % (18 if stage == "screen" else 4),
                                    }
                                )
                                node_orders[node] += 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=initial_jobs) as pool:
        futures = [pool.submit(_initial, record, control) for record in records]
        frozen = [f.result() for f in futures]
    plan = {
        "records": frozen,
        "hosts": hosts,
        "workers_per_node": workers,
        "calibration": calibration,
        "source": source(),
        "stage_arms": {
            "screen": list(ARM_SPECS),
            "confirm": ["S0", "W1", "SELECTED", "R0"],
            "long": ["S0", "W1", "SELECTED", "R0"],
        },
        "statistical_seed": 2026101603,
        "bootstrap": 20000,
        "signflip": 100000,
        "guard": {
            "memory_mib_per_worker": 256,
            "node_root_cap_gib": 16,
            "root_free_min_gib": 20,
            "shared_batch_cap_mib": 1536,
        },
        "hours": 72,
        "parents": 320,
    }
    if plan["source"]["dirty"]:
        raise ValueError("Cannot freeze dirty source")
    atomic_json(control / "source.json", plan["source"])
    atomic_json(control / "plan.json", plan)
    atomic_json(
        control / "plan_hash.json", {"sha256": sha(control / "plan.json"), "records": len(frozen)}
    )
    return plan


def stage_specs(plan: dict, stage: str, selected: dict | None = None) -> list[dict]:
    """Materialize all paired specs, with immutable selection input hashes for holdouts."""
    arms = plan["stage_arms"][stage]
    if stage != "screen":
        if selected is None:
            raise ValueError("Held-out stage needs the frozen screening choice")
        arms = [selected["selected"] if a == "SELECTED" else a for a in arms]
    result = []
    for record in plan["records"]:
        if record["stage"] != stage:
            continue
        ordered = williams(len(arms), record["order_row"] % len(arms))
        for rank, i in enumerate(ordered):
            arm = arms[i]
            config = make_config(
                arm,
                record["seed"],
                record["initial_config"]["seconds"],
                plan["calibration"],
                selected,
            )
            result.append(
                {
                    **{k: v for k, v in record.items() if k != "initial_config"},
                    "id": f"{record['case']}-{arm}",
                    "arm": arm,
                    "config": asdict(config),
                    "position": rank,
                    "selection_hash": digest(selected) if selected else None,
                }
            )
    return result


def _bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def ready_spec(pending: list[dict], active: list[dict]) -> dict | None:
    """Keep each case's Williams sequence serial while running independent cases in parallel."""
    busy = {s["case"] for s in active}
    return next((s for s in pending if s["case"] not in busy), None)


def memory_available(limit: int, current: int, stats: dict[str, int]) -> int:
    """Count only clean inactive file pages as reclaimable; retain dirty/shmem charges."""
    reclaimable = max(
        0,
        stats.get("inactive_file", 0)
        - stats.get("file_dirty", 0)
        - stats.get("file_writeback", 0)
        - stats.get("shmem", 0),
    )
    return max(0, limit - current + min(current, reclaimable))


def supervise(control: Path, output: Path, node: int) -> None:
    """Run finite workers under memory/disk/time guards; fail closed without silent retries."""
    import fcntl

    plan = json.loads((control / "plan.json").read_text())
    expected_hash = json.loads((control / "plan_hash.json").read_text())["sha256"]
    if (
        sha(control / "plan.json") != expected_hash
        or source() != plan["source"]
        or socket.gethostname() != plan["hosts"][node]
    ):
        raise ValueError("Plan/source/host mismatch")
    output.mkdir(parents=True, exist_ok=True)
    ops = control / "ops" / f"node{node}"
    ops.mkdir(parents=True, exist_ok=True)
    lock = (ops / "lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = json.loads((control / "start.json").read_text())
    deadline = start["deadline"]
    stage_number = 0
    processes: dict[str, tuple[subprocess.Popen, dict, object]] = {}
    spawned: dict[str, float] = {}
    done: set[str] = set()
    dispatched: set[str] = set()
    stopped = False
    own_bytes = _bytes(output)
    guard = plan["guard"]
    last_shared_check = 0.0
    while True:
        for key, (process, spec, log) in list(processes.items()):
            code = process.poll()
            if code is None:
                if (
                    time.monotonic() - spawned[key] > spec["config"]["seconds"] * 1.5 + 300
                    or time.time() > deadline
                ):
                    process.terminate()
                    atomic_json(
                        ops / "failure.json",
                        {"id": key, "reason": "hard_timeout", "time": time.time()},
                    )
                    atomic_json(
                        control / "stop.request",
                        {"reason": "hard_timeout", "node": node, "id": key},
                    )
                    stopped = True
                continue
            log.close()
            del processes[key]
            if code:
                atomic_json(ops / "failure.json", {"id": key, "exit": code, "time": time.time()})
                atomic_json(
                    control / "stop.request", {"reason": "worker_failure", "node": node, "id": key}
                )
                stopped = True
            else:
                validate_complete(output / key, spec, plan["source"])
                done.add(key)
                own_bytes += _bytes(output / key)
                # Release only this completed run's clean file pages; no disk deletion.
                for path in (output / key).iterdir():
                    with path.open("rb") as stream:
                        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        selected = None
        if stage_number:
            selected = json.loads((control / "selection.json").read_text())
        stage = STAGES[stage_number]
        all_specs = stage_specs(plan, stage, selected)
        own_specs = [s for s in all_specs if s["node"] == node]
        pending = [s for s in own_specs if s["id"] not in dispatched]
        memory_root = Path("/sys/fs/cgroup")
        memory_free = memory_available(
            int((memory_root / "memory.max").read_text()),
            int((memory_root / "memory.current").read_text()),
            {
                k: int(v)
                for k, v in (
                    line.split() for line in (memory_root / "memory.stat").read_text().splitlines()
                )
            },
        )
        # Reserve a bounded full remaining run for every active slot, plus postprocessing.
        if (
            time.time() >= deadline
            or (control / "stop.request").exists()
            or (ops / "stop.request").exists()
        ):
            stopped = True
        if time.monotonic() - last_shared_check > 60:
            shared_bytes = _bytes(control)
            last_shared_check = time.monotonic()
            if shared_bytes > guard["shared_batch_cap_mib"] * 1024**2:
                atomic_json(
                    control / "stop.request",
                    {"reason": "shared_metadata_guard", "bytes": shared_bytes},
                )
                stopped = True
        while pending and len(processes) < plan["workers_per_node"] and not stopped:
            spec = ready_spec(pending, [v[1] for v in processes.values()])
            if spec is None:
                break
            reserve = spec["config"]["seconds"] * 1.5 + 300
            if time.time() + reserve >= deadline:
                stopped = True
                break
            if (
                own_bytes > guard["node_root_cap_gib"] * 1024**3
                or shutil.disk_usage(output).free < guard["root_free_min_gib"] * 1024**3
            ):
                atomic_json(ops / "failure.json", {"reason": "disk_guard", "own_bytes": own_bytes})
                atomic_json(control / "stop.request", {"reason": "disk_guard", "node": node})
                stopped = True
                break
            if memory_free < 1024**3:
                break
            spec_path = control / "specs" / f"{spec['id']}.json"
            atomic_json(spec_path, spec)
            run_output = output / spec["id"]
            if run_output.exists():
                raise FileExistsError(
                    "Existing attempts require explicit audit, never silent retry"
                )
            log = (ops / f"{spec['id']}.log").open("w")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "research.auxiliary_walk_20261016.worker",
                    "--spec",
                    str(spec_path),
                    "--control",
                    str(control),
                    "--output",
                    str(run_output),
                    "--memory-mib",
                    str(guard["memory_mib_per_worker"]),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            )
            processes[spec["id"]] = (process, spec, log)
            spawned[spec["id"]] = time.monotonic()
            dispatched.add(spec["id"])
            pending.remove(spec)
            memory_free -= guard["memory_mib_per_worker"] * 1024**2
        atomic_json(
            ops / "status.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "time": time.time(),
                "stage": stage,
                "done_total": len(done),
                "stage_target": len(own_specs),
                "active": {k: v[0].pid for k, v in processes.items()},
                "pending": len(pending),
                "stopped": stopped,
                "memory_free": memory_free,
                "own_bytes": own_bytes,
            },
        )
        if stopped and not processes:
            atomic_json(
                ops / "exit.json", {"status": "stopped", "time": time.time(), "done": len(done)}
            )
            return
        if not pending and not processes:
            atomic_json(
                ops / f"{stage}_complete.json", {"time": time.time(), "count": len(own_specs)}
            )
            other_done = all(
                (control / "ops" / f"node{k}" / f"{stage}_complete.json").exists() for k in (0, 1)
            )
            if other_done:
                if stage_number == 2:
                    atomic_json(
                        ops / "exit.json",
                        {"status": "complete", "time": time.time(), "done": len(done)},
                    )
                    return
                if stage_number == 0:
                    with (control / "selection.lock").open("a") as selection_lock:
                        fcntl.flock(selection_lock, fcntl.LOCK_EX)
                        if not (control / "selection.json").exists():
                            rows = []
                            for spec in all_specs:
                                index = json.loads(
                                    (control / "indexes" / f"{spec['id']}.json").read_text()
                                )
                                if (
                                    index["complete"]["spec_hash"] != digest(spec)
                                    or index["complete"]["source"] != plan["source"]
                                ):
                                    raise ValueError("Screening index provenance mismatch")
                                rows.append(index["summary"])
                            choice = choose(rows)
                            choice["screening_index_hash"] = digest([digest(row) for row in rows])
                            choice["frozen_at"] = time.time()
                            atomic_json(control / "selection.json", choice)
                stage_number += 1
                continue
        time.sleep(5)


def main() -> None:
    """Prepare or supervise using explicit control/output paths and node identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "supervise"))
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.control, args.workers, json.loads(args.calibration.read_text()))
    else:
        try:
            supervise(args.control, args.output, args.node)
        except Exception as error:
            atomic_json(
                args.control / "stop.request",
                {
                    "reason": "supervisor_exception",
                    "node": args.node,
                    "error": str(error),
                    "time": time.time(),
                },
            )
            raise


if __name__ == "__main__":
    main()
