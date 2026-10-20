"""Linux detached, run-level resumable initialization comparison.

Complete markers bind input/spec/source and every result byte. Blocks stay on
one host and CPU; their arm order is balanced. Genuine failures stop dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.initial_bill_benchmark import benchmark_population
from geo_llm_scheduler.experiments.initial_bill_speed import ARMS, speed_problem, speed_specs
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.io.loaders import load_instance, save_instance


def digest(path: Path) -> str:
    """Hash a file without retaining a whole raw export in memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            h.update(chunk)
    return h.hexdigest()


def source_digest() -> str:
    """Bind all Python source and this runner independently of Git availability."""
    src = Path(__file__).resolve().parents[1] / "src"
    rows = [(p.relative_to(src).as_posix(), digest(p)) for p in sorted(src.rglob("*.py"))]
    rows.append(("runner", digest(Path(__file__))))
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def prepare(root: Path, *, pilot: bool) -> dict:
    """Create immutable input/spec matrix before any algorithm worker starts."""
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise ValueError("Refuse to overwrite a frozen campaign")
    files = {}
    specs = speed_specs(pilot=pilot)
    for spec in specs:
        key = spec["key"]
        case = root / "inputs" / (key.rsplit("_s", 1)[0] + ".json")
        case.parent.mkdir(exist_ok=True)
        if not case.exists():
            save_instance(
                speed_problem(spec["n"], spec["base_seed"], spec["machines"], spec["scenario"]),
                case,
            )
        spec.update(
            input=case.relative_to(root).as_posix(),
            input_sha256=digest(case),
            host=socket.gethostname(),
        )
        path = root / "specs" / (key + ".json")
        atomic_json(path, spec)
        files[path.relative_to(root).as_posix()] = digest(path)
        files[spec["input"]] = digest(case)
    manifest = {
        "kind": "pilot" if pilot else "formal",
        "source": source_digest(),
        "host": socket.gethostname(),
        "blocks": [s["key"] for s in specs],
        "runs": len(specs) * 4,
        "files": files,
        "population": 100,
        "memory_budget_bytes": 200000000000,
        "child_address_limit_bytes": 4 * 1024**3,
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", {"sha256": digest(root / "manifest.json")})
    return manifest


def validate_inputs(root: Path) -> dict:
    """Reject changed source, matrix, input bytes, or execution host."""
    assert (
        digest(root / "manifest.json")
        == json.loads((root / "manifest.sha256.json").read_text())["sha256"]
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["source"] == source_digest(), "Source changed"
    assert manifest["host"] == socket.gethostname(), "Host changed"
    for name, expected in manifest["files"].items():
        assert digest(root / name) == expected, name
    return manifest


def complete(root: Path, key: str, arm: str, source: str) -> bool:
    """Verify a finished run's input identity, trial counts and all artifact bytes."""
    out = root / "runs" / key / arm
    if not (out / "complete.json").exists():
        return False
    marker = json.loads((out / "complete.json").read_text())
    spec = json.loads((root / "specs" / (key + ".json")).read_text())
    assert marker["source"] == source and marker["host"] == socket.gethostname()
    assert marker["spec_sha256"] == digest(root / "specs" / (key + ".json"))
    assert marker["input_sha256"] == spec["input_sha256"] == digest(root / spec["input"])
    assert marker["arm"] == arm and marker["key"] == key
    assert set(marker["files"]) == {
        "population.json",
        "archive.json",
        "summary.json",
        "config.json",
    }
    for name, expected in marker["files"].items():
        path = out / name
        assert path.stat().st_size == expected["size"] and digest(path) == expected["sha256"], name
    stats = json.loads((out / "summary.json").read_text())
    assert stats["new_candidate_exact"] == 10 * int(arm[-1])
    assert stats["final_independent_exact_verified"] == 100
    assert stats["candidate_independent_exact_verified"] == 10 * int(arm[-1])
    return True


def worker(root: Path, key: str, arm: str, cpu: int | None = None) -> None:
    """Construct one arm from scratch, audit, then atomically seal its artifacts."""
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["source"] == source_digest()
    spec_path = root / "specs" / (key + ".json")
    assert digest(spec_path) == manifest["files"][spec_path.relative_to(root).as_posix()]
    spec = json.loads(spec_path.read_text())
    assert spec["host"] == socket.gethostname()
    assert digest(root / spec["input"]) == spec["input_sha256"]
    out = root / "runs" / key / arm
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def progress(event: dict) -> None:
        atomic_json(out / "progress.json", {"pid": os.getpid(), "updated": time.time(), **event})

    try:
        atomic_json(out / "progress.json", {"pid": os.getpid(), "updated": time.time(), "exact": 0})
        cfg = Config(population=100, seed=spec["seed"])
        atomic_json(
            out / "config.json",
            {
                "seed": spec["seed"],
                "bank_seed": spec["bank_seed"],
                "arm": arm,
                "population": 100,
                "share": 10,
                "trials": int(arm[-1]),
                "accelerated": arm != "REF4",
            },
        )
        pop, front, stats = benchmark_population(
            load_instance(root / spec["input"]),
            cfg,
            "BILL10",
            spec["bank_seed"],
            progress,
            accelerated=arm != "REF4",
            trials=int(arm[-1]),
        )
        stats["parent_sha256"] = hashlib.sha256(
            json.dumps(stats.pop("parents")).encode()
        ).hexdigest()
        stats.update(
            arm=arm,
            pid=os.getpid(),
            started=started,
            completed=time.time(),
            host=socket.gethostname(),
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        atomic_json(out / "population.json", [solution(c) for c in pop])
        atomic_json(out / "archive.json", [solution(c) for c in front])
        atomic_json(out / "summary.json", stats)
        files = {
            p.name: {"size": p.stat().st_size, "sha256": digest(p)}
            for p in (
                out / name
                for name in ("population.json", "archive.json", "summary.json", "config.json")
            )
        }
        atomic_json(
            out / "complete.json",
            {
                "key": key,
                "arm": arm,
                "source": manifest["source"],
                "host": socket.gethostname(),
                "input_sha256": spec["input_sha256"],
                "spec_sha256": digest(spec_path),
                "files": files,
            },
        )
        assert complete(root, key, arm, manifest["source"])
    except BaseException:
        atomic_json(
            out / "failure.json", {"updated": time.time(), "traceback": traceback.format_exc()}
        )
        raise


def validate_block(root: Path, key: str, source: str) -> bool:
    """Require all four runs and strict REF4/FAST4 population/RNG/score equality."""
    if not all(complete(root, key, arm, source) for arm in ARMS):
        return False
    out = root / "runs" / key
    assert digest(out / "REF4/population.json") == digest(out / "FAST4/population.json")
    assert digest(out / "REF4/archive.json") == digest(out / "FAST4/archive.json")
    stats = [json.loads((out / arm / "summary.json").read_text()) for arm in ARMS]
    assert len({s["parent_sha256"] for s in stats}) == 1
    for field in ("observations", "rng_state_sha256"):
        assert stats[0]["bank_stats"][field] == stats[1]["bank_stats"][field]
    return True


def physical_cpus() -> list[int]:
    """Choose one allowed logical processor per physical core on Linux."""
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((topology / "physical_package_id").read_text(), (topology / "core_id").read_text())
        cores.setdefault(key, cpu)
    return list(cores.values())


def supervise(root: Path, parallel: int) -> None:
    """Run balanced sequential-arm blocks in parallel, with safe task-level resume."""
    import fcntl

    lock = (root / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = validate_inputs(root)
    cpus = physical_cpus()
    if not 1 <= parallel <= min(24, len(cpus)):
        raise ValueError("Parallelism must fit 24x4GiB and distinct physical cores")
    if list(root.glob("runs/*/*/failure.json")) or (root / "failure.json").exists():
        raise ValueError("Failures require review; no automatic algorithm retries")
    pending = []
    for key in manifest["blocks"]:
        if validate_block(root, key, manifest["source"]):
            continue
        spec = json.loads((root / "specs" / (key + ".json")).read_text())
        arms = [a for a in spec["order"] if not complete(root, key, a, manifest["source"])]
        for arm in arms:
            out = root / "runs" / key / arm
            if out.exists():
                target = root / "quarantine" / (key + "_" + arm + "_" + str(time.time_ns()))
                target.parent.mkdir(exist_ok=True)
                out.rename(target)
        pending.append((key, arms))
    active: dict[int, dict] = {}
    slots: dict[int, tuple[str, list[str]]] = {}
    started = time.time()
    failed = False
    try:
        while pending or slots or active:
            stopped = (root / "stop.request").exists() or failed
            if stopped and not active:
                break
            available = (
                int(
                    next(
                        line.split()[1]
                        for line in Path("/proc/meminfo").read_text().splitlines()
                        if line.startswith("MemAvailable:")
                    )
                )
                * 1024
            )
            allowed = available > 16 * 1024**3 and shutil.disk_usage(root).free > 10 * 1024**3
            for slot in range(parallel):
                if slot in active or stopped or not allowed:
                    continue
                if slot not in slots and pending:
                    slots[slot] = pending.pop(0)
                if slot not in slots:
                    continue
                key, arms = slots[slot]
                if not arms:
                    assert validate_block(root, key, manifest["source"])
                    del slots[slot]
                    continue
                arm = arms.pop(0)
                log_path = root / "logs" / (key + "_" + arm + ".log")
                log_path.parent.mkdir(exist_ok=True)
                log = log_path.open("a")
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--root",
                    str(root),
                    "--key",
                    key,
                    "--arm",
                    arm,
                    "--cpu",
                    str(cpus[slot]),
                ]
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                active[slot] = {
                    "process": child,
                    "log": log,
                    "key": key,
                    "arm": arm,
                    "started": time.time(),
                    "cpu": cpus[slot],
                }
            for slot, item in list(active.items()):
                child = item["process"]
                if child.poll() is None and time.time() - item["started"] > 7200:
                    child.terminate()
                    child.wait(timeout=30)
                if child.poll() is not None:
                    item["log"].close()
                    if child.returncode != 0:
                        failed = True
                        atomic_json(
                            root / "failure.json",
                            {
                                "key": item["key"],
                                "arm": item["arm"],
                                "code": child.returncode,
                                "updated": time.time(),
                            },
                        )
                    else:
                        assert complete(root, item["key"], item["arm"], manifest["source"])
                    del active[slot]
            done = len(list(root.glob("runs/*/*/complete.json")))
            atomic_json(
                root / "status.json",
                {
                    "pid": os.getpid(),
                    "started": started,
                    "updated": time.time(),
                    "done_runs_observed": done,
                    "total": manifest["runs"],
                    "pending_blocks": len(pending),
                    "failed": failed,
                    "active": [
                        {k: v for k, v in a.items() if k not in ("process", "log")}
                        | {"pid": a["process"].pid}
                        for a in active.values()
                    ],
                    "parallel": parallel,
                    "available_memory_bytes": available,
                    "admission_allowed": allowed,
                    "source": manifest["source"],
                },
            )
            if active or pending or slots:
                time.sleep(2)
        valid = (
            all(validate_block(root, k, manifest["source"]) for k in manifest["blocks"])
            if not failed
            else False
        )
        atomic_json(
            root / "exit.json",
            {
                "state": "complete" if valid else "failure" if failed else "stopped",
                "updated": time.time(),
                "validated_runs": manifest["runs"] if valid else None,
            },
        )
    except BaseException:
        atomic_json(
            root / "failure.json", {"traceback": traceback.format_exc(), "updated": time.time()}
        )
        # Do not strand children after a supervisor verification failure.
        for item in active.values():
            item["process"].wait()
            item["log"].close()
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "worker", "run", "validate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--parallel", type=int, default=24)
    parser.add_argument("--key")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.root, pilot=args.pilot)))
    elif args.command == "worker":
        worker(args.root, args.key, args.arm, args.cpu)
    elif args.command == "run":
        supervise(args.root, args.parallel)
    else:
        frozen = validate_inputs(args.root)
        assert all(validate_block(args.root, key, frozen["source"]) for key in frozen["blocks"])
        print(json.dumps({"validated": frozen["runs"], "source": frozen["source"]}))
