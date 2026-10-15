"""Frozen, run-resumable thirteen-algorithm campaign, isolated from auxiliary walk."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.classic_baselines import run_classic
from geo_llm_scheduler.experiments.comparison13 import (
    ARMS,
    arm_config,
    campaign_specs,
    case_problem,
    initial_trials,
    initialization_factory,
)
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.io.loaders import load_instance, save_instance

REQUIRED = (
    "config.json",
    "summary.json",
    "initial.json.gz",
    "population.json.gz",
    "archive.json.gz",
    "qtable.json",
    "anytime.json",
    "diagnostics.json",
)


def sha(path: Path) -> str:
    """Stream a file SHA256."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_hash() -> str:
    """Bind all package Python and this runner after canonical newline decoding."""
    src = Path(__file__).resolve().parents[1] / "src"
    rows = [
        (
            p.relative_to(src).as_posix(),
            hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest(),
        )
        for p in sorted(src.rglob("*.py"))
    ]
    rows.append(
        ("runner", hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest())
    )
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def write_candidates(path: Path, candidates: list[Candidate]) -> None:
    """Stream complete immutable candidates, avoiding a second full JSON image."""
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write("[")
        for i, candidate in enumerate(candidates):
            if i:
                stream.write(",")
            json.dump(solution(candidate), stream, separators=(",", ":"))
        stream.write("]")


def prepare(root: Path, replicates: int, pilot: bool = False) -> dict:
    """Freeze native prices, paired parents, per-size rotation and thirteen-arm order."""
    if (root / "manifest.json").exists():
        raise FileExistsError("Existing frozen campaign")
    root.mkdir(parents=True, exist_ok=True)
    specs = campaign_specs(replicates, pilot=pilot)
    files = {}
    for spec in specs:
        if pilot:
            spec["seconds"] = {20: 30, 50: 60, 100: 120, 200: 240}[spec["n"]]
        case = root / "inputs" / f"b{spec['base_seed']}_v{spec['variant']}.json"
        case.parent.mkdir(exist_ok=True)
        if not case.exists():
            save_instance(
                case_problem(
                    spec["n"],
                    spec["regions"],
                    spec["machines"],
                    spec["base_seed"],
                    spec["pressure"],
                    spec["variant"],
                ),
                case,
            )
        spec.update(
            input=case.relative_to(root).as_posix(),
            input_sha256=sha(case),
            host=socket.gethostname(),
        )
        target = root / "specs" / (spec["key"] + ".json")
        atomic_json(target, spec)
        files[spec["input"]] = sha(case)
        files[target.relative_to(root).as_posix()] = sha(target)
    commit = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = dict(
        source=source_hash(),
        commit=commit,
        host=socket.gethostname(),
        kind="pilot" if pilot else "formal",
        blocks=[s["key"] for s in specs],
        arms=list(ARMS),
        runs=len(specs) * 13,
        files=files,
        child_address_limit_bytes=2 * 1024**3,
        memory_budget_bytes=200000000000,
        window_seconds=72 * 3600,
    )
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", dict(sha256=sha(root / "manifest.json")))
    return manifest


def validate_inputs(root: Path) -> dict:
    """Reject any input/spec/source/host or sealed matrix mutation."""
    manifest = json.loads((root / "manifest.json").read_text())
    assert (
        sha(root / "manifest.json")
        == json.loads((root / "manifest.sha256.json").read_text())["sha256"]
    )
    assert manifest["source"] == source_hash() and manifest["host"] == socket.gethostname()
    assert tuple(manifest["arms"]) == ARMS
    assert (
        len(manifest["blocks"]) == len(set(manifest["blocks"]))
        and manifest["runs"] == len(manifest["blocks"]) * 13
    )
    for name, expected in manifest["files"].items():
        assert sha(root / name) == expected, name
    return manifest


def complete(root: Path, key: str, arm: str, source: str) -> bool:
    """Require all eight full artifacts, final exact evidence and frozen provenance."""
    out = root / "runs" / key / arm
    if not (out / "complete.json").exists():
        return False
    marker = json.loads((out / "complete.json").read_text())
    spec_path = root / "specs" / (key + ".json")
    spec = json.loads(spec_path.read_text())
    assert marker["source"] == source and marker["host"] == spec["host"] == socket.gethostname()
    assert marker["key"] == key and marker["arm"] == arm
    assert marker["input_sha256"] == spec["input_sha256"] == sha(root / spec["input"])
    assert marker["spec_sha256"] == sha(spec_path)
    assert set(marker["files"]) == set(REQUIRED)
    for name, e in marker["files"].items():
        assert (out / name).stat().st_size == e["size"] and sha(out / name) == e["sha256"], name
    summary = json.loads((out / "summary.json").read_text())
    assert summary["arm"] == arm and summary["source"] == source
    assert summary["offspring"] > 0 and summary["final_exact_verified"] >= 100
    assert summary["termination"] == "time_budget"
    assert summary["initialization"]["trials"] == initial_trials(arm)
    assert json.loads((out / "config.json").read_text()) == json.loads(
        json.dumps(asdict(arm_config(arm, spec["seed"], spec["seconds"])))
    )
    return True


def worker(root: Path, key: str, arm: str, cpu: int) -> None:
    """Initialize and evolve in one clock, independently audit all final solutions."""
    import resource

    from geo_llm_scheduler.engine.run import run
    from geo_llm_scheduler.experiments.nsga2 import run_nsga2

    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    os.sched_setaffinity(0, {cpu})
    # Per-run flock excludes orphan workers after a supervisor interruption.
    import fcntl

    lock_path = root / "locks" / (key + "_" + arm + ".lock")
    lock_path.parent.mkdir(exist_ok=True)
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((root / "manifest.json").read_text())
    spec_path = root / "specs" / (key + ".json")
    assert manifest["source"] == source_hash()
    assert sha(spec_path) == manifest["files"][spec_path.relative_to(root).as_posix()]
    spec = json.loads(spec_path.read_text())
    assert (
        sha(root / spec["input"]) == spec["input_sha256"] and spec["host"] == socket.gethostname()
    )
    out = root / "runs" / key / arm
    out.mkdir(parents=True, exist_ok=False)
    config = arm_config(arm, spec["seed"], spec["seconds"])
    atomic_json(out / "config.json", asdict(config))
    started = time.time()
    clock = time.perf_counter()
    snapshots = []
    thresholds = [spec["seconds"] * f for f in (0.1, 0.25, 0.5, 0.75, 0.9)]
    last = -10.0
    initial_evidence: dict = {}
    diagnostics: dict[str, Counter] = {}
    try:

        def observe(event: str, data: dict) -> None:
            nonlocal last
            gateway = data.get("gateway")
            elapsed = time.perf_counter() - clock
            if event == "initial_ready":
                write_candidates(out / "initial.json.gz", data["population"])
                snapshots.append(
                    dict(
                        target="initialization",
                        elapsed=elapsed,
                        objectives=[c.evaluation.objectives for c in gateway.archive.members],
                    )
                )
            if event == "offspring":
                for step in data["row"].get("steps", []):
                    total = diagnostics.setdefault(str(step["action"]), Counter())
                    for field in (
                        "attempts",
                        "effective",
                        "feasible",
                        "accepted",
                        "reward",
                        "seconds",
                    ):
                        total[field] += step.get(field, 0)
                    total["budget:" + str(step.get("requested_budget", step.get("budget", 0)))] += 1
            if gateway is None:
                return
            if thresholds and elapsed >= thresholds[0]:
                target = thresholds.pop(0)
                snapshots.append(
                    dict(
                        target=target,
                        elapsed=elapsed,
                        objectives=[c.evaluation.objectives for c in gateway.archive.members],
                    )
                )
            if elapsed - last >= 10 or event == "initial_ready":
                atomic_json(
                    out / "heartbeat.json",
                    dict(
                        pid=os.getpid(),
                        host=socket.gethostname(),
                        updated=time.time(),
                        elapsed=elapsed,
                        exact=gateway.counts["exact"],
                        archive=len(gateway.archive.members),
                        phase="search" if initial_evidence else "initialization",
                        offspring=data.get("processed", 0),
                    ),
                )
                last = elapsed

        problem = load_instance(root / spec["input"])
        initialize = initialization_factory(
            problem, config, initial_trials(arm), spec["bank_seed"], initial_evidence, observe
        )
        if arm in ("RK_MOPSO", "SPEA2", "EPS_IBEA"):
            result = run_classic(problem, config, arm, initialize, observer=observe)
        elif arm == "NSGAII":
            result = run_nsga2(
                problem, config, retain_trace=False, observer=observe, initializer=initialize
            )
        else:
            result = run(
                problem, config, retain_trace=False, observer=observe, initializer=initialize
            )
        production = result.elapsed
        production_end = time.time()
        tick = time.perf_counter()
        for candidate in result.population + result.archive.members:
            assert candidate.evaluation.feasible
            assert evaluate(problem, candidate.genotype, candidate.schedule) == candidate.evaluation
        audit = time.perf_counter() - tick
        snapshots.append(
            dict(
                target=spec["seconds"],
                elapsed=production,
                objectives=[c.evaluation.objectives for c in result.archive.members],
            )
        )
        write_candidates(out / "population.json.gz", result.population)
        write_candidates(out / "archive.json.gz", result.archive.members)
        atomic_json(
            out / "qtable.json",
            dict(
                q=result.controller.q,
                visits=result.controller.visits,
                updates=result.controller.updates,
                selections=result.controller.selections,
            ),
        )
        atomic_json(out / "anytime.json", snapshots)
        atomic_json(
            out / "diagnostics.json",
            dict(
                actions=diagnostics,
                counts=dict(result.gateway.counts),
                seconds=result.gateway.seconds,
                adaptive_counts=dict(result.adaptive.counts) if result.adaptive else {},
            ),
        )
        atomic_json(
            out / "summary.json",
            dict(
                arm=arm,
                host=socket.gethostname(),
                source=manifest["source"],
                commit=manifest["commit"],
                seed=spec["seed"],
                pid=os.getpid(),
                cpu=cpu,
                started=started,
                production_end=production_end,
                completed=time.time(),
                production_seconds=production,
                requested_seconds=spec["seconds"],
                budget_overshoot_seconds=max(0.0, production - spec["seconds"]),
                audit_seconds=audit,
                initialization=initial_evidence,
                offspring=result.offspring_count,
                termination=result.termination_reason,
                final_exact_verified=len(result.population) + len(result.archive.members),
                population=len(result.population),
                archive=len(result.archive.members),
                counts=dict(result.gateway.counts),
                max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            ),
        )
        marker = dict(
            arm=arm,
            key=key,
            source=manifest["source"],
            host=socket.gethostname(),
            input_sha256=spec["input_sha256"],
            spec_sha256=sha(spec_path),
            files={
                name: dict(size=(out / name).stat().st_size, sha256=sha(out / name))
                for name in REQUIRED
            },
        )
        atomic_json(out / "complete.json", marker)
        assert complete(root, key, arm, manifest["source"])
    except BaseException:
        atomic_json(
            out / "failure.json", dict(updated=time.time(), traceback=traceback.format_exc())
        )
        raise


def cpus_available(parallel: int) -> list[int]:
    """Prefer one CPU per physical core, then balanced sibling CPUs if requested."""
    groups: dict[tuple[str, str], list[int]] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        p = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        groups.setdefault(
            ((p / "physical_package_id").read_text(), (p / "core_id").read_text()), []
        ).append(cpu)
    cpus = [group[0] for group in groups.values()] + [
        c for group in groups.values() for c in group[1:]
    ]
    if not 1 <= parallel <= min(64, len(cpus)):
        raise ValueError("Parallelism exceeds64 or allowed CPUs")
    return cpus[:parallel]


def supervise(root: Path, parallel: int) -> None:
    """Dispatch round-robin size blocks, stop safely on real failures or user pause."""
    import fcntl

    lock = (root / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = validate_inputs(root)
    if list(root.glob("runs/*/*/failure.json")) or (root / "failure.json").exists():
        raise ValueError("Recorded failure: manual review required")
    # Reject surviving same-root workers before moving any incomplete attempt.
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"worker" in command and str(root).encode() in command:
            raise ValueError("Existing campaign worker; do not duplicate")
    pending = []
    for key in manifest["blocks"]:
        spec = json.loads((root / "specs" / (key + ".json")).read_text())
        arms = [a for a in spec["order"] if not complete(root, key, a, manifest["source"])]
        for arm in arms:
            out = root / "runs" / key / arm
            if out.exists():
                target = root / "quarantine" / (key + "_" + arm + "_" + str(time.time_ns()))
                target.parent.mkdir(exist_ok=True)
                out.rename(target)
        if arms:
            if manifest["kind"] == "pilot":
                pending.extend((key, [arm]) for arm in arms)
            else:
                pending.append((key, arms))
    cpus = cpus_available(parallel)
    now = time.time()
    start_path = root / "start.json"
    if not start_path.exists():
        atomic_json(start_path, dict(started=now, deadline=now + manifest["window_seconds"]))
    start = json.loads(start_path.read_text())
    active: dict[int, dict] = {}
    slots: dict[int, tuple[str, list[str]]] = {}
    done = sum(1 for _ in root.glob("runs/*/*/complete.json"))
    failed = False
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
        resources = available > 32 * 1024**3 and shutil.disk_usage(root).free > 30 * 1024**3
        for slot in range(parallel):
            if slot in active or stopped or not resources:
                continue
            if slot not in slots and pending:
                # Admit a whole thirteen-arm block only if its conservative tail fits.
                key, arms = pending[0]
                spec = json.loads((root / "specs" / (key + ".json")).read_text())
                if time.time() + len(arms) * (spec["seconds"] + 600) + 300 > start["deadline"]:
                    continue
                slots[slot] = pending.pop(0)
            if slot not in slots:
                continue
            key, arms = slots[slot]
            if not arms:
                del slots[slot]
                continue
            arm = arms.pop(0)
            spec = json.loads((root / "specs" / (key + ".json")).read_text())
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
            active[slot] = dict(
                child=child,
                log=log,
                key=key,
                arm=arm,
                started=time.time(),
                limit=spec["seconds"] + 1800,
                cpu=cpus[slot],
            )
        for slot, item in list(active.items()):
            child = item["child"]
            if child.poll() is None and time.time() - item["started"] > item["limit"]:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if child.poll() is not None:
                item["log"].close()
                if child.returncode != 0:
                    failed = True
                    atomic_json(
                        root / "failure.json",
                        dict(
                            key=item["key"],
                            arm=item["arm"],
                            code=child.returncode,
                            updated=time.time(),
                        ),
                    )
                else:
                    assert complete(root, item["key"], item["arm"], manifest["source"])
                    done += 1
                del active[slot]
        atomic_json(
            root / "status.json",
            dict(
                pid=os.getpid(),
                updated=time.time(),
                started=start["started"],
                deadline=start["deadline"],
                done=done,
                total=manifest["runs"],
                pending_blocks=len(pending),
                active=[
                    dict(
                        pid=x["child"].pid,
                        key=x["key"],
                        arm=x["arm"],
                        cpu=x["cpu"],
                        started=x["started"],
                    )
                    for x in active.values()
                ],
                failure=failed,
            ),
        )
        if (
            pending
            and not slots
            and not active
            and time.time() + 13 * (2400 + 600) + 300 > start["deadline"]
        ):
            break
        time.sleep(1)
    state = (
        "failed"
        if failed
        else "stopped"
        if (root / "stop.request").exists()
        else "complete"
        if done == manifest["runs"]
        else "window_complete"
    )
    atomic_json(
        root / "exit.json",
        dict(state=state, updated=time.time(), done=done, total=manifest["runs"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "worker", "validate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--parallel", type=int, default=48)
    parser.add_argument("--key")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--cpu", type=int, default=0)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.root, args.replicates, args.pilot)))
    elif args.action == "run":
        supervise(args.root, args.parallel)
    elif args.action == "worker":
        worker(args.root, args.key, args.arm, args.cpu)
    else:
        m = validate_inputs(args.root)
        count = sum(complete(args.root, k, a, m["source"]) for k in m["blocks"] for a in ARMS)
        print(json.dumps(dict(validated=count, total=m["runs"])))
