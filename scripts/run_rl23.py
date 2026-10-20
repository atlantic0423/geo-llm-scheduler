"""Sealed five-policy study on the original component33 input bytes."""

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
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Candidate
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.mechanism5 import StructuralObserver
from geo_llm_scheduler.experiments.rl23 import (
    ARMS,
    run_policy,
)
from geo_llm_scheduler.experiments.rl23 import (
    study_config as arm_config,
)
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.rng import RNGManager

REQUIRED = (
    "config.json",
    "summary.json",
    "initial.json.gz",
    "population.json.gz",
    "archive.json.gz",
    "qtable.json",
    "anytime.json",
    "diagnostics.json",
    "steps.jsonl.gz",
)


def sha(path: Path) -> str:
    """Hash a regular file without retaining its contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hash() -> str:
    """Bind the installed package and this runner with canonical newlines."""
    src = Path(__file__).resolve().parents[1] / "src"
    paths = [(p.relative_to(src).as_posix(), p) for p in sorted(src.rglob("*.py"))]
    paths.append(("rl23_runner", Path(__file__)))
    return hashlib.sha256(
        json.dumps(
            [
                (name, hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest())
                for name, p in paths
            ]
        ).encode()
    ).hexdigest()


def write_candidates(path: Path, members: list[Candidate]) -> None:
    """Stream complete immutable candidates to avoid a second JSON image."""
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write("[")
        for i, candidate in enumerate(members):
            if i:
                stream.write(",")
            json.dump(solution(candidate), stream, separators=(",", ":"))
        stream.write("]")


def initializer_policy(arm: str) -> str:
    """All study arms use the same90+10 BILL four-choice initializer."""
    if arm not in ARMS:
        raise ValueError("Unknown policy arm")
    return "bill4"


def balanced_order(index: int) -> list[str]:
    """Rotate the five treatments using predeclared Williams entry orders."""
    count = len(ARMS)
    row = [0] + [(j + 1) // 2 if j % 2 else count - j // 2 for j in range(1, count)]
    order = [ARMS[(j + index % count) % count] for j in row]
    return order if (index // count) % 2 == 0 else order[::-1]


def prepare(root: Path, historical: Path, pilot: bool = False) -> dict:
    """Copy only SHA-bound component33 input bytes; never regenerate easier cases."""
    if root.exists():
        raise FileExistsError("Campaign root must be new")
    old_manifest = json.loads((historical / "manifest.json").read_text())
    assert old_manifest["commit"] == "b148c4addd207fe30a6b2a43d953a04afc35d3a1"
    assert (
        sha(historical / "manifest.json")
        == json.loads((historical / "manifest.sha256.json").read_text())["sha256"]
    )
    assert len(old_manifest["blocks"]) == 32 and old_manifest["runs"] == 1056
    root.mkdir(parents=True)
    files = {}
    blocks = []
    originals = old_manifest["blocks"][:4] if pilot else old_manifest["blocks"]
    for seed in (16101,) if pilot else (16101, 26101):
        for old_key in originals:
            old_path = historical / "specs" / (old_key + ".json")
            assert (
                sha(old_path) == old_manifest["files"][old_path.relative_to(historical).as_posix()]
            )
            old = json.loads(old_path.read_text())
            assert old["seed"] == 16101
            original = historical / old["input"]
            assert sha(original) == old["input_sha256"] == old_manifest["files"][old["input"]]
            target = root / old["input"]
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copyfile(original, target)
            assert sha(target) == old["input_sha256"]
            key = old_key.rsplit("_s", 1)[0] + "_s" + str(seed)
            spec = {
                name: old[name]
                for name in (
                    "n",
                    "regions",
                    "machines",
                    "pressure",
                    "variant",
                    "base_seed",
                    "seconds",
                    "input",
                    "input_sha256",
                )
            }
            spec.update(
                key=key,
                seed=seed,
                old_key=old_key,
                historical_spec_sha256=sha(old_path),
                host=socket.gethostname(),
                order=balanced_order(len(blocks)),
            )
            if pilot:
                spec["seconds"] = {20: 60, 50: 90, 100: 150, 200: 300}[spec["n"]]
            spec_path = root / "specs" / (key + ".json")
            atomic_json(spec_path, spec)
            files[spec["input"]] = sha(target)
            files[spec_path.relative_to(root).as_posix()] = sha(spec_path)
            blocks.append(key)
    manifest = dict(
        source=source_hash(),
        host=socket.gethostname(),
        arms=list(ARMS),
        blocks=blocks,
        runs=len(blocks) * len(ARMS),
        kind="pilot" if pilot else "formal",
        files=files,
        commit=subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
        ).strip(),
        historical_commit=old_manifest["commit"],
        historical_manifest_sha256=sha(historical / "manifest.json"),
        memory_reserve_bytes=200 * 1024**3,
        learning_parameters=dict(
            time_reference_seconds={str(n): 1.0 for n in (20, 50, 100, 200)},
            penalty_lambda=0.25,
            rate_floor=0.1,
            time_boundary=0.5,
            active_tie_ratio=1.1,
        ),
        child_address_limit_bytes=2 * 1024**3,
        window_seconds=8 * 3600,
        sampling_probabilities=dict(conditional_ls=0.05, conditional_no_ls=0.10),
        initialization="90+10_BILL4_P100",
        independent_parents=4 if pilot else 16,
    )
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", dict(sha256=sha(root / "manifest.json")))
    return manifest


def validate_inputs(root: Path) -> dict:
    """Reject mutated source, host, input, spec or campaign seal."""
    manifest = json.loads((root / "manifest.json").read_text())
    assert (
        sha(root / "manifest.json")
        == json.loads((root / "manifest.sha256.json").read_text())["sha256"]
    )
    assert manifest["source"] == source_hash() and manifest["host"] == socket.gethostname()
    assert tuple(manifest["arms"]) == ARMS
    assert len(set(manifest["blocks"])) == len(manifest["blocks"])
    assert manifest["runs"] == len(ARMS) * len(manifest["blocks"])
    for name, expected in manifest["files"].items():
        assert sha(root / name) == expected, name
    return manifest


def preflight(root: Path, key: str) -> dict:
    """Fresh independent decode/exact audit of the common BILL4 initialization policy."""
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["source"] == source_hash()
    spec_path = root / "specs" / (key + ".json")
    assert sha(spec_path) == manifest["files"][spec_path.relative_to(root).as_posix()]
    spec = json.loads(spec_path.read_text())
    assert sha(root / spec["input"]) == spec["input_sha256"]
    problem = load_instance(root / spec["input"])
    policies = {}
    for arm in ("BILL_BIRTH",):
        cfg = arm_config(arm, spec["seed"], spec["seconds"])
        captured: list = []
        gateway = EvaluationGateway(problem, Archive())
        gateway.candidate_sink = captured.append
        evidence: dict = {}
        members = charged_initializer(problem, cfg, arm, evidence)(gateway, RNGManager(cfg.seed))
        assert len(members) == len({c.genotype for c in members}) == 100
        for candidate in captured:
            assert decode(problem, candidate.genotype) == candidate.schedule
            assert evaluate(problem, candidate.genotype, candidate.schedule) == candidate.evaluation
            assert candidate.evaluation.feasible
        policy = initializer_policy(arm)
        expected_count = 100 + 10 * int(policy[-1]) if policy.startswith("bill") else 100
        assert len(captured) == gateway.counts["exact"] == expected_count
        policies[initializer_policy(arm)] = dict(
            evidence=evidence, fresh_exact_verified=len(captured)
        )
    result = dict(
        source=manifest["source"],
        input_sha256=spec["input_sha256"],
        key=key,
        policies=policies,
        host=socket.gethostname(),
        completed=time.time(),
    )
    atomic_json(root / "preflight" / (key + ".json"), result)
    return result


def complete(root: Path, key: str, arm: str, source: str) -> bool:
    """Require exact-audited full artifacts, provenance and preflight initialization."""
    out = root / "runs" / key / arm
    if not (out / "complete.json").exists():
        return False
    spec_path = root / "specs" / (key + ".json")
    spec = json.loads(spec_path.read_text())
    marker = json.loads((out / "complete.json").read_text())
    assert marker["source"] == source and marker["host"] == spec["host"] == socket.gethostname()
    assert marker["key"] == key and marker["arm"] == arm
    assert marker["input_sha256"] == spec["input_sha256"] == sha(root / spec["input"])
    assert marker["spec_sha256"] == sha(spec_path) and set(marker["files"]) == set(REQUIRED)
    for name, record in marker["files"].items():
        assert (out / name).stat().st_size == record["size"] and sha(out / name) == record["sha256"]
    summary = json.loads((out / "summary.json").read_text())
    cfg = arm_config(arm, spec["seed"], spec["seconds"])
    assert json.loads((out / "config.json").read_text()) == json.loads(json.dumps(asdict(cfg)))
    assert summary["source"] == source and summary["arm"] == arm
    assert (
        summary["learning_parameters"]
        == json.loads((root / "manifest.json").read_text())["learning_parameters"]
    )
    assert summary["termination"] == "time_budget" and summary["offspring"] > 0
    q = json.loads((out / "qtable.json").read_text())
    assert len(q["q"]) == (84 if arm == "STATE_TIME84" else 6 if arm == "POOLED6" else 42)
    assert all(len(row) == 8 for row in q["q"])
    assert (
        summary["population"] == 100 and summary["final_exact_verified"] == 100 + summary["archive"]
    )
    check = json.loads((root / "preflight" / (key + ".json")).read_text())
    assert check["source"] == source and check["input_sha256"] == spec["input_sha256"]
    assert summary["initialization"]["policy"] == initializer_policy(arm)
    expected = check["policies"][initializer_policy(arm)]["evidence"]
    for field in ("initial_population_sha256", "initial_archive_sha256"):
        assert summary["initialization"][field] == expected[field]
    return True


def worker(root: Path, key: str, arm: str, cpu: int) -> None:
    """Run charged production, then independently audit every final saved solution."""
    import fcntl
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    os.sched_setaffinity(0, {cpu})
    lock_path = root / "locks" / (key + "_" + arm + ".lock")
    lock_path.parent.mkdir(exist_ok=True)
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["source"] == source_hash() and manifest["host"] == socket.gethostname()
    spec_path = root / "specs" / (key + ".json")
    assert sha(spec_path) == manifest["files"][spec_path.relative_to(root).as_posix()]
    spec = json.loads(spec_path.read_text())
    assert sha(root / spec["input"]) == spec["input_sha256"]
    out = root / "runs" / key / arm
    out.mkdir(parents=True, exist_ok=False)
    cfg = arm_config(arm, spec["seed"], spec["seconds"])
    atomic_json(out / "config.json", asdict(cfg))
    evidence: dict = {}
    snapshots: list = []
    step_stream = gzip.open(out / "steps.jsonl.gz", "wt", encoding="utf-8")
    structures = StructuralObserver()
    thresholds = [spec["seconds"] * f for f in (0.1, 0.25, 0.5, 0.75, 0.9)]
    started, clock, last = time.time(), time.perf_counter(), -10.0

    def observe(event: str, data: dict) -> None:
        nonlocal last
        if event == "rl_step":
            row = {
                k: v
                for k, v in data.items()
                if k
                not in ("population", "gateway", "controller", "streams", "stagnation", "context")
            }
            step_stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        gateway = data.get("gateway")
        if gateway is None:
            return
        elapsed = time.perf_counter() - clock
        if event == "candidate":
            structures.candidate(data["candidate"], gateway.counts["ssgs"])
        if event == "initial_ready":
            structures.initialized = True
            write_candidates(out / "initial.json.gz", data["population"])
        if event == "initial_ready" or thresholds and elapsed >= thresholds[0]:
            target = "initialization" if event == "initial_ready" else thresholds.pop(0)
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
                    phase="search" if evidence else "initialization",
                ),
            )
            last = elapsed

    try:
        problem = load_instance(root / spec["input"])
        clock = time.perf_counter()
        result = run_policy(
            problem,
            arm,
            cfg,
            evidence,
            observer=observe,
            parameters=manifest["learning_parameters"],
        )
        step_stream.close()
        audit = time.perf_counter()
        for candidate in result.population + result.archive.members:
            assert candidate.evaluation.feasible
            assert evaluate(problem, candidate.genotype, candidate.schedule) == candidate.evaluation
        audit_seconds = time.perf_counter() - audit
        snapshots.append(
            dict(
                target=spec["seconds"],
                elapsed=result.elapsed,
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
                counts=dict(result.gateway.counts),
                seconds=dict(result.gateway.seconds),
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
                completed=time.time(),
                requested_seconds=spec["seconds"],
                production_seconds=result.elapsed,
                audit_seconds=audit_seconds,
                initialization=evidence,
                learning_parameters=manifest["learning_parameters"],
                offspring=result.offspring_count,
                termination=result.termination_reason,
                population=len(result.population),
                archive=len(result.archive.members),
                final_exact_verified=len(result.population) + len(result.archive.members),
                counts=dict(result.gateway.counts),
                max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                structural_diagnostics=structures.summary(
                    result.population, result.archive.members
                ),
            ),
        )
        atomic_json(
            out / "complete.json",
            dict(
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
            ),
        )
        assert complete(root, key, arm, manifest["source"])
    except BaseException:
        step_stream.close()
        atomic_json(
            out / "failure.json", dict(updated=time.time(), traceback=traceback.format_exc())
        )
        raise


def queue(root: Path, manifest: dict) -> list[tuple[str, str]]:
    """Interleave scales for every arm round, with within-input Williams rotation."""
    specs = [
        json.loads((root / "specs" / (key + ".json")).read_text()) for key in manifest["blocks"]
    ]
    return [(s["key"], a) for s in specs for a in s["order"]]


def cpus_available(parallel: int) -> list[int]:
    """Prefer one available logical CPU per physical core before SMT siblings."""
    groups: dict[tuple[str, str], list[int]] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        p = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        groups.setdefault(
            ((p / "physical_package_id").read_text(), (p / "core_id").read_text()), []
        ).append(cpu)
    cpus = [v[0] for v in groups.values()] + [c for v in groups.values() for c in v[1:]]
    if not 1 <= parallel <= min(96, len(cpus)):
        raise ValueError("Unsupported CPU parallelism")
    return cpus[:parallel]


def supervise(root: Path, parallel: int) -> None:
    """Run-level locked resume; real failures stop new dispatch without retry."""
    import fcntl

    lock = (root / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = validate_inputs(root)
    if list(root.glob("runs/*/*/failure.json")) or (root / "failure.json").exists():
        raise ValueError("Recorded failure; no automatic retry")
    for key in manifest["blocks"]:
        check = json.loads((root / "preflight" / (key + ".json")).read_text())
        assert check["source"] == manifest["source"] and check["host"] == manifest["host"]
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"worker" in command and str(root).encode() in command:
            raise ValueError("Existing same-root worker")
    pending = []
    done = 0
    for key, arm in queue(root, manifest):
        if complete(root, key, arm, manifest["source"]):
            done += 1
            continue
        out = root / "runs" / key / arm
        if out.exists():
            target = root / "quarantine" / (key + "_" + arm + "_" + str(time.time_ns()))
            target.parent.mkdir(exist_ok=True)
            out.rename(target)
        pending.append((key, arm))
    cpus = cpus_available(parallel)
    if not (root / "start.json").exists():
        now = time.time()
        atomic_json(
            root / "start.json", dict(started=now, deadline=now + manifest["window_seconds"])
        )
    start = json.loads((root / "start.json").read_text())
    active: dict = {}
    failed = False
    while pending or active:
        stopped = failed or (root / "stop.request").exists()
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
        cgroup_ok = True
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                current = Path("/sys/fs/cgroup" + line[3:]) / "memory.current"
                if current.exists():
                    cgroup_ok = int(current.read_text()) < 160000000000
        for slot in range(parallel):
            if (
                slot in active
                or stopped
                or not pending
                or available < 220 * 1024**3
                or not cgroup_ok
                or shutil.disk_usage(root).free < 30 * 1024**3
            ):
                continue
            key, arm = pending[0]
            spec = json.loads((root / "specs" / (key + ".json")).read_text())
            if time.time() + spec["seconds"] + 900 > start["deadline"]:
                continue
            pending.pop(0)
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
                cpu=cpus[slot],
                started=time.time(),
                limit=spec["seconds"] + 1800,
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
                pending=len(pending),
                failure=failed,
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
            ),
        )
        if not active and (stopped or time.time() + 3300 > start["deadline"]):
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "run", "worker", "validate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--parallel", type=int, default=48)
    parser.add_argument("--key")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--cpu", type=int, default=0)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.root, args.historical_root, args.pilot)))
    elif args.action == "preflight":
        print(json.dumps(preflight(args.root, args.key)))
    elif args.action == "run":
        supervise(args.root, args.parallel)
    elif args.action == "worker":
        worker(args.root, args.key, args.arm, args.cpu)
    else:
        m = validate_inputs(args.root)
        print(
            json.dumps(
                dict(
                    validated=sum(
                        complete(args.root, k, a, m["source"]) for k in m["blocks"] for a in ARMS
                    ),
                    total=m["runs"],
                )
            )
        )
