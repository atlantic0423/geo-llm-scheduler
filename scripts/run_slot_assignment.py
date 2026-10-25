"""Sealed initial slot-assignment study on the original component33 development input bytes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import socket
import subprocess
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
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.experiments.slot_assignment import (
    ARMS,
    reference_policy,
    run_arm,
    slot_policy,
)
from geo_llm_scheduler.experiments.slot_assignment import (
    study_config as arm_config,
)
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
)
FORMAL_SEEDS = (16501,)
PILOT_SECONDS = {20: 30, 50: 45, 100: 60, 200: 90}


def sha(path: Path) -> str:
    """Hash a regular file without retaining its contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hash() -> str:
    """Bind the installed package and this runner with canonical newlines."""
    src = Path(__file__).resolve().parents[1] / "src"
    paths = [(p.relative_to(src).as_posix(), p) for p in sorted(src.rglob("*.py"))]
    paths.extend(("scripts/" + p.name, p) for p in sorted(Path(__file__).parent.glob("*.py")))
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
    """All study arms use the same 90+10 BILL four-choice initializer."""
    if arm not in ARMS:
        raise ValueError("Unknown policy arm")
    return "bill4"


def balanced_order(index: int) -> list[str]:
    """Rotate the four treatments using predeclared Williams entry orders."""
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
    for seed in FORMAL_SEEDS:
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
                spec["seconds"] = PILOT_SECONDS[spec["n"]]
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
        slot_policies={arm: slot_policy(arm) for arm in ARMS},
        reference_policies={arm: reference_policy(arm) for arm in ARMS},
        child_address_limit_bytes=2 * 1024**3,
        window_seconds=6 * 3600,
        polish_policy="TWO_FOCUS_except_BILL_control",
        disk_reserve_bytes=200 * 1024**3,
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
    for arm in ("BILL_RANDOM",):
        cfg = arm_config(arm, spec["seed"], spec["seconds"])
        captured: list = []
        gateway = EvaluationGateway(problem, Archive())
        gateway.candidate_sink = captured.append
        evidence: dict = {}
        members = charged_initializer(problem, cfg, "BILL_MOEAD", evidence)(
            gateway, RNGManager(cfg.seed)
        )
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
    assert summary["slot_policy"] == slot_policy(arm)
    assert summary["reference_policy"] == reference_policy(arm)
    assert summary["termination"] == "time_budget" and summary["offspring"] > 0
    q = json.loads((out / "qtable.json").read_text())
    assert q["q"] and all(len(row) == 8 for row in q["q"])
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
    structures = StructuralObserver()
    thresholds = [spec["seconds"] * f for f in (0.1, 0.25, 0.5, 0.75, 0.9)]
    started, clock, last = time.time(), time.perf_counter(), -10.0

    def observe(event: str, data: dict) -> None:
        nonlocal last
        gateway = data.get("gateway")
        if gateway is None:
            return
        elapsed = time.perf_counter() - clock
        if event == "candidate":
            structures.candidate(data["candidate"], gateway.counts["ssgs"])
        if event == "initial_ready":
            structures.initialized = True
            write_candidates(out / "initial.json.gz", data["population"])
        population = data.get("population")
        if (
            event == "initial_ready"
            or thresholds
            and elapsed >= thresholds[0]
            and population is not None
        ):
            target = "initialization" if event == "initial_ready" else thresholds.pop(0)
            snapshots.append(
                dict(
                    target=target,
                    elapsed=elapsed,
                    objectives=[c.evaluation.objectives for c in gateway.archive.members],
                    population=[c.evaluation.objectives for c in population],
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
        result = run_arm(problem, arm, cfg, evidence, observer=observe)
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
                population=[c.evaluation.objectives for c in result.population],
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
                slot_policy=slot_policy(arm),
                reference_policy=reference_policy(arm),
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
        atomic_json(
            out / "failure.json", dict(updated=time.time(), traceback=traceback.format_exc())
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "worker", "validate"))
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
