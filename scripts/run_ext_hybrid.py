"""Sealed hybrid-initialized external baselines on the original component33 input bytes.

Five ``EXTH_`` arms (weakened hybrid initializer + per-generation old Polish on a
random 20% of the population) with the original seed 16101 and the original
300/600/1200/2400 s budgets. Our own algorithms are compared from their own
batches on the same input bytes; no reference arm is rerun here. Reuses the
slot-assignment runner's hashing and streaming helpers.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from run_slot_assignment import sha, source_hash, write_candidates

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.ext_hybrid import (
    ARMS,
    POLICY,
    arm_config,
    hybrid_initializer,
    run_arm,
)
from geo_llm_scheduler.experiments.mechanism5 import StructuralObserver
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.rng import RNGManager

REQUIRED = (
    "config.json",
    "summary.json",
    "initial.json.gz",
    "population.json.gz",
    "archive.json.gz",
    "anytime.json",
    "diagnostics.json",
)
SEED = 16101
PILOT_SECONDS = {20: 45, 50: 60, 100: 120, 200: 300}
FRACTIONS = (0.1, 0.25, 0.5, 0.75, 0.9)
# gateway exact evaluations of the hybrid initializer: 20 heuristic + 1 BILL + 79 random
EXPECTED_INIT_EXACT = 100


def initializer_policy(arm: str) -> str:
    """All five hybrid arms share one weakened initializer."""
    if arm not in ARMS:
        raise ValueError("Unknown study arm")
    return POLICY


def balanced_order(index: int) -> list[str]:
    """Rotate the five arms by block and reverse every other cycle."""
    count = len(ARMS)
    order = [ARMS[(j + index) % count] for j in range(count)]
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
    for old_key in originals:
        old_path = historical / "specs" / (old_key + ".json")
        assert sha(old_path) == old_manifest["files"][old_path.relative_to(historical).as_posix()]
        old = json.loads(old_path.read_text())
        assert old["seed"] == SEED
        original = historical / old["input"]
        assert sha(original) == old["input_sha256"] == old_manifest["files"][old["input"]]
        target = root / old["input"]
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copyfile(original, target)
        assert sha(target) == old["input_sha256"]
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
                "seed",
            )
        }
        spec.update(
            key=old_key,
            old_key=old_key,
            historical_spec_sha256=sha(old_path),
            host=socket.gethostname(),
            order=balanced_order(len(blocks)),
        )
        if pilot:
            spec["seconds"] = PILOT_SECONDS[spec["n"]]
        spec_path = root / "specs" / (old_key + ".json")
        atomic_json(spec_path, spec)
        files[spec["input"]] = sha(target)
        files[spec_path.relative_to(root).as_posix()] = sha(spec_path)
        blocks.append(old_key)
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
        child_address_limit_bytes=3 * 1024**3,
        window_seconds=20 * 3600,
        disk_reserve_bytes=200 * 1024**3,
        initialization={arm: initializer_policy(arm) for arm in ARMS},
        polish="old right-shift Polish on random 20% after every complete generation",
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
    assert manifest["runs"] == len(manifest["arms"]) * len(manifest["blocks"])
    for name, expected in manifest["files"].items():
        assert sha(root / name) == expected, name
    return manifest


def preflight(root: Path, key: str) -> dict:
    """Fresh independent decode/exact audit of the shared hybrid initializer."""
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["source"] == source_hash()
    spec_path = root / "specs" / (key + ".json")
    assert sha(spec_path) == manifest["files"][spec_path.relative_to(root).as_posix()]
    spec = json.loads(spec_path.read_text())
    assert sha(root / spec["input"]) == spec["input_sha256"]
    problem = load_instance(root / spec["input"])
    policies = {}
    for arm in ("EXTH_MOEAD",):
        cfg = arm_config(arm, spec["seed"], spec["seconds"])
        captured: list = []
        gateway = EvaluationGateway(problem, Archive())
        gateway.candidate_sink = captured.append
        evidence: dict = {}
        members = hybrid_initializer(problem, cfg, evidence)(gateway, RNGManager(cfg.seed))
        assert len(members) == len({c.genotype for c in members}) == 100
        for candidate in captured:
            assert decode(problem, candidate.genotype) == candidate.schedule
            assert evaluate(problem, candidate.genotype, candidate.schedule) == candidate.evaluation
            assert candidate.evaluation.feasible
        policy = initializer_policy(arm)
        assert len(captured) == gateway.counts["exact"] == EXPECTED_INIT_EXACT
        policies[policy] = dict(evidence=evidence, fresh_exact_verified=len(captured))
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
    assert summary["termination"] == "time_budget" and summary["offspring"] > 0
    assert (
        summary["population"] == 100 and summary["final_exact_verified"] == 100 + summary["archive"]
    )
    assert summary["counts"].get("polish_calls", 0) > 0
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

    manifest = json.loads((root / "manifest.json").read_text())
    limit = manifest["child_address_limit_bytes"]
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    os.sched_setaffinity(0, {cpu})
    lock_path = root / "locks" / (key + "_" + arm + ".lock")
    lock_path.parent.mkdir(exist_ok=True)
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
    thresholds = [spec["seconds"] * f for f in FRACTIONS]
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
        if event == "initial_ready" or thresholds and elapsed >= thresholds[0]:
            target = "initialization" if event == "initial_ready" else thresholds.pop(0)
            snapshots.append(
                dict(
                    target=target,
                    elapsed=elapsed,
                    objectives=[c.evaluation.objectives for c in gateway.archive.members],
                    population=None
                    if population is None
                    else [c.evaluation.objectives for c in population],
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
                    polish_calls=gateway.counts["polish_calls"],
                    archive=len(gateway.archive.members),
                    phase="search" if evidence else "initialization",
                ),
            )
            last = elapsed

    try:
        problem = load_instance(root / spec["input"])
        clock = time.perf_counter()
        result = run_arm(problem, cfg, arm, evidence, observer=observe)
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
                        complete(args.root, k, a, m["source"])
                        for k in m["blocks"]
                        for a in m["arms"]
                    ),
                    total=m["runs"],
                )
            )
        )
