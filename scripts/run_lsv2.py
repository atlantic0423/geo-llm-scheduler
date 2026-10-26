"""Sealed four-arm LSV2 development study on the original component33 input bytes.

Arms: BILL_RANDOM and MAIN_RANDOM (unchanged references), LSV2 and LSV2_NOGAP
(time-share LS on the BILL engine; gap vs Flow-end seed rule). Every run also
logs Archive insertions with their exact count and time for equal-exact and
equal-wall-clock analyses; LSV2 runs additionally log every LS step.
"""

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
from geo_llm_scheduler.experiments.lsv2_study import (
    ARMS,
    LSV2_ARMS,
    lsv2_config,
    operator_config,
    run_arm,
)
from geo_llm_scheduler.experiments.lsv2_study import (
    study_config as arm_config,
)
from geo_llm_scheduler.experiments.mechanism5 import StructuralObserver
from geo_llm_scheduler.experiments.runner import solution
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.search import LSV2Search
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.rng import RNGManager

BASE_REQUIRED = (
    "config.json",
    "summary.json",
    "initial.json.gz",
    "population.json.gz",
    "archive.json.gz",
    "archive_events.json.gz",
    "qtable.json",
    "anytime.json",
    "diagnostics.json",
)
LSV2_REQUIRED = ("lsv2_steps.json.gz", "lsv2_summary.json")
FORMAL_SEEDS = (16601,)
PILOT_SECONDS = {20: 45, 50: 60, 100: 120, 200: 240}
DEFAULT_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 0.9)


def required(arm: str) -> tuple[str, ...]:
    """Artifacts of one arm; LSV2 arms add their per-step log and search summary."""
    return BASE_REQUIRED + (LSV2_REQUIRED if arm in LSV2_ARMS else ())


def arm_record(arm: str, seed: int, seconds: float) -> dict:
    """Full declared configuration written to ``config.json``."""
    ls = lsv2_config(arm)
    return dict(
        engine=asdict(arm_config(arm, seed, seconds)),
        lsv2=None if ls is None else asdict(ls),
        operators=None if ls is None else asdict(operator_config(seed, seconds)),
    )


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


def write_json_gz(path: Path, value: object) -> None:
    """Compact gzip JSON for large row logs."""
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, separators=(",", ":"))


def write_candidates(path: Path, members: list[Candidate]) -> None:
    """Stream complete immutable candidates to avoid a second JSON image."""
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write("[")
        for i, candidate in enumerate(members):
            if i:
                stream.write(",")
            json.dump(solution(candidate), stream, separators=(",", ":"))
        stream.write("]")


def balanced_order(index: int) -> list[str]:
    """Rotate the four treatments using predeclared Williams entry orders."""
    count = len(ARMS)
    row = [0] + [(j + 1) // 2 if j % 2 else count - j // 2 for j in range(1, count)]
    order = [ARMS[(j + index % count) % count] for j in row]
    return order if (index // count) % 2 == 0 else order[::-1]


def prepare(root: Path, historical: Path, pilot: bool = False) -> dict:
    """Copy only SHA-bound component33 input bytes; never regenerate easier cases.

    The pilot uses the first four historical blocks (one per size) at short budgets.
    """
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
        lsv2_configs={arm: arm_record(arm, FORMAL_SEEDS[0], 1.0)["lsv2"] for arm in ARMS},
        child_address_limit_bytes=3 * 1024**3,
        window_seconds=6 * 3600,
        polish_policy="TWO_FOCUS for MAIN_RANDOM only; LSV2 arms have no Polish",
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
    assert manifest["runs"] == len(manifest["arms"]) * len(manifest["blocks"])
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
    cfg = arm_config("BILL_RANDOM", spec["seed"], spec["seconds"])
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
    assert len(captured) == gateway.counts["exact"] == 140
    result = dict(
        source=manifest["source"],
        input_sha256=spec["input_sha256"],
        key=key,
        policies=dict(bill4=dict(evidence=evidence, fresh_exact_verified=len(captured))),
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
    assert marker["spec_sha256"] == sha(spec_path) and set(marker["files"]) == set(required(arm))
    for name, record in marker["files"].items():
        assert (out / name).stat().st_size == record["size"] and sha(out / name) == record["sha256"]
    summary = json.loads((out / "summary.json").read_text())
    declared = arm_record(arm, spec["seed"], spec["seconds"])
    assert json.loads((out / "config.json").read_text()) == json.loads(json.dumps(declared))
    assert summary["source"] == source and summary["arm"] == arm
    assert summary["termination"] == "time_budget" and summary["offspring"] > 0
    if arm in LSV2_ARMS:
        ls = json.loads((out / "lsv2_summary.json").read_text())
        assert (
            ls["config"] == json.loads(json.dumps(declared["lsv2"]))
            and ls["counts"].get("steps", 0) > 0
        )
        assert summary["lsv2"]["steps"] == ls["counts"]["steps"]
    q = json.loads((out / "qtable.json").read_text())
    assert q["q"] and all(len(row) == 8 for row in q["q"])
    assert (
        summary["population"] == 100 and summary["final_exact_verified"] == 100 + summary["archive"]
    )
    check = json.loads((root / "preflight" / (key + ".json")).read_text())
    assert check["source"] == source and check["input_sha256"] == spec["input_sha256"]
    assert summary["initialization"]["policy"] == "bill4"
    expected = check["policies"]["bill4"]["evidence"]
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
    atomic_json(out / "config.json", arm_record(arm, spec["seed"], spec["seconds"]))
    evidence: dict = {}
    snapshots: list = []
    # (exact count, elapsed, Flow, Bill) of every Archive insertion; the Archive
    # front at exact count k is the non-dominated set of rows with exact <= k.
    events: list[tuple[int, float, float, float]] = []
    inserted = 0
    searches: list[LSV2Search] = []
    structures = StructuralObserver()
    thresholds = [spec["seconds"] * f for f in DEFAULT_FRACTIONS]
    started, clock, last = time.time(), time.perf_counter(), -10.0

    def observe(event: str, data: dict) -> None:
        nonlocal last, inserted
        gateway = data.get("gateway")
        if gateway is None:
            return
        elapsed = time.perf_counter() - clock
        if event == "candidate":
            structures.candidate(data["candidate"], gateway.counts["ssgs"])
            if gateway.archive.insertions != inserted:
                inserted = gateway.archive.insertions
                flow, bill = data["candidate"].evaluation.objectives
                events.append((gateway.counts["exact"], elapsed, flow, bill))
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
                    exact=gateway.counts["exact"],
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
                    ls_seconds=searches[0].ls_seconds if searches else None,
                    phase="search" if evidence else "initialization",
                ),
            )
            last = elapsed

    try:
        problem = load_instance(root / spec["input"])
        clock = time.perf_counter()
        result = run_arm(problem, arm, cfg, evidence, observer=observe, search_sink=searches.append)
        audit = time.perf_counter()
        for candidate in result.population + result.archive.members:
            assert candidate.evaluation.feasible
            assert evaluate(problem, candidate.genotype, candidate.schedule) == candidate.evaluation
        audit_seconds = time.perf_counter() - audit
        snapshots.append(
            dict(
                target=spec["seconds"],
                elapsed=result.elapsed,
                exact=result.gateway.counts["exact"],
                objectives=[c.evaluation.objectives for c in result.archive.members],
                population=[c.evaluation.objectives for c in result.population],
            )
        )
        write_candidates(out / "population.json.gz", result.population)
        write_candidates(out / "archive.json.gz", result.archive.members)
        write_json_gz(out / "archive_events.json.gz", events)
        lsv2: dict | None = None
        if arm in LSV2_ARMS:
            search = searches[0]
            write_json_gz(out / "lsv2_steps.json.gz", search.steps)
            ls_summary = search.summary()
            atomic_json(out / "lsv2_summary.json", ls_summary)
            search_seconds = result.elapsed - (search.search_start or 0.0)
            lsv2 = dict(
                steps=ls_summary["counts"].get("steps", 0),
                trajectories=search.trajectories,
                ls_seconds=search.ls_seconds,
                search_seconds=search_seconds,
                share=search.ls_seconds / search_seconds if search_seconds > 0 else None,
            )
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
                lsv2=lsv2,
                archive_events=len(events),
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
                    for name in required(arm)
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
