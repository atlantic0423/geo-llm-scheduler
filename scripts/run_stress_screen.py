"""Baseline-only screening of stress-generator v2 candidates (no LSV2 run is involved).

Each of the 64 candidates (16 design points x 20/50/100/200 jobs) gets three plain
BILL4+MOEA/D (BILL_PERMUTED) runs: two seeds at the original budget and one at
twice the budget. Headroom = HV gap of the 1x runs to the union of all three; the
panel is selected from these baseline numbers only (see docs/experiments).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from run_lsv2 import sha, source_hash, write_candidates, write_json_gz

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.component_random import charged_initializer
from geo_llm_scheduler.experiments.slot_assignment import run_arm, study_config
from geo_llm_scheduler.experiments.stress_instances import candidate_specs, stress_problem
from geo_llm_scheduler.io.loaders import load_instance, save_instance
from geo_llm_scheduler.scheduling.ssgs import decode
from geo_llm_scheduler.utils.rng import RNGManager

ARMS = ("BILL_A", "BILL_B", "BILL_C2X")
RUNS = {"BILL_A": (16801, 1), "BILL_B": (16802, 1), "BILL_C2X": (16803, 2)}
BUDGETS = {20: 300, 50: 600, 100: 1200, 200: 2400}
PILOT_SECONDS = {20: 45, 50: 60, 100: 120, 200: 240}
REQUIRED = (
    "config.json",
    "summary.json",
    "archive.json.gz",
    "population.json.gz",
    "archive_events.json.gz",
)


def arm_config(arm: str, base_seconds: float):
    """Unchanged BILL_PERMUTED reference configuration for one screening run."""
    seed, scale = RUNS[arm]
    return study_config("BILL_RANDOM", seed, base_seconds * scale)


def prepare(root: Path, pilot: bool = False) -> dict:
    """Generate every candidate instance, seal input/spec SHA and the run matrix."""
    if root.exists():
        raise FileExistsError("Campaign root must be new")
    (root / "inputs").mkdir(parents=True)
    files, blocks = {}, []
    specs = candidate_specs()
    if pilot:
        specs = [s for s in specs if s["key"].endswith("_d15")]
    for s in specs:
        problem = stress_problem(s["n"], s["regions"], s["levels"], s["seed"])
        target = root / "inputs" / (s["key"] + ".json")
        save_instance(problem, target)
        base = PILOT_SECONDS[s["n"]] if pilot else BUDGETS[s["n"]]
        spec = dict(
            s,
            input=target.relative_to(root).as_posix(),
            input_sha256=sha(target),
            base_seconds=base,
            seconds=2 * base,  # dispatch ordering and kill limit use the longest arm
            machines=len(problem.instances),
            host=socket.gethostname(),
            order=list(ARMS),
        )
        spec_path = root / "specs" / (s["key"] + ".json")
        atomic_json(spec_path, spec)
        files[spec["input"]] = spec["input_sha256"]
        files[spec_path.relative_to(root).as_posix()] = sha(spec_path)
        blocks.append(s["key"])
    manifest = dict(
        source=source_hash(),
        host=socket.gethostname(),
        arms=list(ARMS),
        runs_spec=RUNS,
        blocks=blocks,
        runs=len(blocks) * len(ARMS),
        kind="pilot" if pilot else "formal",
        files=files,
        commit=subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
        ).strip(),
        child_address_limit_bytes=3 * 1024**3,
        window_seconds=6 * 3600,
        memory_reserve_bytes=200 * 1024**3,
        disk_reserve_bytes=200 * 1024**3,
        initialization="90+10_BILL4_P100",
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
    assert manifest["runs"] == len(ARMS) * len(manifest["blocks"])
    for name, expected in manifest["files"].items():
        assert sha(root / name) == expected, name
    return manifest


def preflight(root: Path, key: str) -> dict:
    """Fresh independent decode/exact audit of the BILL4 initialization on one candidate."""
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["source"] == source_hash()
    spec = json.loads((root / "specs" / (key + ".json")).read_text())
    assert sha(root / spec["input"]) == spec["input_sha256"]
    problem = load_instance(root / spec["input"])
    regenerated = stress_problem(spec["n"], spec["regions"], spec["levels"], spec["seed"])
    assert regenerated == problem, "Instance bytes do not match the generator"
    checks = {}
    for arm in ARMS:
        cfg = arm_config(arm, spec["base_seconds"])
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
        checks[arm] = dict(fresh_exact_verified=len(captured), evidence=evidence)
    result = dict(
        source=manifest["source"],
        key=key,
        input_sha256=spec["input_sha256"],
        host=socket.gethostname(),
        arms=checks,
        completed=time.time(),
    )
    atomic_json(root / "preflight" / (key + ".json"), result)
    return result


def complete(root: Path, key: str, arm: str, source: str) -> bool:
    """Require exact-audited artifacts with matching provenance and initialization."""
    out = root / "runs" / key / arm
    if not (out / "complete.json").exists():
        return False
    spec_path = root / "specs" / (key + ".json")
    spec = json.loads(spec_path.read_text())
    marker = json.loads((out / "complete.json").read_text())
    assert marker["source"] == source and marker["host"] == spec["host"] == socket.gethostname()
    assert marker["key"] == key and marker["arm"] == arm and set(marker["files"]) == set(REQUIRED)
    assert marker["input_sha256"] == spec["input_sha256"] == sha(root / spec["input"])
    for name, record in marker["files"].items():
        assert (out / name).stat().st_size == record["size"] and sha(out / name) == record["sha256"]
    summary = json.loads((out / "summary.json").read_text())
    cfg = arm_config(arm, spec["base_seconds"])
    assert json.loads((out / "config.json").read_text()) == json.loads(json.dumps(asdict(cfg)))
    assert summary["termination"] == "time_budget" and summary["offspring"] > 0
    assert summary["final_exact_verified"] == summary["population"] + summary["archive"]
    check = json.loads((root / "preflight" / (key + ".json")).read_text())
    expected = check["arms"][arm]["evidence"]
    for field in ("initial_population_sha256", "initial_archive_sha256"):
        assert summary["initialization"][field] == expected[field]
    return True


def worker(root: Path, key: str, arm: str, cpu: int) -> None:
    """Run one charged baseline production run and audit every saved solution."""
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
    spec = json.loads((root / "specs" / (key + ".json")).read_text())
    assert sha(root / spec["input"]) == spec["input_sha256"]
    out = root / "runs" / key / arm
    out.mkdir(parents=True, exist_ok=False)
    cfg = arm_config(arm, spec["base_seconds"])
    atomic_json(out / "config.json", asdict(cfg))
    evidence: dict = {}
    events: list[tuple[int, float, float, float]] = []
    inserted = 0
    clock, last, started = time.perf_counter(), -10.0, time.time()

    def observe(event: str, data: dict) -> None:
        nonlocal inserted, last
        gateway = data.get("gateway")
        if gateway is None:
            return
        elapsed = time.perf_counter() - clock
        if event == "candidate" and gateway.archive.insertions != inserted:
            inserted = gateway.archive.insertions
            flow, bill = data["candidate"].evaluation.objectives
            events.append((gateway.counts["exact"], elapsed, flow, bill))
        if elapsed - last >= 10:
            atomic_json(
                out / "heartbeat.json",
                dict(
                    pid=os.getpid(),
                    updated=time.time(),
                    elapsed=elapsed,
                    exact=gateway.counts["exact"],
                    archive=len(gateway.archive.members),
                ),
            )
            last = elapsed

    try:
        problem = load_instance(root / spec["input"])
        clock = time.perf_counter()
        result = run_arm(problem, "BILL_RANDOM", cfg, evidence, observer=observe)
        for candidate in result.population + result.archive.members:
            assert candidate.evaluation.feasible
            assert evaluate(problem, candidate.genotype, candidate.schedule) == candidate.evaluation
        write_candidates(out / "population.json.gz", result.population)
        write_candidates(out / "archive.json.gz", result.archive.members)
        write_json_gz(out / "archive_events.json.gz", events)
        atomic_json(
            out / "summary.json",
            dict(
                arm=arm,
                key=key,
                source=manifest["source"],
                commit=manifest["commit"],
                seed=cfg.seed,
                cpu=cpu,
                pid=os.getpid(),
                started=started,
                completed=time.time(),
                requested_seconds=cfg.seconds,
                production_seconds=result.elapsed,
                initialization=evidence,
                offspring=result.offspring_count,
                termination=result.termination_reason,
                population=len(result.population),
                archive=len(result.archive.members),
                final_exact_verified=len(result.population) + len(result.archive.members),
                counts=dict(result.gateway.counts),
                max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
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
                files={
                    n: dict(size=(out / n).stat().st_size, sha256=sha(out / n)) for n in REQUIRED
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
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--key")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--cpu", type=int, default=0)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.root, args.pilot)))
    elif args.action == "preflight":
        print(json.dumps(preflight(args.root, args.key)))
    elif args.action == "worker":
        worker(args.root, args.key, args.arm, args.cpu)
    else:
        m = validate_inputs(args.root)
        done = sum(complete(args.root, k, a, m["source"]) for k in m["blocks"] for a in m["arms"])
        print(json.dumps(dict(validated=done, total=m["runs"])))
