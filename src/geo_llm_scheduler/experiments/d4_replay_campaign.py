"""Frozen, task/unit-resumable offline D4 prototypes, disconnected from online MOEA/D."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import (
    atomic_json,
    available_memory_gb,
    free_disk_gb,
    process_peak_rss_gb,
)
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.campaign.worker import TUPLE_FIELDS
from geo_llm_scheduler.experiments.d1_campaign import campaign_lease
from geo_llm_scheduler.experiments.d4_peak import audit_residual
from geo_llm_scheduler.experiments.d4_prototypes import (
    ResearchPeak,
    cached_certificate,
    certificate_cache,
)
from geo_llm_scheduler.experiments.p1_observation import restore_candidate
from geo_llm_scheduler.experiments.p2_worker import campaign_path, canonical_source_hash
from geo_llm_scheduler.experiments.runner import digest, git_metadata, solution
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.operators.base import ProposalBatch
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition
from geo_llm_scheduler.utils.numeric import TOL

PROTOCOL = "docs/experiments/d4_two_prototypes.md"


def prepare_replay(
    parent: Path,
    root: Path,
    role: str,
    seeds: int = 4,
    cycles: int = 4,
    keys: list[str] | None = None,
) -> dict:
    """Extract only frozen panels; bind verified parent inputs, source and protocol."""
    if role not in {"guard", "positions"} or seeds < 1 or cycles < 1:
        raise ValueError("Invalid replay design")
    if root.exists():
        raise ValueError("Refuse overwriting a replay campaign")
    repo = Path(__file__).resolve().parents[3]
    provenance = git_metadata(repo)
    if provenance["git_dirty"]:
        raise ValueError("Prepare requires a clean committed checkout")
    original = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    selected = original["keys"] if keys is None else keys
    if not set(selected) <= set(original["keys"]):
        raise ValueError("Unknown parent key")
    root.mkdir(parents=True)
    for key in selected:
        specfile = parent / "specs" / f"{key}.json"
        if file_hash(specfile) != original["input_files"][f"specs/{key}.json"]:
            raise ValueError("Parent spec hash mismatch")
        spec = json.loads(specfile.read_text(encoding="utf-8"))
        sample = parent / "samples" / key
        marker = json.loads((sample / "complete.json").read_text(encoding="utf-8"))
        if file_hash(sample / "data.json") != marker["files"]["data.json"]:
            raise ValueError("Parent data artifact mismatch")
        instance = parent / spec["instance"]
        if file_hash(instance) != spec["instance_hash"]:
            raise ValueError("Parent instance hash mismatch")
        target = root / spec["instance"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(instance, target)
        panels = json.loads((sample / "data.json").read_text(encoding="utf-8"))["panels"]
        atomic_json(
            root / "inputs" / f"{key}_panels.json",
            {
                "panels": panels,
                "parent_data_sha256": marker["files"]["data.json"],
                "parent_complete_sha256": file_hash(sample / "complete.json"),
                "parent_spec_sha256": file_hash(specfile),
                "spec": spec,
            },
        )
    shutil.copyfile(repo / PROTOCOL, root / "inputs/protocol.md")
    manifest = {
        "schema": "D4_two_prototypes_v1",
        "role": role,
        "keys": selected,
        "seeds": seeds,
        "cycles": cycles,
        "source_commit": provenance["git_commit"],
        "source_hash": canonical_source_hash(),
        "parent_source": original["source_commit"],
        "parent_manifest_sha256": file_hash(parent / "manifest.json"),
        "scope": "D4 reused development panels; no online HV or novelty claim",
        "input_files": {
            p.relative_to(root).as_posix(): file_hash(p)
            for p in sorted((root / "inputs").glob("*"))
        },
    }
    atomic_json(root / "manifest.json", manifest)
    return manifest


def frozen_replay(root: Path) -> dict:
    """Validate the entire frozen input map and clean source before dispatch."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    provenance = git_metadata(Path(__file__).resolve().parents[3])
    if (
        provenance["git_dirty"]
        or provenance["git_commit"] != manifest["source_commit"]
        or canonical_source_hash() != manifest["source_hash"]
    ):
        raise ValueError("Replay source mismatch")
    for name, sha in manifest["input_files"].items():
        if file_hash(campaign_path(root, name)) != sha:
            raise ValueError("Replay input hash mismatch")
    return manifest


def replay_unit(
    problem: ProblemInstance, panel: dict, config: Config, role: str, seed: int, cycles: int
) -> dict:
    """Pair true construction costs, exact-audit outputs and keep empty-source denominators."""
    operators: dict[str, PeakCoalition | ResearchPeak] = (
        {
            "LEGACY": PeakCoalition(True),
            "REFERENCE": ResearchPeak(),
            "GUARD_ALL": ResearchPeak("all"),
            "GUARD_SINGLE": ResearchPeak("single"),
        }
        if role == "guard"
        else {
            "REFERENCE": ResearchPeak(),
            "RANDOM_SCORED": ResearchPeak(positions="random_scored"),
            "PEAK_RANK": ResearchPeak(positions="peak_rank"),
        }
    )
    measured: dict[str, PeakCoalition | ResearchPeak] = {
        arm: (
            PeakCoalition()
            if isinstance(op, PeakCoalition)
            else ResearchPeak(op.guard, op.positions, diagnose=False)
        )
        for arm, op in operators.items()
    }
    rows, audits = [], 0
    for index, item in enumerate(panel["sources"]):
        source = restore_candidate(item["candidate"])
        if evaluate(problem, source.genotype, source.schedule) != source.evaluation:
            raise ValueError("Fresh source exact mismatch")
        audits += 1
        batches: dict[str, tuple[ProposalBatch, object, str]] = {}
        timing: dict[str, list[dict]] = {arm: [] for arm in operators}
        for cycle in range(cycles):
            order = list(operators)
            random.Random(digest([seed, index, cycle, "arm_order"])).shuffle(order)
            for arm in order:
                rng = random.Random(seed)
                wall, cpu = time.perf_counter(), time.process_time()
                batch = measured[arm].propose(problem, source, 3, config, rng)
                timing[arm].append(
                    {
                        "wall": time.perf_counter() - wall,
                        "cpu": time.process_time() - cpu,
                        "order": order.index(arm),
                    }
                )
                signature = digest([repr(batch.proposals), batch.attempts, rng.getstate()])
                if arm in batches and signature != batches[arm][2]:
                    raise ValueError("Nondeterministic repeated invocation")
                batches[arm] = (batch, rng.getstate(), signature)
        # Trace/counterfactual evidence is collected OUTSIDE construction timing.
        for arm, op in operators.items():
            rng = random.Random(seed)
            audited = op.propose(problem, source, 3, config, rng)
            if (
                digest([repr(audited.proposals), audited.attempts, rng.getstate()])
                != batches[arm][2]
            ):
                raise ValueError("Tracing changed measured operator semantics")
            batches[arm] = (audited, rng.getstate(), batches[arm][2])
        reference, reference_rng, _ = batches["REFERENCE"]
        reference_trace = reference.diagnostics.get("trace", [])
        if role == "guard":
            region = reference.diagnostics.get("region")
            cache = None if region is None else certificate_cache(problem, source, region)
            for arm in ("GUARD_ALL", "GUARD_SINGLE"):
                batch, state, _ = batches[arm]
                if (
                    batch.proposals != reference.proposals
                    or batch.attempts != reference.attempts
                    or state != reference_rng
                ):
                    raise ValueError("Guard changed proposal/budget/RNG semantics")
                traces = batch.diagnostics.get("trace", [])
                for ref, guarded in zip(reference_trace, traces):
                    if ref["members"] != guarded["members"]:
                        raise ValueError("Guard shifted later recipe")
                    members = frozenset(ref["members"])
                    if cache is None or region is None or not members:
                        continue
                    cert = cached_certificate(cache, members)
                    independent = audit_residual(problem, source, members, region)
                    if any(
                        abs(a - b) > TOL.power * 0.01
                        for a, b in zip(independent, cert["lower_windows"])
                    ):
                        raise ValueError("Certificate disagrees with independent integration")
                    if guarded["stage"] == "certificate" and ref["stage"] in {
                        "proposal",
                        "duplicate",
                    }:
                        raise ValueError("False certificate rejection")
                    audits += 1
        else:
            if batches["RANDOM_SCORED"][0].proposals != reference.proposals:
                raise ValueError("Scored control changed first-feasible output")
        outputs, armrows = {}, {}
        for arm, (batch, _, _) in batches.items():
            candidates = []
            for recipe in batch.diagnostics.get("trace", []):
                if "repaired_starts" in recipe:
                    schedule = Schedule(tuple(recipe["repaired_starts"]))
                    exact = evaluate(problem, source.genotype, schedule)
                    repaired_members = set(recipe["members"])
                    if (
                        not exact.feasible
                        or schedule.horizon(problem) > source.schedule.horizon(problem) + TOL.time
                        or any(
                            a != b
                            for o, (a, b) in enumerate(zip(schedule.starts, source.schedule.starts))
                            if o not in repaired_members
                        )
                    ):
                        raise ValueError("Repaired candidate violates exact/freeze/horizon")
                    audits += 1
            for proposal in batch.proposals:
                if proposal.schedule is None:
                    raise ValueError("Timing-only proposal missing schedule")
                exact = evaluate(problem, proposal.genotype, proposal.schedule)
                if not exact.feasible:
                    raise ValueError("Infeasible proposal")
                candidates.append(Candidate(proposal.genotype, proposal.schedule, exact, arm))
                audits += 1
            outputs[arm] = candidates
            armrows[arm] = {
                "timing": timing[arm],
                "attempts": batch.attempts,
                "recipes": batch.diagnostics.get("trace", []),
                "candidates": [solution(c) for c in candidates],
            }
        ideal = tuple(
            min(
                [panel["context"]["ideal"][k], source.evaluation.objectives[k]]
                + [c.evaluation.objectives[k] for cs in outputs.values() for c in cs]
            )
            for k in range(2)
        )
        context = NormalizationContext(ideal, tuple(panel["context"]["maximum"]))
        baseline = context.scalar(source, tuple(panel["weight"]))
        for arm, candidates in outputs.items():
            armrows[arm]["best_scalar_gain"] = max(
                [0.0] + [baseline - context.scalar(c, tuple(panel["weight"])) for c in candidates]
            )
        rows.append({"source_index": index, "arms": armrows, "joint_ideal": ideal})
    return {"seed": seed, "role": role, "rows": rows, "exact_audits": audits}


def validate_replay_job(root: Path, manifest: dict, key: str) -> bool:
    """Require the full declared unit matrix, hashes, clean frozen identity and one host."""
    path = campaign_path(root, f"results/{key}")
    if not (path / "complete.json").exists():
        return False
    marker = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    inp = json.loads((root / "inputs" / f"{key}_panels.json").read_text(encoding="utf-8"))
    expected = {
        f"p{p}_s{s}.json" for p in range(len(inp["panels"])) for s in range(manifest["seeds"])
    }
    if (
        marker["manifest_hash"] != digest(manifest)
        or set(marker["files"]) != expected
        or any(file_hash(path / name) != sha for name, sha in marker["files"].items())
    ):
        raise ValueError("Invalid replay completion evidence")
    for name in expected:
        unit = json.loads((path / name).read_text(encoding="utf-8"))
        if unit["manifest_hash"] != digest(manifest) or unit["host"] != marker["host"]:
            raise ValueError("Replay unit identity/host mismatch")
    return True


def execute_replay(root: Path, key: str) -> None:
    """Resume verified units only; interruption never counts as a completed attempt."""
    manifest = frozen_replay(root)
    if key not in manifest["keys"]:
        raise ValueError("Unknown key")
    with campaign_lease(root / "ops/locks" / f"{key}.lock"):
        if validate_replay_job(root, manifest, key):
            return
        data = json.loads((root / "inputs" / f"{key}_panels.json").read_text(encoding="utf-8"))
        problem = load_instance(root / data["spec"]["instance"])
        config_values: dict[str, Any] = {
            k: tuple(v) if k in TUPLE_FIELDS else v for k, v in data["spec"]["config"].items()
        }
        config = Config(**config_values)
        files, started = {}, time.time()
        for p, panel in enumerate(data["panels"]):
            for s in range(manifest["seeds"]):
                if (root / "stop.request").exists():
                    return
                name = f"p{p}_s{s}.json"
                path = root / "results" / key / name
                sidecar = path.with_suffix(".sha.json")
                seed = int(digest([key, p, s, "D4 prototypes"])[:16], 16)
                if sidecar.exists():
                    marker = json.loads(sidecar.read_text(encoding="utf-8"))
                    if file_hash(path) != marker["sha256"]:
                        raise ValueError("Checkpoint artifact hash mismatch")
                    unit = json.loads(path.read_text(encoding="utf-8"))
                    if (
                        unit["manifest_hash"] != digest(manifest)
                        or unit["host"] != platform.node()
                        or unit["seed"] != seed
                    ):
                        raise ValueError("Checkpoint identity/host mismatch")
                else:
                    if path.exists():
                        quarantine = root / "quarantine" / f"{time.time_ns()}_{key}_{name}"
                        quarantine.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(path), quarantine)
                    unit = replay_unit(
                        problem, panel, config, manifest["role"], seed, manifest["cycles"]
                    )
                    unit.update(
                        manifest_hash=digest(manifest),
                        host=platform.node(),
                        panel=p,
                        repeat=s,
                        source_commit=manifest["source_commit"],
                    )
                    atomic_json(path, unit)
                    atomic_json(sidecar, {"sha256": file_hash(path)})
                files[name] = file_hash(path)
                atomic_json(
                    root / "ops/heartbeats" / f"{key}.json",
                    {
                        "pid": os.getpid(),
                        "host": platform.node(),
                        "timestamp": time.time(),
                        "units": len(files),
                        "total": len(data["panels"]) * manifest["seeds"],
                        "elapsed": time.time() - started,
                        "peak_rss_gib": process_peak_rss_gb(),
                    },
                )
        atomic_json(
            root / "results" / key / "complete.json",
            {
                "manifest_hash": digest(manifest),
                "host": platform.node(),
                "files": files,
                "elapsed": time.time() - started,
                "peak_rss_gib": process_peak_rss_gb(),
            },
        )
        if not validate_replay_job(root, manifest, key):
            raise ValueError("Completion verification failed")


def supervise_replay(root: Path, workers: int) -> None:
    """Exclusive supervisor; resource guard, graceful stop and failure-without-retry."""
    manifest = frozen_replay(root)
    if workers < 1:
        raise ValueError("Invalid concurrency")
    with campaign_lease(root / "ops/supervisor.lock"):
        if list((root / "ops/failures").glob("*.json")):
            raise ValueError("Preserved failure: explicit review required")
        done = {key for key in manifest["keys"] if validate_replay_job(root, manifest, key)}
        active: dict[str, subprocess.Popen] = {}
        launched: dict[str, float] = {}
        failed, started = False, time.time()
        while len(done) < len(manifest["keys"]) or active:
            for key, child in list(active.items()):
                if child.poll() is None and time.time() - launched[key] > 7200:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)
                    failed = True
                    atomic_json(
                        root / "ops/failures" / f"timeout_{key}.json",
                        {"timestamp": time.time(), "pid": child.pid, "reason": "2h task deadline"},
                    )
                if child.poll() is None:
                    continue
                del active[key]
                if child.returncode == 0 and validate_replay_job(root, manifest, key):
                    done.add(key)
                elif not (root / "stop.request").exists():
                    failed = True
                    atomic_json(
                        root / "ops/failures" / f"supervisor_{key}.json",
                        {"exit": child.returncode, "timestamp": time.time()},
                    )
            stopping = failed or (root / "stop.request").exists()
            memory = available_memory_gb()
            cgroup = Path("/sys/fs/cgroup")
            if (cgroup / "memory.max").exists() and (
                cgroup / "memory.max"
            ).read_text().strip().isdigit():
                memory = min(
                    memory,
                    (
                        int((cgroup / "memory.max").read_text())
                        - int((cgroup / "memory.current").read_text())
                    )
                    / 1024**3,
                )
            if not stopping and memory > 6 and free_disk_gb(root) > 8:
                for key in manifest["keys"]:
                    if len(active) >= workers:
                        break
                    if key in done or key in active:
                        continue
                    log = root / "logs" / f"{key}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open("ab") as stream:
                        launched[key] = time.time()
                        active[key] = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "geo_llm_scheduler.experiments.d4_replay_campaign",
                                "worker",
                                "--root",
                                str(root),
                                "--key",
                                key,
                            ],
                            cwd=Path(__file__).resolve().parents[3],
                            stdout=stream,
                            stderr=stream,
                            stdin=subprocess.DEVNULL,
                            env=dict(
                                os.environ,
                                OMP_NUM_THREADS="1",
                                OPENBLAS_NUM_THREADS="1",
                                MKL_NUM_THREADS="1",
                            ),
                        )
            state = "failed_draining" if failed else "draining" if stopping else "running"
            atomic_json(
                root / "ops/status.json",
                {
                    "pid": os.getpid(),
                    "host": platform.node(),
                    "timestamp": time.time(),
                    "started": started,
                    "state": state,
                    "completed": len(done),
                    "total": len(manifest["keys"]),
                    "active": [{"key": k, "pid": v.pid} for k, v in active.items()],
                    "workers": workers,
                    "available_memory_gib": memory,
                },
            )
            if stopping and not active:
                break
            time.sleep(2)
        final: dict[str, Any] = {
            "state": "complete"
            if len(done) == len(manifest["keys"])
            else "failed"
            if failed
            else "paused",
            "timestamp": time.time(),
            "elapsed": time.time() - started,
            "completed": len(done),
            "active": [],
        }
        atomic_json(root / "ops/exit.json", final)
        status = (
            json.loads((root / "ops/status.json").read_text(encoding="utf-8"))
            if (root / "ops/status.json").exists()
            else {}
        )
        atomic_json(root / "ops/status.json", {**status, **final})


def main() -> None:
    """CLI for frozen prepare, worker, supervisor and independent completion validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "worker", "run", "validate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--role", choices=("guard", "positions"))
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--key")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.operation == "prepare":
        if args.parent is None or args.role is None:
            parser.error("prepare requires parent and role")
        print(json.dumps(prepare_replay(args.parent, root, args.role, args.seeds, args.cycles)))
    elif args.operation == "run":
        supervise_replay(root, args.workers)
    elif args.operation == "validate":
        manifest = frozen_replay(root)
        if not all(validate_replay_job(root, manifest, key) for key in manifest["keys"]):
            raise ValueError("Incomplete campaign")
        print(json.dumps({"state": "validated", "jobs": len(manifest["keys"])}))
    else:
        try:
            if args.key is None:
                parser.error("worker requires key")
            execute_replay(root, args.key)
        except Exception:
            atomic_json(
                root / "ops/failures" / f"{args.key}.json",
                {"timestamp": time.time(), "pid": os.getpid(), "traceback": traceback.format_exc()},
            )
            raise


if __name__ == "__main__":
    main()
