"""Frozen terminal-source D1 routing campaign, independent of P2 and the engine."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json, process_peak_rss_gb
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.d1 import extract_intent, perturb_structure
from geo_llm_scheduler.experiments.d1_campaign import campaign_lease
from geo_llm_scheduler.experiments.d1_routing import (
    GROUPS,
    matched_random_routes,
    routing_probe,
)
from geo_llm_scheduler.experiments.p1_observation import restore_candidate, sample_population
from geo_llm_scheduler.experiments.p2_worker import (
    campaign_path,
    canonical_source_hash,
    validate_complete,
)
from geo_llm_scheduler.experiments.runner import digest, git_metadata
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.moead.core import NormalizationContext, maximum, weights
from geo_llm_scheduler.utils.rng import RNGManager

MASTER_SEED = 20261004
DEVELOPMENT_BASES = frozenset(range(810001, 810005)) | frozenset(range(820001, 820005))
HOLDOUT_BASES = frozenset(range(810005, 810013)) | frozenset(range(820005, 820013))
SEEDS = (1101, 2202, 3303)
PILOT_KEYS = (
    "n50_i810001_H_a1101_F6",
    "n50_i810002_T_a2202_F6",
    "n100_i820001_H_a1101_F6",
    "n100_i820002_T_a2202_F6",
)


def _frozen_manifest(root: Path) -> dict:
    expected = json.loads((root / "manifest.sha256.json").read_text(encoding="utf-8"))
    if file_hash(root / "manifest.json") != expected["manifest.json"]:
        raise ValueError("Frozen manifest checksum mismatch")
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def prepare_holdout(root: Path, p2_root: Path, pilot: bool = False) -> dict:
    """Validate original P2 evidence and copy only frozen inputs and sampled sources.

    Formal selection is exactly sixteen prespecified bases, both tariffs and the
    first three prespecified algorithm seeds. Pilot uses four old development
    runs, preventing implementation tuning on formal routing outcomes.
    """
    provenance = git_metadata()
    if provenance["git_dirty"]:
        raise ValueError("Freeze a clean source commit before preparing a campaign")
    if root.exists():
        raise FileExistsError("Do not overwrite an existing campaign")
    specs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((p2_root / "specs").glob("*.json"))
    ]
    selected = [
        s
        for s in specs
        if s["arm"] == "F6"
        and (
            s["key"] in PILOT_KEYS
            if pilot
            else s["base_seed"] in HOLDOUT_BASES
            and s["algorithm_seed"] in SEEDS
            and s["tariff"] in ("H", "T")
        )
    ]
    if len(selected) != (4 if pilot else 96):
        raise ValueError("P2 does not contain the complete prespecified source matrix")
    if not pilot and {s["base_seed"] for s in selected} != HOLDOUT_BASES:
        raise ValueError("Formal bases do not match the fixed holdout")
    if not pilot:
        expected_matrix = {
            (50 if b < 820000 else 100, b, t, a)
            for b in HOLDOUT_BASES
            for t in ("H", "T")
            for a in SEEDS
        }
        actual_matrix = {
            (s["jobs"], s["base_seed"], s["tariff"], s["algorithm_seed"]) for s in selected
        }
        if actual_matrix != expected_matrix or len({s["key"] for s in selected}) != 96:
            raise ValueError("Incomplete or duplicated formal source conditions")
    root.mkdir(parents=True)
    for name in ("inputs", "sources", "jobs", "ops", "ops/heartbeats", "analysis", "logs"):
        (root / name).mkdir()
    begun = time.perf_counter()
    source_files: dict[str, str] = {}
    inputs: dict[str, str] = {}
    sources: dict[str, str] = {}
    for spec in selected:
        key = spec["key"]
        original = p2_root / "runs" / key
        valid, reason = validate_complete(original, spec)
        if not valid:
            raise ValueError(f"P2 source {key}: {reason}")
        for name in ("instance", "initial"):
            relative = spec[name]
            if file_hash(campaign_path(p2_root, relative)) != spec[f"{name}_hash"]:
                raise ValueError("P2 source input mismatch")
            source_files[relative] = spec[f"{name}_hash"]
        marker = json.loads((original / "complete.json").read_text(encoding="utf-8"))
        for name, sha in marker["files"].items():
            source_files[f"runs/{key}/{name}"] = sha
        for name in (f"specs/{key}.json", f"runs/{key}/complete.json"):
            source_files[name] = file_hash(p2_root / name)
        population = [
            restore_candidate(c)
            for c in json.loads((original / "population.json").read_text(encoding="utf-8"))
        ]
        archive = [
            restore_candidate(c)
            for c in json.loads((original / "archive.json").read_text(encoding="utf-8"))
        ]
        problem = load_instance(p2_root / spec["instance"])
        if len(problem.jobs) != spec["jobs"]:
            raise ValueError("Source job count does not match the imported instance")
        stream = RNGManager(MASTER_SEED).stream(f"D1H:sample:{key}")
        sampled = sample_population(population, stream)
        raw_population = json.loads((original / "population.json").read_text(encoding="utf-8"))
        for row in sampled:
            index = row["index"]
            row["candidate"] = raw_population[index] if index is not None else None
            if index is not None:
                c = population[index]
                if evaluate(problem, c.genotype, c.schedule) != c.evaluation:
                    raise ValueError("Sampled P2 source failed exact re-evaluation")
        instance_name = spec["instance"]
        if instance_name not in inputs:
            shutil.copyfile(p2_root / instance_name, root / instance_name)
            inputs[instance_name] = file_hash(root / instance_name)
        payload = {
            "key": key,
            "jobs": spec["jobs"],
            "base_seed": spec["base_seed"],
            "tariff": spec["tariff"],
            "algorithm_seed": spec["algorithm_seed"],
            "instance": instance_name,
            "population": len(population),
            "ideal": [
                min(c.evaluation.objectives[m] for c in population + archive) for m in range(2)
            ],
            "maximum": maximum(population),
            "sample": sampled,
            "p2_spec_hash": digest(spec),
            "p2_source_commit": spec["source_commit"],
            "p2_source_hash": spec["source_hash"],
            "p2_terminal_seconds": spec["config"]["seconds"],
        }
        name = f"sources/{key}.json"
        atomic_json(root / name, payload)
        sources[name] = file_hash(root / name)
    source_files["manifest.json"] = file_hash(p2_root / "manifest.json")
    manifest = {
        "schema": 1,
        "kind": "D1_terminal_routing_holdout",
        "pilot": pilot,
        "source_commit": provenance["git_commit"],
        "source_hash": canonical_source_hash(),
        "master_seed": MASTER_SEED,
        "groups": GROUPS,
        "keys": [s["key"] for s in selected],
        "bases": sorted({s["base_seed"] for s in selected}),
        "cases_planned": len(selected) * 4 * 6,
        "sources_planned": len(selected) * 4,
        "inputs": inputs,
        "sources": sources,
        "original_p2_files": source_files,
        "quota_seconds": {"50": 1.0, "100": 3.0},
        "exact_cap": 3,
        "intent_cap_seconds": 900.0,
        "normalization": "Freeze population/archive ideal plus base, before proposals",
        "rule": "Positive residual wait AND identical actual MS AND changed actual OS",
        "random_allocation": "Exact gated TRUE count per source target pool; positive-wait guard",
        "sampling": "Unique phenotype per Flow/Balanced/Electricity stratum and one remainder",
        "statistics": {
            "independent_unit": "base instance",
            "bootstrap": 50000,
            "bootstrap_strata": "50/100 jobs",
            "primary_comparisons": ["CONDITIONAL-ALWAYS_A7", "CONDITIONAL-RANDOM"],
            "p_values": "exact two-sided base sign-flip; Holm two quality comparisons",
        },
        "scope": "Terminal internal holdout, not unseen independent instances or online MOEA/D",
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", {"manifest.json": file_hash(root / "manifest.json")})
    atomic_json(
        root / "ops/import.json",
        {
            "p2_root": str(p2_root.resolve()),
            "import_seconds": time.perf_counter() - begun,
            "source_runs": len(selected),
            "original_files": len(source_files),
            "sampled_source_exact_checks": sum(
                sum(r["index"] is not None for r in json.loads((root / n).read_text())["sample"])
                for n in sources
            ),
        },
    )
    return manifest


def validate_holdout_job(root: Path, key: str) -> tuple[bool, str]:
    """Check published hashes, frozen routing identity and complete case accounting."""
    try:
        manifest = _frozen_manifest(root)
        output = root / "jobs" / key
        marker = json.loads((output / "complete.json").read_text(encoding="utf-8"))
        if marker["manifest_hash"] != file_hash(root / "manifest.json"):
            return False, "manifest mismatch"
        if marker["source_hash"] != manifest["source_hash"] or marker["key"] != key:
            return False, "source/key mismatch"
        if set(marker["files"]) != {"data.json", "summary.json"}:
            return False, "missing artifact"
        if any(file_hash(output / n) != sha for n, sha in marker["files"].items()):
            return False, "artifact hash mismatch"
        source_name = f"sources/{key}.json"
        if file_hash(root / source_name) != manifest["sources"][source_name]:
            return False, "source snapshot mismatch"
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        rows = json.loads((output / "data.json").read_text(encoding="utf-8"))["rows"]
        valid_rows = [r for r in rows if "groups" in r]
        skipped = sum(6 if "missing_source" in r else 1 for r in rows if "groups" not in r)
        if (
            summary["cases"] != len(valid_rows)
            or summary["skipped"] != skipped
            or len(valid_rows) + skipped != 24
        ):
            return False, "case accounting mismatch"
        if any(set(r["groups"]) != set(GROUPS) for r in valid_rows):
            return False, "missing policy group"
        for sample_index in range(4):
            pool = [r for r in valid_rows if r["sample_index"] == sample_index]
            if sum(r["groups"]["CONDITIONAL"]["route"] == "TRUE" for r in pool) != sum(
                r["groups"]["RANDOM"]["route"] == "TRUE" for r in pool
            ):
                return False, "random route count mismatch"
        return True, "complete"
    except (OSError, ValueError, KeyError, TypeError):
        return False, "missing/unreadable result"


def execute_holdout_job(root: Path, key: str) -> dict:
    """Execute one whole terminal-source run; publish no incomplete results."""
    manifest = _frozen_manifest(root)
    provenance = git_metadata()
    if (
        provenance["git_dirty"]
        or provenance["git_commit"] != manifest["source_commit"]
        or canonical_source_hash() != manifest["source_hash"]
        or key not in manifest["keys"]
    ):
        raise ValueError("Worker requires the clean frozen campaign source")
    source_name = f"sources/{key}.json"
    if file_hash(root / source_name) != manifest["sources"][source_name]:
        raise ValueError("Source snapshot hash mismatch")
    payload = json.loads((root / source_name).read_text(encoding="utf-8"))
    if file_hash(root / payload["instance"]) != manifest["inputs"][payload["instance"]]:
        raise ValueError("Copied instance hash mismatch")
    output = root / "jobs" / key
    temporary = root / "jobs" / f".attempt-{key}-{os.getpid()}"
    if output.exists() or temporary.exists():
        raise FileExistsError("Existing output must be audited before another attempt")
    temporary.mkdir()
    started = time.perf_counter()
    problem = load_instance(root / payload["instance"])
    context = NormalizationContext(tuple(payload["ideal"]), tuple(payload["maximum"]))
    lambdas = weights(payload["population"])
    streams = RNGManager(manifest["master_seed"])
    rows: list[dict] = []
    planning_seconds = 0.0
    for sample_index, sample in enumerate(payload["sample"]):
        if sample["candidate"] is None:
            rows.append({"sample_index": sample_index, "missing_source": sample["missing"]})
            continue
        source = restore_candidate(sample["candidate"])
        tick = time.perf_counter()
        intent = extract_intent(problem, source)
        targets, descriptors = [], []
        for kind in ("OS", "INSTANCE", "REGION"):
            for magnitude in (1, max(2, (len(problem.jobs) + 9) // 10)):
                label = f"{key}:{sample_index}:{kind}:{magnitude}"
                seed = streams.stream(f"D1H:case:{label}").getrandbits(63)
                target = perturb_structure(
                    problem,
                    source.genotype,
                    kind,
                    magnitude,
                    RNGManager(seed).stream("D1:structure"),
                )
                metadata = {
                    "sample_index": sample_index,
                    "preference": sample["stratum"],
                    "kind": kind,
                    "magnitude": magnitude,
                    "seed": seed,
                }
                if target is None:
                    rows.append({**metadata, "unchanged_structure": True})
                else:
                    targets.append(target)
                    descriptors.append(metadata)
        random_routes = matched_random_routes(
            source.genotype, targets, intent, streams.stream(f"D1H:routes:{key}:{sample_index}")
        )
        planning_seconds += time.perf_counter() - tick
        for target, metadata, random_route in zip(targets, descriptors, random_routes):
            row = routing_probe(
                problem,
                source,
                target,
                lambdas[sample["index"]],
                context,
                metadata["seed"],
                random_route,
                manifest["quota_seconds"][str(payload["jobs"])],
                manifest["intent_cap_seconds"],
            )
            rows.append({**metadata, **row})
            atomic_json(
                root / "ops/heartbeats" / f"{key}.json",
                {
                    "pid": os.getpid(),
                    "timestamp": time.time(),
                    "cases": sum("groups" in r for r in rows),
                    "elapsed": time.perf_counter() - started,
                },
            )
    count = sum("groups" in r for r in rows)
    summary = {
        "key": key,
        "jobs": payload["jobs"],
        "base_seed": payload["base_seed"],
        "tariff": payload["tariff"],
        "algorithm_seed": payload["algorithm_seed"],
        "cases": count,
        "skipped": sum(6 if "missing_source" in r else 1 for r in rows if "groups" not in r),
        "source_snapshots": sum(s["index"] is not None for s in payload["sample"]),
        "planning_seconds": planning_seconds,
        "elapsed": time.perf_counter() - started,
        "rss_peak_gib": process_peak_rss_gb(),
        "host": platform.node(),
    }
    atomic_json(temporary / "data.json", {"rows": rows})
    atomic_json(temporary / "summary.json", summary)
    atomic_json(
        temporary / "complete.json",
        {
            "key": key,
            "source_hash": manifest["source_hash"],
            "manifest_hash": file_hash(root / "manifest.json"),
            "files": {n: file_hash(temporary / n) for n in ("data.json", "summary.json")},
        },
    )
    temporary.rename(output)
    valid, reason = validate_holdout_job(root, key)
    if not valid:
        raise ValueError(reason)
    return summary


def run_holdout(root: Path, workers: int = 4) -> dict:
    """Supervise a bounded queue with process timeout and durable failure evidence."""
    if not 1 <= workers <= 8:
        raise ValueError("Use one to eight preflighted workers")
    manifest = _frozen_manifest(root)
    queue = []
    completed = []
    for key in manifest["keys"]:
        valid, _ = validate_holdout_job(root, key)
        if valid:
            completed.append(key)
        else:
            if (root / "jobs" / key).exists() or (root / "ops" / f"failure_{key}.json").exists():
                raise ValueError("Invalid/failed attempt requires review; no automatic retry")
            queue.append(key)
    begun = time.time()
    atomic_json(
        root / "ops/started.json", {"pid": os.getpid(), "timestamp": begun, "workers": workers}
    )
    failure = None
    with campaign_lease(root / "ops/campaign.lock"):
        pending: dict[str, tuple[subprocess.Popen, float]] = {}
        stopped = False
        while queue or pending:
            if (root / "stop.request").exists():
                stopped = True
            while queue and len(pending) < workers and not stopped and failure is None:
                key = queue.pop(0)
                with (root / "logs" / f"{key}.log").open("ab") as handle:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "geo_llm_scheduler.experiments.d1_holdout",
                            "worker",
                            "--root",
                            str(root.resolve()),
                            "--key",
                            key,
                        ],
                        cwd=Path(__file__).resolve().parents[3],
                        stdin=subprocess.DEVNULL,
                        stdout=handle,
                        stderr=handle,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                        if sys.platform == "win32"
                        else 0,
                    )
                pending[key] = (child, time.time())
            if (stopped or failure is not None) and not pending:
                break
            for key, (child, launched) in list(pending.items()):
                timed_out = time.time() - launched > 600 and child.poll() is None
                if timed_out:
                    child.kill()
                    child.wait()
                code = child.poll()
                if code is None:
                    continue
                del pending[key]
                valid, reason = validate_holdout_job(root, key)
                if code or not valid or timed_out:
                    failure = f"{key}: timeout" if timed_out else f"{key}: exit {code}; {reason}"
                    atomic_json(root / "ops" / f"failure_{key}.json", {"error": failure})
                else:
                    completed.append(key)
            atomic_json(
                root / "ops/status.json",
                {
                    "timestamp": time.time(),
                    "completed": len(completed),
                    "total": len(manifest["keys"]),
                    "pending": list(pending),
                    "worker_pids": {key: child.pid for key, (child, _) in pending.items()},
                    "remaining": len(queue),
                    "elapsed": time.time() - begun,
                },
            )
            if pending:
                time.sleep(0.5)
    result = {
        "status": "failed" if failure else "paused" if stopped else "complete",
        "completed": len(completed),
        "total": len(manifest["keys"]),
        "elapsed": time.time() - begun,
    }
    atomic_json(root / "ops/exit.json", result)
    if failure:
        raise RuntimeError(failure)
    return result


def main() -> None:
    """Portable CLI for preparation, execution, validation and base-level analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "worker", "validate", "analyse"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--p2-root", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--key")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.p2_root is None:
            parser.error("prepare requires --p2-root")
        result = prepare_holdout(args.root, args.p2_root, args.pilot)
        print(json.dumps({"sources": len(result["keys"]), "cases": result["cases_planned"]}))
    elif args.command == "run":
        print(json.dumps(run_holdout(args.root, args.workers)))
    elif args.command == "worker":
        if args.key is None:
            parser.error("worker requires --key")
        try:
            print(json.dumps(execute_holdout_job(args.root, args.key)))
        except Exception:
            atomic_json(
                args.root / "ops" / f"failure_{args.key}.json", {"error": traceback.format_exc()}
            )
            raise
    elif args.command == "validate":
        manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
        results = {k: validate_holdout_job(args.root, k) for k in manifest["keys"]}
        if not all(ok for ok, _ in results.values()):
            raise ValueError(results)
        if args.p2_root is not None and any(
            file_hash(args.p2_root / n) != sha for n, sha in manifest["original_p2_files"].items()
        ):
            raise ValueError("Original P2 files changed")
        print(
            json.dumps(
                {"valid": len(results), "original_inputs_unchanged": args.p2_root is not None}
            )
        )
    else:
        from geo_llm_scheduler.experiments.d1_holdout_analysis import analyse_holdout

        print(json.dumps(analyse_holdout(args.root)["primary"]))


if __name__ == "__main__":
    main()
