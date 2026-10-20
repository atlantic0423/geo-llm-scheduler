"""Prepare disjoint D6 matrices for the already frozen D5 runtime."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

from geo_llm_scheduler.config import Config, load_config
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.d5_worker import canonical_source_hash
from geo_llm_scheduler.experiments.e15_instances import generate_e15_pair
from geo_llm_scheduler.experiments.runner import git_metadata
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.io.loaders import save_instance
from geo_llm_scheduler.utils.rng import RNGManager

_ARMS = ("LEGACY", "REFERENCE", "W", "GW")
_COMMIT = "3a1342cef98dae4b09cd370f40417a119efb4703"
_HASH = "592cf56992cfd0079f7dd9f3e581efdf01215af4cbe5f6d66ccba443cdd9e2d9"


def _prepare(root: Path, repository: Path, node: int, pilot: bool) -> dict:
    """Freeze hashes, balanced orders and host-assigned independent base blocks."""
    if root.exists():
        raise FileExistsError("Refusing to replace an existing D6 matrix")
    provenance = git_metadata(repository)
    if provenance["git_commit"] != _COMMIT or provenance["git_dirty"]:
        raise ValueError("The original clean runtime is required")
    if canonical_source_hash(repository / "src/geo_llm_scheduler") != _HASH:
        raise ValueError("Frozen runtime source hash mismatch")
    for directory in ("inputs", "specs", "runs", "ops", "logs", "quarantine"):
        (root / directory).mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "docs/experiments/d6_independent_confirmation.md",
        root / "inputs/protocol.md",
    )
    configs = load_config(repository / "configs/p2_local/F6.yaml")
    blocks = []
    for jobs in (50, 100):
        start = (
            (1120101 if jobs == 50 else 1121101) if pilot else (1105001 if jobs == 50 else 1115001)
        )
        bases = [start + i for i in range(6 if pilot else 32) if i % 2 == node]
        for base in bases:
            pair = generate_e15_pair(jobs, base)
            for tariff, problem in zip(("H", "T"), pair):
                save_instance(problem, root / f"inputs/{jobs}_{base}_{tariff}.json")
            for seed in (6606, 7707) if pilot else (1101, 2202):
                initial = f"inputs/initial_{jobs}_{base}_{seed}.json"
                genotypes = initial_genotypes(
                    pair[0], Config(seed=seed), RNGManager(seed).stream("initialization")
                )
                atomic_json(root / initial, [asdict(g) for g in genotypes])
                for tariff in ("H", "T"):
                    block = f"n{jobs}_i{base}_{tariff}_a{seed}"
                    instance = f"inputs/{jobs}_{base}_{tariff}.json"
                    offset = len(blocks) % 4
                    order = _ARMS[offset:] + _ARMS[:offset]
                    keys = []
                    for arm in order:
                        key = block + "_" + arm
                        config = replace(
                            configs,
                            seed=seed,
                            instance=instance,
                            output="runs/" + key,
                            seconds=(60.0 if jobs == 50 else 120.0)
                            if pilot
                            else (1200.0 if jobs == 50 else 2400.0),
                            generations=1_000_000,
                        )
                        spec = {
                            "key": key,
                            "block": block,
                            "arm": arm,
                            "jobs": jobs,
                            "base_seed": base,
                            "tariff": tariff,
                            "algorithm_seed": seed,
                            "config": asdict(config),
                            "instance": instance,
                            "instance_hash": file_hash(root / instance),
                            "initial": initial,
                            "initial_hash": file_hash(root / initial),
                            "source_commit": _COMMIT,
                            "source_hash": _HASH,
                        }
                        atomic_json(root / "specs" / (key + ".json"), spec)
                        keys.append(key)
                    blocks.append({"key": block, "jobs": jobs, "runs": keys})
    RNGManager(2026100806).stream("d6_block_order").shuffle(blocks)
    blocks.sort(key=lambda b: -b["jobs"])
    manifest = {
        "schema": 1,
        "kind": "D6_technical_pilot" if pilot else "D6_64base_fourarm_1024",
        "node": node,
        "nodes": 2,
        "assignment": "base-position modulo 2; H/T and both seeds remain on host",
        "source_commit": _COMMIT,
        "source_hash": _HASH,
        "source_hash_format": "UTF8-universal-newline-JSON-sha256-v1",
        "arms": _ARMS,
        "blocks": blocks,
        "run_count": len(blocks) * 4,
        "input_files": {
            p.relative_to(root).as_posix(): file_hash(p)
            for directory in ("inputs", "specs")
            for p in sorted((root / directory).glob("*"))
        },
        "nominal_worker_hours": sum(
            json.loads((root / "specs" / (key + ".json")).read_text())["config"]["seconds"]
            for block in blocks
            for key in block["runs"]
        )
        / 3600,
        "resume": "task-level; complete blocks portable; partial migration reruns block",
        "intervention": "frozen D5 A8 LEGACY/REFERENCE/W/GUARD_ALL+W; no new behavior",
        "analysis": "four primary W-reference/W-legacy HV/IGD+ joint Holm; 64 bases",
        "runtime_entry": "scripts/run_d5_campaign.py run; original immutable entry",
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(
        root / "manifest.sha256.json", {"manifest_sha256": file_hash(root / "manifest.json")}
    )
    return manifest


def main() -> None:
    """Write a new host shard without editing the runtime or any prior campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-repository", type=Path, required=True)
    parser.add_argument("--node", choices=(0, 1), type=int, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    m = _prepare(args.root.resolve(), args.source_repository.resolve(), args.node, args.pilot)
    print(json.dumps({k: m[k] for k in ("kind", "node", "run_count", "nominal_worker_hours")}))


if __name__ == "__main__":
    main()
