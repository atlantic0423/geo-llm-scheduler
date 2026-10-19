"""Immutable campaign manifests and streamed, bounded-memory artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments.runner import digest

ROOT = Path(__file__).resolve().parents[2]
TUPLES = (
    "mutation_weights",
    "severity_thresholds",
    "budgets",
    "severity_budget_cutoffs",
    "static_budgets",
    "enabled_operators",
    "auxiliary_cutoffs",
    "auxiliary_random_probabilities",
)


def sha(path: Path) -> str:
    """Hash a file without reading the whole file into memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source() -> dict:
    """Bind tracked text to canonical hashes independent of CRLF checkout style."""

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()

    paths = git("ls-files", "src", "research/auxiliary_walk_20261016").splitlines()
    return {
        "head": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
        "canonical": digest({p: digest((ROOT / p).read_text(encoding="utf-8")) for p in paths}),
    }


def config_from(data: dict) -> Config:
    """Restore explicitly serialized tuple fields before Config validation."""
    return Config(**{k: tuple(v) if k in TUPLES else v for k, v in data.items()})


def stream_json(path: Path, values: Iterable[object]) -> dict:
    """Write a compressed JSON array a single item at a time, retaining both hashes."""
    h = hashlib.sha256()
    count = 0
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as z:

        def write(block: bytes) -> None:
            h.update(block)
            z.write(block)

        write(b"[")
        for value in values:
            if count:
                write(b",")
            write(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())
            count += 1
        write(b"]")
    return {
        "count": count,
        "raw_sha256": h.hexdigest(),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }


def validate_complete(root: Path, spec: dict, provenance: dict) -> dict:
    """Recheck every committed artifact and immutable run/source identity."""
    marker = json.loads((root / "complete.json").read_text())
    if marker["spec_hash"] != digest(spec) or marker["source"] != provenance:
        raise ValueError("Run/source provenance mismatch")
    for name, expected in marker["files"].items():
        if name != Path(name).name or not name or name == "complete.json":
            raise ValueError("Unsafe manifest member")
        path = root / name
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError(f"Artifact changed: {name}")
    if set(marker["files"]) != {"summary.json", "q.json", "diagnostics.json", "solutions.json.gz"}:
        raise ValueError("Incomplete artifact set")
    summary = json.loads((root / "summary.json").read_text())
    if summary["independent_exact"] != summary["population"] + summary["archive_size"]:
        raise ValueError("Final independent exact audit incomplete")
    if summary["offspring"] < 1 or summary["initial_hash"] != spec["initial_hash"]:
        raise ValueError("No evolution or wrong initialization")
    return marker
