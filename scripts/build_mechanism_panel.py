"""Generate the declared 120-input panel, then optionally run resumable exact audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geo_llm_scheduler.domain.models import ProblemInstance  # noqa: E402
from geo_llm_scheduler.experiments.mechanism_audit import audit_problem  # noqa: E402
from geo_llm_scheduler.experiments.mechanism_instances import (  # noqa: E402
    FAMILIES,
    PANEL_SIZES,
    input_features,
    mechanism_problem,
    panel_specs,
)
from geo_llm_scheduler.io.loaders import load_instance, save_instance  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _frozen_input(problem: ProblemInstance, path: Path) -> None:
    # The first 120 published fingerprints use CRLF. Generate these same bytes
    # on Linux and Windows while keeping the shared JSON-v1 writer authoritative.
    with TemporaryDirectory(dir=path.parent) as tmpdir:
        candidate = Path(tmpdir) / "input.json"
        save_instance(problem, candidate)
        frozen = candidate.read_text(encoding="utf-8").replace("\n", "\r\n").encode()
        if path.exists():
            if path.read_bytes() != frozen:
                raise ValueError(f"Refusing changed input: {path}")
        else:
            candidate.write_bytes(frozen)
            candidate.replace(path)
    assert load_instance(path) == problem


def _audit(path: str, output: str, operators: bool) -> str:
    source = Path(path)
    result = audit_problem(load_instance(source), operators=operators)
    result["input_sha256"] = _sha(source)
    result["protocol"] = "bill4-140-front5-b3-v1"
    result["operators_probed"] = operators
    _write(Path(output), result)
    return source.stem


def main() -> None:
    """CLI entry; inputs are frozen before any audit, and audit never changes them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--rightshift-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--keys", nargs="*")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(PANEL_SIZES),
        help="job counts to generate (default: the frozen 50/100/200 panel; 500 extends it)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = args.output / "instances"
    audits = args.output / "audits"
    inputs.mkdir(exist_ok=True)
    audits.mkdir(exist_ok=True)
    entries = []
    for spec in panel_specs(tuple(args.sizes)):
        problem = mechanism_problem(spec)
        path = inputs / (spec.key + ".json")
        _frozen_input(problem, path)
        entries.append(
            {
                **asdict(spec),
                "key": spec.key,
                "path": "instances/" + path.name,
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
                "features": input_features(problem),
            }
        )
    manifest = {
        "schema_version": 1,
        "purpose": "mechanism-oriented synthetic stress panel",
        "selection": "all declared cells retained; no solver-win filtering",
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_files_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[1])).replace("\\", "/"): _sha(p)
            for p in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1]
                / "src/geo_llm_scheduler/experiments/mechanism_instances.py",
                Path(__file__).resolve().parents[1]
                / "src/geo_llm_scheduler/experiments/mechanism_audit.py",
            )
        },
        "families": [asdict(f) for f in FAMILIES],
        "instances": entries,
    }
    if tuple(args.sizes) != PANEL_SIZES:
        manifest["sizes"] = list(args.sizes)
    _write(args.output / "manifest.json", manifest)
    print(f"Generated/verified {len(entries)} independent inputs", flush=True)
    if not args.audit:
        return
    selected = [e for e in entries if args.keys is None or e["key"] in args.keys]
    if args.keys and len(selected) != len(set(args.keys)):
        parser.error("Unknown input key")
    pending = []
    for entry in selected:
        output = audits / (entry["key"] + ".json")
        if output.exists():
            old = json.loads(output.read_text(encoding="utf-8"))
            assert old["input_sha256"] == entry["sha256"]
            if args.rightshift_only or any(r["operator"] == "A1" for r in old["rows"]):
                continue
        pending.append(
            (str(inputs / (entry["key"] + ".json")), str(output), not args.rightshift_only)
        )
    _write(args.output / "audit_status.json", {"status": "running", "pending": len(pending)})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_audit, *job) for job in pending]
        for done, future in enumerate(as_completed(futures), 1):
            key = future.result()
            _write(
                args.output / "audit_status.json",
                {
                    "status": "running",
                    "finished_this_call": done,
                    "scheduled_this_call": len(pending),
                    "last": key,
                },
            )
            print(f"[{done}/{len(pending)}] {key}", flush=True)
    _write(args.output / "audit_status.json", {"status": "complete", "selected": len(selected)})


if __name__ == "__main__":
    main()
