"""Fresh four-LLM restart with an explicit index reusing the eight frozen references."""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
from dataclasses import asdict
from pathlib import Path

from run_round_llm import ARM, REQUIRED, complete, preflight, sha, source_hash, study_config, worker

from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.round_responses import ResponsesProfile

REFERENCE_SOURCE = "591186f08b4d1c428f5472e6814ff6492c611791401ce208dc67b1f2dd674796"
REFERENCE_HEAD = "a90004aadfafcb335728d90268c45c54ed4269f2"


def reference_binding(previous: Path, key: str, arm: str) -> dict:
    """Bind a completed reference's full saved artifacts; never count aborted Flash runs."""
    if arm not in ("BILL_MOEAD", "MAIN_Q"):
        raise ValueError("Only completed references may be reused")
    m = json.loads((previous / "manifest.json").read_text())
    spec_path = previous / "specs" / (key + ".json")
    spec = json.loads(spec_path.read_text())
    out = previous / "runs" / key / arm
    marker = json.loads((out / "complete.json").read_text())
    assert sha(spec_path) == m["files"]["specs/" + spec_path.name] == marker["spec_sha256"]
    assert sha(previous / spec["input"]) == spec["input_sha256"] == marker["input_sha256"]
    assert marker["source"] == m["source"] == REFERENCE_SOURCE
    assert m["commit"] == REFERENCE_HEAD and marker["arm"] == arm and marker["key"] == key
    assert marker["host"] == m["host"] == spec["host"] == socket.gethostname()
    assert set(marker["files"]) == set(REQUIRED)
    for name, record in marker["files"].items():
        assert (out / name).stat().st_size == record["size"] and sha(out / name) == record["sha256"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["termination"] == "time_budget" and summary["offspring"] > 0
    assert summary["search_elapsed_s"] >= spec["seconds"] and summary["api_blocked_s"] == 0
    assert summary["final_exact_verified"] == summary["population"] + summary["archive"]
    assert json.loads((out / "config.json").read_text()) == json.loads(
        json.dumps(asdict(study_config(arm, spec["seed"], spec["seconds"])))
    )
    expected = json.loads((previous / "preflight" / (key + ".json")).read_text())["policies"][
        "bill4"
    ]["evidence"]
    assert all(
        summary["initialization"][k] == expected[k]
        for k in ("initial_population_sha256", "initial_archive_sha256")
    )
    return dict(
        key=key,
        arm=arm,
        path=str(out.resolve()),
        source=REFERENCE_SOURCE,
        commit=REFERENCE_HEAD,
        marker_sha256=sha(out / "complete.json"),
    )


def prepare(root: Path, previous: Path, profile: ResponsesProfile) -> dict:
    """Create four fresh LLM specs plus eight immutable reference bindings, exactly once."""
    if root.exists():
        raise FileExistsError("New restart root required")
    old = json.loads((previous / "manifest.json").read_text())
    assert (
        sha(previous / "manifest.json")
        == json.loads((previous / "manifest.sha256.json").read_text())["sha256"]
    )
    assert old["commit"] == REFERENCE_HEAD and old["source"] == REFERENCE_SOURCE
    assert old["runs"] == 12 and len(old["blocks"]) == 4 and old["seeds"] == [17401, 27401]
    references = [
        reference_binding(previous, key, arm)
        for key in old["blocks"]
        for arm in ("BILL_MOEAD", "MAIN_Q")
    ]
    root.mkdir(parents=True)
    files = {}
    for name, expected in old["files"].items():
        source = previous / name
        assert sha(source) == expected
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[name] = sha(target)
    manifest = dict(
        old,
        source=source_hash(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        responses_profile=asdict(profile),
        prompt_version=profile.version,
        runs=4,
        comparison_runs=12,
        arms=[ARM],
        reference_arms=["BILL_MOEAD", "MAIN_Q"],
        previous_root=str(previous.resolve()),
        files=files,
        reused_references=references,
    )
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "manifest.sha256.json", dict(sha256=sha(root / "manifest.json")))
    index = references + [
        dict(
            key=key,
            arm=ARM,
            path=str((root / "runs" / key / ARM).resolve()),
            source=manifest["source"],
            commit=manifest["commit"],
        )
        for key in old["blocks"]
    ]
    atomic_json(
        root / "formal_execution_index.json",
        dict(
            runs=12,
            entries=index,
            aborted_flash_attempts="Excluded; retained at previous_root and never relabeled",
        ),
    )
    return manifest


def validate(root: Path) -> bool:
    """Verify all four new markers and the original eight references using the explicit index."""
    manifest = json.loads((root / "manifest.json").read_text())
    assert (
        sha(root / "manifest.json")
        == json.loads((root / "manifest.sha256.json").read_text())["sha256"]
    )
    assert manifest["source"] == source_hash() and manifest["runs"] == 4
    assert all(complete(root, key, ARM) for key in manifest["blocks"])
    previous = Path(manifest["previous_root"])
    assert manifest["reused_references"] == [
        reference_binding(previous, key, arm)
        for key in manifest["blocks"]
        for arm in ("BILL_MOEAD", "MAIN_Q")
    ]
    index = json.loads((root / "formal_execution_index.json").read_text())
    assert index["runs"] == len(index["entries"]) == 12
    assert len({(r["key"], r["arm"]) for r in index["entries"]}) == 12
    assert index["entries"] == manifest["reused_references"] + [
        dict(
            key=key,
            arm=ARM,
            path=str((root / "runs" / key / ARM).resolve()),
            source=manifest["source"],
            commit=manifest["commit"],
        )
        for key in manifest["blocks"]
    ]
    return True


def main() -> None:
    """Expose a separately named four-run restart, never launch reference workers."""
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "preflight", "worker", "validate"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--previous", type=Path)
    p.add_argument("--model", default="gpt-6.1-sol")
    p.add_argument("--key")
    p.add_argument("--cpu", type=int, default=0)
    args = p.parse_args()
    if args.action == "prepare":
        assert args.previous is not None
        print(json.dumps(prepare(args.root, args.previous, ResponsesProfile(model=args.model))))
    elif args.action == "preflight":
        assert args.key is not None
        print(json.dumps(preflight(args.root, args.key)))
    elif args.action == "worker":
        assert args.key is not None
        worker(args.root, args.key, ARM, args.cpu)
    else:
        print(json.dumps(dict(accepted=validate(args.root))))


if __name__ == "__main__":
    main()
