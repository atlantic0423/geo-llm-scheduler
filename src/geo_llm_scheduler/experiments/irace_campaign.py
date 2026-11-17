"""Freeze and verify local irace campaigns before execution and recovery."""

import json
import subprocess
from pathlib import Path

from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.irace_target import sha256, validate_case


def verify_seal(path: Path, case: Path | None = None) -> dict:
    """Reject modified code, configuration, or inputs; optionally restrict a target case."""
    seal = json.loads(path.read_text(encoding="utf-8"))
    if case is not None and str(case.resolve()) not in seal["cases"]:
        raise ValueError("Case is not in the frozen campaign")
    for name, expected in seal["files"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Campaign checksum mismatch: {name}")
    return seal


def seal_campaign(root: Path, repo: Path) -> Path:
    """Seal a prepared dataset and source tree; refuse to replace an existing seal."""
    root, repo = root.resolve(), repo.resolve()
    target = root / "seal.json"
    if target.exists():
        verify_seal(target)
        return target
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if manifest["purpose"] == "formal":
        for split, count in (("configuration", 60), ("validation", 30), ("evaluation_locked", 30)):
            selected = [c for c in cases if c["split"] == split]
            if len(selected) != count:
                raise ValueError(f"Invalid {split} count")
            if any(sum(c["jobs"] == n for c in selected) != count // 3 for n in (50, 100, 200)):
                raise ValueError("Unbalanced instance sizes")
        groups: dict[tuple, str] = {}
        for c in cases:
            group = (c["family"], c["replicate"])
            if group in groups and groups[group] != c["split"]:
                raise ValueError("Instance group crosses splits")
            groups[group] = c["split"]
    files = list((repo / "src").rglob("*.py"))
    files += [
        repo / "scripts" / n
        for n in (
            "irace_target.py",
            "run_irace.R",
            "prepare_irace.py",
            "irace_campaign.py",
            "evaluate_irace.py",
        )
    ]
    files += [root / "manifest.json", root / "preparation.json", root / "training.txt"]
    files += list(root.glob("parameters_*.txt")) + list(root.glob("default_*.txt"))
    training = (root / "training.txt").read_text(encoding="utf-8").splitlines()
    expected_training = {
        str(Path(c["path"]).resolve()) for c in cases if c["split"] == "configuration"
    }
    if (
        len(training) != len(expected_training)
        or {str(Path(p).resolve()) for p in training} != expected_training
    ):
        raise ValueError("Training file must contain exactly the configuration split")
    for c in cases:
        p = Path(c["path"])
        if sha256(p) != c["sha256"]:
            raise ValueError("Manifest case checksum mismatch")
        validate_case(p)
        files += list(p.parent.glob("*.json"))
    # Three independent races per arm. Each uses the same inputs and target budgets.
    for mode in ("numeric", "joint"):
        base = json.loads((root / f"campaign_{mode}.json").read_text(encoding="utf-8"))
        for repetition in range(1, 4):
            settings = {
                **base,
                "seed": 17100 + repetition,
                "seal": str(target),
                "output": str(root / f"{mode}_rep{repetition}"),
            }
            p = root / f"campaign_{mode}_rep{repetition}.json"
            atomic_json(p, settings)
            files.append(p)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    atomic_json(
        target,
        dict(
            source_base_commit=commit,
            purpose=manifest["purpose"],
            cases=[str(Path(c["path"]).resolve()) for c in cases],
            files={str(p.resolve()): sha256(p) for p in files},
        ),
    )
    return target
