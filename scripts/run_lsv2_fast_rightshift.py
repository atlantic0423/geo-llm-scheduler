"""Run one explicit fast-right-shift job locally; campaign dispatch is separate."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from run_lsv2 import sha, source_hash, write_candidates, write_json_gz

from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.lsv2_fast_rightshift import (
    ARMS,
    RUN_SEED,
    arm_config,
    configuration_record,
    make_search,
)
from geo_llm_scheduler.experiments.lsv2_qscreen import run_screen
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.qlearning import QTable


def run_job(
    input_path: Path,
    calibration_path: Path,
    out: Path,
    arm: str,
    seconds: float,
    seed: int = RUN_SEED,
) -> dict:
    """Run from an empty Q table and persist separate learning/polish evidence.

    Calibration is a previously frozen single-fold mapping, not pretraining.
    Existing output directories are rejected; failure never writes complete.json.
    """
    calibration = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
    config = arm_config(arm, seed, seconds)
    table = QTable()
    search = make_search(arm, config, calibration, table, training=True)
    problem = load_instance(input_path)
    out.mkdir(parents=True, exist_ok=False)
    record = configuration_record(arm, seed, seconds, calibration)
    record.update(
        arm=arm,
        input_sha256=sha(input_path),
        calibration_sha256=sha(calibration_path),
        source_sha256=source_hash(),
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        learning_mode="online_from_zero",
        pretraining_runs=0,
    )
    atomic_json(out / "config.json", record)
    evidence: dict = {}
    result = run_screen(problem, config, evidence, search)
    write_candidates(out / "archive.json.gz", result.archive.members)
    write_candidates(out / "population.json.gz", result.population)
    write_json_gz(out / "lsv2_steps.json.gz", search.steps)
    write_json_gz(out / "rightshift_steps.json.gz", search.polish_steps)
    atomic_json(out / "qtable.json", asdict(table))
    summary = dict(
        arm=arm,
        initialization=evidence,
        lsv2=search.summary(),
        counts=dict(result.gateway.counts),
        gateway_seconds=dict(result.gateway.seconds),
    )
    atomic_json(out / "summary.json", summary)
    files = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.iterdir())}
    atomic_json(out / "complete.json", dict(status="complete", files=files))
    return summary


def main() -> None:
    """Require explicit input, frozen calibration and a new destination directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--seed", type=int, default=RUN_SEED)
    args = parser.parse_args()
    result = run_job(args.input, args.calibration, args.out, args.arm, args.seconds, args.seed)
    print(json.dumps(dict(arm=args.arm, lsv2=result["lsv2"])))


if __name__ == "__main__":
    main()
