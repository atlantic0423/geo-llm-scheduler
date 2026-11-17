"""Prepare an execution seal, check it, or launch/recover official R irace."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.irace_campaign import seal_campaign, verify_seal
from geo_llm_scheduler.experiments.irace_target import sha256


def main() -> None:
    """Execute one explicit local campaign; no implicit server dispatch or formal race."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("seal", "check", "run", "recover"))
    parser.add_argument(
        "path", type=Path, help="Dataset directory for seal/check; campaign JSON for run/recover"
    )
    parser.add_argument("--rscript", type=Path)
    parser.add_argument("--rlibrary", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.action == "seal":
        print(seal_campaign(args.path, repo))
        return
    if args.action == "check":
        seal = verify_seal(args.path / "seal.json")
        print(
            json.dumps(dict(status="verified", purpose=seal["purpose"], cases=len(seal["cases"])))
        )
        return
    settings = json.loads(args.path.read_text(encoding="utf-8"))
    verify_seal(Path(settings["seal"]))
    if args.rscript is None or not args.rscript.is_file():
        parser.error("--rscript must name the installed Rscript executable")
    output = Path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "launch.json"
    identity = dict(campaign_sha256=sha256(args.path), seal_sha256=sha256(Path(settings["seal"])))
    if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")) != identity:
        raise ValueError("Launch identity changed; recovery is forbidden")
    if args.action == "recover" and not (output / "irace.Rdata").exists():
        raise ValueError("No irace recovery state exists")
    atomic_json(receipt, identity)
    env = {**os.environ, "LC_ALL": "English_United States.utf8"}
    if args.rlibrary:
        env["R_LIBS_USER"] = str(args.rlibrary.resolve())
    command = [str(args.rscript), str(repo / "scripts/run_irace.R"), str(args.path.resolve())]
    if args.action == "recover":
        command.append("recover")
    with (output / "console.log").open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode:
        raise RuntimeError(
            f"irace failed ({completed.returncode}); inspect {output / 'console.log'}"
        )
    print(output / "elites.csv")


if __name__ == "__main__":
    main()
