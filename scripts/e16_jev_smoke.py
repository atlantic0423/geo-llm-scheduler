"""Live API prerequisite check; never creates formal results or starts a campaign."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.controllers.jev import JevClient, JevError
from geo_llm_scheduler.controllers.jev_questions import QUESTION_VERSION, question_hash, questions
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.utils.rng import RNGManager


def main() -> int:
    """Probe combined Choice and post-step Noul using a secret read from environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("TYPESAFE_MODEL", "jev-1.13.0"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report: dict = {
        "formal": False,
        "stage": "api-prerequisite-smoke",
        "question_version": QUESTION_VERSION,
        "question_hash": question_hash(),
    }
    try:
        client = JevClient(args.model, RNGManager(160930).stream("jev-retry"))
        state = {
            "Preference": "Balanced",
            "DominantCondition": "Resource",
            "SearchProgress": "Improving",
        }
        report["routing"] = asdict(client.decide(state, questions(budget=True)))
        report["continuation"] = asdict(
            client.decide(
                {**state, "step_index": 0, "accepted": False, "remaining_steps": 4},
                questions(continuation=True),
            )
        )
        report["status"] = "passed"
    except JevError as error:
        report.update(status="blocked", error=str(error))
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
