"""JSON bridge called by the irace R targetRunner; stdout contains only its score."""

import argparse
import json
from pathlib import Path

from geo_llm_scheduler.experiments.irace_target import evaluate_request


def main() -> None:
    """Read an immutable request and emit the finite minimization cost."""
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = evaluate_request(request, Path(request["case"]), Path(request["run_dir"]))
    print(result["cost"])


if __name__ == "__main__":
    main()
