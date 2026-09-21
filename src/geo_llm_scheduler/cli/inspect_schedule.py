"""Re-evaluate a saved candidate using the exact reference evaluator."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.domain.models import Genotype, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.io.loaders import load_instance


def main() -> None:
    """Read one candidate from a saved archive and print objective decomposition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    data = json.loads(Path(args.archive).read_text(encoding="utf-8"))[args.index]
    g = Genotype(tuple(data["genotype"]["ms"]), tuple(data["genotype"]["os"]))
    s = Schedule(tuple(data["schedule"]["starts"]))
    print(json.dumps(asdict(evaluate(load_instance(args.instance), g, s)), indent=2))


if __name__ == "__main__":
    main()
