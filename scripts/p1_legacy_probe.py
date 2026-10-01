"""Run a frozen old checkout without modifying it, tapping exact-return order."""

import hashlib
import json
import sys
from pathlib import Path

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Genotype
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.experiments.e15_worker import TUPLE_FIELDS
from geo_llm_scheduler.experiments.nsga2 import run_nsga2
from geo_llm_scheduler.experiments.runner import digest, solution
from geo_llm_scheduler.io.loaders import load_instance


def semantic(value):
    """Exclude runtime fields only, using the P0 recorder's precise projection."""
    if isinstance(value, dict):
        return {
            k: semantic(v)
            for k, v in value.items()
            if k not in ("elapsed", "archive_objectives") and not k.endswith("seconds")
        }
    if isinstance(value, (list, tuple)):
        return [semantic(v) for v in value]
    return value


def main() -> None:
    """Execute a legacy spec and export a directly comparable semantic signature."""
    spec_path, output = map(Path, sys.argv[1:])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    config_data = dict(spec["config"])
    for key in TUPLE_FIELDS:
        config_data[key] = tuple(config_data[key])
    config = Config(**config_data)
    problem = load_instance(spec["instance_path"])
    initial = [
        Genotype(tuple(g["ms"]), tuple(g["os"]))
        for g in json.loads(Path(spec["initial_path"]).read_text(encoding="utf-8"))
    ]
    chain = hashlib.sha256()
    original = EvaluationGateway.evaluate

    def tapped(self, *args, **kwargs):
        candidate = original(self, *args, **kwargs)
        chain.update(digest(solution(candidate)).encode())
        return candidate

    EvaluationGateway.evaluate = tapped
    runner = run_nsga2 if config.method == "nsga2" else run
    result = runner(problem, config, initial)
    decisions = hashlib.sha256()
    for row in result.trace:
        decisions.update(digest(semantic(row)).encode())
    signature = {
        "chain": chain.hexdigest(),
        "decision_chain": decisions.hexdigest(),
        "offspring": len(result.trace),
        "population": [digest(solution(c)) for c in result.population],
        "archive": sorted(digest(solution(c)) for c in result.archive.members),
        "q": result.controller.q,
        "visits": result.controller.visits,
        "updates": result.controller.updates,
        "selections": result.controller.selections,
        "counts": dict(result.gateway.counts),
        "reason": result.termination_reason,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(signature), encoding="utf-8")


if __name__ == "__main__":
    main()
