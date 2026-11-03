"""Paired old-development W benchmark with exact proposal and RNG invariance."""

import argparse
import json
import random
from pathlib import Path
from time import process_time

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.d4_prototypes import ResearchPeak
from geo_llm_scheduler.experiments.d5_peak import OnlinePeak
from geo_llm_scheduler.experiments.p1_observation import restore_candidate
from geo_llm_scheduler.io.loaders import load_instance


def benchmark(parent: Path, output: Path) -> dict:
    """Measure cold per-invocation costs, preserving source and result artifacts."""
    rows = []
    paths = sorted((parent / "inputs").glob("*_H_a1101_panels.json"))
    selected = [p for n in (50, 100) for p in [x for x in paths if x.name.startswith(f"n{n}_")][:4]]
    assert len(selected) == 8
    for path in selected:
        data = json.loads(path.read_text(encoding="utf-8"))
        problem = load_instance(parent / data["spec"]["instance"])
        for panel in data["panels"]:
            for item in panel["sources"][:2]:
                source = restore_candidate(item["candidate"])
                assert evaluate(problem, source.genotype, source.schedule) == source.evaluation
                for repeat in range(2):
                    operators = {
                        "slow": ResearchPeak(positions="peak_rank", diagnose=False),
                        "fast": OnlinePeak(positions="peak_rank"),
                        "combined": OnlinePeak("all", "peak_rank"),
                    }
                    results, timings, states = {}, {}, {}
                    order = (
                        ("slow", "fast", "combined")
                        if repeat == 0
                        else ("combined", "fast", "slow")
                    )
                    for name in order:
                        rng = random.Random(55100 + repeat)
                        start = process_time()
                        results[name] = operators[name].propose(problem, source, 6, Config(), rng)
                        timings[name] = process_time() - start
                        states[name] = rng.getstate()
                    assert (
                        results["slow"].proposals
                        == results["fast"].proposals
                        == results["combined"].proposals
                    )
                    assert (
                        results["slow"].attempts
                        == results["fast"].attempts
                        == results["combined"].attempts
                    )
                    assert states["slow"] == states["fast"] == states["combined"]
                    for proposal in results["fast"].proposals:
                        assert proposal.schedule is not None
                        assert evaluate(problem, proposal.genotype, proposal.schedule).feasible
                    rows.append(
                        {
                            "file": path.name,
                            "panel": panel["panel"],
                            "repeat": repeat,
                            "cpu": timings,
                            "proposals": len(results["fast"].proposals),
                            "equivalent": True,
                        }
                    )
    total = {name: sum(r["cpu"][name] for r in rows) for name in ("slow", "fast", "combined")}
    result = {
        "scope": "8 previously seen development bases; CPU, not online quality",
        "pairs": len(rows),
        "total_cpu_seconds": total,
        "speedup": total["slow"] / total["fast"],
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.parent, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
