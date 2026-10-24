"""Execute one explicitly configured Jev epsilon arm; no automatic paired campaign."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.jev_decisions import JevSettings
from geo_llm_scheduler.experiments.jev_search import run_policy, study_config
from geo_llm_scheduler.io.loaders import load_instance


def main() -> None:
    """Write replay evidence and independently audit every final population/Archive member."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--epsilon-g", type=float, choices=(0.5, 0.9), required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=4_000_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence: dict = {}
    problem = load_instance(args.instance)
    arm = "JEV_E05" if args.epsilon_g == 0.5 else "JEV_E09"
    settings = JevSettings(
        epsilon_g=args.epsilon_g,
        max_requests=args.max_requests,
        max_input_tokens=args.max_input_tokens,
    )
    with (args.output / "requests.jsonl").open("w", encoding="utf-8") as handle:

        def sink(record: dict) -> None:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()

        result = run_policy(
            problem,
            arm,
            study_config(arm, args.seed, args.seconds),
            evidence,
            settings=settings,
            request_sink=sink,
        )
    for member in result.population + result.archive.members:
        exact = evaluate(problem, member.genotype, member.schedule)
        if exact != member.evaluation or not exact.feasible:
            raise ValueError("Final independent exact audit failed")
    payload = dict(
        arm=arm,
        settings=asdict(settings),
        seed=args.seed,
        elapsed_s=result.elapsed,
        termination_reason=result.termination_reason,
        final_exact_audit=True,
        jev_treatment_valid=evidence["jev"]["jev_treatment_valid"],
        population=[asdict(c) for c in result.population],
        archive=[asdict(c) for c in result.archive.members],
        evidence=evidence,
    )
    (args.output / "result.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    if not evidence["jev"]["jev_treatment_valid"]:
        raise RuntimeError("No valid Jev action decision; invalid treatment evidence preserved")
    print(json.dumps(dict(arm=arm, elapsed_s=result.elapsed, final_exact_audit=True)))


if __name__ == "__main__":
    main()
