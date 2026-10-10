"""Bounded local engineering validation; these runs do not estimate P2 gains."""

import json
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.engine.run import run
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.p1_observation import SemanticRecorder
from geo_llm_scheduler.experiments.p2 import P2_ARMS, p2_config
from geo_llm_scheduler.experiments.runner import digest, git_metadata, source_digest
from geo_llm_scheduler.experiments.synthetic import synthetic
from geo_llm_scheduler.initialization.generators import initial_genotypes
from geo_llm_scheduler.utils.rng import RNGManager


def validate_local_p2(output: Path, seeds: tuple[int, ...] = (17, 23)) -> dict:
    """Run all arms twice with shared initialization and audit every exact return.

    Uses six synthetic jobs, population 12, three generations and always-trigger
    trajectories of length two to exercise pathways. No time-budget comparison,
    statistical claim, server connection or formal campaign is performed.
    Existing output is rejected to preserve prior evidence.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Nonempty unique validation seeds required")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in seeds:
        problem = synthetic(6, 2, 2, seed)
        base = Config(
            method="full",
            population=12,
            neighborhood=4,
            generations=3,
            rl_steps=2,
            trigger_mode="always",
            seed=seed,
        )
        initial = initial_genotypes(problem, base, RNGManager(seed).stream("initialization"))
        for arm in P2_ARMS:
            config = p2_config(arm, base)
            signatures = []
            audited_counts = []
            elapsed = []
            for _ in range(2):
                recorder = SemanticRecorder()
                audited = 0

                def observer(event: str, data: dict) -> None:
                    nonlocal audited
                    recorder(event, data)
                    if event == "candidate":
                        c = data["candidate"]
                        if c.evaluation != evaluate(problem, c.genotype, c.schedule):
                            raise AssertionError("Candidate differs from exact ground truth")
                        audited += 1

                result = run(problem, config, initial, retain_trace=False, observer=observer)
                if result.trace or result.offspring_count != 36:
                    raise AssertionError("Compact run lost offspring or retained history")
                if result.termination_reason != "generation_limit":
                    raise AssertionError("Unexpected local validation termination")
                if audited != result.gateway.counts["exact"]:
                    raise AssertionError("Missing exact candidate audit")
                signatures.append(recorder.signature(result))
                audited_counts.append(audited)
                elapsed.append(result.elapsed)
            if signatures[0] != signatures[1]:
                raise AssertionError(f"Nondeterministic replay: {arm}/{seed}")
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "config": asdict(config),
                    "config_hash": digest(asdict(config)),
                    "instance_hash": digest(asdict(problem)),
                    "initial_hash": digest([asdict(g) for g in initial]),
                    "signature_hash": digest(signatures[0]),
                    "replay_equal": True,
                    "audited_exact": audited_counts,
                    "elapsed_seconds": elapsed,
                    "counts": signatures[0]["counts"],
                    "archive_size": len(signatures[0]["archive"]),
                }
            )
    report = {
        "kind": "local_engineering_validation",
        "scientific_gain_verified": False,
        "server_deployed": False,
        "source_hash": source_digest(),
        **git_metadata(),
        "seeds": seeds,
        "arms": P2_ARMS,
        "runs": len(rows) * 2,
        "rows": rows,
    }
    (output / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
