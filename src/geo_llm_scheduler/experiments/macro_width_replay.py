"""Fresh isolated exact replay of saved calls; never feeds the online solver."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, ProblemInstance, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.campaign.validation import file_hash
from geo_llm_scheduler.experiments.campaign.worker import TUPLE_FIELDS
from geo_llm_scheduler.experiments.macro_width import gzip_json, read_json
from geo_llm_scheduler.experiments.macro_width_worker import validate_complete
from geo_llm_scheduler.experiments.runner import digest
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.macrosearch.search import execute
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.operators.active_pack import ActivePack
from geo_llm_scheduler.operators.base import ProposalBatch
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition
from geo_llm_scheduler.operators.structural import StructuralOperator
from geo_llm_scheduler.utils.numeric import TOL, identity


def tuples(value: Any) -> Any:
    """Restore JSON-encoded RNG tuples without pickle or executing external data."""
    return tuple(tuples(v) for v in value) if isinstance(value, list) else value


def proposal_data(batch: ProposalBatch) -> list[dict]:
    """Return the frozen proposal identities, including selected timing operations."""
    return [
        {
            "ms": list(p.genotype.ms),
            "os": list(p.genotype.os),
            "starts": None if p.schedule is None else list(p.schedule.starts),
            "selected": list(p.selected),
        }
        for p in batch.proposals
    ]


def replay_call(problem: ProblemInstance, config: Config, row: dict) -> dict:
    """Verify original construction, then fresh-evaluate the common ten-candidate pool."""
    start = time.perf_counter()
    source = row["incumbent"]
    genotype = Genotype(tuple(source["ms"]), tuple(source["os"]))
    schedule = Schedule(
        tuple(source["starts"]), tuple(source["resource_wait"]), tuple(source["intentional_wait"])
    )
    evaluation = evaluate(problem, genotype, schedule)
    if (
        evaluation.objectives != tuple(source["objectives"])
        or evaluation.feasible != source["feasible"]
    ):
        raise ValueError("Saved incumbent exact mismatch")
    incumbent = Candidate(genotype, schedule, evaluation)
    action = row["action"]
    op = (
        StructuralOperator(action)
        if action < 7
        else ActivePack()
        if action == 7
        else PeakCoalition()
    )
    rng = random.Random(0)
    rng.setstate(tuples(row["rng_before"]))
    original = op.propose(problem, incumbent, row["generation_budget"], config, rng)
    if (
        proposal_data(original) != row["proposals"]
        or original.attempts != row["attempts"]
        or rng.getstate() != tuples(row["rng_after"])
    ):
        raise ValueError("Original batch/RNG replay mismatch")
    rng = random.Random(0)
    rng.setstate(tuples(row["rng_before"]))
    tick = time.perf_counter()
    full = op.propose(problem, incumbent, 10, config, rng)
    construction_seconds = time.perf_counter() - tick
    gateway = EvaluationGateway(problem, Archive())
    gateway.ideal = tuple(row["ideal"])
    context = NormalizationContext(tuple(row["ideal"]), tuple(row["maximum"]))
    weight = tuple(row["weight"])
    result = execute(full, incumbent, 10, weight, context, gateway, "offline_probe")
    common = result.context
    before = common.scalar(incumbent, weight)
    prefix = {}
    for budget in (3, 6, 10):
        candidates = result.evaluated[:budget]
        score = min(
            [before] + [common.scalar(c, weight) for c in candidates if c.evaluation.feasible]
        )
        prefix[str(budget)] = {
            "evaluated": len(candidates),
            "gain": before - score,
            "relative_gain": (before - score) / (before + 1e-12),
        }
    # Native6 has a different generator; verify its actual evaluated prefix separately.
    reference_gateway = EvaluationGateway(problem, Archive())
    reference_gateway.ideal = tuple(row["ideal"])
    online = execute(
        original, incumbent, row["budget"], weight, context, reference_gateway, "offline_original"
    )
    if len(online.evaluated) != len(row["online_evaluated"]):
        raise ValueError("Online candidate count mismatch")
    for candidate, saved in zip(online.evaluated, row["online_evaluated"]):
        if (
            candidate.genotype.ms != tuple(saved["ms"])
            or candidate.genotype.os != tuple(saved["os"])
            or candidate.schedule.starts != tuple(saved["starts"])
            or candidate.evaluation.objectives != tuple(saved["objectives"])
            or candidate.evaluation.feasible != saved["feasible"]
        ):
            raise ValueError("Online exact prefix mismatch")
    # Use the original implementation, including prefix-specific temporary ideal.
    seq_gateway = EvaluationGateway(problem, Archive())
    seq_gateway.ideal = tuple(row["ideal"])
    sequential = execute(
        full, incumbent, 10, weight, context, seq_gateway, "offline_seq", sequential=True
    )
    selected = [c for c in sequential.evaluated if c.evaluation.feasible]
    seq_common_score = min([before] + [common.scalar(c, weight) for c in selected])
    return {
        "action": action,
        "state": row["state"],
        "progress": row["progress"],
        "prefix": prefix,
        "late_4_6_after_zero3": prefix["3"]["gain"] <= TOL.scalar
        and prefix["6"]["gain"] > TOL.scalar,
        "late_7_10_after_zero6": prefix["6"]["gain"] <= TOL.scalar
        and prefix["10"]["gain"] > TOL.scalar,
        "seq": sequential.sequential,
        "seq_common_gain": before - seq_common_score,
        "seq_missed_gain": prefix["10"]["gain"] - (before - seq_common_score),
        "full_objectives": [
            c.evaluation.objectives for c in result.evaluated if c.evaluation.feasible
        ],
        "archive_before_objectives": row["archive_objectives"],
        "original_identity": identity(online.best) if online.best else None,
        "common_generation_seconds": construction_seconds,
        "offline_timings": dict(gateway.seconds),
        "total_replay_seconds": time.perf_counter() - start,
        "online_rng_verified": True,
    }


def replay_run(root: Path, key: str) -> dict:
    """Bind an idempotent offline result to every original compressed online artifact."""
    spec = read_json(root / "specs" / (key + ".json"))
    output = root / "runs" / key
    valid, reason = validate_complete(output, spec)
    if not valid:
        raise ValueError(reason)
    target = root / "replay" / (key + ".json.gz")
    marker = target.with_suffix(".complete.json")
    original_hash = file_hash(output / "complete.json")
    if marker.exists():
        checked = read_json(marker)
        if checked["source_marker_hash"] != original_hash or checked["sha256"] != file_hash(target):
            raise ValueError("Existing replay changed; retain evidence")
        return checked
    config_data = dict(spec["config"])
    for field in TUPLE_FIELDS:
        config_data[field] = tuple(config_data[field])
    config = Config(**config_data)
    problem = load_instance(root / spec["instance"])
    sample = read_json(output / "diagnostics.json.gz")
    rows = {bucket: replay_call(problem, config, row) for bucket, row in sample["calls"].items()}
    target.parent.mkdir(exist_ok=True)
    gzip_json(
        target,
        {
            "schema": 1,
            "key": key,
            "counts": sample["counts"],
            "calls": rows,
            "source_marker_hash": original_hash,
            "input_hash": digest(spec),
            "online_feedback": False,
        },
    )
    evidence = {
        "key": key,
        "source_marker_hash": original_hash,
        "sha256": file_hash(target),
        "calls": len(rows),
    }
    atomic_json(marker, evidence)
    return evidence


if __name__ == "__main__":
    root, key = Path(sys.argv[1]).resolve(), sys.argv[2]
    try:
        print(json.dumps(replay_run(root, key)))
    except Exception as error:
        atomic_json(
            root / "ops/failures" / ("replay_" + key + ".json"),
            {"error": repr(error), "timestamp": time.time()},
        )
        raise
