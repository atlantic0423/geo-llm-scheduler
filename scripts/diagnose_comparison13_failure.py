"""Replay only a frozen failed initializer and inspect it without changing limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import FrameType
from typing import Any

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments.comparison13 import arm_config, initialization_factory
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.utils.rng import RNGManager


def diagnose(raw: Path, output: Path) -> dict:
    """Reproduce the recorded error, preserving source, seed, quota and attempt limit."""
    key = "r0_n20_R5M12_cheap_capacity_v1_s16101"
    spec = json.loads((raw / "specs" / (key + ".json")).read_text())
    problem = load_instance(raw / spec["input"])
    gateway = EvaluationGateway(problem, Archive())
    calls: list[dict] = []
    failed: dict[str, Any] = {}

    def trace(frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code.co_name == "guided_genotype" and event == "return" and arg:
            parent = frame.f_back
            if parent is not None and parent.f_code.co_name == "seed_bank":
                loc = parent.f_locals
                genotype, evidence = arg
                calls.append(
                    dict(
                        slot=loc["slot"],
                        weight=loc["weight"],
                        attempt=loc["attempt"],
                        duplicate=genotype in loc["seen"],
                        seen=len(loc["seen"]),
                        style=evidence["order_style"],
                        sha256=hashlib.sha256(repr(genotype).encode()).hexdigest(),
                    )
                )
        if frame.f_code.co_name == "seed_bank" and event == "exception":
            if isinstance(arg[1], ValueError):
                loc = frame.f_locals
                failed.update(
                    slot=loc["slot"],
                    attempt=loc["attempt"],
                    pool_size=len(loc["pool"]),
                    required=loc["trials"],
                    seen=len(loc["seen"]),
                    completed_slots=len(loc["members"]),
                    accepted_trials=len(loc["observations"]),
                    error=str(arg[1]),
                )
        return trace

    old = sys.gettrace()
    config = arm_config("EPS_IBEA", spec["seed"], spec["seconds"])
    try:
        sys.settrace(trace)
        initialization_factory(problem, config, 4, spec["bank_seed"], {})(
            gateway, RNGManager(config.seed)
        )
    except ValueError as exc:
        if str(exc) != "Unique seed trials unavailable within frozen attempt limit":
            raise
    else:
        raise AssertionError("Frozen failure did not reproduce")
    finally:
        sys.settrace(old)
    slots = []
    for slot in sorted({c["slot"] for c in calls}):
        rows = [c for c in calls if c["slot"] == slot]
        slots.append(
            dict(
                slot=slot,
                weight=rows[0]["weight"],
                calls=len(rows),
                accepted=sum(not c["duplicate"] for c in rows),
                unique_generated=len({c["sha256"] for c in rows}),
                styles=dict(Counter(c["style"] for c in rows)),
            )
        )
    result = dict(
        reproduced=True,
        key=key,
        arm="EPS_IBEA",
        seed=spec["seed"],
        bank_seed=spec["bank_seed"],
        failure=failed,
        slots=slots,
        calls=calls,
        exact=gateway.counts["exact"],
        context="initializer only; no evolution, limit change, fallback, or formal retry",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.raw, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "calls"}))
