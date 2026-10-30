"""Replay fixed real inputs against frozen COMMON, recording exact and RNG equality."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from random import Random
from time import perf_counter

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.group_polish import ConcurrentRightShiftPolish


def main() -> None:
    """Measure cold/repeated construction separately; this is not online quality evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--golden", type=Path)
    args = parser.parse_args()
    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    gold = json.loads(args.golden.read_text())["rows"] if args.golden else None
    problems, rows = {}, []
    for i, item in enumerate(samples):
        key = item["key"]
        if key not in problems:
            spec = json.loads((args.historical / "specs" / (key + ".json")).read_text())
            problems[key] = load_instance(args.historical / spec["input"])
        problem = problems[key]
        raw = item["input_schedule"]
        g, s = Genotype(tuple(raw["ms"]), tuple(raw["os"])), Schedule(tuple(raw["starts"]))
        c = Candidate(g, s, evaluate(problem, g, s))
        assert c.evaluation.feasible
        operators = [("reference", ConcurrentRightShiftPolish())]
        if not args.reference_only:
            from geo_llm_scheduler.lsv2.common_refine import ReusedCommonPolish

            operators += [("fast", ReusedCommonPolish()), ("peak", ReusedCommonPolish(True))]
        for repeat in range(3):
            reference_signature = None
            for name, op in operators:
                rng = Random(16901 + repeat)
                tick = perf_counter()
                batch = op.propose(problem, c, 3, Config(), rng)
                seconds = perf_counter() - tick
                signature = dict(
                    proposals=[asdict(p) for p in batch.proposals],
                    rng_sha256=hashlib.sha256(repr(rng.getstate()).encode()).hexdigest(),
                )
                signature = json.loads(json.dumps(signature))
                if name == "reference":
                    reference_signature = signature
                    if gold:
                        assert signature == gold[i * 3 + repeat]["signature"], (i, repeat)
                if name == "fast":
                    assert signature == reference_signature, (i, repeat, name)
                bills = []
                for prop in batch.proposals:
                    ev = evaluate(problem, prop.genotype, prop.schedule)
                    assert ev.feasible and ev.flow == c.evaluation.flow
                    assert ev.bill < c.evaluation.bill
                    assert prop.schedule.starts[1::2] == c.schedule.starts[1::2]
                    bills.append(ev.bill)
                row = dict(
                    sample=i,
                    key=key,
                    repeat=repeat,
                    method=name,
                    seconds=seconds,
                    best_bill=min([c.evaluation.bill, *bills]),
                    signature=signature,
                    diagnostics=batch.diagnostics,
                )
                rows.append(row)
            if not args.reference_only:
                assert rows[-1]["best_bill"] <= rows[-2]["best_bill"] + 1e-9
        atomic_json(args.out, dict(accepted=False, rows=rows))
    totals = {
        name: dict(
            cold=sum(r["seconds"] for r in rows if r["method"] == name and r["repeat"] == 0),
            repeat=sum(r["seconds"] for r in rows if r["method"] == name and r["repeat"] > 0),
        )
        for name in {r["method"] for r in rows}
    }
    atomic_json(args.out, dict(accepted=True, inputs=len(samples), rows=rows, seconds=totals))
    print(json.dumps(dict(inputs=len(samples), seconds=totals)))


if __name__ == "__main__":
    main()
