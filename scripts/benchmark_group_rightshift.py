"""Fixed-input cost and exact audit for three right-shift neighborhoods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from time import perf_counter

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, Genotype, Schedule
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.group_polish import ConcurrentRightShiftPolish
from geo_llm_scheduler.lsv2.polish import FastRightShiftPolish
from geo_llm_scheduler.utils.numeric import TOL, close


def main() -> None:
    """Audit recovered physical inputs; biased hotspot samples are not online trials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    unique = {(r["key"], r["snapshot"]["physical_fingerprint"]): r for r in samples}
    rows = []
    for (key, fingerprint), row in unique.items():
        spec = json.loads((args.historical / "specs" / (key + ".json")).read_text())
        problem = load_instance(args.historical / spec["input"])
        raw = row["snapshot"]["input_schedule"]
        g, s = Genotype(tuple(raw["ms"]), tuple(raw["os"])), Schedule(tuple(raw["starts"]))
        current = Candidate(g, s, evaluate(problem, g, s))
        assert current.evaluation.feasible
        for method, operator in (
            ("single", FastRightShiftPolish()),
            ("common", ConcurrentRightShiftPolish()),
            ("coordinated", ConcurrentRightShiftPolish(coordinated=True)),
        ):
            begin = perf_counter()
            batch = operator.propose(problem, current, 3, Config(), Random(16901))
            seconds = perf_counter() - begin
            values = []
            for prop in batch.proposals:
                assert prop.schedule is not None
                ev = evaluate(problem, prop.genotype, prop.schedule)
                assert ev.feasible and ev.flow == current.evaluation.flow
                assert ev.bill < current.evaluation.bill
                assert prop.schedule.starts[1::2] == s.starts[1::2]
                assert all(a >= b for a, b in zip(prop.schedule.starts, s.starts))
                values.append(ev.bill - current.evaluation.bill)
            if method != "single":
                assert all(
                    close(a, b, TOL.cost)
                    for a, b in zip(values, batch.diagnostics["incremental_deltas"])
                )
            rows.append(
                dict(
                    key=key,
                    fingerprint=fingerprint,
                    method=method,
                    seconds=seconds,
                    deltas=values,
                    diagnostics=batch.diagnostics,
                )
            )
            atomic_json(args.out, dict(scope="fixed biased hotspots, not online effect", rows=rows))
    print(
        json.dumps(
            {
                m: dict(
                    inputs=sum(r["method"] == m for r in rows),
                    improved=sum(r["method"] == m and bool(r["deltas"]) for r in rows),
                    seconds=sum(r["seconds"] for r in rows if r["method"] == m),
                )
                for m in ("single", "common", "coordinated")
            }
        )
    )


if __name__ == "__main__":
    main()
