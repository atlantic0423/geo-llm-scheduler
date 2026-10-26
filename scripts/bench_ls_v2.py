"""Layer-2 single-call microbenchmark of LS v2 prototypes against the existing operators.

Seeds are front members of completed equal-budget run Archives (end-of-run
states, where LS actually operates). Every operator call starts from the same frozen seed, all proposals go
through the normal EvaluationGateway (SSGS for structural, exact for all), and the
call is scored by the HV gain its feasible candidates add to the frozen Archive
front, in that Archive's own frozen frame. Descriptive only; no hypothesis test.

E1: operator comparison, independent calls (front not updated between calls).
E2: seed-rule comparison, a fixed operator portfolio applied sequentially with the
    front updated after every call (diminishing returns included).
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.domain.models import Candidate, Genotype, Schedule
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.rl23_support import study_config
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.destroy_repair import TargetedDestroyRepair
from geo_llm_scheduler.lsv2.flowsafe import FlowSafeBillMoves
from geo_llm_scheduler.lsv2.frontier import FrozenFrame, SeedSampler, demand_gate, nondominated
from geo_llm_scheduler.operators.active_pack import ActivePack
from geo_llm_scheduler.operators.peak_coalition import PeakCoalition
from geo_llm_scheduler.operators.right_shift import PolishStats, polish
from geo_llm_scheduler.operators.structural import StructuralOperator

OPERATORS = (
    *(f"A{a}" for a in range(1, 9)),
    "POLISH",
    "P3_FLOWSAFE",
    "P2_FLOW",
    "P2_TOU",
    "P2_PEAK",
    "P2_REGION",
    "P2_MIXED",
)
PORTFOLIO = ("A7", "A5", "A1", "P2_MIXED", "P3_FLOWSAFE")
RULES = ("uniform", "flow_end", "gap")


def load_members(path: Path, problem) -> list[Candidate]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        records = json.load(f)
    out = []
    for r in records:
        g = Genotype(tuple(r["genotype"]["ms"]), tuple(r["genotype"]["os"]))
        s = Schedule(
            *(tuple(r["schedule"][k]) for k in ("starts", "resource_wait", "intentional_wait"))
        )
        e = evaluate(problem, g, s)
        assert e.feasible
        out.append(Candidate(g, s, e, r.get("origin", "archive")))
    return out


def call(name: str, problem, seed: Candidate, frame: FrozenFrame, config, rng: random.Random):
    """Run one operator call; return (feasible candidates, seconds, exact count, proposals)."""
    captured: list[Candidate] = []
    gateway = EvaluationGateway(problem, Archive())
    gateway.candidate_sink = captured.append
    start = time.perf_counter()
    if name == "POLISH":
        polish(problem, seed, gateway, PolishStats())
        proposals = None
    else:
        if name.startswith("A"):
            a = int(name[1:])
            op = StructuralOperator(a) if a <= 6 else ActivePack() if a == 7 else PeakCoalition()
        elif name == "P3_FLOWSAFE":
            op = FlowSafeBillMoves()
        else:
            span = (frame.nadir[0] - frame.ideal[0], frame.nadir[1] - frame.ideal[1])
            op = TargetedDestroyRepair(
                mode=name.split("_", 1)[1],
                weight=frame.weight_for(seed.evaluation.objectives),
                scale=span,
            )
        batch = op.propose(problem, seed, 6, config, rng)
        proposals = len(batch.proposals)
        seen = set()
        for prop in batch.proposals[:6]:
            key = (prop.genotype, None if prop.schedule is None else prop.schedule.starts)
            if key in seen:
                continue
            seen.add(key)
            gateway.evaluate(prop.genotype, prop.schedule, name)
    seconds = time.perf_counter() - start
    feasible = [c for c in captured if c.evaluation.feasible]
    return feasible, seconds, gateway.counts["exact"], proposals


def run_input(task: dict) -> dict:
    problem = load_instance(task["input"])
    members = load_members(Path(task["archive"]), problem)
    by_obj: dict = {}
    for c in members:
        by_obj.setdefault(c.evaluation.objectives, c)
    front = nondominated(list(by_obj))
    frame = FrozenFrame.of(front)
    config = study_config("STRICT_TWO_FOCUS", 16502, 2400)
    rng = random.Random(task["seed"])
    seeds = [
        by_obj[front[i]] for i in rng.sample(range(len(front)), min(task["seeds"], len(front)))
    ]
    rows = []
    # E1：每个算子从同一组起点各调用一次，互不影响
    for name in OPERATORS:
        for k, seed in enumerate(seeds):
            cands, sec, exact, props = call(
                name, problem, seed, frame, config, random.Random(task["seed"] * 1000 + k)
            )
            pts = [c.evaluation.objectives for c in cands]
            gain = frame.hv_gain(front, pts)
            rows.append(
                dict(
                    exp="E1",
                    key=task["key"],
                    n=task["n"],
                    op=name,
                    seed=k,
                    seed_x=frame.normalize(seed.evaluation.objectives)[0],
                    gate=demand_gate(seed),
                    hv_gain=gain,
                    success=gain > 1e-12,
                    seconds=sec,
                    exact=exact,
                    proposals=props,
                    candidates=len(pts),
                )
            )
    # E2：三种起点规则，固定算子组合轮换，前沿随调用更新
    for rule in RULES:
        sampler = SeedSampler(rule, random.Random(task["seed"] + 7))
        current = list(front)
        pool = dict(by_obj)
        total_sec = 0.0
        for k in range(task["e2_calls"]):
            i = sampler.pick(frame, current)
            point = current[i]
            seed = pool[point]
            name = PORTFOLIO[k % len(PORTFOLIO)]
            cands, sec, exact, _ = call(
                name, problem, seed, frame, config, random.Random(task["seed"] * 7919 + k)
            )
            pts = [c.evaluation.objectives for c in cands]
            gain = frame.hv_gain(current, pts)
            sampler.report(point, gain > 1e-12)
            # 新产出的完整解并入前沿，之后可以继续作为起点
            for c in cands:
                pool.setdefault(c.evaluation.objectives, c)
            current = nondominated(current + pts)
            total_sec += sec
            rows.append(
                dict(
                    exp="E2",
                    key=task["key"],
                    n=task["n"],
                    op=name,
                    rule=rule,
                    call=k,
                    seed_x=frame.normalize(point)[0],
                    hv_gain=gain,
                    success=gain > 1e-12,
                    seconds=sec,
                    exact=exact,
                )
            )
        rows.append(
            dict(
                exp="E2_total",
                key=task["key"],
                n=task["n"],
                rule=rule,
                hv_gain=frame.hv(current) - frame.hv(front),
                seconds=total_sec,
            )
        )
    return dict(key=task["key"], n=task["n"], arm=task["arm"], front=len(front), rows=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-root", type=Path, required=True)
    parser.add_argument("--suffix", default="s16502")
    parser.add_argument("--arms", default="BILL_RANDOM,MAIN_RANDOM")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--e2-calls", type=int, default=30)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    tasks = []
    for topo in ("R2M12_kv", "R3M24_arrival", "R3M6_arrival", "R5M12_arrival"):
        for n in (20, 50, 100, 200):
            for variant in ("v0", "v1"):
                key = f"r0_n{n}_{topo}_{variant}_{args.suffix}"
                run = next(
                    (
                        args.long_root / "runs" / key / arm
                        for arm in args.arms.split(",")
                        if (args.long_root / "runs" / key / arm / "complete.json").exists()
                    ),
                    None,
                )
                if run is None:
                    continue
                spec = json.loads((args.long_root / "specs" / f"{key}.json").read_text())
                tasks.append(
                    dict(
                        key=key,
                        n=n,
                        arm=run.name,
                        archive=str(run / "archive.json.gz"),
                        input=str(args.long_root / spec["input"]),
                        seed=zlib.crc32(key.encode()) % 100003,
                        seeds=args.seeds,
                        e2_calls=args.e2_calls,
                    )
                )
                break
    tasks.sort(key=lambda t: -t["n"])
    args.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_input, tasks))
    (args.out / "bench_rows.json").write_text(json.dumps(results), encoding="utf-8")
    print(json.dumps(dict(inputs=len(results), rows=sum(len(r["rows"]) for r in results))))


if __name__ == "__main__":
    main()
