"""Compare G1 with BILL4 under common retention and the same local runtime."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.initial_gap_construction import gap_initializer
from geo_llm_scheduler.experiments.initial_pool_retention import pool_initializer
from geo_llm_scheduler.experiments.s4_ablation import arm_config
from geo_llm_scheduler.io.loaders import load_instance
from geo_llm_scheduler.lsv2.frontier import FrozenFrame, nondominated
from geo_llm_scheduler.utils.rng import RNGManager


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def _check(key: str, campaign: Path, out: Path) -> str:
    spec = _read(campaign / "formal/specs" / (key + ".json"))
    path = campaign / "formal" / spec["input"]
    expected = _read(campaign / "server_startup/formal/preflight" / (key + ".json"))["policies"][
        "bill4"
    ]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["input_sha256"]
    cfg = arm_config("A0_FULL_S4", spec["seed"], spec["seconds"])
    pb = load_instance(path)
    pair = {}
    order = ("bill4", "g1") if int(key.rsplit("r", 1)[-1]) % 2 else ("g1", "bill4")
    for name in order:
        gw = EvaluationGateway(pb, Archive(), decode_cache_size=cfg.decode_cache_size)
        trace: list = []
        gw.candidate_sink = trace.append
        e: dict = {}
        factory = gap_initializer if name == "g1" else pool_initializer
        selected = factory(pb, cfg, e)(gw, RNGManager(cfg.seed))
        assert len(trace) == len({c.genotype for c in trace}) == gw.counts["exact"] == 140
        assert len(selected) == 100
        for c in trace:
            assert evaluate(pb, c.genotype, c.schedule) == c.evaluation
        front = nondominated([c.evaluation.objectives for c in gw.archive.members])
        assert set(front) <= {c.evaluation.objectives for c in selected}
        if name == "bill4":
            assert e["legacy_initial_population_sha256"] == expected["initial_population_sha256"]
            assert e["initial_archive_sha256"] == expected["initial_archive_sha256"]
        pair[name] = (trace, e, front)
    old_trace, old, before = pair["bill4"]
    trace, e, after = pair["g1"]
    assert trace[:100] == old_trace[:100]
    both = before + after
    frame = FrozenFrame(tuple(map(min, zip(*both))), tuple(map(max, zip(*both))))

    def middle(front: list[tuple[float, float]]) -> int:
        return sum(1 / 3 <= frame.normalize(p)[0] < 2 / 3 for p in front)

    base_hv, new_hv = frame.hv(before), frame.hv(after)
    # Recompute hits using only the earlier points plus that round's four trials.
    past = [c.evaluation.objectives for c in trace[:100]]
    hits = targets = 0
    for row in e["rounds"]:
        past += row["points"]
        if row["edge"] is not None:
            targets += 1
            left, right = row["edge"]
            current = set(nondominated(past))
            hits += any(
                left[0] < f < right[0] and right[1] < b < left[1] and (f, b) in current
                for f, b in row["points"]
            )
    record = dict(
        key=key,
        n=spec["n"],
        seed=cfg.seed,
        input_sha256=spec["input_sha256"],
        exact_checks=len(old_trace) + len(trace),
        order=order,
        old_seconds=old["initialization_seconds"],
        new_seconds=e["initialization_seconds"],
        old_archive_sha256=old["initial_archive_sha256"],
        original_matches=100,
        old_front=before,
        new_front=after,
        old_hv=base_hv,
        new_hv=new_hv,
        hv_delta=new_hv - base_hv,
        old_middle=middle(before),
        new_middle=middle(after),
        targeted_rounds=targets,
        targeted_hits=hits,
        evidence=e,
    )
    (out / (key + ".json")).write_text(json.dumps(record), encoding="utf8")
    return key


def main() -> None:
    """Audit the whole manifest with explicit local paths and a bounded worker count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    keys = _read(args.campaign / "formal/manifest.json")["blocks"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_check, k, args.campaign, args.out) for k in keys]
        for f in as_completed(futures):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
