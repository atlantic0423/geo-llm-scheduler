"""Read-only reduction of every P1 event and initialization control."""

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import time

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.evaluation import EvaluationGateway
from geo_llm_scheduler.experiments.p1_observation import restore_candidate
from geo_llm_scheduler.io.loaders import load_instance
from research.p1.reduce import portable_path


def one(arg: tuple[Path, int, dict]) -> dict:
    """Audit a complete stream with bounded accumulators and an initialization control."""
    base, node, item = arg
    root = base / f"campaign_p1_20261002_node{node}"
    d = root / "runs" / item["key"]
    spec = json.loads(portable_path(base, item["path"]).read_text())
    problem = load_instance(portable_path(base, spec["instance_path"]))
    gateway = EvaluationGateway(problem, Archive())
    control = Counter()
    maxdiff = 0.0
    with gzip.open(d / "initial_population.json.gz", "rt") as f:
        initial = json.load(f)
    for x in initial:
        c = restore_candidate(x)
        dec = gateway.evaluate(c.genotype, origin="initial_control")
        control["total"] += 1
        control["starts_changed"] += c.schedule.starts != dec.schedule.starts
        diff = max(abs(a - b) for a, b in zip(c.evaluation.objectives, dec.evaluation.objectives))
        maxdiff = max(diff, maxdiff)
        control["objective_changed"] += diff > 1e-9
    repl = Counter()
    gains = defaultdict(float)
    actions = {i: Counter() for i in range(1, 9)}
    states = Counter()
    front = {}
    overshoot_front = {}
    candidate_origins = Counter()
    seconds = spec["config"]["seconds"]
    last_gen = -1
    with gzip.open(d / "events.jsonl.gz", "rt") as f:
        for line in f:
            x = json.loads(line)
            event = x["event"]
            if event == "replacement":
                improving = x["improving"]
                outside = x["outside_birth"]
                repl["total"] += 1
                repl["improving"] += bool(improving)
                repl["outside"] += bool(outside)
                repl["missed"] += bool(improving) and not x["original"]
                repl["cap"] += x["cap_truncated"] > 0
                repl["pairs"] += len(improving)
                repl["outside_pairs"] += len(outside)
                repl["cap_pairs"] += x["cap_truncated"]
                repl["direction_distance"] += abs(x["direction"] - x["birth"])
                repl["direction_missed"] += bool(improving) and not x["direction_neighbors"]
                for k in ["original", "random", "gain_order", "direction_neighbors"]:
                    repl[k] += len(x[k])
                    gains[k] += sum(x["gains"][j] for j in x[k])
            elif event == "offspring":
                last_gen = max(last_gen, x["generation"])
                for step in x["steps"]:
                    a = actions[step["action"]]
                    a["calls"] += 1
                    a["accepted"] += step["accepted"]
                    a["empty"] += step["effective"] == 0
                    a["effective"] += step["effective"]
                    a["explore"] += step["explore"]
                    a["reward"] += step["reward"]
                    a["seconds"] += step["seconds"]
                    a["construction_seconds"] += step["construction_seconds"]
                    a["archive_net_retained"] += step["archive_net_retained"]
                    a["archive_only"] += not step["accepted"] and step["archive_net_retained"] > 0
                    for k, v in step.get("operator_instrumentation", {}).items():
                        if isinstance(v, (int, float)):
                            a["instrument:" + k] += v
                    states[step["state"]] += 1
            elif event == "candidate" and x["feasible"]:
                candidate_origins[x["origin"]] += 1
                flow, bill = x["flow"], x["bill"]
                overshoot_front[flow] = min(bill, overshoot_front.get(flow, float("inf")))
                if x["elapsed"] <= seconds:
                    front[flow] = min(bill, front.get(flow, float("inf")))

    def pareto(points: dict[float, float]) -> list[list[float]]:
        """Deduplicate objectives for descriptive metrics without pruning raw phenotypes."""
        best = float("inf")
        out = []
        for flow, bill in sorted(points.items()):
            if bill < best:
                out.append([flow, bill])
                best = bill
        return out

    signature = json.loads((d / "semantic_signature.json").read_text())
    visited = sum(v > 0 for v in signature["visits"])
    return {
        **{k: item[k] for k in ["key", "arm", "jobs", "base_seed", "algorithm_seed", "tariff"]},
        "node": node,
        "initial_control": dict(control),
        "initial_maxdiff": maxdiff,
        "replacement": dict(repl),
        "replacement_gain": dict(gains),
        "actions": {str(k): dict(v) for k, v in actions.items()},
        "states": dict(states),
        "last_generation": last_gen,
        "q_visited": visited,
        "visits": signature["visits"],
        "selections": signature["selections"],
        "candidate_origins": dict(candidate_origins),
        "front_at_T": pareto(front),
        "front_soft_stop": pareto(overshoot_front),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    base = parser.parse_args().base
    items = []
    for n in (0, 1):
        m = json.loads((base / f"campaign_p1_20261002_node{n}/manifest.json").read_text())
        items.extend((base, n, x) for x in m["specs"] if x["stage"] == "P1")
    out = base / "p1_analysis_20261002/online.jsonl.gz"
    begin = time()
    with gzip.open(out, "wt", compresslevel=1) as f, ProcessPoolExecutor(max_workers=8) as pool:
        for i, row in enumerate(pool.map(one, items, chunksize=1)):
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            f.flush()
            print(json.dumps({"online_reduced": i + 1, "seconds": time() - begin}), flush=True)
    manifest = {
        "file": out.name,
        "size": out.stat().st_size,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "runs": len(items),
        "elapsed": time() - begin,
    }
    out.with_name("online_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
