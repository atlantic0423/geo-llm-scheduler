"""Reduce frozen P1 diagnostics without altering any search run or candidate."""

import argparse
import gzip
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path


def objectives(candidate: dict) -> tuple[float, float]:
    """Recover Flow and the complete Bill from a serialized exact candidate."""
    e = candidate["evaluation"]
    return (e["flow"], e["tou"] + sum(e["demand"]))


def score(
    point: Sequence[float], ideal: Sequence[float], maximum: Sequence[float], index: int
) -> float:
    """Match frozen P1 normalized Tchebycheff, including endpoint protection."""
    w = (index / 99, 1 - index / 99)
    return max(w[k] * abs((point[k] - ideal[k]) / max(maximum[k] - ideal[k], 1e-9)) for k in (0, 1))


def positive(value: float) -> bool:
    """Use the frozen absolute scalar audit tolerance."""
    return value > 1e-9


def portable_path(base: Path, original: str) -> Path:
    """Relocate archived campaign paths without changing their relative names."""
    parts = Path(original).parts
    for i, part in enumerate(parts):
        if part.startswith("campaign_p1_20261002_node"):
            return base.joinpath(*parts[i:])
    raise ValueError("Input is outside the archived P1 campaigns")


def prefix_improvement(
    parent: Sequence[float],
    candidates: list[tuple[float, float]],
    ideal: Sequence[float],
    maximum: Sequence[float],
    index: int,
    early: int,
    late: int,
) -> float:
    """Isolate additional candidates by scoring both prefixes in one context."""
    original = score(parent, ideal, maximum, index)
    first = min([original] + [score(p, ideal, maximum, index) for p in candidates[:early]])
    second = min([original] + [score(p, ideal, maximum, index) for p in candidates[:late]])
    return first - second


def reduce(base: Path) -> None:
    """Reduce complete frozen raw diagnostics into five bounded JSONL streams."""
    out = base / "p1_analysis_20261002"
    out.mkdir(exist_ok=True)
    handles = {
        k: gzip.open(out / f"{k}.jsonl.gz", "wt", encoding="utf-8", compresslevel=1)
        for k in ["runs", "h1", "actions", "h3", "fronts"]
    }
    counts = {k: 0 for k in handles}

    def write(kind: str, row: dict) -> None:
        handles[kind].write(json.dumps(row, separators=(",", ":")) + "\n")
        counts[kind] += 1

    for node in [0, 1]:
        root = base / f"campaign_p1_20261002_node{node}"
        manifest = json.loads((root / "manifest.json").read_text())
        items = [x for x in manifest["specs"] if x["stage"] == "P1"]
        for item in items:
            d = root / "runs" / item["key"]
            summary = json.loads((d / "summary.json").read_text())
            spec = json.loads(portable_path(base, item["path"]).read_text())
            meta = {
                **{
                    k: item[k]
                    for k in ["key", "arm", "jobs", "base_seed", "algorithm_seed", "tariff"]
                },
                "node": node,
            }
            write("runs", {**meta, "summary": summary, "config": spec["config"]})
            for fraction in [0.1, 0.5, 0.9]:
                with gzip.open(d / f"checkpoint_{fraction}.json.gz", "rt") as f:
                    cp = json.load(f)
                write(
                    "fronts",
                    {
                        **meta,
                        "fraction": fraction,
                        "elapsed": cp["elapsed"],
                        "points": [objectives(c) for c in cp["archive"]],
                        "origins": [c["origin"] for c in cp["archive"]],
                        "population_origins": [c["origin"] for c in cp["population"]],
                        "ideal": cp["ideal"],
                        "maximum": cp["maximum"],
                        "sample": cp["sample"],
                    },
                )
                with gzip.open(d / f"diagnostic_{fraction}.json.gz", "rt") as f:
                    rows = json.load(f)
                for s, row in enumerate(rows):
                    smeta = {
                        **meta,
                        "fraction": fraction,
                        "sample_id": s,
                        "stratum": row["sample"]["stratum"],
                    }
                    if row.get("missing"):
                        write("h1", {**smeta, "missing": True, "reason": row["sample"]["missing"]})
                        continue
                    inc = row["incumbent"]
                    op = objectives(inc)
                    dec = objectives(row["redecoded"])
                    index = row["sample"]["index"]
                    ideal = cp["ideal"]
                    maximum = cp["maximum"]
                    h1 = {
                        **smeta,
                        "missing": False,
                        "index": index,
                        "origin": inc["origin"],
                        "bill_loss": row["bill_loss_ratio"],
                        "flow_change": (dec[0] - op[0]) / max(op[0], 1e-9),
                        "tou_change": row["redecoded"]["evaluation"]["tou"]
                        - inc["evaluation"]["tou"],
                        "demand_change": sum(row["redecoded"]["evaluation"]["demand"])
                        - sum(inc["evaluation"]["demand"]),
                        "scalar_change": row["scalar_redecoded"] - row["scalar_original"],
                        "scalar_relative_change": (row["scalar_redecoded"] - row["scalar_original"])
                        / max(abs(row["scalar_original"]), 1e-9),
                        "timing_changed": inc["schedule"]["starts"]
                        != row["redecoded"]["schedule"]["starts"],
                    }
                    if "continuation_retained" in row:
                        ret = objectives(row["continuation_retained"]["final"])
                        rec = objectives(row["continuation_redecoded"]["final"])
                        common_ideal = [min(ideal[k], ret[k], rec[k]) for k in (0, 1)]
                        h1.update(
                            retained_final=ret,
                            redecoded_final=rec,
                            continuation_scalar_difference=score(rec, ideal, maximum, index)
                            - score(ret, ideal, maximum, index),
                            continuation_updated_scalar_difference=score(
                                rec, common_ideal, maximum, index
                            )
                            - score(ret, common_ideal, maximum, index),
                            continuation_bill_difference=(rec[1] - ret[1]) / max(ret[1], 1e-9),
                            continuation_flow_difference=(rec[0] - ret[0]) / max(ret[0], 1e-9),
                            continuation_seconds=row["continuation_seconds"],
                        )
                    write("h1", h1)
                    for a in row["actions"]:
                        pts = [objectives(c) for c in a["candidates"]]
                        frozen_scores = [score(p, ideal, maximum, index) for p in pts]
                        inc_score = score(op, ideal, maximum, index)
                        reps = a["replacement"]
                        action_row = {
                            **smeta,
                            "index": index,
                            "origin": inc["origin"],
                            "action": a["action"],
                            "exact": a["exact"],
                            "attempts": a["attempts"],
                            "construction_seconds": a["construction_seconds"],
                            "timings": a["timings"],
                            "construction": a["construction"],
                            "instrumentation": a["instrumentation"],
                            "scalar_success": any(positive(inc_score - v) for v in frozen_scores),
                            "bill_success": any(positive(op[1] - p[1]) for p in pts),
                            "bill_increases": sum(positive(p[1] - op[1]) for p in pts),
                            "improving_candidates": sum(bool(x["improving"]) for x in reps),
                            "outside_candidates": sum(bool(x["outside_birth"]) for x in reps),
                            "completely_missed": sum(
                                bool(x["improving"]) and not x["original"] for x in reps
                            ),
                            "cap_candidates": sum(x["cap_truncated"] > 0 for x in reps),
                            "improving_pairs": sum(len(x["improving"]) for x in reps),
                            "outside_pairs": sum(len(x["outside_birth"]) for x in reps),
                            "cap_pairs": sum(x["cap_truncated"] for x in reps),
                            "replacement_counts": {
                                k: sum(len(x[k]) for x in reps)
                                for k in ["original", "random", "gain_order", "direction_neighbors"]
                            },
                            "replacement_gain": {
                                k: sum(sum(x["gains"][j] for j in x[k]) for x in reps)
                                for k in ["original", "random", "gain_order", "direction_neighbors"]
                            },
                            "prefixes": a["prefixes"],
                        }
                        for early, late in [(3, 6), (6, 10), (3, 10)]:
                            p_early = next(x for x in a["prefixes"] if x["prefix"] == early)
                            p_late = next(x for x in a["prefixes"] if x["prefix"] == late)
                            # Re-score old and new prefixes in the later prefix's context.
                            # This distinguishes new candidates from a normalization-only change.
                            ctx = p_late["ideal"]
                            gain = prefix_improvement(op, pts, ctx, maximum, index, early, late)
                            action_row[f"late_{early}_{late}"] = positive(gain)
                            action_row[f"late_gain_{early}_{late}"] = gain
                            action_row[f"late_hv_{early}_{late}"] = (
                                p_late["hv_gain"] - p_early["hv_gain"]
                            )
                        action_row["rescue_after3"] = (
                            not positive(a["prefixes"][0]["scalar_gain"])
                            and action_row["late_3_10"]
                        )
                        action_row["rescue_after_flat6"] = (
                            positive(a["prefixes"][0]["scalar_gain"])
                            and not action_row["late_3_6"]
                            and action_row["late_6_10"]
                        )
                        write("actions", action_row)
                        if a["action"] == 7:
                            raw = a["raw_audit"]
                            sample = raw["sample"]
                            raw_points = [objectives(c["candidate"]) for c in sample]
                            updated = [
                                min([ideal[k], op[k]] + [p[k] for p in pts + raw_points])
                                for k in (0, 1)
                            ]
                            parent_score = score(op, updated, maximum, index)
                            selected_best = min(
                                [parent_score] + [score(p, updated, maximum, index) for p in pts]
                            )
                            extra = [
                                {
                                    "score": score(p, updated, maximum, index),
                                    "bill_change": (p[1] - op[1]) / max(op[1], 1e-9),
                                    "flow_change": (p[0] - op[0]) / max(op[0], 1e-9),
                                    "tou_change": c["candidate"]["evaluation"]["tou"]
                                    - inc["evaluation"]["tou"],
                                    "demand_change": sum(c["candidate"]["evaluation"]["demand"])
                                    - sum(inc["evaluation"]["demand"]),
                                    "representative": c["representative"],
                                    "pool": c["pool"],
                                }
                                for c, p in zip(sample, raw_points)
                            ]
                            best_all = min([selected_best] + [x["score"] for x in extra])
                            layers = {}
                            for label, predicate in [
                                ("representative", lambda x: not x["representative"]),
                                ("pool", lambda x: x["representative"] and not x["pool"]),
                                ("draw", lambda x: x["pool"]),
                            ]:
                                group = [x for x in extra if predicate(x)]
                                layers[label] = {
                                    "count": len(group),
                                    "better": sum(
                                        positive(selected_best - x["score"]) for x in group
                                    ),
                                }
                            write(
                                "h3",
                                {
                                    **smeta,
                                    "index": index,
                                    "raw_count": raw["raw_count"],
                                    "selected_count": len(pts),
                                    "representatives_count": len(raw["representatives"]),
                                    "pool_count": len(raw["pool"]),
                                    "sampling_probability": raw["sampling_probability"],
                                    "extra_count": len(extra),
                                    "parent_score": parent_score,
                                    "selected_best": selected_best,
                                    "sampled_best": best_all,
                                    "sampled_regret": selected_best - best_all,
                                    "layers": layers,
                                    "selected_points": pts,
                                    "raw": extra,
                                },
                            )
            print(json.dumps({"reduced": item["key"], "counts": counts}), flush=True)
    for f in handles.values():
        f.close()
    files = []
    for p in out.glob("*.jsonl.gz"):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        files.append({"name": p.name, "size": p.stat().st_size, "sha256": h})
    (out / "reduction_manifest.json").write_text(
        json.dumps({"counts": counts, "files": files, "schema_version": 1}, indent=2)
    )
    print(json.dumps({"finished": True, "counts": counts, "files": files}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    reduce(parser.parse_args().base)
