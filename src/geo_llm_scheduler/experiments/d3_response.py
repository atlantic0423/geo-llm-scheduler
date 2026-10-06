"""Prospective source/action/direction screening using an identical raw candidate pool."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import Candidate, ProblemInstance
from geo_llm_scheduler.evaluation.exact import evaluate
from geo_llm_scheduler.experiments.campaign.support import atomic_json
from geo_llm_scheduler.experiments.d2_analysis import paired_summary, ridge_predictions
from geo_llm_scheduler.experiments.d2_observation import opportunity_features
from geo_llm_scheduler.experiments.d2_probe import action_probe
from geo_llm_scheduler.experiments.p1_observation import restore_candidate
from geo_llm_scheduler.moead.core import NormalizationContext
from geo_llm_scheduler.utils.numeric import EPS_NORM, TOL, identity, less
from geo_llm_scheduler.utils.rng import RNGManager

POLICIES = ("GEOMETRIC", "RANDOM", "RGAIN", "BASE", "RESPONSE", "SHUFFLED", "POST_RGAIN")


def replacement_value(
    candidate: Candidate,
    targets: list[Candidate],
    weights: list[tuple[float, float]],
    context: NormalizationContext,
) -> float:
    """Sum at most two positive own-weight gains within exactly the shared target pool."""
    if not targets or len(targets) != len(weights):
        raise ValueError("Require nonempty matched targets and weights")
    gains = [
        max(0.0, context.scalar(p, w) - context.scalar(candidate, w))
        for p, w in zip(targets, weights, strict=True)
    ]
    return sum(sorted(gains, reverse=True)[:2]) / len(targets)


def response_features(
    problem: ProblemInstance,
    source: Candidate,
    targets: list[Candidate],
    items: list[dict],
    panel: dict,
) -> tuple[list[dict], float]:
    """Extract all pre-action controls and source-opportunity/direction interactions."""
    tick = perf_counter()
    context = NormalizationContext(
        tuple(panel["context"]["ideal"]), tuple(panel["context"]["maximum"])
    )
    opp = opportunity_features(problem, source)

    def normalized(c: Candidate) -> list[float]:
        return [
            (c.evaluation.objectives[k] - context.ideal[k])
            / max(EPS_NORM, context.maximum[k] - context.ideal[k])
            for k in range(2)
        ]

    source_values = normalized(source)
    rows = []
    for index, (target, item) in enumerate(zip(targets, items, strict=True)):
        w = tuple(item["weight"])
        distance = abs(w[0] - panel["weight"][0])
        gain = max(0.0, context.scalar(target, w) - context.scalar(source, w))
        controls = [
            *source_values,
            *normalized(target),
            context.scalar(source, w),
            context.scalar(target, w),
            w[0],
            distance,
            item["history"],
            float(item["stagnation"]),
            float(panel["stage"]),
            float(panel["preference"]),
            len(problem.jobs) / 100,
            gain,
        ]
        rows.append(
            {
                "index": index,
                "slot": item["slot"],
                "controls": controls,
                "source_opportunities": opp,
                "opportunities": [v * power for power in (w[0], w[0] ** 2) for v in opp],
                "weight_flow": w[0],
                "distance": distance,
                "source_gain": gain,
            }
        )
    return rows, perf_counter() - tick


def probe_response_panel(problem: ProblemInstance, panel: dict, config: Config, seed: int) -> dict:
    """Generate each A1-A8 raw B3 pool once; score all six prospective directions.

    Current generators do not accept a direction. Routing therefore changes only
    best-of-B retention. The post-generation RGAIN bound has access to the same
    outputs and is explicitly an oracle diagnostic, not a prospective predictor.
    """
    source = restore_candidate(panel["mate"])
    targets = [restore_candidate(s["candidate"]) for s in panel["sources"]]
    for c in [source, *targets]:
        if evaluate(problem, c.genotype, c.schedule) != c.evaluation:
            raise ValueError("D3 source/target exact mismatch")
    context = NormalizationContext(
        tuple(panel["context"]["ideal"]), tuple(panel["context"]["maximum"])
    )
    weights = [tuple(s["weight"]) for s in panel["sources"]]
    prepared, feature_seconds = response_features(problem, source, targets, panel["sources"], panel)
    baseline = replacement_value(source, targets, weights, context)
    rows: list[dict] = []
    pools: list[dict] = []
    audits = len(targets) + 1
    order = list(range(1, 9))
    RNGManager(seed).stream("D3:action_order").shuffle(order)
    for action in order:
        diagnostic = action_probe(
            problem,
            source,
            source,
            action,
            config,
            tuple(panel["weight"]),
            context,
            seed + 10007 * action,
        )
        outputs = [source] + [
            restore_candidate(c) for c in diagnostic["outputs"] if c["evaluation"]["feasible"]
        ]
        selection = NormalizationContext(
            tuple(diagnostic["selection_context"]["ideal"]),
            tuple(diagnostic["selection_context"]["maximum"]),
        )
        post_gain = max(replacement_value(c, targets, weights, context) - baseline for c in outputs)
        for entry, weight in zip(prepared, weights, strict=True):
            best = min(outputs, key=lambda c: (selection.scalar(c, weight), identity(c)))
            retained = (
                best
                if less(
                    selection.scalar(best, weight), selection.scalar(source, weight), TOL.scalar
                )
                else source
            )
            gain = replacement_value(retained, targets, weights, context) - baseline
            if gain > post_gain + 1e-12:
                raise ValueError("Prospective route exceeded same-pool RGAIN bound")
            rows.append(
                {
                    **entry,
                    "action": action,
                    "gain": gain,
                    "empty": diagnostic["empty"],
                    "post_gain": post_gain,
                    "retained_identity": identity(retained),
                }
            )
        pools.append({"action": action, "raw": diagnostic, "post_gain": post_gain})
        audits += diagnostic["audit_checks"]
    return {
        "panel": panel["panel"],
        "rows": rows,
        "pools": pools,
        "pool_size": len(targets),
        "feature_seconds": feature_seconds,
        "audit_checks": audits,
        "actions": 8,
        "exact": sum(p["raw"]["exact"] for p in pools),
    }


def shuffle_opportunities(rows: list[dict], seed: int) -> list[dict]:
    """Shuffle source vectors across matched source panels, preserving direction interactions.

    A source vector is constant within its six-direction pool. Shuffling within
    that pool would be a no-op, so panels are exchanged within job/stage/preference
    and action strata. Training and held-out calls are made separately.
    """
    groups: dict[tuple, dict[tuple, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        groups[row["jobs"], row["stage"], row["preference"], row["action"]][
            row["key"], row["panel"]
        ].append(i)
    result = [dict(r) for r in rows]
    for group, pools in groups.items():
        names = sorted(pools)
        permutation = list(names)
        RNGManager(seed).stream(f"D3:shuffle:{group}").shuffle(permutation)
        for a, b in zip(names, permutation, strict=True):
            vector = rows[pools[b][0]]["source_opportunities"]
            for i in pools[a]:
                w = rows[i]["weight_flow"]
                result[i]["opportunities"] = [v * power for power in (w, w**2) for v in vector]
    return result


def analyse_response(root: Path) -> dict:
    """Validate one shard; retain training/test separation for the joint final analysis."""
    from geo_llm_scheduler.experiments.d2_campaign import frozen_manifest, validate_opportunity_job

    manifest = frozen_manifest(root)
    if manifest["kind"] != "D3_response_v1":
        raise ValueError("Require D3 manifest")
    counts = {"pairs": 0, "panels": 0, "rows": 0, "exact": 0, "audits": 0}
    for key in manifest["keys"]:
        for role in ("samples", "probes"):
            valid, reason = validate_opportunity_job(root, role, key)
            if not valid:
                raise ValueError(reason)
        data = json.loads((root / "probes" / key / "data.json").read_text())
        counts["pairs"] += 1
        for panel in data["panels"]:
            d = panel["d2"]
            counts["panels"] += 1
            counts["rows"] += len(d["rows"])
            counts["exact"] += d["exact"]
            counts["audits"] += d["audit_checks"]
    report = {
        "status": "SHARD_VALIDATED",
        "source_commit": manifest["source_commit"],
        "source_hash": manifest["source_hash"],
        "node": manifest["node"],
        **counts,
        "joint_quality": "UNVERIFIED until both shards jointly analyzed",
        "online_moead": "UNVERIFIED",
    }
    atomic_json(root / "analysis/shard_acceptance.json", report)
    return report


def analyse_response_joint(roots: list[Path], output: Path) -> dict:
    """Fit fixed per-action ridge on 32 development bases and screen 16 held-out bases."""
    from geo_llm_scheduler.experiments.d2_campaign import frozen_manifest

    if len(roots) != 2 or any(output.resolve().is_relative_to(r.resolve()) for r in roots):
        raise ValueError("Use two distinct shards and an independent analysis output")
    manifests = [frozen_manifest(r) for r in roots]
    if (
        {m["node"] for m in manifests} != {0, 1}
        or any(m["pilot"] for m in manifests)
        or len({m["source_hash"] for m in manifests}) != 1
    ):
        raise ValueError("Require paired final shards from the identical source")
    for root in roots:
        analyse_response(root)
    rows = []
    for root, manifest in zip(roots, manifests, strict=True):
        for key in manifest["keys"]:
            spec = json.loads((root / "specs" / f"{key}.json").read_text())
            data = json.loads((root / "probes" / key / "data.json").read_text())
            for panel in data["panels"]:
                name = panel["d2"]["panel"]
                for row in panel["d2"]["rows"]:
                    rows.append(
                        {
                            **row,
                            "key": key,
                            "base": spec["base_seed"],
                            "jobs": spec["jobs"],
                            "split": spec["split"],
                            "panel": name,
                            "stage": int(name[1]),
                            "preference": int(name[-1]),
                        }
                    )
    rows.sort(key=lambda r: (r["key"], r["panel"], r["action"], r["index"]))
    train = [r for r in rows if r["split"] == "development"]
    test = [r for r in rows if r["split"] == "heldout"]
    if len({r["base"] for r in train}) != 32 or len({r["base"] for r in test}) != 16:
        raise ValueError("Frozen independent-base split changed")
    shuffled_train, shuffled_test = (
        shuffle_opportunities(train, 20261006031),
        shuffle_opportunities(test, 20261006032),
    )
    decisions = []
    prediction_seconds = 0.0
    for action in range(1, 9):
        tr, te = (
            [r for r in train if r["action"] == action],
            [r for r in test if r["action"] == action],
        )
        trs, tes = (
            [r for r in shuffled_train if r["action"] == action],
            [r for r in shuffled_test if r["action"] == action],
        )
        tick = perf_counter()
        predictions = {
            "BASE": ridge_predictions(tr, te, False),
            "RESPONSE": ridge_predictions(tr, te, True),
            "SHUFFLED": ridge_predictions(trs, tes, True),
        }
        prediction_seconds += perf_counter() - tick
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, row in enumerate(te):
            groups[row["key"], row["panel"]].append(i)
        for group, indices in groups.items():
            chosen = {
                "GEOMETRIC": min(indices, key=lambda i: (te[i]["distance"], te[i]["index"])),
                "RGAIN": min(
                    indices,
                    key=lambda i: (-te[i]["source_gain"], te[i]["distance"], te[i]["index"]),
                ),
                "RANDOM": RNGManager(20261006033)
                .stream(f"D3:random:{action}:{group}")
                .choice(indices),
            }
            chosen.update(
                {
                    p: max(indices, key=lambda i: (values[i], -te[i]["index"]))
                    for p, values in predictions.items()
                }
            )
            decisions.append(
                {
                    "key": group[0],
                    "panel": group[1],
                    "base": te[indices[0]]["base"],
                    "action": action,
                    **{p: te[i]["gain"] for p, i in chosen.items()},
                    **{p + "_index": te[i]["index"] for p, i in chosen.items()},
                    "POST_RGAIN": te[indices[0]]["post_gain"],
                }
            )
    basemeans = {
        base: {p: float(np.mean([d[p] for d in decisions if d["base"] == base])) for p in POLICIES}
        for base in sorted({d["base"] for d in decisions})
    }
    tests = {}
    for control in ("BASE", "RGAIN", "SHUFFLED"):
        tests["RESPONSE-" + control] = paired_summary(
            [v["RESPONSE"] - v[control] for v in basemeans.values()], 20261006034
        )
    order = sorted(tests, key=lambda k: tests[k]["p"])
    adjusted = 0.0
    for i, name in enumerate(order):
        adjusted = max(adjusted, (len(order) - i) * tests[name]["p"])
        tests[name]["p_holm"] = min(1.0, adjusted)
    report = {
        "status": "ANALYZED",
        "source_commit": manifests[0]["source_commit"],
        "unit": "16 wholly held-out independent bases, equal panel/action/tariff/seed",
        "primary": tests,
        "pass_screen": all(
            v["mean"] >= 0.001 and v["ci"][0] > 0 and v["p_holm"] < 0.05 for v in tests.values()
        ),
        "policy_means": {p: float(np.mean([v[p] for v in basemeans.values()])) for p in POLICIES},
        "prediction_seconds": prediction_seconds,
        "training_bases": 32,
        "heldout_bases": 16,
        "rows": len(rows),
        "limits": "Same raw B3 pool; six sampled neighbor slots, not global direction search. Current generators are weight-independent. POST_RGAIN is post-generation upper bound. No online HV/IGD benefit or novelty inference.",
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    with (output / "heldout_decisions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    atomic_json(output / "base_means.json", basemeans)
    return report
