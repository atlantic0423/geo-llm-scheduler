"""Reconcile regional-ledger predictions with exact evaluation on recorded R1 samples.

Read-only over a recovered R1 live-observation campaign. For every sampled donor,
receiver and eligible region it predicts the spliced objectives from regional
ledgers, then runs the reference path (splice genotype, SSGS decode, exact
evaluation) and compares objectives, scalar decisions and front gains. Timing of
both paths is recorded; nothing is fed back into any search.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import r1_offline as r1  # noqa: E402

from geo_llm_scheduler.domain.models import (  # noqa: E402
    Candidate,
    EvaluationResult,
    ProblemInstance,
)
from geo_llm_scheduler.evaluation.exact import evaluate  # noqa: E402
from geo_llm_scheduler.evaluation.regional import (  # noqa: E402
    RegionalPrediction,
    combine,
    evaluate_with_ledgers,
    region_ledgers,
    splice,
    tariff_indexes,
)
from geo_llm_scheduler.experiments.p1_observation import restore_candidate  # noqa: E402
from geo_llm_scheduler.experiments.trans_runner import digest, write_json  # noqa: E402
from geo_llm_scheduler.io.loaders import load_instance  # noqa: E402
from geo_llm_scheduler.moead.core import NormalizationContext  # noqa: E402
from geo_llm_scheduler.scheduling.ssgs import decode  # noqa: E402
from geo_llm_scheduler.utils.numeric import TOL, less  # noqa: E402

ARM = "GS4_GUIDED_RAND_SWITCH_D4"


def _as_candidate(base: Candidate, prediction: RegionalPrediction) -> Candidate:
    """Wrap predicted objectives for scalar helpers; never inserted anywhere."""
    result = EvaluationResult(True, prediction.flow, prediction.tou, prediction.demand, ())
    return Candidate(base.genotype, base.schedule, result, "R1_PREDICTION")


def _decisions(
    frozen: NormalizationContext,
    weight: tuple[float, float],
    donor: Candidate,
    recipient: Candidate,
    mixed: Candidate,
    front: list[tuple[float, float]],
    ref: tuple[float, float],
    span: tuple[float, float],
    base_hv: float,
) -> tuple[bool, bool, float, float]:
    """Direction improvement, rescue, scalar and scaled HV gain, as in the live replay."""
    objectives = mixed.evaluation.objectives
    ideal = (min(frozen.ideal[0], objectives[0]), min(frozen.ideal[1], objectives[1]))
    context = NormalizationContext(ideal, frozen.maximum)
    base = context.scalar(recipient, weight)
    scalar = context.scalar(mixed, weight)
    improve = less(scalar, base, TOL.scalar)
    whole = less(context.scalar(donor, weight), base, TOL.scalar)
    gain = r1.hypervolume(front + [objectives], ref) - base_hv
    return improve, improve and not whole, scalar, max(0.0, gain / (span[0] * span[1]))


def reconcile_run(task: tuple[str, str]) -> dict:
    """Verify one sealed run and write its per-candidate reconciliation rows."""
    run, output = (Path(x) for x in task)
    key = run.parent.name
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    for entry in json.loads((run / "manifest.json").read_text(encoding="utf-8")):
        path = run / entry["path"]
        assert path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"]
    instance = run.parents[2] / "instances" / cfg["instance"]
    assert digest(instance) == cfg["instance_sha256"]
    problem: ProblemInstance = load_instance(instance)
    with gzip.open(run / "regional_observation.json.gz", "rt", encoding="utf-8") as f:
        record = json.load(f)
    tick = perf_counter()
    indexes = tariff_indexes(problem)
    index_seconds = perf_counter() - tick
    rows: list[dict] = []
    ledger_seconds: list[float] = []
    exact_ref_seconds: list[float] = []
    fused_seconds: list[float] = []
    for snapshot in record["snapshots"]:
        donor = restore_candidate(snapshot["donor"])
        front = [tuple(p) for p in snapshot["front"]]
        recorded = snapshot["context"]
        frozen = NormalizationContext(tuple(recorded["ideal"]), tuple(recorded["maximum"]))
        tick = perf_counter()
        d_led = region_ledgers(problem, donor.genotype, donor.schedule)
        ledger_seconds.append(perf_counter() - tick)
        # Reference cost of one exact evaluation of an already decoded offspring,
        # and of the fused evaluation that also yields its ledgers.
        tick = perf_counter()
        assert evaluate(problem, donor.genotype, donor.schedule) == donor.evaluation
        exact_ref_seconds.append(perf_counter() - tick)
        tick = perf_counter()
        fused, fused_ledgers = evaluate_with_ledgers(problem, donor.genotype, donor.schedule)
        fused_seconds.append(perf_counter() - tick)
        assert fused == donor.evaluation and fused_ledgers == d_led
        for receiver in snapshot["receivers"]:
            recipient = restore_candidate(receiver["candidate"])
            weight = tuple(receiver["weight"])
            tick = perf_counter()
            r_led = region_ledgers(problem, recipient.genotype, recipient.schedule)
            ledger_seconds.append(perf_counter() - tick)
            anchors = front + [recipient.evaluation.objectives, donor.evaluation.objectives]
            lo = tuple(min(p[k] for p in anchors) for k in range(2))
            hi = tuple(max(p[k] for p in anchors) for k in range(2))
            span = tuple(max(hi[k] - lo[k], 1e-9) for k in range(2))
            ref = tuple(hi[k] + 0.1 * span[k] for k in range(2))
            base_hv = r1.hypervolume(front, ref)
            for region in range(len(problem.regions)):
                if not d_led[region].jobs or d_led[region].jobs != r_led[region].jobs:
                    continue
                tick = perf_counter()
                prediction = combine(problem, indexes, splice(r_led, d_led, region))
                predict_seconds = perf_counter() - tick
                neutral = d_led[region].same_core(r_led[region])
                tick = perf_counter()
                genotype, _ = r1.mix_region(problem, donor, recipient, region)
                schedule = decode(problem, genotype)
                result = evaluate(problem, genotype, schedule)
                reference_seconds = perf_counter() - tick
                assert result.feasible
                mixed = Candidate(genotype, schedule, result, "R1_RECONCILE")
                predicted = _as_candidate(recipient, prediction)
                e_imp, e_res, e_scalar, e_gain = _decisions(
                    frozen, weight, donor, recipient, mixed, front, ref, span, base_hv
                )
                p_imp, p_res, p_scalar, p_gain = _decisions(
                    frozen, weight, donor, recipient, predicted, front, ref, span, base_hv
                )
                rows.append(
                    dict(
                        stage=snapshot["stage"],
                        birth=snapshot["subproblem"],
                        slot=receiver["slot"],
                        region=region,
                        flow_error=prediction.flow - result.flow,
                        bill_error=prediction.bill - result.bill,
                        tou_error=prediction.tou - result.tou,
                        demand_error=max(
                            abs(a - b) for a, b in zip(prediction.demand, result.demand)
                        ),
                        horizon_error=prediction.horizon - schedule.horizon(problem),
                        horizon_changed=prediction.horizon != recipient.schedule.horizon(problem),
                        scalar_error=p_scalar - e_scalar,
                        neutral=neutral,
                        neutral_exact=result.objectives == recipient.evaluation.objectives,
                        exact_improve=e_imp,
                        predicted_improve=p_imp,
                        exact_rescue=e_res,
                        predicted_rescue=p_res,
                        exact_hv=e_gain > 1e-12,
                        predicted_hv=p_gain > 1e-12,
                        exact_gain=e_gain,
                        predicted_gain=p_gain,
                        predict_seconds=predict_seconds,
                        reference_seconds=reference_seconds,
                    )
                )
    result = dict(
        key=key,
        n=len(problem.jobs),
        regions=len(problem.regions),
        input_sha256=cfg["instance_sha256"],
        source_commit=cfg["source"]["commit"],
        index_seconds=index_seconds,
        ledger_seconds=ledger_seconds,
        exact_reference_seconds=exact_ref_seconds,
        fused_seconds=fused_seconds,
        rows=rows,
    )
    write_json(output / (key + ".json"), result)
    return dict(key=key, n=result["n"], trials=len(rows))


def _q(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))] if ordered else float("nan")


def summarize(reports: list[dict]) -> dict:
    """Aggregate agreement, decision confusion and cost per task count."""
    out = {}
    groups: dict[int, list[dict]] = defaultdict(list)
    for report in reports:
        groups[report["n"]].append(report)
    for n, group in sorted(groups.items()):
        rows = [row for r in group for row in r["rows"]]
        ledgers = [s for r in group for s in r["ledger_seconds"]]
        exacts = [s for r in group for s in r["exact_reference_seconds"]]
        fused = [s for r in group for s in r["fused_seconds"]]
        screen = [row["predicted_improve"] or row["predicted_hv"] for row in rows]
        out[str(n)] = dict(
            runs=len(group),
            trials=len(rows),
            max_abs_flow_error=max(abs(r["flow_error"]) for r in rows),
            max_abs_bill_error=max(abs(r["bill_error"]) for r in rows),
            max_abs_demand_error=max(r["demand_error"] for r in rows),
            max_abs_horizon_error=max(abs(r["horizon_error"]) for r in rows),
            max_abs_scalar_error=max(abs(r["scalar_error"]) for r in rows),
            horizon_changed=sum(r["horizon_changed"] for r in rows),
            neutral_ledgers=sum(r["neutral"] for r in rows),
            neutral_exact=sum(r["neutral_exact"] for r in rows),
            neutral_ledger_not_exact=sum(r["neutral"] and not r["neutral_exact"] for r in rows),
            improve_disagree=sum(r["exact_improve"] != r["predicted_improve"] for r in rows),
            rescue_disagree=sum(r["exact_rescue"] != r["predicted_rescue"] for r in rows),
            hv_disagree=sum(r["exact_hv"] != r["predicted_hv"] for r in rows),
            exact_improve=sum(r["exact_improve"] for r in rows),
            exact_hv=sum(r["exact_hv"] for r in rows),
            screen_pass=sum(screen),
            screen_pass_rate=sum(screen) / len(rows),
            missed_by_screen=sum(
                (r["exact_improve"] or r["exact_hv"]) and not s for r, s in zip(rows, screen)
            ),
            predict_ms_median=1000 * statistics.median(r["predict_seconds"] for r in rows),
            predict_ms_p90=1000 * _q([r["predict_seconds"] for r in rows], 0.9),
            reference_ms_median=1000 * statistics.median(r["reference_seconds"] for r in rows),
            ledger_build_ms_median=1000 * statistics.median(ledgers),
            exact_eval_ms_median=1000 * statistics.median(exacts),
            fused_eval_ms_median=1000 * statistics.median(fused),
            fused_over_exact_median=statistics.median(f / e for f, e in zip(fused, exacts)),
        )
    return out


def main() -> None:
    """Reconcile an explicitly supplied recovered campaign; never launches search."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    runs = sorted(args.results.glob(f"formal_panel_shard*/runs/*/{ARM}"))
    if args.limit:
        runs = runs[: args.limit]
    elif len(runs) != 120:
        raise ValueError(f"Expected all 120 sealed runs, got {len(runs)}")
    per_run = args.out / "runs"
    per_run.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for done in pool.map(reconcile_run, [(str(r), str(per_run)) for r in runs]):
            print(json.dumps(done), flush=True)
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(per_run.glob("*.json"))]
    summary = dict(
        status="offline reconciliation; prediction is a screen, exact remains ground truth",
        script_sha256=digest(Path(__file__)),
        by_size=summarize(reports),
    )
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
