"""Declared Q screen entry points and train-only calibration (no holdout fitting)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from statistics import median

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.domain.models import ProblemInstance
from geo_llm_scheduler.engine.run import RunResult
from geo_llm_scheduler.experiments.lsv2_study import operator_config
from geo_llm_scheduler.experiments.rl23 import run_policy
from geo_llm_scheduler.lsv2.qlearning import DESIGNS, FixedA7Search, QLearningSearch, QSpec, QTable
from geo_llm_scheduler.lsv2.search import LSV2Config, LSV2Search

LEARNING_ARMS = tuple(f"{d}_G{g}" for d in DESIGNS for g in (0, 9))
REFERENCES = ("UNIFORM", "BANDIT", "FIXED_A7")


def calibrate(rows: Sequence[dict], features: Sequence[dict], keys: Sequence[str]) -> dict:
    """Fit reward scales and opportunity cutoffs only on declared training instance keys."""
    selected = [r for r in rows if r["key"] in keys]
    points = [r for r in features if r["key"] in keys]
    positives = [float(r["gain"]) for r in selected if float(r["gain"]) > 1e-12]
    if not selected or not points or not positives:
        raise ValueError("Insufficient training-fold calibration evidence")
    total_time = sum(float(r["seconds"]) for r in selected)
    if total_time <= 0:
        raise ValueError("Invalid calibration durations")
    return dict(
        keys=list(keys),
        rows=len(selected),
        feature_rows=len(points),
        feature_keys=sorted({r["key"] for r in points}),
        rho=sum(float(r["gain"]) for r in selected) / total_time,
        h=median(positives),
        p_cutoff=median(float(r["prefill_window_fraction"]) for r in points),
        w_cutoff=median(float(r["resource_wait_fraction"]) for r in points),
    )


def make_search(
    arm: str, config: Config, calibration: dict, table: QTable, *, training: bool
) -> LSV2Search:
    """Construct a research controller; references keep their existing LSV2 paths."""
    operators = operator_config(config.seed, config.seconds or 1.0)
    if arm in REFERENCES:
        cls = FixedA7Search if arm == "FIXED_A7" else LSV2Search
        return cls(
            LSV2Config(selector="uniform" if arm == "UNIFORM" else "bandit"), config, operators
        )
    if arm not in LEARNING_ARMS:
        raise ValueError("Unknown Q-screen arm")
    design, discount = arm.rsplit("_G", 1)
    cutoff = calibration["p_cutoff" if design == "Q2_P_NET" else "w_cutoff"]
    spec = QSpec(design, int(discount) / 10, calibration["rho"], calibration["h"], cutoff)
    return QLearningSearch(config, operators, spec, table, training=training)


def run_screen(
    problem: ProblemInstance,
    config: Config,
    evidence: dict,
    search: LSV2Search,
    *,
    observer: Callable[[str, dict], None] | None = None,
) -> RunResult:
    """Run the unchanged charged BILL engine with the supplied research controller."""
    return run_policy(
        problem, "BILL_PERMUTED", config, evidence, observer=observer, interleave=search
    )
