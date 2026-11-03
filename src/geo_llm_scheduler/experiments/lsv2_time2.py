"""Matched doubled-budget experiment: five external methods, BILL and four B3L3 arms."""

from __future__ import annotations

from dataclasses import asdict, replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments import ext_hybrid, lsv2_ablation3, lsv2_budget12
from geo_llm_scheduler.experiments.lsv2_study import operator_config, study_config
from geo_llm_scheduler.lsv2.qlearning import QTable
from geo_llm_scheduler.lsv2.search import LSV2Config, LSV2Search

Q_FULL = "Q1_NET_G9_B3L3"
Q_NOP3 = Q_FULL + "_NO_P3"
INTERNAL_ARMS = (Q_FULL, Q_NOP3, "UNIFORM_B3L3", "UNIFORM_B3L3_NO_P3")
LEARNING_ARMS = INTERNAL_ARMS[:2]
EXTERNAL_ARMS = ext_hybrid.ARMS
BILL = "BILL_RANDOM"
ARMS = INTERNAL_ARMS + (BILL,) + EXTERNAL_ARMS
RUN_SEED = 16901
BASE_BUDGETS = {20: 300, 50: 600, 100: 1200, 200: 2400}


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Preserve each frozen method's engine and change only the total time limit."""
    if arm not in ARMS:
        raise ValueError("Unknown time2 arm")
    if arm in EXTERNAL_ARMS:
        return ext_hybrid.arm_config(arm, seed, seconds)
    return study_config("LSV2" if arm in INTERNAL_ARMS else BILL, seed, seconds)


def search_config(arm: str) -> LSV2Config:
    """Match Q and uniform on B3L3, initialization, action pool and A8 gate."""
    if arm not in INTERNAL_ARMS:
        raise ValueError("Not a local-search time2 arm")
    base = lsv2_budget12.search_config("UNIFORM_B3L3" if arm.startswith("UNIFORM") else Q_FULL)
    return replace(
        base,
        operators=tuple(op for op in base.operators if op != "P3" or not arm.endswith("NO_P3")),
    )


def make_search(
    arm: str, config: Config, calibration: dict, table: QTable, *, training: bool
) -> LSV2Search:
    """Reuse the previously tested Q arms and the exactly matched uniform control."""
    if arm in LEARNING_ARMS:
        return lsv2_ablation3.make_search(arm, config, calibration, table, training=training)
    return LSV2Search(
        search_config(arm),
        config,
        operator_config(config.seed, config.seconds or 1.0),
        retain_steps=True,
    )


def configuration_record(arm: str, seed: int, seconds: float, calibration: dict) -> dict:
    """Seal the full configuration without modifying frozen reward calibration."""
    if arm in LEARNING_ARMS:
        return lsv2_ablation3.configuration_record(arm, seed, seconds, calibration)
    cfg = arm_config(arm, seed, seconds)
    if arm in EXTERNAL_ARMS:
        return dict(
            engine=asdict(cfg),
            initialization=ext_hybrid.POLICY,
            polish_share=ext_hybrid.POLISH_SHARE,
            lsv2=None,
            qspec=None,
        )
    if arm == BILL:
        return dict(engine=asdict(cfg), initialization="bill4", lsv2=None, qspec=None)
    return dict(
        engine=asdict(cfg),
        lsv2=asdict(search_config(arm)),
        operators=asdict(operator_config(seed, seconds)),
        qspec=None,
    )


def select_specs(existing: list[dict], pool: list[dict], per_size: int = 10) -> list[dict]:
    """Keep the old panel and add lowest unselected IDs, never ranking new outcomes."""
    chosen = {s["base_seed"]: dict(s) for s in existing}
    if per_size < 8 or len(chosen) != 32:
        raise ValueError("Require the existing 32-instance panel and at least eight per size")
    for n in BASE_BUDGETS:
        candidates = sorted(
            (s for s in pool if s["n"] == n and s["key"] not in chosen), key=lambda s: s["key"]
        )
        needed = per_size - sum(s["n"] == n for s in chosen.values())
        if len(candidates) < needed:
            raise ValueError("Insufficient additional instances")
        for old in candidates[:needed]:
            base = old["key"]
            chosen[base] = dict(
                old,
                base_seed=base,
                key=f"{base}_s{RUN_SEED}",
                panel="extension",
                seed=RUN_SEED,
                seconds=BASE_BUDGETS[n],
            )
    return sorted(chosen.values(), key=lambda s: (s["n"], s["base_seed"]))


def calibration_for(spec: dict, calibrations: dict, nondev_index: int) -> tuple[str, dict]:
    """Reuse dev cross-fitting; assign existing folds alternately to unseen instances."""
    if spec["panel"] == "dev":
        matched = [(k, v) for k, v in calibrations.items() if spec["key"] in v["validation_keys"]]
        if len(matched) != 1:
            raise ValueError("Expected exactly one frozen dev calibration")
        name, cal = matched[0]
    else:
        name = f"n{spec['n']}_f{nondev_index % 2}"
        cal = calibrations[name]
    if spec["key"] in cal["keys"]:
        raise ValueError("Calibration contains this evaluation instance")
    return name, dict(cal)
