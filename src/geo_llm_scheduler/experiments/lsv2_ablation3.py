"""Authorized B3L3 action-pool factorial: seven removals plus a matched full control."""

from __future__ import annotations

from dataclasses import asdict, replace

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments import lsv2_budget12 as baseline
from geo_llm_scheduler.experiments.lsv2_study import operator_config
from geo_llm_scheduler.lsv2.qlearning import QLearningSearch, QSpec, QTable
from geo_llm_scheduler.lsv2.search import LSV2Config

BASE = "Q1_NET_G9_B3L3"
REMOVALS = {
    BASE: (),
    BASE + "_NO_P3": ("P3",),
    BASE + "_NO_A4": ("A4",),
    BASE + "_NO_P2M": ("P2_MIXED",),
    BASE + "_NO_P3_A4": ("P3", "A4"),
    BASE + "_NO_P3_P2M": ("P3", "P2_MIXED"),
    BASE + "_NO_A4_P2M": ("A4", "P2_MIXED"),
    BASE + "_NO_P3_A4_P2M": ("P3", "A4", "P2_MIXED"),
}
ARMS = tuple(REMOVALS)
LEARNING_ARMS = ARMS
EXTERNAL_ARMS: tuple[str, ...] = ()
RUN_SEED = baseline.RUN_SEED


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Keep the original engine, initialization, seed and wall-clock budget."""
    if arm not in ARMS:
        raise ValueError("Unknown ablation3 arm")
    return baseline.arm_config(BASE, seed, seconds)


def search_config(arm: str) -> LSV2Config:
    """Remove exactly the named actions, retaining original action order and A3/A8."""
    if arm not in ARMS:
        raise ValueError("Unknown ablation3 arm")
    cfg = baseline.search_config(BASE)
    return replace(cfg, operators=tuple(op for op in cfg.operators if op not in REMOVALS[arm]))


def make_search(
    arm: str, config: Config, calibration: dict, table: QTable, *, training: bool
) -> QLearningSearch:
    """Construct unchanged online Q1 NET with a restricted selectable/bootstrap mask."""
    spec = QSpec("Q1_NET", 0.9, calibration["rho"], calibration["h"], calibration["w_cutoff"])
    return QLearningSearch(
        config,
        operator_config(config.seed, config.seconds or 1.0),
        spec,
        table,
        training=training,
        lsv2=search_config(arm),
    )


def configuration_record(arm: str, seed: int, seconds: float, calibration: dict) -> dict:
    """Seal the exact baseline configuration with only its operator tuple replaced."""
    record = baseline.configuration_record(BASE, seed, seconds, calibration)
    record["lsv2"] = asdict(search_config(arm))
    return record
