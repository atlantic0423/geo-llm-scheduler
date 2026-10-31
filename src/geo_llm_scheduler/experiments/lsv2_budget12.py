"""Twelve authorized arms: unchanged Q1, B/L, A8 removal and two hybrid baselines."""

from __future__ import annotations

from dataclasses import asdict

from geo_llm_scheduler.config import Config
from geo_llm_scheduler.experiments import ext_hybrid
from geo_llm_scheduler.experiments.lsv2_study import operator_config, study_config
from geo_llm_scheduler.lsv2.qlearning import QLearningSearch, QSpec, QTable
from geo_llm_scheduler.lsv2.search import OPERATORS, LSV2Config, LSV2Search

INTERNAL_ARMS = (
    "Q1_NET_G9_B6L5",
    "Q1_NET_G0_B6L5",
    "UNIFORM_B6L5",
    "Q1_NET_G9_B3L3",
    "Q1_NET_G0_B3L3",
    "UNIFORM_B3L3",
    "Q1_NET_G0_B1L1",
    "UNIFORM_B1L1",
    "Q1_NET_G9_B6L5_NOA8",
    "UNIFORM_B6L5_NOA8",
)
EXTERNAL_ARMS = ("EXTH_SPEA2", "EXTH_MOEAD")
ARMS = INTERNAL_ARMS + EXTERNAL_ARMS
LEARNING_ARMS = tuple(a for a in INTERNAL_ARMS if a.startswith("Q1_"))
RUN_SEED = 16901


def arm_config(arm: str, seed: int, seconds: float) -> Config:
    """Return the existing BILL engine or the unchanged external hybrid configuration."""
    if arm not in ARMS:
        raise ValueError("Unknown budget12 arm")
    if arm in EXTERNAL_ARMS:
        return ext_hybrid.arm_config(arm, seed, seconds)
    return study_config("LSV2", seed, seconds)


def search_config(arm: str) -> LSV2Config:
    """Change only B, L and the action set; all other LSV2 semantics stay fixed."""
    if arm not in INTERNAL_ARMS:
        raise ValueError("Not an internal budget12 arm")
    b, steps = (1, 1) if "B1L1" in arm else (3, 3) if "B3L3" in arm else (6, 5)
    return LSV2Config(
        budget=b,
        trajectory_steps=steps,
        selector="uniform" if arm.startswith("UNIFORM") else "bandit",
        operators=tuple(a for a in OPERATORS if a != "A8" or not arm.endswith("NOA8")),
    )


def _qspec(arm: str, calibration: dict) -> QSpec:
    return QSpec(
        "Q1_NET",
        0.9 if "G9" in arm else 0.0,
        calibration["rho"],
        calibration["h"],
        calibration["w_cutoff"],
    )


def make_search(
    arm: str, config: Config, calibration: dict, table: QTable, *, training: bool
) -> LSV2Search:
    """Build an unchanged Q1-net or uniform selector with the declared LS configuration."""
    ls = search_config(arm)
    operators = operator_config(config.seed, config.seconds or 1.0)
    if arm not in LEARNING_ARMS:
        return LSV2Search(ls, config, operators, retain_steps=True)
    # Preserve the previous Q1 calibration payload exactly, including its unused cutoff.
    spec = _qspec(arm, calibration)
    return QLearningSearch(config, operators, spec, table, training=training, lsv2=ls)


def configuration_record(arm: str, seed: int, seconds: float, calibration: dict) -> dict:
    """Serialize all configuration inputs so the campaign seal also binds the treatment."""
    config = arm_config(arm, seed, seconds)
    if arm in EXTERNAL_ARMS:
        return dict(
            engine=asdict(config),
            initialization=ext_hybrid.POLICY,
            polish_share=ext_hybrid.POLISH_SHARE,
            lsv2=None,
            qspec=None,
        )
    return dict(
        engine=asdict(config),
        lsv2=asdict(search_config(arm)),
        operators=asdict(operator_config(seed, seconds)),
        qspec=asdict(_qspec(arm, calibration)) if arm in LEARNING_ARMS else None,
    )
