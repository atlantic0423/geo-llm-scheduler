"""Adopted working algorithm, reusing the frozen GS4 implementation and audit runner."""

from __future__ import annotations

import json
import math
from pathlib import Path

from geo_llm_scheduler.experiments import s4_ablation, trans_runner
from geo_llm_scheduler.io.loaders import load_instance

DEFAULT_ALGORITHM = "GS4_GUIDED_RAND_SWITCH_D4"
DEFAULT_PROFILE = "working-default/v1"


def run_default(
    instance: str | Path,
    output: str | Path,
    *,
    calibration: dict,
    seed: int = 1,
    seconds: float | None = None,
) -> dict:
    """Run B2 + P1 + N7 + GS4 with explicit calibration and non-overwriting artifacts.

    The inclusive budgets are 600/1200/2400 seconds for 50/100/200 Jobs.
    Other sizes require an explicit budget; no calibration is estimated online.
    """
    if not isinstance(calibration, dict) or not {"rho", "h", "w_cutoff"} <= calibration.keys():
        raise ValueError("Supply an explicit frozen S4 calibration record")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if seconds is None:
        size = len(load_instance(instance).jobs)
        if size not in s4_ablation.BUDGETS:
            raise ValueError("No default budget for this instance size; supply seconds explicitly")
        seconds = s4_ablation.BUDGETS[size]
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Finite positive budget required")
    return trans_runner.run_case(
        DEFAULT_ALGORITHM,
        Path(instance),
        Path(output),
        seed=seed,
        seconds=seconds,
        calibration=calibration,
    )


def run_default_config(data: dict, **overrides: object) -> dict:
    """Resolve a default-profile mapping and CLI overrides, then run the adopted method.

    Paths follow the legacy CLI convention: relative to the working directory.
    Calibration may be a JSON record or a keyed collection of frozen records.
    """
    fields = {"profile", "instance", "output", "calibration", "calibration_key", "seed", "seconds"}
    unknown = (data.keys() | overrides.keys()) - fields
    if unknown or data.get("profile") != DEFAULT_PROFILE:
        raise ValueError(f"Invalid default profile or unknown fields: {sorted(unknown)}")
    values = {**data, **{key: value for key, value in overrides.items() if value is not None}}
    for key in ("instance", "output", "calibration"):
        if not isinstance(values.get(key), (str, Path)) or not str(values[key]).strip():
            raise ValueError(f"Default algorithm requires {key}")
    calibration = json.loads(Path(values["calibration"]).read_text(encoding="utf-8"))
    if values.get("calibration_key") is not None:
        calibration = calibration[values["calibration_key"]]
    return run_default(
        values["instance"],
        values["output"],
        calibration=calibration,
        seed=values.get("seed", 1),
        seconds=values.get("seconds"),
    )
