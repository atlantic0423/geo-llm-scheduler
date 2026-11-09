"""Versioned JSON input converted to canonical units at the boundary."""

import json
from dataclasses import asdict
from pathlib import Path

from geo_llm_scheduler.domain.models import (
    Job,
    PowerModel,
    ProblemInstance,
    Profile,
    Region,
    ServingInstance,
    Tariff,
)
from geo_llm_scheduler.io.validation import validate_problem

POWER_MODEL_KEYS = ("beta", "kappa_prefill", "kappa_decode")


def load_instance(path: str | Path) -> ProblemInstance:
    """Load schema v1; time s/ms/h, power kW/W, memory GB, currency CNY."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported schema version")
    units = data.get("units", {})
    if units.get("memory", "GB") != "GB" or units.get("currency", "CNY") != "CNY":
        raise ValueError("Preconvert memory/currency explicitly to GB/CNY")
    try:
        time = {"s": 1.0, "ms": 0.001, "h": 3600.0}[units.get("time", "s")]
        power = {"kW": 1.0, "W": 0.001}[units.get("power", "kW")]
    except KeyError as exc:
        raise ValueError("Unsupported unit") from exc

    def profile(p: dict) -> Profile:
        return Profile(p["duration"] * time, p["compute"], p["vram"])

    # Optional load-dependent power; absent means the original two-state model.
    raw_power = data.get("power_model")
    if raw_power is None:
        power_model = PowerModel()
    else:
        if not isinstance(raw_power, dict) or set(raw_power) != set(POWER_MODEL_KEYS):
            raise ValueError("power_model requires exactly beta, kappa_prefill, kappa_decode")
        power_model = PowerModel(*(raw_power[k] for k in POWER_MODEL_KEYS))

    problem = ProblemInstance(
        tuple(
            Job(
                j["name"],
                j["release"] * time,
                profile(j["prefill"]),
                profile(j["decode"]),
                j.get("kv_delay", 0) * time,
            )
            for j in data["jobs"]
        ),
        tuple(
            Region(
                r["name"],
                tuple(Tariff(t["start"] * time, t["end"] * time, t["price"]) for t in r["tariffs"]),
                r["demand_rate"],
            )
            for r in data["regions"]
        ),
        tuple(
            ServingInstance(
                m["name"],
                m["region"],
                m["vram"],
                m["idle_kw"] * power,
                m["active_kw"] * power,
                tuple(m.get("phases", [0, 1])),
            )
            for m in data["instances"]
        ),
        power_model,
    )
    validate_problem(problem)
    return problem


def save_instance(problem: ProblemInstance, path: str | Path) -> None:
    """Serialize canonical units without introducing mutable domain state.

    The original two-state power model is written without a ``power_model`` key,
    so files saved before the load-dependent model existed stay byte-identical.
    """
    body = asdict(problem)
    if problem.power_model.legacy:
        body.pop("power_model")
    data = {
        "schema_version": 1,
        "units": {"time": "s", "power": "kW", "memory": "GB", "currency": "CNY"},
        **body,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
