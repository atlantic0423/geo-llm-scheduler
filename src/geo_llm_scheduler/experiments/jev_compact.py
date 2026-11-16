"""Lossless tabular wire states: no rounding, candidate filtering or hidden evaluation."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

SCHEMA = "jev_tables_v3"


def _table(rows: list[dict]) -> dict:
    columns = list(rows[0]) if rows else []
    if any(list(row) != columns for row in rows):
        raise ValueError("Nonuniform table schema")
    return dict(columns=columns, rows=[[row[k] for k in columns] for row in rows])


def _untable(value: dict) -> list[dict]:
    return [dict(zip(value["columns"], row, strict=True)) for row in value["rows"]]


def compact_state(state: dict) -> dict:
    """Factor repeated fields while retaining every Job and every original numeric value."""
    result = deepcopy(state)
    result["wire_schema"] = SCHEMA
    history = result.get("action_history")
    if history and history.get("by_group"):
        groups = history.pop("by_group")
        first = next(iter(next(iter(groups.values())).values()))
        columns = list(first)
        values = [
            (g, a, tuple(row[k] for k in columns))
            for g, items in groups.items()
            for a, row in items.items()
        ]
        default = Counter(v for _, _, v in values).most_common(1)[0][0]
        history["metric_columns"] = columns
        history["groups"] = list(groups)
        history["actions"] = list(next(iter(groups.values())))
        history["default_metrics"] = list(default)
        history["overrides"] = [[g, a, list(v)] for g, a, v in values if v != default]
    if "instance" in result:
        regions = result["instance"].get("regions", [])
        for region in regions:
            for key in ("tou_segments", "instances"):
                if key in region:
                    region[key] = _table(region[key])
        if regions:
            result["instance"]["regions"] = _table(regions)
    jobs = result.get("all_jobs")
    if jobs and isinstance(jobs[0], dict) and "operations" in jobs[0]:
        profiles: list[list] = []
        profile_ids, resource_wait, flow_wait = [], [], []
        for j, job in enumerate(jobs):
            if job["job_id"] != j or [op[0] for op in job["operations"]] != [2 * j, 2 * j + 1]:
                raise ValueError("Noncanonical Job/operation identity")
            for op in job["operations"]:
                profile = op[6:9]
                if profile not in profiles:
                    profiles.append(profile)
                profile_ids.append(profiles.index(profile))
                resource_wait.append(op[9])
                flow_wait.append(op[10])
        result["all_jobs"] = dict(
            columns=["job_id", "release_s", "kv_if_separate_s"],
            rows=[[j[k] for k in ("job_id", "release_s", "kv_if_separate_s")] for j in jobs],
            profile_columns=["duration_s", "compute", "vram_gb"],
            profiles=profiles,
            operation_profile_ids=profile_ids,
            resource_wait_s=resource_wait,
            flow_wait_s=flow_wait,
        )
    return result


def expand_state(state: dict) -> dict:
    """Reconstruct the original snapshot for independent audits and evidence readers."""
    result = deepcopy(state)
    if result.pop("wire_schema", None) != SCHEMA:
        raise ValueError("Unknown compact state schema")
    instance = result.get("instance")
    if instance and isinstance(instance.get("regions"), dict):
        instance["regions"] = _untable(instance["regions"])
        for region in instance["regions"]:
            for key in ("tou_segments", "instances"):
                if key in region:
                    region[key] = _untable(region[key])
    history = result.get("action_history")
    if history and "metric_columns" in history:
        columns = history.pop("metric_columns")
        default = history.pop("default_metrics")
        overrides = {(g, a): v for g, a, v in history.pop("overrides")}
        groups, actions = history.pop("groups"), history.pop("actions")
        history["by_group"] = {
            g: {a: dict(zip(columns, overrides.get((g, a), default), strict=True)) for a in actions}
            for g in groups
        }
    jobs = result.get("all_jobs")
    if isinstance(jobs, dict) and "profiles" in jobs:
        solution = result["current_solution"]
        machines = {
            m["id"]: r["geographic_region_id"]
            for r in result["instance"]["regions"]
            for m in r["instances"]
        }
        restored = []
        for job_id, release, kv in jobs["rows"]:
            operations = []
            for o in (2 * job_id, 2 * job_id + 1):
                duration, compute, vram = jobs["profiles"][jobs["operation_profile_ids"][o]]
                machine, start = solution["MS"][o], solution["starts_s"][o]
                operations.append(
                    [
                        o,
                        "P" if o % 2 == 0 else "D",
                        machine,
                        machines[machine],
                        start,
                        start + duration,
                        duration,
                        compute,
                        vram,
                        jobs["resource_wait_s"][o],
                        jobs["flow_wait_s"][o],
                    ]
                )
            restored.append(
                dict(
                    job_id=job_id,
                    release_s=release,
                    kv_if_separate_s=kv,
                    actual_kv_s=kv * (solution["MS"][2 * job_id] != solution["MS"][2 * job_id + 1]),
                    operations=operations,
                )
            )
        result["all_jobs"] = restored
    return result
