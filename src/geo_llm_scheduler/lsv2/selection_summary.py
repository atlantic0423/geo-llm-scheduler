"""Run-scoped diagnostics: keep skipped opportunities separate from empty searches."""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

STATUSES = frozenset(
    (
        "empty",
        "evaluated",
        "no_opportunity",
        "deadline_partial",
        "deadline_after_selection",
        "deadline_skip",
        "region_skip",
    )
)


def summarize_selection(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize chronological records from ONE run with a fixed search scope.

    Pool exclusion counts are repeated exposures, never distinct saved calls.
    Raw MS/starts distinguish physical inputs exactly, including tiny changes.
    Changing HV frames forbid adding per-call HV gains into a final-HV estimate.
    """
    statuses: Counter[str] = Counter()
    selected: set[tuple] = set()
    known_empty: set[tuple] = set()
    repeat_selected = after_empty = missing_input = selected_calls = 0
    excluded = raw_exposures = eligible_exposures = measured_pools = 0
    exact = proposals = accepted = hv_positive = reflow = 0
    seconds = selection_seconds = 0.0
    region_calls: Counter[str] = Counter()
    region_seconds: dict[str, float] = {}
    for row in rows:
        status = row["status"]
        if status not in STATUSES:
            raise ValueError(f"Unknown polish status: {status}")
        statuses[status] += 1
        cost = float(row.get("seconds", 0.0))
        seconds += cost
        selection_seconds += float(row.get("seconds_selection", 0.0))
        exact += int(row.get("exact", 0))
        proposals += int(row.get("proposals", 0))
        accepted += bool(row.get("accepted", False))
        hv_positive += float(row.get("hv_gain", 0.0)) > 0
        reflow += int(row.get("reflow_replaced", 0))
        if "raw_pool_size" in row:
            raw, eligible = int(row["raw_pool_size"]), int(row["eligible_pool_size"])
            removed = int(row["certified_empty_excluded"])
            if raw < 0 or eligible < 0 or removed < 0 or raw != eligible + removed:
                raise ValueError("Inconsistent eligibility pool denominators")
            measured_pools += 1
            raw_exposures += raw
            eligible_exposures += eligible
            excluded += removed
        if status in ("no_opportunity", "deadline_skip", "region_skip"):
            continue
        selected_calls += 1
        region = str(row.get("segment", "unknown"))
        region_calls[region] += 1
        region_seconds[region] = region_seconds.get(region, 0.0) + cost
        if "input_schedule" not in row:
            missing_input += 1
            continue
        schedule = row["input_schedule"]
        key = (tuple(schedule["ms"]), tuple(schedule["starts"]))
        repeat_selected += key in selected
        after_empty += key in known_empty
        selected.add(key)
        if status == "empty":
            known_empty.add(key)
    complete = statuses["empty"] + statuses["evaluated"]
    generated = complete + statuses["deadline_partial"]
    attempts = sum(statuses.values())
    return dict(
        module_attempts=attempts,
        statuses=dict(statuses),
        generator_calls=generated,
        non_generated_calls=attempts - generated,
        non_generated_rate=(attempts - generated) / attempts if attempts else None,
        complete_searches=complete,
        empty_searches=statuses["empty"],
        empty_rate_complete=statuses["empty"] / complete if complete else None,
        no_opportunity_calls=statuses["no_opportunity"],
        selected_calls=selected_calls,
        unique_selected_inputs=len(selected),
        repeated_selected_calls=repeat_selected,
        selections_after_complete_empty=after_empty,
        selected_without_input_record=missing_input,
        measured_pool_calls=measured_pools,
        raw_pool_exposures=raw_exposures,
        eligible_pool_exposures=eligible_exposures,
        certified_empty_exposures=excluded,
        seconds_full=seconds,
        seconds_selection=selection_seconds,
        seconds_selected_by_region=dict(region_seconds),
        selected_calls_by_region=dict(region_calls),
        exact=exact,
        proposals=proposals,
        accepted_bill_calls=accepted,
        positive_hv_calls=hv_positive,
        reflow_replacements=reflow,
        notes=[
            "single fixed-scope run; pool exposures may repeatedly count the same input",
            "complete-empty denominator excludes skips and partial evaluation",
            "per-call HV gains are not summed across changing frames",
        ],
    )
