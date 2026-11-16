"""Restore the 582237f scheduling, exact and LS-billing code in memory for differential checks.

The files under ``frozen/`` are the exact original Git blobs (checked by a test). Their
imports of the live ``scheduling.resources`` are rebound here to the frozen copy, so the
reference path can never reach a new fast path.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from geo_llm_scheduler.evaluation import exact as live_exact
from geo_llm_scheduler.lsv2 import billing as live_billing
from geo_llm_scheduler.scheduling import resources as live_resources
from geo_llm_scheduler.scheduling import ssgs as live_ssgs

from .frozen import billing, exact, resources, ssgs


def reference_calendar_fits(calendar, machine, operation, start, others):  # noqa: ANN001, ANN201
    """The 582237f destroy-repair slot check: a full scan of ``others``."""
    return resources._fits_intervals(calendar.problem, machine, operation, start, others)


for _module in (ssgs, exact):
    for _name in ("ResourceCalendar", "ResourceView", "est_for"):
        if hasattr(_module, _name):
            setattr(_module, _name, getattr(resources, _name))

_NAMES = {
    (live_resources, resources): (
        "ResourceView",
        "ResourceCalendar",
        "_fits_intervals",
        "_earliest_intervals",
        "fits",
        "earliest",
        "intervals",
        "est_for",
        "union_length",
    ),
    (live_ssgs, ssgs): ("decode", "refresh_diagnostics"),
    (live_exact, exact): (
        "violations",
        "_checked_resources",
        "power_segments",
        "_power_segments",
        "_load_steps",
        "_machine_power",
        "regional_windows",
        "evaluate",
        "tied_peaks",
    ),
    (live_billing, billing): ("merge", "difference", "PowerProfile"),
}
REPLACEMENTS = tuple(
    (getattr(live, name), getattr(frozen, name))
    for (live, frozen), names in _NAMES.items()
    for name in names
) + ((live_resources.calendar_fits, reference_calendar_fits),)


@contextmanager
def reference_runtime() -> Iterator[None]:
    """Temporarily rebind every loaded ``geo_llm_scheduler`` reference to the frozen code.

    Import the intended solver before entering. Serialized tests and benchmarks only;
    never a shared-thread runtime option.
    """
    patches = []
    try:
        for name, module in tuple(sys.modules.items()):
            if module is None or not name.startswith("geo_llm_scheduler."):
                continue
            for attribute, value in tuple(vars(module).items()):
                for current, reference in REPLACEMENTS:
                    if value is current:
                        replacement = patch.object(module, attribute, reference)
                        replacement.start()
                        patches.append(replacement)
                        break
        yield
    finally:
        for replacement in reversed(patches):
            replacement.stop()
