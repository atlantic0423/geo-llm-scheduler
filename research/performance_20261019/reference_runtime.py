"""Restore bf719d6 implementations in memory for local differential checks."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from geo_llm_scheduler.archive.pareto import Archive
from geo_llm_scheduler.engine.trajectory import improve
from geo_llm_scheduler.operators.structural import StructuralOperator
from geo_llm_scheduler.scheduling.resources import ResourceCalendar, earliest, fits
from geo_llm_scheduler.scheduling.ssgs import decode

from .frozen import initial_bill_index, pareto, resources, ssgs, structural, trajectory

# The frozen .py files are exact original Git blobs. Resolve the original
# SSGS's resource import here so it cannot accidentally use the new fast path.
ssgs.earliest = resources.earliest
trajectory.StructuralOperator = structural.StructuralOperator


@contextmanager
def reference_runtime() -> Iterator[None]:
    """Temporarily replace loaded core bindings, without RNG or input changes.

    Import the intended solver/initializer before entering. This helper is only
    for serialized tests and benchmarks, never a shared-thread runtime option.
    """
    replacements = (
        (Archive, pareto.Archive),
        (StructuralOperator, structural.StructuralOperator),
        (decode, ssgs.decode),
        (earliest, resources.earliest),
        (fits, resources.fits),
        (ResourceCalendar, initial_bill_index.ResourceCalendar),
        (improve, trajectory.improve),
    )
    patches = []
    try:
        for name, module in tuple(sys.modules.items()):
            if module is None or not name.startswith("geo_llm_scheduler."):
                continue
            for attribute, value in tuple(vars(module).items()):
                for current, reference in replacements:
                    if value is current:
                        replacement = patch.object(module, attribute, reference)
                        replacement.start()
                        patches.append(replacement)
                        break
        yield
    finally:
        for replacement in reversed(patches):
            replacement.stop()
