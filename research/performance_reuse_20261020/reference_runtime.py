"""Serialized differential/benchmark context for exact b148c4a Git blobs."""

import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

MODULES = {
    "resources": "scheduling.resources",
    "validation": "io.validation",
    "exact": "evaluation.exact",
    "ssgs": "scheduling.ssgs",
    "timing": "scheduling.timing",
    "pareto": "archive.pareto",
    "trajectory": "engine.trajectory",
    "active_pack": "operators.active_pack",
    "right_shift": "operators.right_shift",
    "severity": "diagnostics.severity",
    "packing_proxy": "operators.packing_proxy",
    "variation": "moead.variation",
    "generators": "initialization.generators",
    "peak_coalition": "operators.peak_coalition",
    "state": "rl.state",
}
current = {
    name: importlib.import_module("geo_llm_scheduler." + path) for name, path in MODULES.items()
}
original = {name: importlib.import_module(__package__ + ".frozen." + name) for name in MODULES}


@contextmanager
def reference_runtime() -> Iterator[None]:
    """Use frozen functions, including their loaded import aliases, in one thread.

    This helper never runs in production. Import intended callers before entry;
    the original and optimized work are always executed sequentially.
    """
    replacements = []
    for name, module in current.items():
        for attribute, value in vars(original[name]).items():
            if (
                not attribute.startswith("_")
                and callable(value)
                and getattr(value, "__module__", None) == original[name].__name__
                and hasattr(module, attribute)
            ):
                replacements.append((getattr(module, attribute), value))
    patches = []
    try:
        for name, module in tuple(sys.modules.items()):
            if module is None or not (
                name.startswith("geo_llm_scheduler.") or name.startswith(__package__ + ".frozen.")
            ):
                continue
            for attribute, value in tuple(vars(module).items()):
                for optimized, reference in replacements:
                    if value is optimized:
                        replacement = patch.object(module, attribute, reference)
                        replacement.start()
                        patches.append(replacement)
                        break
        yield
    finally:
        for replacement in reversed(patches):
            replacement.stop()
