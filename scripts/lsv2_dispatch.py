"""Operations-only flat dispatch of LSV2-study or stress-screen workers (slot dispatcher).

Longest budgets first, physical cores before SMT siblings, excluded cores skipped
entirely, fail-stop on any worker failure and 200 GB memory/disk reserve.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slot_assignment_dispatch import cpu_order, dispatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source"


def main() -> None:
    """Run only the authorized campaign roots with the frozen LSV2 worker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=48)
    parser.add_argument("--exclude-cpus", default="")
    parser.add_argument("--runner", choices=("run_lsv2", "run_stress_screen"), default="run_lsv2")
    args = parser.parse_args()
    assert args.root.resolve() in {(ROOT / name).resolve() for name in ("formal", "pilot")}
    sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "scripts")]
    import importlib

    runner = importlib.import_module(args.runner)

    assert not subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain"], text=True
    ).strip()
    allowed = set(os.sched_getaffinity(0))
    topology = {
        c: tuple(
            (Path(f"/sys/devices/system/cpu/cpu{c}/topology") / f).read_text().strip()
            for f in ("physical_package_id", "core_id")
        )
        for c in allowed
    }
    excluded = frozenset(int(c) for c in args.exclude_cpus.split(",") if c.strip())
    cpus = cpu_order(allowed, topology, args.parallel, excluded)
    print(
        json.dumps(dispatch(args.root, runner, cpus, SOURCE / "scripts" / (args.runner + ".py"))),
        flush=True,
    )


if __name__ == "__main__":
    main()
