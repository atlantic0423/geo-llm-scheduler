"""Operations-only flat dispatch of hybrid-baseline workers, reusing the slot dispatcher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source"


def main() -> None:
    """Run only this batch's campaign roots with the frozen worker, longest budgets first."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=12)
    parser.add_argument("--exclude-cpus", default="")
    args = parser.parse_args()
    assert args.root.resolve() in {(ROOT / name).resolve() for name in ("formal", "pilot")}
    sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "scripts")]
    import run_ext_hybrid as runner
    from slot_assignment_dispatch import cpu_order, dispatch

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
        json.dumps(dispatch(args.root, runner, cpus, SOURCE / "scripts/run_ext_hybrid.py")),
        flush=True,
    )


if __name__ == "__main__":
    main()
