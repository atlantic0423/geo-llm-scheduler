"""Resume a frozen campaign while handling a disappearing heartbeat in its quota scan."""

from __future__ import annotations

import argparse
import re
import runpy
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def transient_heartbeat_error(stderr: str, campaign: Path) -> bool:
    """Recognize only ENOENT errors for this campaign's atomic heartbeat files."""
    prefix = re.escape(str(campaign))
    pattern = re.compile(
        rf"du: cannot access '{prefix}/formal_node[01]/ops/heartbeats/"
        r"\.[^/']+\.json\.\d+\.tmp': No such file or directory"
    )
    lines = stderr.strip().splitlines()
    return bool(lines) and all(pattern.fullmatch(line) for line in lines)


def quota_check_output(
    original: Callable[..., Any],
    campaign: Path,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Retry a transient quota read, then report no free quota rather than dispatch."""
    command = ["du", "-sb", str(campaign.parent)]
    if not args or args[0] != command:
        return original(*args, **kwargs)
    options = dict(kwargs, stderr=subprocess.PIPE)
    for attempt in range(4):
        try:
            return original(*args, **options)
        except subprocess.CalledProcessError as error:
            stderr = error.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            if error.returncode != 1 or not transient_heartbeat_error(stderr or "", campaign):
                raise
            print(
                f"Quota read encountered an atomic heartbeat replacement; retry {attempt + 1}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < 3:
                time.sleep(0.05 * 2**attempt)
    # The frozen dispatcher reserves 2 GiB. This forces its existing resource guard.
    conservative = f"{100 * 1024**3}\t{campaign.parent}\n"
    print(
        "Quota read remained transiently unavailable; keep the existing dispatch guard closed",
        file=sys.stderr,
        flush=True,
    )
    return (
        conservative
        if options.get("text") or options.get("universal_newlines")
        else conservative.encode()
    )


def main() -> None:
    """Run the original source; change only the supervisor's quota subprocess wrapper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    root = args.root.resolve()
    original = subprocess.check_output
    subprocess.check_output = lambda *a, **kw: quota_check_output(original, root.parent, *a, **kw)
    sys.path.insert(0, str(source / "src"))
    sys.argv = [
        str(source / "scripts/run_macro_width_campaign.py"),
        "run",
        "--root",
        str(root),
        "--workers",
        str(args.workers),
        "--deadline",
        str(args.deadline),
    ]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        subprocess.check_output = original


if __name__ == "__main__":
    main()
