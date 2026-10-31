"""Replay trusted development snapshots to compare numeric tie behavior across runtimes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

from geo_llm_scheduler.experiments.lsv2_delayed import (
    audit_candidates,
    fingerprint,
    force_step,
    restore,
)
from geo_llm_scheduler.utils.numeric import identity


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.inputs.glob("st_n*_s16901/snapshot_p1_s1.pkl.gz"))
    if len(paths) != 8:
        raise ValueError("Expected the eight original development snapshots")
    rows = []
    for path in paths:
        key = path.parent.name
        # Inputs are our own SHA-verified campaign pickles, never untrusted uploads.
        with gzip.open(path, "rb") as stream:
            snapshot = pickle.load(stream)
        seed0 = int.from_bytes(hashlib.sha256((key + path.name).encode()).digest()[:8])
        for repeat in range(3):
            context = restore(snapshot)
            steps = []
            for action, seed in (
                ("P2_FLOW", seed0 + 1000 + repeat),
                ("A7", seed0 + 100000 + repeat),
            ):
                row = force_step(context, action, seed)
                steps.append(
                    dict(
                        action=action,
                        seed=seed,
                        chain=row["chain"],
                        current_sha256=hashlib.sha256(
                            repr(identity(context.current)).encode()
                        ).hexdigest(),
                        current_objectives=context.current.evaluation.objectives,
                        hv_gain=row["hv_gain"],
                        state_sha256=fingerprint(context),
                    )
                )
            rows.append(
                dict(
                    key=key,
                    snapshot=path.name,
                    repeat=repeat,
                    input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    steps=steps,
                    fresh_exact=audit_candidates(context),
                )
            )
        print(key, flush=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = dict(
        python=sys.version,
        source=source,
        rows=rows,
        fresh_exact=sum(row["fresh_exact"] for row in rows),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(paths=len(paths), pairs=len(rows), fresh_exact=result["fresh_exact"])))


if __name__ == "__main__":
    _main()
