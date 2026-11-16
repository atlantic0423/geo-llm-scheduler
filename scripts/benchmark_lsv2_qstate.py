"""Compare exact-equivalent Q feature/identity overhead on trusted saved snapshots."""

import argparse
import gzip
import json
import pickle
from pathlib import Path
from statistics import median
from time import perf_counter

from geo_llm_scheduler.experiments.lsv2_delayed import opportunity_features
from geo_llm_scheduler.experiments.lsv2_qscreen import make_search
from geo_llm_scheduler.lsv2.qlearning import QTable
from geo_llm_scheduler.utils.numeric import identity


def _timed(function, repeats):
    times = []
    for _ in range(3):
        tick = perf_counter()
        for _ in range(repeats):
            function()
        times.append((perf_counter() - tick) / repeats)
    return median(times)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.snapshots.glob("*/snapshot_p1_s1.pkl.gz")):
        with gzip.open(path, "rb") as f:
            s = pickle.load(f)
        p, c, pop, archive = s["problem"], s["current"], s["population"], s["archive"]
        for design, field in [
            ("Q2_P_NET", "prefill_window_fraction"),
            ("Q2_W_NET", "resource_wait_fraction"),
            ("Q3_NET", None),
        ]:
            search = make_search(
                design + "_G9",
                s["engine"],
                dict(rho=0, h=1, p_cutoff=0, w_cutoff=0),
                QTable(),
                training=True,
            )
            if field:

                def old():
                    return opportunity_features(s)[field]

                def fast():
                    return search._opportunity(p, c)
            else:

                def old():
                    return (
                        identity(c),
                        [identity(x) for x in pop],
                        {identity(x) for x in archive},
                    )

                def fast():
                    return search._identity_snapshot(c, pop, archive)

            assert old() == fast()
            before, after = _timed(old, args.repeats), _timed(fast, args.repeats)
            rows.append(
                dict(
                    key=path.parent.name,
                    n=len(p.jobs),
                    design=design,
                    reference_seconds=before,
                    optimized_seconds=after,
                    speedup=before / after,
                    exact_equal=True,
                )
            )
    assert rows
    args.output.write_text(
        json.dumps(
            dict(
                rows=rows,
                repeats=args.repeats,
                scope="Component overhead only; not an end-to-end solver speedup.",
            ),
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                d: median(r["speedup"] for r in rows if r["design"] == d)
                for d in ("Q2_P_NET", "Q2_W_NET", "Q3_NET")
            }
        )
    )


if __name__ == "__main__":
    _main()
