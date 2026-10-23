"""Descriptive endpoint, state-cell and anytime checks; no additional hypothesis tests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geo_llm_scheduler.experiments.history_rl import ARMS


def supplement(data: Path, output: Path) -> dict:
    """Summarize all runs without choosing cases or adding inferential comparisons."""
    with (output / "run_metrics.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fronts = json.loads((output / "normalized_fronts.json").read_text())
    references = json.loads((output / "references.json").read_text())
    lookup = {(r["key"], r["arm"]): r for r in rows}
    records = []
    for r in rows:
        baseline = lookup[(r["key"], "BILL_PERMUTED")]
        f = fronts[r["key"] + "|" + r["arm"]]
        b = fronts[r["key"] + "|BILL_PERMUTED"]
        ref = references[f["reference"]]

        def bill_in_window(front: list) -> float:
            limit = 1.1 * float(baseline["min_flow"])
            bills = [
                ref["ideal"][1] + y * ref["span"][1]
                for x, y in front
                if ref["ideal"][0] + x * ref["span"][0] <= limit + 1e-9
            ]
            assert bills
            return min(bills)

        q = json.loads((data / "raw/formal/runs" / r["key"] / r["arm"] / "qtable.json").read_text())
        counts = np.asarray(q["selections"])
        positive = counts[counts > 0]
        records.append(
            dict(
                key=r["key"],
                arm=r["arm"],
                base_seed=r["base_seed"],
                n=int(r["n"]),
                min_flow_vs_bill_pct=100 * (float(r["min_flow"]) / float(baseline["min_flow"]) - 1),
                min_bill_vs_bill_pct=100 * (float(r["min_bill"]) / float(baseline["min_bill"]) - 1),
                fast_window_bill_saving_pct=100
                * (1 - bill_in_window(f["archive"]) / bill_in_window(b["archive"])),
                state_action_visited=int((counts > 0).sum()),
                state_action_fraction=float((counts > 0).mean()),
                mean_calls_per_visited_cell=float(positive.mean()) if len(positive) else 0,
                singleton_visited_cell_fraction=float((positive == 1).mean())
                if len(positive)
                else 0,
            )
        )
    fields = list(records[0])
    with (output / "endpoint_and_cell_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summaries = []
    for n in (0, 20, 50, 100, 200):
        for arm in ARMS:
            chosen = [r for r in records if r["arm"] == arm and (not n or r["n"] == n)]
            summaries.append(
                dict(
                    n=n, arm=arm, **{k: float(np.mean([r[k] for r in chosen])) for k in fields[4:]}
                )
            )
    result = dict(
        descriptive_only=True,
        threshold="Flow <=1.1 times same-run ordinary fastest Flow; actual saved points",
        rows=len(records),
        summaries=summaries,
    )
    (output / "supplementary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    with (output / "anytime_metrics.csv").open(encoding="utf-8") as stream:
        snaps = list(csv.DictReader(stream))
    durations = {20: 300, 50: 600, 100: 1200, 200: 2400}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, n in zip(axes.flat, durations):
        for arm in (
            "BILL_PERMUTED",
            "STRICT_TWO_FOCUS",
            "HISTORY_ACTION",
            "PREF_HISTORY",
            "PREF_HISTORY_STOP2",
            "PREF_HISTORY_FRONTIER_STOP2",
        ):
            groups = defaultdict(list)
            for s in snaps:
                if int(s["n"]) == n and s["arm"] == arm and s["target"] != "initialization":
                    groups[float(s["target"]) / durations[n]].append(float(s["hv"]))
            x = sorted(groups)
            ax.plot(x, [np.mean(groups[k]) for k in x], marker="o", markersize=3, label=arm)
        ax.set_title(f"{n} jobs | four development parents")
        ax.set_xlabel("Requested wall-clock fraction (observations may be delayed)")
        ax.set_ylabel("Mean normalized Archive HV")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Anytime observations: common per-input empirical-union coordinates")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / ("05_anytime." + suffix), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {k: v for k, v in supplement(args.data, args.output).items() if k != "summaries"}
        )
    )
