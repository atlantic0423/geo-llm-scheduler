"""Plot each seed-11001 P100 front separately in physical objective units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("BASE", "BILL10", "BILL20", "DEMAND20")
COLORS = ("#333333", "#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "s", "^", "D")


def draw(ax: plt.Axes, data: dict, title: str, zoom: bool = False) -> None:
    """Show a single seed's observed points, with lines serving only as visual guides."""
    baseline = min(x for x, _ in data["BASE"])
    left = min(x for arm in ARMS for x, _ in data[arm]) * 0.99
    limit = baseline * 1.5
    for arm, color, marker in zip(ARMS, COLORS, MARKERS, strict=True):
        points = np.asarray(data[arm])
        if zoom:
            # Do not let a segment to an out-of-window point suggest an unobserved fast solution.
            points = points[(points[:, 0] >= left) & (points[:, 0] <= limit)]
        ax.plot(
            points[:, 0],
            points[:, 1],
            marker=marker,
            color=color,
            markersize=3.4,
            linewidth=0.8,
            alpha=0.88,
            label=arm,
        )
    if zoom:
        visible = [y for arm in ARMS for x, y in data[arm] if left <= x <= limit]
        lo, hi = min(visible), max(visible)
        gap = max((hi - lo) * 0.08, hi * 0.015)
        ax.set_xlim(left, limit)
        ax.set_ylim(max(0, lo - gap), hi + gap)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Total Flow Time (s)")
    ax.set_ylabel("Electricity Bill (CNY)")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)


def plot(data: Path, out: Path) -> dict:
    """Export full and fast-end PNG/PDF plots for all 72 native cases."""
    fronts = json.loads((data / "fronts_seed11001.json").read_text(encoding="utf-8"))
    assert len(fronts) == 72
    out.mkdir(exist_ok=True, parents=True)
    for key, points in sorted(fronts.items()):
        for zoom in (False, True):
            fig, ax = plt.subplots(figsize=(6.4, 4.3), layout="constrained")
            draw(ax, points, key + (" | fast end" if zoom else " | full front"), zoom)
            ax.legend(fontsize=8)
            fig.suptitle(
                "Seed 11001, independent P100 populations; lower-left is better", fontsize=9
            )
            name = key + ("_fast" if zoom else "_full")
            for ext in ("png", "pdf"):
                fig.savefig(out / (name + "." + ext), dpi=180)
            plt.close(fig)
    patterns = ("50_compute_tight_", "100_mixed_", "100_mixed_", "200_demand_high_")
    tariffs = ("T", "H", "T", "T")
    chosen = []
    for prefix, tariff in zip(patterns, tariffs, strict=True):
        chosen.append(
            min(k for k in fronts if k.startswith(prefix) and k.endswith("_" + tariff + "_11001"))
        )
    for zoom in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), layout="constrained")
        for ax, key in zip(axes.flat, chosen, strict=True):
            draw(ax, fronts[key], key, zoom)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(
            "Prespecified examples | " + ("fast end" if zoom else "complete initial fronts"),
            fontsize=13,
        )
        for ext in ("png", "pdf"):
            fig.savefig(out / ("prespecified_" + ("fast" if zoom else "full") + "." + ext), dpi=180)
        plt.close(fig)
    receipt = {
        "native_cases": 72,
        "seed": 11001,
        "individual_png_pdf": 288,
        "overview_png_pdf": 4,
        "prespecified_keys": chosen,
        "fast_zoom_flow_bound": "1.5 times BASE fastest; identical rule for every case",
        "interpolation_is_not_feasibility": True,
    }
    (out / "figure_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.data, args.out)))
