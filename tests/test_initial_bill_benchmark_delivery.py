"""Guard fast-end plots against unobserved interpolation and audit against altered data."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fast_plot_never_connects_to_unobserved_outside_point():
    plotter = load("plot_initial_bill_benchmark")
    fig, ax = plotter.plt.subplots()
    data = {a: [(10, 100), (100, 1)] for a in plotter.ARMS}
    plotter.draw(ax, data, "test", zoom=True)
    assert all(list(line.get_xdata()) == [10] for line in ax.lines)
    plotter.plt.close(fig)


def test_plot_rejects_incomplete_case_matrix(tmp_path):
    plotter = load("plot_initial_bill_benchmark")
    (tmp_path / "fronts_seed11001.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError):
        plotter.plot(tmp_path, tmp_path / "figures")


def test_independent_audit_detects_changed_cluster_effect(tmp_path):
    auditor = load("audit_initial_bill_benchmark")
    arms = ("BILL10", "BILL20", "DEMAND20")
    metrics = ("hv_diff", "igd_plus_gain", "igd_gain", "wall_diff")
    report = {
        "statistics": {
            a: {
                m: {
                    "mean": 0,
                    "ci95": [0, 0],
                    "p_signflip": 1,
                    "p_holm": 1,
                    "base_effects": [0] * 36,
                }
                for m in metrics
            }
            for a in arms
        },
        "families": {
            "primary": [(a, m) for a in arms for m in metrics[:2]],
            "igd_secondary": [(a, "igd_gain") for a in arms],
            "wall_secondary": [(a, "wall_diff") for a in arms],
        },
    }
    path = tmp_path / "pairs.csv"
    rows = [
        {
            "arm": a,
            "jobs": jobs,
            "base_seed": jobs * 100 + i,
            "tariff": t,
            "seed": s,
            **{m: 0 for m in metrics},
        }
        for a in arms
        for jobs in (50, 100, 200)
        for i in range(12)
        for t in ("H", "T")
        for s in (11001, 11002)
    ]

    def write():
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write()
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    assert auditor.audit(tmp_path)["independent_statistics"] == 12
    rows[0]["hv_diff"] = 1
    write()
    with pytest.raises(AssertionError):
        auditor.audit(tmp_path)
