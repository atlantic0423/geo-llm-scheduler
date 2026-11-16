"""Summarize bench_ls_v2 rows: per size and operator (E1) and per size and seed rule (E2)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

SEGMENTS = (("Flow端", -1e9, 1 / 3), ("中段", 1 / 3, 2 / 3), ("电费端", 2 / 3, 1e9))


def segment(x: float) -> str:
    return next(name for name, lo, hi in SEGMENTS if lo <= x < hi)


def summarize(results: list[dict]) -> dict:
    rows = [r for res in results for r in res["rows"]]
    out: dict = {"E1": {}, "E1_segment": {}, "E2": {}, "A8_gate": {}}
    groups = defaultdict(list)
    for r in rows:
        if r["exp"] == "E1":
            groups[(r["n"], r["op"])].append(r)
    for (n, op), rs in sorted(groups.items()):
        sec = sum(r["seconds"] for r in rs)
        out["E1"].setdefault(str(n), {})[op] = dict(
            calls=len(rs),
            success=mean(r["success"] for r in rs),
            hv_gain_per_call=mean(r["hv_gain"] for r in rs),
            hv_gain_per_second=sum(r["hv_gain"] for r in rs) / max(sec, 1e-9),
            seconds=sec / len(rs),
            exact=mean(r["exact"] for r in rs),
            empty=mean((r["candidates"] == 0) for r in rs),
        )
        by_seg = defaultdict(list)
        for r in rs:
            by_seg[segment(r["seed_x"])].append(r)
        out["E1_segment"].setdefault(str(n), {})[op] = {
            s: dict(
                calls=len(v),
                success=mean(r["success"] for r in v),
                hv_gain=mean(r["hv_gain"] for r in v),
            )
            for s, v in by_seg.items()
        }
        if op == "A8":
            for gate in (True, False):
                v = [r for r in rs if r["gate"] == gate]
                if v:
                    out["A8_gate"].setdefault(str(n), {})[str(gate)] = dict(
                        calls=len(v),
                        empty=mean(r["candidates"] == 0 for r in v),
                        success=mean(r["success"] for r in v),
                    )
    e2 = defaultdict(list)
    tot = defaultdict(list)
    for r in rows:
        if r["exp"] == "E2":
            e2[(r["n"], r["rule"])].append(r)
        elif r["exp"] == "E2_total":
            tot[(r["n"], r["rule"])].append(r)
    for (n, rule), rs in sorted(e2.items()):
        t = tot[(n, rule)]
        shares = defaultdict(int)
        for r in rs:
            shares[segment(r["seed_x"])] += 1
        out["E2"].setdefault(str(n), {})[rule] = dict(
            calls=len(rs),
            success=mean(r["success"] for r in rs),
            total_hv_gain=mean(x["hv_gain"] for x in t),
            hv_gain_per_second=sum(x["hv_gain"] for x in t)
            / max(sum(x["seconds"] for x in t), 1e-9),
            seed_share={k: round(v / len(rs), 3) for k, v in shares.items()},
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(json.loads(args.rows.read_text(encoding="utf-8")))
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for n, ops in summary["E1"].items():
        print(f"\n== E1 {n} 任务：成功率 / 每次HV增量 / 每秒HV增量 / 每次秒 / exact / 空调用")
        for op, v in sorted(ops.items(), key=lambda kv: -kv[1]["hv_gain_per_second"]):
            print(
                f"  {op:12s} {v['success']:.3f}  {v['hv_gain_per_call']:.2e}  {v['hv_gain_per_second']:.2e}  "
                f"{v['seconds']:.3f}  {v['exact']:.1f}  {v['empty']:.2f}"
            )
    for n, rules in summary["E2"].items():
        print(f"\n== E2 {n} 任务：起点规则")
        for rule, v in rules.items():
            print(
                f"  {rule:9s} 成功率{v['success']:.3f} 总HV增量{v['total_hv_gain']:.2e} 每秒{v['hv_gain_per_second']:.2e} 起点分布{v['seed_share']}"
            )
    print("\nA8 门控:", json.dumps(summary["A8_gate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
