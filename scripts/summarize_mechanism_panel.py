"""Verify all frozen inputs and produce the Chinese handoff with exact witness statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import median
from zipfile import ZIP_DEFLATED, ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geo_llm_scheduler.domain.models import Genotype, ProblemInstance, Schedule  # noqa: E402
from geo_llm_scheduler.evaluation.exact import evaluate  # noqa: E402
from geo_llm_scheduler.io.loaders import load_instance  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rate(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}（{100 * numerator / denominator:.1f}%）"


def _busy_seconds(solution: dict, jobs: list[dict]) -> float:
    calendars: dict[int, list[tuple[float, float]]] = {}
    for op, (machine, start) in enumerate(
        zip(solution["genotype"]["ms"], solution["schedule"]["starts"])
    ):
        duration = jobs[op // 2]["decode" if op % 2 else "prefill"]["duration"]
        calendars.setdefault(machine, []).append((start, start + duration))
    total = 0.0
    for intervals in calendars.values():
        right = 0.0
        for start, end in sorted(intervals):
            total += max(0.0, end - max(right, start))
            right = max(right, end)
    return total


def _replay_witness(problem: ProblemInstance, witness: dict) -> None:
    before, after = witness["before"], witness["after"]
    assert before["genotype"] == after["genotype"]
    genotype = Genotype(tuple(before["genotype"]["ms"]), tuple(before["genotype"]["os"]))
    for candidate in (before, after):
        schedule = Schedule(tuple(candidate["schedule"]["starts"]))
        checked = evaluate(problem, genotype, schedule)
        assert (
            checked.feasible and json.loads(json.dumps(asdict(checked))) == candidate["evaluation"]
        )
    a, b = (c["schedule"]["starts"] for c in (before, after))
    assert a[1::2] == b[1::2] and all(y >= x for x, y in zip(a, b))


def _bill_mechanisms(base: Path, entries: list[dict], records: dict) -> dict:
    counts = dict(
        witnesses=0,
        tou_down=0,
        demand_down=0,
        demand_up=0,
        busy_seconds_down=0,
        tou_down_without_less_busy_time=0,
        common_best=0,
    )
    for entry in entries:
        jobs = _load(base / entry["path"])["jobs"]
        problem = load_instance(base / entry["path"])
        record = records[entry["key"]]
        shift = {r["parent_index"]: r for r in record["rows"] if r["operator"] == "RIGHTSHIFT"}
        for w in record["rightshift_witnesses"]:
            _replay_witness(problem, w)
            a, b = w["before"], w["after"]
            x, y = a["evaluation"], b["evaluation"]
            dtou = y["tou"] - x["tou"]
            ddem = sum(y["demand"]) - sum(x["demand"])
            dbusy = _busy_seconds(b, jobs) - _busy_seconds(a, jobs)
            assert dtou + ddem < -1e-9 and y["flow"] == x["flow"]
            counts["witnesses"] += 1
            counts["tou_down"] += dtou < -1e-8
            counts["demand_down"] += ddem < -1e-8
            counts["demand_up"] += ddem > 1e-8
            counts["busy_seconds_down"] += dbusy < -1e-8
            counts["tou_down_without_less_busy_time"] += dtou < -1e-8 and dbusy >= -1e-8
            row = shift[w["parent_index"]]["diagnostics"]
            # Source of the winning schedule is only identified if the best delta
            # has a unique source; equality ties across sources stay unclassified.
            sources = {
                s
                for delta, s in zip(row["incremental_deltas"], row["proposal_sources"])
                if abs(delta - dtou - ddem) < 1e-8
            }
            counts["common_best"] += sources == {"common"}
    return counts


def _stats(records: list[dict]) -> dict:
    rows = [r for d in records for r in d["rows"] if r["operator"] == "RIGHTSHIFT"]
    feasible_proof = [
        r
        for r in rows
        if r["proposals"]
        or r["diagnostics"]["candidate_positions"] > r["diagnostics"]["resource_rejected"]
    ]
    return {
        "instances": len(records),
        "parents": len(rows),
        "slack": sum(r["positive_slack_jobs"] > 0 for r in rows),
        "feasible_lower_bound": len(feasible_proof),
        "hit": sum(r["proposals"] > 0 for r in rows),
        "hit_instances": sum(bool(d["rightshift_witnesses"]) for d in records),
        "group": sum("common" in r["diagnostics"]["proposal_sources"] for r in rows),
        "groups_available": sum(r["diagnostics"]["groups"] > 0 for r in rows),
        "group_instances": sum(
            any(
                "common" in r["diagnostics"]["proposal_sources"]
                for r in d["rows"]
                if r["operator"] == "RIGHTSHIFT"
            )
            for d in records
        ),
        "front_gain_instances": sum(d["all_probe_hv_gain"] > 1e-12 for d in records),
        "median_call_gain_pct": median(100 * r["best_bill_reduction_fraction"] for r in rows),
        "median_instance_best_gain_pct": median(
            max(
                100 * r["best_bill_reduction_fraction"]
                for r in d["rows"]
                if r["operator"] == "RIGHTSHIFT"
            )
            for d in records
        ),
        "median_union_hv_gain": median(d["all_probe_hv_gain"] for d in records),
    }


def main() -> None:
    """Produce the input-only distribution and report after all declared audits finish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    base = parser.parse_args().output
    manifest = _load(base / "manifest.json")
    records = {}
    for e in manifest["instances"]:
        assert hashlib.sha256((base / e["path"]).read_bytes()).hexdigest() == e["sha256"]
        d = _load(base / "audits" / (e["key"] + ".json"))
        assert d["status"] == "complete" and d["input_sha256"] == e["sha256"]
        assert len(d["rows"]) == 10 * d["sampled_front_count"] and d["initial_exact_checks"] == 140
        records[e["key"]] = d
    assert len(records) == len(set(e["sha256"] for e in manifest["instances"])) == 120
    baselines = [
        _load(base / "baseline_probes" / (f"mx_n100_f{i:02d}_r{1 if i % 2 else 3:02d}.json"))
        for i in range(1, 11)
    ]
    assert all(
        b["offspring"] == 500 and b["input_sha256"] == records[b["key"]]["input_sha256"]
        for b in baselines
    )
    for b in baselines:
        problem = load_instance(base / "instances" / (b["key"] + ".json"))
        for w in b["rightshift_witnesses"]:
            _replay_witness(problem, w)
    total = _stats(list(records.values()))
    sizes = {
        n: _stats([records[e["key"]] for e in manifest["instances"] if e["jobs"] == n])
        for n in (50, 100, 200)
    }
    families = {
        f["key"]: _stats(
            [records[e["key"]] for e in manifest["instances"] if e["family"] == f["key"]]
        )
        for f in manifest["families"]
    }
    ops = {}
    for name in {r["operator"] for d in records.values() for r in d["rows"]}:
        per_case = [[r for r in d["rows"] if r["operator"] == name] for d in records.values()]
        rows = [r for group in per_case for r in group]
        ops[name] = {
            "calls": len(rows),
            "nonempty": sum(bool(r["proposals"]) for r in rows),
            "gain_calls": sum(r["initial_hv_gain"] > 1e-12 for r in rows),
            "gain_instances": sum(
                any(r["initial_hv_gain"] > 1e-12 for r in group) for group in per_case
            ),
        }
    baseline = {
        "cases": len(baselines),
        "offspring_per_case": 500,
        "baseline_gain": sum(b["baseline_hv_gain_from_initial"] > 1e-12 for b in baselines),
        "probe_gain": sum(b["probe_hv_gain_from_baseline"] > 1e-12 for b in baselines),
        "shift_instances": sum(bool(b["rightshift_witnesses"]) for b in baselines),
        "shift_hit": sum(len(b["rightshift_witnesses"]) for b in baselines),
        "shift_calls": sum(
            sum(r["operator"] == "RIGHTSHIFT" for r in b["rows"]) for b in baselines
        ),
    }
    bills = _bill_mechanisms(base, manifest["instances"], records)
    historical = {}
    if (base / "historical_probes").is_dir():
        for n in (100, 200):
            files = sorted((base / "historical_probes").glob(f"st_n{n}_*.json"))
            assert len(files) == 4
            historical[n] = _stats([_load(path) for path in files])
    summary = {
        "historical_rightshift_only": historical,
        "verified_bill_mechanisms": bills,
        "total": total,
        "sizes": sizes,
        "families": families,
        "operators": ops,
        "baseline": baseline,
    }
    (base / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (base / "per_instance.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        rows = [{"key": k, **_stats([d])} for k, d in records.items()]
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "# 面向算子和右移的 120 个复杂实例",
        "",
        f"本地已生成 50、100、200 任务各 40 个输入。全部保留。每个实例重放 BILL4 初始化，合计独立复查 {140 * len(records):,} 次 exact 评价。输入种子独立，求解器种子统一为 16901。",
        "",
        "## 初始化后能否右移",
        "",
        "每个实例在 BILL4 初始化的外部前沿按 Flow 等分位取至多五个父解，每个解调用一次当前 FAST COMMON。一个调用可返回至多三个候选。下表的降费均为 Flow 固定、完整电费下降、exact 可行。空调用按零计入中位数。",
        "",
        "|规模|有降费的实例|有正余量的父解|证实有可行右移位置的父解（下限）|实际降费的父解|调用最佳降费中位数|至少一个共同平移候选的父解|",
        "|---|---|---|---|---|---|---|",
    ]
    for label, s in [*sizes.items(), ("合计", total)]:
        text.append(
            f"|{label}|{s['hit_instances']}/{s['instances']}|{s['slack']}/{s['parents']}|≥{s['feasible_lower_bound']}/{s['parents']}|{_rate(s['hit'], s['parents'])}|{s['median_call_gain_pct']:.3f}%|{s['group']}/{s['parents']}|"
        )
    text += [
        "",
        "可行位置采用可复核的下限。凡返回严格降费的候选，或共同平移枚举中有位移通过容量检查，就计入。有些空调用中可能有可行但不省钱的单任务位移，原日志不足以逐一否定。共同平移只计入最终保留且来源标为 common 的候选。单任务分支也可以连续移动多个 Prefill，不能简单用移动任务数区分两种机制。",
        "",
        "## 十类结构",
        "",
        "|类型|重点|有降费的实例|抽查父解中降费|至少一个共同平移候选的实例|每实例最佳降费中位数|",
        "|---|---|---|---|---|---|",
    ]
    for f in manifest["families"]:
        s = families[f["key"]]
        text.append(
            f"|{f['key']}|{f['title']}|{s['hit_instances']}/12|{_rate(s['hit'], s['parents'])}|{s['group_instances']}/12|{s['median_instance_best_gain_pct']:.3f}%|"
        )
    text += [
        "",
        "## 电费从哪里省下来",
        "",
        f"每个成功父解只取总费用最低的一个候选，共 {bills['witnesses']} 个。逐项复算，{bills['tou_down']} 个减少了分时电费，{bills['demand_down']} 个降低了需量费，{bills['demand_up']} 个虽抬高需量费、分时节省仍足以抵消。",
        f"将每台机器所有操作区间取并集再求和，有 {bills['busy_seconds_down']} 个候选缩短了总忙碌时间。另有 {bills['tou_down_without_less_busy_time']} 个候选在总忙碌时间持平或增加时仍降低分时电费，体现了移向便宜时段的贡献。两类分支中，{bills['common_best']} 个最佳候选的最低费用仅由共同平移分支返回。",
    ]
    if historical:
        text += [
            "",
            "## 同口径重放上一轮八个实例",
            "",
            "上一轮 100 和 200 任务各有四个固定实例。两批都用种子 16901、原始 BILL4、最多五个按 Flow 等分位选择的前沿父解、B=3 的 FAST COMMON。各实例的初始前沿点数可能不同。下表是两批样本的描述性比较。",
            "",
            "|任务数|旧批有降费的父解|新批有降费的父解|旧批/新批调用降费中位数（空计零）|旧批/新批每实例最佳降费中位数|",
            "|---|---|---|---|---|",
        ]
        for n, old in historical.items():
            new = sizes[n]
            text.append(
                f"|{n}|{_rate(old['hit'], old['parents'])}|{_rate(new['hit'], new['parents'])}|{old['median_call_gain_pct']:.3f}% / {new['median_call_gain_pct']:.3f}%|{old['median_instance_best_gain_pct']:.3f}% / {new['median_instance_best_gain_pct']:.3f}%|"
            )
    text += [
        "",
        "## 算子还能否推进初始化前沿",
        "",
        f"全部探针合并后，有 {total['front_gain_instances']}/120 个实例相对初始前沿取得严格正 HV 增量。每个算子从同一组父解独立提议，彼此没有先后增益归属。",
        "",
        "|算子|非空调用|扩大初始 HV 的调用|扩大初始 HV 的实例数|",
        "|---|---|---|---|",
    ]
    for name in sorted(ops):
        o = ops[name]
        text.append(
            f"|{name}|{o['nonempty']}/{o['calls']}|{o['gain_calls']}/{o['calls']}|{o['gain_instances']}/120|"
        )
    text += [
        "",
        "## 五代纯 MOEA/D 后的机会",
        "",
        f"每类预定一个 100 任务输入，复用同一 BILL4 初始种群指纹，再运行 5 代（500 个子代）。有 {baseline['baseline_gain']}/10 个基线前沿相对初始化扩大，之后十种独立探针又在 {baseline['probe_gain']}/10 个实例上扩大前沿。单独右移在 {baseline['shift_instances']}/10 个实例、{_rate(baseline['shift_hit'], baseline['shift_calls'])} 个抽查父解上降低电费。",
        "",
        "|实例|基线相对初始 HV 增量|探针相对五代基线 HV 增量|有降费的右移父解|",
        "|---|---|---|---|",
    ]
    for b in baselines:
        text.append(
            f"|{b['key']}|{b['baseline_hv_gain_from_initial']:.6f}|{b['probe_hv_gain_from_baseline']:.6f}|{len(b['rightshift_witnesses'])}/{sum(r['operator'] == 'RIGHTSHIFT' for r in b['rows'])}|"
        )
    text += [
        "",
        "各实例采用固定的自身初始化评价范围计算 HV，所以增量只用于检验同一实例的前沿是否继续推进。正式预算下的净收益需要有/无右移的等墙钟、多种子在线对照。",
        "",
        "## 可复核的右移例子",
        "",
        "从 100 任务 F09 第一重复的已验证候选中取降费最大的一个，举例展示账单组成。",
    ]
    example = records["mx_n100_f09_r01"]

    def bill(e: dict) -> float:
        return e["tou"] + sum(e["demand"])

    w = max(
        example["rightshift_witnesses"],
        key=lambda w: bill(w["before"]["evaluation"]) - bill(w["after"]["evaluation"]),
    )
    a, b = w["before"], w["after"]
    x, y = a["evaluation"], b["evaluation"]
    moved = [
        i // 2
        for i, (t, u) in enumerate(zip(a["schedule"]["starts"], b["schedule"]["starts"]))
        if u > t
    ]
    text += [
        "",
        f"父解编号 {w['parent_index']}，移动 {len(moved)} 个 Prefill。总 Flow 保持 {x['flow']:,.0f} 秒。TOU 从 {x['tou']:.6f} 到 {y['tou']:.6f} 元，需量费用从 {sum(x['demand']):.6f} 到 {sum(y['demand']):.6f} 元，总电费从 {bill(x):.6f} 到 {bill(y):.6f} 元，下降 {(1 - bill(y) / bill(x)) * 100:.3f}%。具体时刻见 `audits/mx_n100_f09_r01.json` 中的 `rightshift_witnesses`。",
        "",
        "这批输入用于检验机制能否解决其针对的拥塞、迁移和计费结构。单个初始化种子和前沿分位抽样决定了这里的分母，在线选择器面对的状态分布需要另测。",
        "",
        "`instances/` 是求解器可直接读取的输入。`manifest.json` 给出参数与哈希，`per_instance.csv` 给出逐实例统计，`audits/` 与 `baseline_probes/` 保存完整诊断。",
    ]
    (base / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    bundle = base / "mechanism_stress120_inputs.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
        for rel in [
            "manifest.json",
            "REPORT.md",
            "SUMMARY.json",
            "per_instance.csv",
            *(e["path"] for e in manifest["instances"]),
        ]:
            z.write(base / rel, rel)
    with ZipFile(bundle) as z:
        assert z.testzip() is None
        assert all(
            hashlib.sha256(z.read(e["path"])).hexdigest() == e["sha256"]
            for e in manifest["instances"]
        )
    (base / "INPUT_ZIP_SHA256.txt").write_text(
        hashlib.sha256(bundle.read_bytes()).hexdigest() + "  " + bundle.name + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
