# S4 吞吐加速（第 1–3 级）｜语义保持优化与验收

## 概要

- **任务**：用户 2026-11-09 选定"吞吐加速"，确认先做第 1–3 级。设计草案在本地 `outputs/s4_throughput_20261109/DESIGN_DRAFT.md`。
- **参考版本**：`582237f49a7487ba5d5a2d23086a68f2f3df7046`。被改动的四个文件原样冻结在 `frozen/`，测试逐个核对 Git blob 哈希。
- **分支**：`research/s4-throughput-20261109`。
- **性质**：纯工程改动。同一基因型得到的 SSGS 排程、exact 结果（全部字段与违例列表）、LS 计费数值都与参考实现逐比特相同。没有改动任何研究语义、参数、候选数量或随机流。
- **范围**：本地实现与验收。没有部署服务器，也没有启动实验。

## 改了什么

| 级 | 位置 | 原来的重复工作 | 现在 | 为什么结果不变 |
|---|---|---|---|---|
| 1 | `_fits_intervals` | 窗口内每个小区间都扫一遍机器上的全部工序 | 先只保留与 [start, end) 重叠的工序 | 窗口内处于运行状态、或在窗口内产生断点的工序，必与窗口重叠；过滤保持原列表顺序，求和顺序不变 |
| 1 | `ResourceCalendar` | 每次 `earliest` 对每个候选时刻重扫全部工序 | 按机器增量维护基本区间边界和各段的 Compute/VRAM 和；`earliest`/`fits` 用二分定位 | 每段的和都是对该段运行集合按 `items` 顺序新算的 `sum`，与参考实现表达式相同；带 `temporary` 或工序已提交时走原路径 |
| 1 | SSGS `decode` | 每放置一个工序都复制整条起点元组，只为了算 EST | 直接按 `est_for` 的同一表达式计算 | 表达式和加法顺序相同 |
| 1 | exact `_checked_resources` | 每个工序重建一次"其余工序"列表，再逐窗口扫描 | `ResourceView.self_fits` 每台机器扫一遍，在每段对每个运行中的工序按原顺序求"其余工序之和" | 断点和运行集合与逐工序检查相同；起点为负的工序继续走原路径 |
| 1 | P2 定向破坏重建 `_slots` | 同上，全量扫描 | 通过 `calendar_fits` 使用日历索引 | 同上 |
| 2 | exact `_power_segments` | 每个地区断点对每台机器做三次二分查找 | 每台机器只在自身边界上用参考表达式算一次功率，再对地区断点做前向扫描；地区功率仍按原机器顺序相加 | 机器的忙碌状态和负载只在它自己的边界处变化 |
| 2 | exact `_load_steps` | 每个时刻重新计算 `load_kw` | 每次评价对每个工序只算一次（纯函数） | 加数相同，顺序相同 |
| 3 | LS `PowerProfile` | 窗口重叠逐个用 min/max 计算；电价积分和 `load_kw` 重复计算 | 完整内部窗口的增量每次调用只算一次（重叠量恰为 900.0）；电价积分和 `load_kw` 按纯函数做缓存，副本之间共享 | 窗口累加顺序不变，每个加数的浮点值相同 |

## 等价性证据

- **单元与性质测试**：`tests/test_throughput_equivalence.py`，43 项，显式 RNG。
  - 冻结文件逐个核对 Git blob 哈希；参考切换能替换并恢复全部绑定。
  - `_fits_intervals`：打乱顺序的随机区间列表。
  - 日历：随机提交序列，穿插已提交、未提交和带 `temporary` 的查询。
  - 解码：在"紧"输入上测试。
  - exact：随机或违例时序（含未排工序），新旧两种功率模型。
  - `PowerProfile`：`delta_for`、`window_values`、`apply`、`with_horizon`、`copy`。
  - 面板实例上的解码与评价。
  - **工作版整程**：确定性假时钟下运行 N7_P1_SHARE（50 任务面板实例，3 代，含 P2 与 LS）。档案、计数、LS 步记录、MOEA/D 计数和初始化证据都与参考路径完全相同；只去掉了测量出来的耗时字段。
- **紧输入的构造**：时长取粗网格，使端点频繁重合；Compute 恰在 $1\pm2\times10^{-10}$ 附近，VRAM 恰在容量附近。
- **golden 回放**（`data/golden_replay.json`）：
  - 语料：`s4ge3fix_c253_20261108` 与 `s4invest_c253_20261109` 两批，每个分片每种规模各取 3 块。
  - 每个档案基因型都做三件事：重新解码并评价、评价档案中的原时序、构造 LS 计费（含 `with_horizon`）。同时为每个基因型生成一个随机结构邻居（一次 OS 交换，加一次同地区 MS 改动），一起回放。
  - 共 16,226 例，与参考实现全部相同：

    | 规模 | 用例数 |
    |---|---|
    | 50 | 6,636 |
    | 100 | 5,634 |
    | 200 | 3,956 |

- **本地完整质量门**（`scripts/validate_all.py`）：
  - 格式、lint、mypy 全部通过；1874 passed、11 skipped；
  - 覆盖率 91.01%，核心 96.43%（门槛分别为 80%、90%）；改动前为 91.05% 和 97.24%。

## 测时（本机，无 profiler）

### 分组件，固定工作量

数据：`data/benchmark_components.json`。语料为档案时序，每种规模 2 块。每项新旧路径交替各跑 3 次，取中位数，表中为参考耗时除以新耗时。

| 规模 | 用例数 | SSGS 解码 | exact 评价 | LS 计费构造 + `with_horizon` |
|---|---|---|---|---|
| 50 | 94 | 1.73× | 1.51× | 1.11× |
| 100 | 183 | 2.11× | 1.65× | 1.10× |
| 200 | 167 | 2.19× | 1.66× | 1.13× |

### 整程，固定墙钟

数据：`data/benchmark_runs.json`。

- **设置**：工作版 N7_P1_SHARE，每次 120 秒（含初始化），seed 13/14，每块 2 轮。每轮新旧两条路径在同一时间窗口内并发运行，各自是独立进程。
- **为什么并发**：这台本机是日常桌面机，同一份代码的初始化墙钟在 43–115 秒之间漂移。最初顺序交替运行时，一块 200 任务的结果因此反常（0.63×）。查明原因后改为成对并发，让两条路径经历同样的机器状态。

相同墙钟内完成的 exact 次数之比（新/参考）：

| 规模 | 各轮比值 | 中位数 |
|---|---|---|
| 50 | 1.41, 1.40, 1.42, 1.42 | 1.42× |
| 100 | 1.71, 1.73, 1.84, 1.77 | 1.75× |
| 200 | 2.51, 2.06, 2.04, 1.92 | 2.05× |

- **结果解读**：
  - 120 秒预算下，200 任务的初始化占了很大比例，正式预算（2400 秒）下的比值会更接近搜索阶段的组件加速。
  - 初始化本身也变快了：同一实例的进程 CPU 时间，新路径约 80 秒，参考路径约 100 秒。
  - 绝对次数随机器状态变化，比值是稳定的。
- **这只说明吞吐提高了**：更多评价能否转化为更高的最终 HV，要靠后续实验判断（见设计草案第 6 节）。

## 复现

```bash
python -m pytest tests/test_throughput_equivalence.py -q
```

```bash
python -m research.throughput_20261109.golden_replay RESULTS_ROOT_1 RESULTS_ROOT_2 --per-size 3 --out research/throughput_20261109/data/golden_replay.json
```

```bash
python -m research.throughput_20261109.benchmark_components RESULTS_ROOT --per-size 2 --repeats 3 --out research/throughput_20261109/data/benchmark_components.json
```

```bash
python -m research.throughput_20261109.benchmark_runs plan CAMPAIGN_ROOT --keys mx_n050_f01_r02_s9 mx_n050_f05_r01_s9 mx_n100_f01_r03_s9 mx_n100_f05_r02_s9 mx_n200_f01_r01_s9 mx_n200_f05_r03_s9 --seconds 120 --seed 13 --repeats 2 --out research/throughput_20261109/data/benchmark_runs.json
```

`RESULTS_ROOT` 指某批的 `results/` 目录（内含 `formal_panel_shard*`）；`CAMPAIGN_ROOT` 指其中一个 `formal_panel_shard*`。运行时需要把 `src` 和仓库根目录加入 `PYTHONPATH`。

## 未做与待定

- 第 4 级（解码前缀复用、exact 复核分层）没有做，等待用户决定。
- 没有在 253 上测时，也没有做 C0F 对 C0 的 HV 实验。
- 核心覆盖率下降 0.81 个百分点。可能来自新增的回退分支（`temporary` 查询、已提交工序的检查），没有逐行核对；这些分支的行为都有等价测试覆盖。
