# MOEA/D P2 本地实现验收

日期：2026-10-02。结论：计划与本地工程验收完成，等待用户审阅。算法收益仍待正式 P2；服务器未部署。

## 已交付的实现

在独立工作树 `E:\LLM调度\worktrees\moead-p2-local-20261002`、分支 `research/moead-p2-local-20261002` 实现三种默认关闭的研究机制和必要控制组：

| 组 | 实现 | 本地核对重点 |
| --- | --- | --- |
| F6 | 原完整 MOEA/D、固定预算 6 | 对未修改基线的默认行为一致 |
| CG | 20% 普通 genotype 复制，重新 SSGS | 作为减少变异/复制的控制；消耗 exact |
| CT | 20% 保留同一 incumbent 的完整时间表 | 不拼接父代 timing；仍 exact、检查可行性、尝试 Archive |
| RPERM | 出生邻域随机替换顺序 | 独立 RNG、原邻域、own-weight、cap |
| RDIR | 最终子代方向关联的替换邻域 | 逆权重参考方向、冻结归一化、自己的邻居权重、cap |
| A7B | 每工序第一代表使用局部电费代理 | 后续筛选规则保留；最终 complete candidate 仍 exact |

原 `Config` 的默认路径和现有配置不启用三种机制。六份审阅模板位于 [configs/p2_local](../../configs/p2_local/README.md)。其中 smoke 输入是占位，正式实例、共用初始化、种子、时间预算与任务矩阵尚未冻结。

实施计划：[moead_p2_local_plan.md](moead_p2_local_plan.md)。新增验证入口：`scripts/validate_p2_local.py`。本批没有修改其它工作树，也没有执行远程操作。

## 真实验收结果

最终代码验收提交：`fd4f9161b32db881bf377e5b6835ffd9aa060160`。验收运行时工作树 clean。Python 3.13.5 / Windows。

源码模块内容哈希：`a54ce42cfc97c329399e5b9538929c0d51be263c853fec29f3affb2c75a1c72f`。该哈希采用项目 `source_digest()` 的规范 JSON 聚合方法，并非 Git SHA。

| 检查 | 结果 |
| --- | --- |
| ruff check / format | 通过；147 个文件满足格式 |
| mypy | 80 个源文件通过 |
| pytest | **253 passed，131.07 秒**；含新增 42 项机制/集成检查 |
| 总体行覆盖率 | **86.64%**，门槛 80% |
| 核心行覆盖率 | **96.80%**，门槛 90% |
| 六组回放 | 两个 seed，每组运行两次，共 **24 次**；全部语义一致 |
| exact ground truth | 逐候选核对 **10,100 次**在线 exact 返回，包括重复回放 |
| 原版独立进程比较 | plain MOEA/D、full MOEA/D、NSGA-II 三组与冻结基线一致 |
| Markdown 公式 | 新增 Markdown 不含旧式数学分隔符 |

基线为 `3ec2355fbc43993375daab06ec0ea39fb9d86991`。从该提交导出未修改源码，在独立进程、共用输入与初始化、固定代数下比较完整候选顺序、决策哈希、最终 population、Archive、Q/访问/更新/选择计数及终止原因；三个方法均相同。full 比较包含 126 次 exact，plain/NSGA-II 各 18 次。这不保证按秒停止的运行具有相同吞吐，正式对照必须共用本版代码。

机器可读证据：[moead_p2_local_validation.json](moead_p2_local_validation.json)。质量门原始日志：[moead_p2_local_quality.txt](moead_p2_local_quality.txt)。文件校验清单：[moead_p2_local_manifest.json](moead_p2_local_manifest.json)。原始本地输出另保留在 `outputs/p2_local_review_fd4f916`。

## 反例与边界

CG/CT 测试使用不同父代时间表，确认复制对象为当前出生子问题的 incumbent。CG 重解码，CT 保留完整 timing；两组共用独立 routing stream，关闭时不改变原变异随机流。候选仍经过统一 gateway，评价 cap 不超额。

RDIR 覆盖端点、零目标范围、零向量稳定 tie；RPERM 验证集合不变与独立随机流；反例验证真正被替换的是在邻居自身权重下改善的成员，且不突破 cap。

A7 包含两个真实调度反例。分时电价反例中，原最大压缩代表减少 200 秒 active union 却提高电费；新代理第一代表减少 100 秒并降低电费，说明新筛选路径确实会改变候选。新峰反例中，代理预测降低 100 元，但完整评价反而增加约 77.78 元，说明不能依赖代理接受候选。另有单位换算、原并列峰取均值、重叠不重复计费和 raw pool 不调用完整 evaluator 的检查。

## 验收的含义与下一步

本地回放仅使用 6-job 合成实例、population 12、3 代、两步 always-trigger 轨迹，目的是覆盖工程路径。它不是等时间质量比较，也不是统计实验。短回放中的 A7 调用很少，A7B 与 F6 的结果相同不能解释为改进有效或无效；实际筛选变化另由分时电价反例验真。

20% 复制概率尚未调优；A7 代理忽略新峰接管及 horizon/idle-tail 的全局变化。当前工程验收不能替代 50/100-job 正式在线检验、服务器资源预跑或 Python 3.11 CI。

用户审阅确认后才进入下一批：先确定 GitHub/服务器交付版本与冻结输入，测量内存和单任务资源，再优先运行 F6/CG/CT；随后独立筛选 RPERM/RDIR 和 A7B。组合实验由单项结果决定，不自动展开完整长 campaign。

本批交付是本地提交与 Notion 记录。GitHub push/PR/CI、服务器部署与正式 P2 均未执行；不宣称三端同步或新的算法正式采用。

## Material Passport

- Type: LOCAL_ENGINEERING_ACCEPTANCE
- Verification: VERIFIED for the listed local checks
- Source: code commit fd4f916 and content hash above
- Scientific gains: UNVERIFIED
- Formal P2: NOT RUN
- Server: NOT DEPLOYED; awaits user review confirmation
