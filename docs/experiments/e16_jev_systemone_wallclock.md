# E16 Jev System-One 等墙钟实验：实现与前置检查

研究规格见 [Notion E16](https://app.notion.com/p/3eb8878b801981eaacd7e399848f76e0) 与 [实现任务](https://app.notion.com/p/3eb8878b801981c9bed4fc71ee578a96)。本页是工作版实现记录，**正式实验未启动**，不包含效果结论。

## 固定设计

15 个新的独立 100-job 基础实例，各 8 个处理组，每条 1200 秒等墙钟，共 120 条。Jev 网络调用、有限重试与等待计入墙钟。实例是独立统计单位；所有处理组共享同一实例、algorithm seed 和 100 个初始 genotype。

| Arm | 控制 | MacroSearch | 局部搜索 |
| --- | --- | --- | --- |
| C0 Current-Fixed | 当前 Q-learning | 固定 6 | 5 步 |
| C1 Current-Full | 当前 main 的完整配置 | main 当前权威预算 | 5 步 |
| J1 Jev-Choice | Jev operator Choice | 固定 6 | 5 步 |
| J2 Jev-Control | operator + budget Choice，同一请求 | 3/6/10 | 5 步 |
| J3 Jev-ClosedLoop | 同 J2；每步后重新观察并调用 Noul | 3/6/10 | 最多 5 步，可提前停止 |
| M0 plain MOEA/D | plain | 无 | 无 |
| N0 plain NSGA-II | rank + crowding | 无 | 无 |
| P0 HPSO-adapted | particle/velocity/pbest/archive leader | 自有机制 | 独立基础扰动 |

2026-09-30 核对 main `3c59ada`：E15 的 adaptive 策略仍为实验选项，默认 `budget_policy=fixed`、`fixed_budget=6`。没有证据表明 E15 已将某个 winner 升级为全局默认，不能按 E16 结果挑选 C1。若启动前 main 默认仍为固定 6，必须如实记录 C0/C1 配置相同。main 尚无 HPSO、Jev trajectory 或 E16 campaign。

## 已实现的基础模块

- `controllers/jev.py`：typed Choice/Noul 请求与响应；固定官方 HTTPS endpoint；显式 RNG；每 decision 最多三次请求；网络错误、429、5xx 有界退避；可注入 transport/clock/sleep；deadline 包含重试和 API 等待；记录实际 model、usage、latency、retry 与 HTTP 状态。
- 必须提供版本化 model ID，禁止 latest/preview；响应 model 与请求不一致时失败。服务端错误正文、transport 异常与未知响应字段不写入错误/产物；发现 key 出现在请求或响应中时失败。API 失败没有 Q-learning/default-action fallback。
- `controllers/jev_questions.py`：operator、combined operator/budget 和 post-step continuation 三种版本化工作模板，以及模板/预算映射 SHA256。措辞尚未经过 live pilot，不视为 formal freeze。
- `scripts/e16_jev_smoke.py`：只做真实 API 前置检查，无 formal run。可选输出 secret-free 报告；缺少凭据以退出码 2 明确阻塞。

官方契约于 2026-09-30 核对：[API reference](https://docs.typesafe.ai/api)、[Models](https://docs.typesafe.ai/models)、[Noul](https://docs.typesafe.ai/primitives/noul)。Choice 返回 choice/probabilities/confidence；Noul 返回 raw noul 数值，没有额外 confidence。保留 raw 数值，不声称其已在调度问题上校准。客户端仅保存官方返回的 token usage，不编造 cost。

## 恢复前置条件

先恢复安全 SSH 公钥登录，在服务器的安全环境提供 `TYPESAFE_API_KEY`；不要把 key 放入 Git、YAML、Notion、shell history 或日志。下面命令只读取已有环境变量：

```bash
export PYTHONPATH=src
python scripts/e16_jev_smoke.py --model jev-1.13.0 --output outputs/e16_api_smoke.json
```

这不代表 J1/J2/J3 solver integration 或 100-job pilot 通过。服务器无外网时，先验证 API 网络可达性；不能换用未批准的第三方转发站，也不能用 mock 顶替 live API。

## 尚未完成的正式验收

1. 将独立 Jev 控制层接入现行 trajectory，保持 exact/SSGS、同源 MacroSearch、local acceptance、Polish、Archive 与 replacement 语义；补 J3 early stop/max 5/deadline/integration 测试。
2. 文献核对并实现可审计 HPSO-adapted、latent-key 映射、pbest/archive leader 与统一 exact 评价，补 baseline 测试。
3. 扫描历史正式 seeds，冻结 15 个新实例、paired algorithm seeds、shared initial pools 与全部 hashes。
4. 实现完整 E16 campaign、watchdog/supervisor、heartbeat/resume/quarantine/retry、资源门、Jev 并发门、完整日志和含 phenotype 的 reservoir snapshots。
5. 至少两个独立非正式 100-job 实例，8 arms 全覆盖，每条 60–120 秒，Jev live API；根据实际 RSS、CPU、吞吐、latency/429 冻结 workers 和并发，并据此估算 ETA。
6. 全仓质量门与 replay，通过 PR/CI/merge 后从 clean main 部署正式 120 条；同步 Notion 09/E16/task，并验证脱离会话运行。
7. 完成后回收 SHA256 manifest、发布 GitHub Release 完整结果资产并回读校验；现在没有结果可发布。

总计算预算为 40 worker-hours；尚无 pilot，不能给实测 ETA。也不能因前置条件阻塞而缩小矩阵、缩短时间或删除处理组。

## 本批次工程验收（2026-09-30）

34 项新增 Jev mock 测试通过；`python scripts/validate_all.py` 的 lint、format、mypy、229 项全仓测试与覆盖率门禁通过，整体覆盖率 85.62%、核心 96.08%，未修改门槛。`python scripts/reproduce.py` 确定性回放通过，标准 build-isolation wheel 构建通过。文档旧式数学公式分隔符为 0。真实 API 前置检查因未配置 `TYPESAFE_API_KEY` 返回 blocked/退出码 2，不能称 live smoke 通过；没有 100-job pilot、formal manifest 或正式运行。
