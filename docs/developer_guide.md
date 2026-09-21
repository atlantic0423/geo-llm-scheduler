# 开发指南

先读domain/models.py，再读scheduling/resources.py和evaluation/exact.py，沿engine/run.py→trajectory.py→macrosearch/search.py理解完整调用。候选经EvaluationGateway统一精确评价与Archive分发；算子不直接修改population或Q-table。

新增structural算子实现Operator.propose，返回只有genotype的Proposal；timing算子返回原genotype、新schedule、selected。ProposalBatch.attempts 统一表示实际检查的 raw construction moves，proposals 表示送往 MacroSearch 的完整候选；diagnostics 记录候选空间、qualifying pool 和选择数。失败不冒充exact，一次invocation不链式构造。大位置空间必须 lazy/bounded，禁止先完整物化再截断。

模型变化先写specifications和Notion框架/决策，再独立手算golden与反例。优化保留reference对照。随机函数接收RNG，Markdown公式使用美元分隔符。

```powershell
.venv/Scripts/python scripts/validate_all.py
.venv/Scripts/python -m pytest --cov=geo_llm_scheduler --cov-report=term-missing --cov-report=json:outputs/coverage.json
.venv/Scripts/python scripts/reproduce.py
.venv/Scripts/python scripts/profile_run.py
```

golden覆盖手算/反例；unit覆盖数据/config/RL；integration覆盖生命周期/回放，全部旧测试组成回归集。ruff格式和lint、mypy src为质量门。语法兼容3.11，CI使用同一入口。

每批次按AGENTS同步Notion09：目的、文件、规格、命令/结果、限制、下一步、完整源码ZIP/校验和并回读；未验收不得标完成。
