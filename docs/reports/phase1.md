# Phase 1 数据模型验收

2026-09-21：ruff passed；mypy 21 source files passed；pytest 7 passed。新增 models、validation、loaders、numeric、rng，验证不可变对象、MS 唯一权威、非法 assignment/OS、负值/NaN/Inf 拒绝、JSON roundtrip、随机子流隔离。

所有时间进入核心前归一化为秒；当前货币要求为人民币，输入必须显式预转换为 CNY，不联网使用隐式汇率。输入 tariff 有限且连续；评价超出输入 tariff 覆盖将明确报错，不静默延拓。
