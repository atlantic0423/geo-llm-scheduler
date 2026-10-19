# D3 / D4-G / D4-W 交付入口与验收边界

## 冻结源码与完整原始结果

| 项目 | source commit | 完整原始文件 | 原始字节 | tar.gz SHA256 |
|---|---|---:|---:|---|
| D3 | 2725e1d8c34ca5f003fbfc3ebb058a1cbdc96e79 | 8612 | 7796038850 | 9d741561925c5cf49e369e7efb8b8e1bcad9312a82dd52c80528862b940aa985 |
| D4-G | 46bcb3385b39969b82b46b2b46fe13136d1badad | 14855 | 4516040064 | da36a8560349ee4aac14f1d0708a0429ce47cd8175de9d096f980f0dfb575b2b |
| D4-W | 46bcb3385b39969b82b46b2b46fe13136d1badad | 28679 | 7077204699 | 29e581fde714ee1a26645de99cbfdadd910894f7089f56150b498009d9ba1c8b |

D3 canonical a03e0f58bc247a629eef1d41ee144dad4bdc3d741672cf21c0e6cd2ffc3845c8；D4原型 canonical6dbbb9cdd2702b60cd36205eae3d461dd097149eca81bddb336e6134a7d05469。上表原始数据在本地逐项大小/SHA256及tar成员验真，原冻结validator在服务器和本地通过。原exact/RNG审计证据由完整产物hash支持；重新核对标签与完整exact重新执行的区别见详细报告。

完整可复现 raw、manifest、校验和、原源码bundle和复现说明入口：

- [D3 Complete Release](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/d3-complete-20261007-2725e1d)
- [D4-G/W Complete Release](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/d4-prototypes-complete-20261007-46bcb33)

Release 的逐资产实际大小/digest回读、抽样下载校验、报告分支确切commit/PR/双PythonCI由该批最新 `FINAL_DELIVERY_RECEIPT.json` 资产与 Notion09记录确定。源码bundle分为原运行bundle和独立报告bundle；报告提交不改变运行源码。

## 本地分析与工程验收

原joint D3分析与独立CSV重算：6912留出决策、16base、三项均值/CI/全部65536符号翻转/Holm一致。D4独立CSV base聚合/区间、完整caller标量标签、门控/打分控制输出、recipe成员/attempts和审计分母复核均通过。

报告程序接口具有类型标注和docstring；新增5项测试覆盖空调用分母、base bootstrap、零尺度边界、反事实门控计数、产物篡改/错误标签拒绝，以及候选存在但标量收益下降的反例。

最终本地 `scripts/validate_all.py`：365 passed；总体覆盖84.58%（门槛80%）、core96.70%（门槛90%）；lint/format/mypy通过。默认deterministic replay `aeb9824d2e671fe76771297ece91af40576124c930721193c18127cbdb814fa7`；D03历史发布验真57/57 runs、41 PNG/PDF pairs、134 hashed files通过。科学总览PNG/PDF已渲染检查。

完整质量门中一次旧P1测试（0.15秒预算）缺少规定checkpoint：保留失败记录，未改运行算法或测试预算；单项复核通过，最终完整365项通过。首次报告读错序列化objects属性已按真实Flow/TOU/Demand修正并更新反例，完整标签复核通过。D4打包父目录缺失运维错误已修正；这都不属于正式算法任务崩溃重试。

## 六项闭环的范围

本地报告分支与对应GitHub提交、原运行源码/服务器clean状态、三项完整raw、Release资产回读、PR双PythonCI及Notion04/05/06/09与总览/框架的最新状态，由最终receipt记实。不清理服务器原件，不改变旧E15/E16或默认main。主仓库既有E16工作分支独立保留。

当前科学结论限于：D3v1冻结主比较未通过；D4-G开发回放净构造省时且保持输出；D4-W开发回放质量提高但CPU成本增加。新base等时间HV/IGD、空闲冷启动、guard×位置组合和文献创新判定仍待验证。新实验尚未启动。
