# D6 正式结果回收与完整性验收

用户本轮要求先回收服务器结果并保存到本地与 GitHub，暂不做质量分析。本交付仅报告数据完整性；HV、IGD+、覆盖率和统计检验尚未执行。

## 已验收数据

- 1024/1024 正式运行、256/256 同主机四组块，64 个全新 base；50/100 jobs 各32个，H/T、seed1101/2202、LEGACY/REFERENCE/W/GW。
- 两节点于2026-10-08北京时间19:37:55全部完成，08:56:19启动，计算约10小时42分钟；原supervisor与真实worker均退出，status/exit complete，failure/stop/OOM为0。
- 原P2/D5 validate_complete 在服务器与本地冻结源码入口均通过；每run八项产物、输入、初始化、配置、源码、seed及每块同host全部验证。
- 完整 12,562 文件、4,367,630,002 字节。tar每一成员与本地每一文件逐size/SHA256一致，目录计数没有代替验真。
- 原worker最终独立exact检查共545,341个解，其证据由完整产物hash绑定；本轮没有fresh重新评价全部解。

## 归档与版本

完整raw、manifest、checksums、冻结运行bundle、数据交付bundle和复现说明见 [D6正式原始数据Release](https://github.com/atlantic0423/geo-llm-scheduler/releases/tag/d6-formal-raw-20261008-3a1342c)；GitHub最终上传/回读/下载核验以Release附带的最终receipt为准。本文件先完成本地验收，发布后的核验单独追加，已发布资产保持不可变。

| 对象 | 值 |
|---|---|
| 运行commit | 3a1342cef98dae4b09cd370f40417a119efb4703 |
| canonical source hash | 592cf56992cfd0079f7dd9f3e581efdf01215af4cbe5f6d66ccba443cdd9e2d9 |
| raw tar大小 | 278,790,471 字节 |
| raw tar SHA256 | 5b513fa9ebd5e7359810da68131e03458ceb9dca0748a0cffb3cab792f148f6b |
| 完整manifest SHA256 | a43220ab5df8c0d1ec2e0e9d2c4d589ab9cc669869015f7e54bad1393474bded |

服务器和本地原件均保留。192次技术pilot在原启动Release单列，不混正式。数据交付使用独立分支，不合并默认算法，不安排新实验；质量分析应用户要求暂缓，归档完成后暂停原自动跟进。

复现入口见 [REPRODUCE_D6_RAW.md](REPRODUCE_D6_RAW.md)，机器可读验收摘要见 [data_integrity_summary.json](data_integrity_summary.json)。
