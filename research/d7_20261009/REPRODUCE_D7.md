# D7 正式结果与分析复现说明

1. 下载正式Release全部资产，按照SHA256SUMS验证每个资产。完整raw的manifest列出12434文件，所有相对path、bytes、SHA256及成员集合必须一致。
2. 将d7_complete_raw.tar.gz解压到新目录RAW，保留formal_node0/1和全部ops/quarantine历史。原数据不能在分析时改写；384技术pilot不混入。
3. git clone d7_runtime_f01b141.bundle并checkout f01b1417cf6be1cb5e1f375dbc710eb575c06123，安装dev依赖，将PYTHONPATH明确指向src。canonical_source_hash应为6e4385dd639db34aec25f526943eda84bb66bb9961ca12252ef138d7888da114。用原P2/D5 validate_complete、每个spec/input/init/config/source/seed/host和八项manifest验真。456272最终解是原workerexact证据，验hash不等于fresh全解重评。
4. git clone d7_analysis_source.bundle并检出source_identity.json记录的分析commit。安装dev依赖，用PYTHONPATH指向该checkout/src；执行python scripts/analyze_d7_results.py --data RAW --output NEW_ANALYSIS。输出应与完整分析ZIP中的report.json和CSV一致；这是固定结果后处理，未重跑在线wall-clock搜索。
5. frozen seed2026100907，32base内平均H/T和seed1101/2202，按50/100规模分层20k bootstrap、100k signflip；七HV主Holm与七IGD+次级Holm。完整anytime记录实际delay，不假设严格同刻。主程序中BASE/1101的IGD+还与独立项目实现交叉核对。
6. 完整ZIP中checks/保存验收/CI及ops/中独立统计与A3近零审计脚本。独立审计需显式调整输入根目录常量，不引入主分析实现；A3审计仅重建每块第一个初始genotype，不重跑完整算法，原source不修改。
7. A3当前收益包含冻结实现的近零tie扰动，不能把尚未修正的版本当纯电价因果证据。B10/SEQ/STATE84/RPERM与A3单项比较不能相加当组合提升。没有自动启动后续实验，默认main/F6不升级。
8. server/local原件保留。正式Release新建，不覆盖启动Release；最终发布CI、逐资产uploaded/size/SHA256 digest和抽样download回读以最终receipt为准。
