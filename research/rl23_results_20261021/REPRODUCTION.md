# RL23结果复现说明

本报告分析23组正式1472次，16开发父、两电价变体、两算法seed。92次技术pilot另列，不混入正式。完整原始结果保留本地与服务器，原运行源码不改。

## 依赖与完整性

运行源码commit为27198e88ae0f670f29c9f15caa2fe4af3bb196c6，分析源码另一个独立交付分支。完整source bundle在启动Release，完整分析bundle与结果在本结果Release；先git clone、fsck和指定checkout。安装pyproject.toml开发依赖；明确将此检出src的绝对路径加入PYTHONPATH，避免误用其他editable源码。

重建RL23_FORMAL_RAW.tar.gz时，按RAW_RELEASE_MANIFEST.json的有序18分卷逐卷核对size/SHA再拼接，完整tar核对size/SHA。拒绝绝对路径、上级目录、符号链接和重复成员，按SERVER_VALIDATION_AND_MANIFEST.json验证全部19308成员；不是按目录数验收。原始运行complete.json还绑定九项产物、初始化、输入/spec/source/host。

分析程序读取结果目录中的LOCAL_RECOVERY_ACCEPTANCE.json、SERVER_VALIDATION_AND_MANIFEST.json及raw/formal。首次分析会重新计算完整清单全部文件SHA。它不重新运行算法，也不fresh重评所有最终候选；原worker1248896最终成员独立exact证据由验真原产物绑定。

## 复现命令

在本分析Git检出的根目录，用数据目录与输出目录替换下列占位参数：

```text
python research/rl23_results_20261021/analyze_results.py --data <RESULTS> --output <ANALYSIS>
python research/rl23_results_20261021/audit_results.py --data <RESULTS> --output <ANALYSIS>
python research/rl23_results_20261021/plot_results.py --output <ANALYSIS> --figures <FIGURES>
```

完整原输入面板及算法配置都来自原component33，未换算例。指标归一化使用每原输入、全部23臂和两seed最终Archive的共同经验ND union之ideal/nadir；这是经验参照，不是true Pareto。HV参考(1.1,1.1)，超出矩形点贡献0；IGD+低为好。P100、初始P100和初始Archive各自单列。

## 统计口径

16父为独立单位；每父两变体×两seed均值，四规模各4父等权。20000分规模配对bootstrap、100000双侧配对signflip，根seed2026102101加比较名/指标SHA派生种子。四个预定Holm家族为主32/Polish12/普通底座8/组合4项，共56项。保存逐父差与完整p值、边际CI及效应；边际CI不是同时区间。独立audit重新从run CSV计算parent均值和全部统计，并以65536种符号模式核查精确p，不用它替换预定Monte Carlo主p。未显著不是等价，分规模与未预定比较仅描述。

步骤/动作收益以实际到访上下文为条件，不能视为同一父解随机分配八动作的因果收益。前缀诊断使用同次MacroSearch保存候选的冻结比较context/权重，只估计已观察候选的边际标量改善；不代表截断后整程最终质量。TIME42须按数值state和wall_progress解码，原通用search_progress文字标签未改；TIME84保留原42并额外编码进度，与旧D8定义不同。

## 耗时与边界

初始化、LS已测构造/SSGS/exact/Archive/预算分量、Polish、外层评价及剩余工作组成互斥分解。整步action_seconds包含前者LS分量及支持，不能把它们相加。剩余工作还含外层变异/替换/观察和其他支持，不能叫“Q-learning耗时”。最终exact审计、导出另计，不计入生产wall。

固定前沿图选择每规模最早父、variant0/seed16101，23组各四面板；没有按最好结果挑实例。PNG/PDF为真实存储点散点，未把连接线当可行插值。默认算法不因本开发筛选自动晋级，也没有自动启动下一轮实验。
