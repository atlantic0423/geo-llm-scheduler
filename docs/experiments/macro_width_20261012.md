# MacroSearch同池预算与真实成本：九小时诊断

2026-10-12用户在侧面会话接受本方案并明确授权开始。来源为D8 d5374fe；独立研究旁路，默认main/F6、已完成批次和初始化保持。

## 冻结研究语义

四组NATIVE6/POOL3/POOL6/POOL10。全部A3=tariff、Q42、原A8、五步轨迹、原Trigger/Polish/出生邻域替换。NATIVE6原生生成6；后三组采用10的生成规则，只送前3/6/10个不同完整候选进入统一exact。先构造全部候选，全部构造成本计入；不是惰性生成。相同冻结incumbent及算子RNG状态的回放才保证同池前缀，线上轨迹自然分叉。全局ideal只看线上实际评价的候选；余下候选不进入线上ideal/Archive/Q。

正式拟40全新独立base，50/100任务各20，base1525001–1525020/1535001–1535020，H/T和算法seed1112/2212，共640run、160四组块、320solver-hour。技术pilot另12base、192run，base1545001–1545006/1555001–1555006，seed6612/7712、30/60秒。pilot只决定工程/资源是否可运行，不按质量调参；正式最终选择40或预声明32base必须在首次正式派发前冻结。

六类压力按位置轮转：mixed/burst/KV×4/resource（compute×1.5截0.95、VRAM×2截15.5）/region-asymmetric（Region0仅一台其余三台）/demand×5。每规模各4/4/3/3/3/3base；名字只是输入干预，不保证生成对应状态。H/T共享所有非电价参数；H/T各按native输入生成原100成员混合，四组共用冻结genes。初始化基因生成在离线输入准备中，正式引擎时间包含初始化解码/exact、全部搜索、控制和同步observer；最终全量exact复验与导出另计，均单列wall。

当前两host b518ece5cd01/878be0fed7b0，各8GiB、共享物理128逻辑CPU。拟16/32solver，根据同步预跑确认；按base位置mod3分14/26base，同base的HT/两seed/全部组同host。四阶Williams顺序：0123中的(0,1,3,2)循环平移，每base四个HT/seed块覆盖四顺序。每块内部按该顺序；每run完成将块重新排队，避免整块占用导致长尾。

## 观测与离线补查

所有组同样开启research observer。每run按动作×三个wall-progress阶段各保留一个均匀reservoir，最多24调用；独立seed异或2026101209的RNG，不消费算法RNG，不强行填空桶。记录冻结incumbent/MS/OS/starts/waits、初始ideal与population maximum、权重、完整batch、算子RNG前后态、线上exact结果、构造时间及当时Archive目标点。保存的候选本身不被observer额外评价；开销计入在线总wall。

阶段内的reservoir只代表该动作/阶段的线上调用分布；不同run、动作或阶段不直接合并为独立样本。无访问桶报告缺失。离线独立重新生成原batch，验证proposal身份、attempts与RNG后态一致；再从同一incumbent及RNG前态构造10，统一exact补查。所有补查只在独立gateway进行，不反馈线上轨迹。共同10候选ideal下比较prefix局部收益；另外精确回放旧SEQ分阶段的动态归一化停止判断。Archive目标点仅作objective-envelope诊断，不宣称完整phenotype Archive因果净保留。

成本区分构造、SSGS、exact、Archive、observer、budget和导出。实际每调用插入/净保留汇总独立记录，插入不是最终贡献。压缩archive/population保存完整原JSON字节的SHA与size，解压逐字节round-trip；其他证据与快照各自SHA绑定。压缩是存储格式，不裁剪Archive。

## 统计与判定

主对照POOL3−POOL6、POOL10−POOL6、POOL6−NATIVE6的HV/IGD+共六项联合Holm。40base独立，先平均HT与两seed，按规模分层20k配对bootstrap、100k signflip，seed2026101209。共同经验非支配union作reference，不称true Pareto；所有组/seed等规则。anytime、吞吐、端点、成本及候选追加价值为诊断。未显著不是等价，不事后加样、挑状态或改停止阈值。预测如需训练按base隔离，不能把同base调用随机拆训练/测试；本轮不部署控制器。

## 运维与完成门

源码commit/canonical、全部输入/配置/初始化/hash与host冻结。各root排他锁，完成产物验真后才复用；意外中断保留attempt/quarantine，任务级续跑不宣称内部搜索状态恢复。真实失败留证据并停新派发，不自动算法重试；另一shard同父stop.request安全收尾。新用户暂停优先。

nice10/单线程依赖、256MiB每child虚拟内存上限，1.5GiB空余内存派发门、100GiB台账额度保留2GiB。不修改其他项目/全局cache，不删旧数据。正式共九小时窗口；在deadline前2700秒停止派发未开始run，已运行安全收尾；未完成必须如实标待续跑，不能改预算凑验收。正式全部640/160的输入/初始化/source/config/seed/九项产物/hash、每base同host、原worker最终独立exact证据和后续离线replay完整绑定，实际worker/supervisor0及exitcomplete才能验收。阶段计算与交付/质量分开。

当前尚未正式派发，校准后的精确并发、矩阵、启动证据与ETA写专用receipt及Notion；无Codex定时跟进。
